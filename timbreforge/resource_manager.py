"""Serial heavy-task execution, guaranteed cleanup and memory-pressure release.

Only one heavy model task runs at a time. Every task owns a private workspace
that is removed in ``finally``; a failed or cancelled task also asks the
resource layer to release idle models. Release order intentionally never touches
audio that is currently being generated or auditioned.
"""
from __future__ import annotations

import ctypes
import gc
import os
from pathlib import Path
import queue as _queue
import shutil
import threading
import time

from timbreforge.config import get_config
from timbreforge.job_queue import JobQueue, JobQueueError, TERMINAL_STATES
from timbreforge.resources import is_memory_error, release_idle_models

ACTIVE = ('waiting', 'queued', 'preparing', 'generating', 'saving', 'cleaning')
_RELEASE_PAGE_CACHE = 10
_RELEASE_HISTORY_PREVIEW = 20
_RELEASE_IDLE_MODELS = 30
_MISSING = object()


class TaskCancelled(RuntimeError):
    """Raised inside a task when cancellation or shutdown was requested."""


class TaskContext:
    """Handed to a task function: workspace, cancellation and progress."""

    def __init__(self, manager, job, workspace):
        self.manager = manager
        self.job = job
        self.job_id = job['id']
        self.workspace = workspace
        self.cancel_event = threading.Event()
        self._lock = threading.Lock()

    @property
    def cancelled(self):
        return self.cancel_event.is_set()

    def raise_if_cancelled(self):
        if self.cancel_event.is_set():
            raise TaskCancelled('任务已取消')

    def step(self, state, message=''):
        """Advance the persisted lifecycle; ordering rules live in JobQueue."""
        self.manager._transition(self.job_id, state, message=message or None)
        return state

    def progress(self, fraction, message=''):
        if not isinstance(fraction, (int, float)) or not 0 <= float(fraction) <= 1:
            raise ValueError('progress fraction must be between 0 and 1')
        if message:
            self.manager._transition(self.job_id, self.manager._state(self.job_id),
                                     message=f'{message} ({float(fraction) * 100:.0f}%)')


class ResourceManager:
    """One-worker scheduler used by every heavy generation/training entry."""

    def __init__(self, *, jobs=None, workspace_root=None):
        self.jobs = jobs or JobQueue()
        self.workspace_root = Path(
            workspace_root if workspace_root is not None
            else Path(get_config().outputs_dir) / 'jobs').expanduser().resolve()
        self._queue = _queue.PriorityQueue()
        self._lock = threading.Lock()
        self._thread = None
        self._stop = threading.Event()
        self._sequence = 0
        self._contexts = {}
        self._tasks = {}
        self._recovered = self.jobs.recover_interrupted()

    # ------------------------------------------------------------------ submit
    def submit(self, kind, task, *, role_id='', priority=0, payload=None):
        """Create a persisted job and run ``task(context)`` when a slot is free."""
        if not callable(task):
            raise TypeError('task must be callable')
        job = self.jobs.create(kind, role_id=role_id, payload=payload or {}, priority=priority)
        with self._lock:
            self._tasks[job['id']] = task
            self._sequence += 1
            self._ensure_thread()
            self._queue.put((-int(priority), self._sequence, job['id']))
        return job

    def _ensure_thread(self):
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._work, name='timbreforge-heavy-task',
                                            daemon=True)
            self._thread.start()

    # -------------------------------------------------------------------- cancel
    def cancel(self, job_id):
        job = self.jobs.request_cancel(job_id)
        with self._lock:
            context = self._contexts.get(job_id)
        if context is not None:
            context.cancel_event.set()
        return job

    # ------------------------------------------------------------------ queries
    def get(self, job_id):
        return self.jobs.get(job_id)

    def list(self, limit=50):
        return self.jobs.list(limit=limit)

    def wait(self, job_id, timeout=30.0):
        """Block until a job reaches a terminal state (used by CLI and tests)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = self.jobs.get(job_id)
            if job['state'] in TERMINAL_STATES:
                return job
            time.sleep(.02)
        raise TimeoutError(f'任务未在 {timeout:g}s 内结束。')

    # ------------------------------------------------------------------- worker
    def _work(self):
        while not self._stop.is_set():
            try:
                _priority, _sequence, job_id = self._queue.get(timeout=.2)
            except _queue.Empty:
                continue
            try:
                self._execute(job_id)
            except JobQueueError:
                # A record removed or corrupted externally must not kill the worker.
                continue
            finally:
                self._queue.task_done()

    def _execute(self, job_id):
        try:
            job = self.jobs.get(job_id)
        except JobQueueError:
            return
        if job['state'] in TERMINAL_STATES:
            return
        self._transition(job_id, 'queued', message='已排队，等待模型资源')
        if job.get('cancel_requested'):
            self._transition(job_id, 'cancelled', message='排队时已取消')
            return
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        workspace = self.workspace_root / job_id
        context = TaskContext(self, job, workspace)
        if workspace.exists():
            shutil.rmtree(workspace, ignore_errors=True)
        workspace.mkdir(parents=True, exist_ok=False)
        with self._lock:
            self._contexts[job_id] = context
        result = None
        try:
            self._transition(job_id, 'preparing', message='准备模型与临时文件')
            self._transition(job_id, 'generating', message='生成中')
            result = self._invoke(context)
            if context.cancelled:
                raise TaskCancelled('任务已取消')
            state = self._state(job_id)
            if state == 'preparing':
                self._transition(job_id, 'generating', message='生成中')
                state = 'generating'
            if state == 'generating':
                self._transition(job_id, 'saving', message='保存生成结果')
                state = 'saving'
            if state == 'saving':
                self._transition(job_id, 'cleaning', message='清理临时资源')
            # Cleanup happens before the terminal state is published, so "failed"
            # and "completed" always mean the temporary workspace is already gone.
            self._cleanup_workspace(workspace)
            self._transition(job_id, 'completed', message='完成', result=_jsonable(result))
        except TaskCancelled as exc:
            self._cleanup_workspace(workspace)
            self._transition(job_id, 'cancelled', message=str(exc) or '任务已取消')
        except Exception as exc:  # noqa: BLE001 - persisted, user-facing failure
            detail = _failure_message(exc)
            try:
                state = self._state(job_id)
                if state in TERMINAL_STATES:
                    return
                # Only states reachable from the current one are used, so a
                # failure during preparation is not forced through saving.
                if state == 'generating':
                    self._transition(job_id, 'saving', message='生成失败，记录错误')
                    state = 'saving'
                if state == 'saving':
                    self._transition(job_id, 'cleaning', message='生成失败，清理临时资源')
                self._cleanup_workspace(workspace)
                self._transition(job_id, 'failed', message='失败', error=detail)
            except JobQueueError:
                pass
        finally:
            with self._lock:
                self._contexts.pop(job_id, None)
                self._tasks.pop(job_id, None)
            self._cleanup_workspace(workspace)
            self._release_idle_after_task()

    @staticmethod
    def _cleanup_workspace(workspace):
        """Idempotent removal of a task's private temporary directory."""
        shutil.rmtree(workspace, ignore_errors=True)
        gc.collect()

    def _invoke(self, context):
        with self._lock:
            callable_task = self._tasks.get(context.job_id)
        if callable_task is None:
            raise RuntimeError('任务函数丢失，无法继续；已清理临时文件。')
        return callable_task(context)

    # --------------------------------------------------------------- transitions
    def _state(self, job_id):
        return self.jobs.get(job_id)['state']

    def _transition(self, job_id, state, *, message=None, error=None, result=_MISSING):
        kwargs = dict(message=message, error=error)
        if result is not _MISSING:
            kwargs['result'] = result
        return self.jobs.transition(job_id, state, **kwargs)

    def _release_idle_after_task(self):
        with self._lock:
            if self._contexts:
                return
        try:
            release_idle_models()
        except (OSError, RuntimeError, ValueError):
            # Releasing idle models is best effort; a busy ComfyUI must not turn a
            # completed generation into a failure.
            pass

    def shutdown(self, wait=True, timeout=10.0):
        self._stop.set()
        with self._lock:
            contexts = list(self._contexts.values())
        for context in contexts:
            context.cancel_event.set()
        thread = self._thread
        if thread is not None and wait:
            thread.join(timeout=timeout)


def _jsonable(value):
    """Keep task results JSON-safe; never let a result break the job log."""
    import json
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
        return value
    except (TypeError, ValueError):
        return {'summary': str(value)[:1000]}


def _failure_message(exc):
    if is_memory_error(exc):
        return ('内存或显存不足：已停止模型进程并清理临时文件。请关闭其它占用显存的程序、'
                '缩短单次台词或改用更小的模型后重试。')
    if isinstance(exc, FileNotFoundError):
        return f'本地文件缺失：{exc}。请检查设置中的模型路径；本工具不会自动下载模型。'
    return f'{type(exc).__name__}: {exc}；已清理临时文件，可修正后重试。'


# ------------------------------------------------------------ memory pressure
_CLEANUP_LOCK = threading.Lock()
_CLEANUPS = {}


def register_cleanup(name, priority, callback):
    """Register a low-priority releasable resource.

    Priorities follow the documented release order: 10 = UI/page caches,
    20 = historical previews. Active generation and the current audition are
    never registered here and therefore never released by pressure.
    """
    if not isinstance(name, str) or not name:
        raise ValueError('cleanup name must be a nonempty string')
    if type(priority) is not int or not 1 <= priority <= 100:
        raise ValueError('cleanup priority must be an integer in 1-100')
    if not callable(callback):
        raise TypeError('cleanup callback must be callable')
    with _CLEANUP_LOCK:
        _CLEANUPS[name] = (priority, callback)


def unregister_cleanup(name):
    with _CLEANUP_LOCK:
        _CLEANUPS.pop(name, None)


def clear_cleanup_callbacks():
    """Test helper: drop every registered low-priority cleanup."""
    with _CLEANUP_LOCK:
        _CLEANUPS.clear()


def release_low_priority_resources(*, include_idle_models=True):
    """Release page caches, then history previews, then idle models."""
    released = []
    with _CLEANUP_LOCK:
        callbacks = sorted(((priority, name, callback)
                            for name, (priority, callback) in _CLEANUPS.items()),
                           key=lambda item: (item[0], item[1]))
    for _priority, name, callback in callbacks:
        try:
            callback()
            released.append(name)
        except Exception:  # noqa: BLE001 - cleanup must never mask the real error
            continue
    if include_idle_models:
        try:
            release_idle_models()
            released.append('idle-models')
        except (OSError, RuntimeError, ValueError):
            pass
    gc.collect()
    return released


def available_memory_bytes():
    """Physical memory available to new allocations, or None when unknown."""
    if os.name == 'nt':
        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                        ('ullTotalPhys', ctypes.c_ulonglong),
                        ('ullAvailPhys', ctypes.c_ulonglong),
                        ('ullTotalPageFile', ctypes.c_ulonglong),
                        ('ullAvailPageFile', ctypes.c_ulonglong),
                        ('ullTotalVirtual', ctypes.c_ulonglong),
                        ('ullAvailVirtual', ctypes.c_ulonglong),
                        ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        try:
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return None
        except (AttributeError, OSError):
            return None
        return int(status.ullAvailPhys)
    try:
        with open('/proc/meminfo', encoding='ascii') as stream:
            for line in stream:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def memory_snapshot():
    available = available_memory_bytes()
    total = None
    if os.name == 'nt':
        try:
            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                            ('ullTotalPhys', ctypes.c_ulonglong),
                            ('ullAvailPhys', ctypes.c_ulonglong),
                            ('ullTotalPageFile', ctypes.c_ulonglong),
                            ('ullAvailPageFile', ctypes.c_ulonglong),
                            ('ullTotalVirtual', ctypes.c_ulonglong),
                            ('ullAvailVirtual', ctypes.c_ulonglong),
                            ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                total = int(status.ullTotalPhys)
        except (AttributeError, OSError):
            total = None
    elif os.path.exists('/proc/meminfo'):
        try:
            with open('/proc/meminfo', encoding='ascii') as stream:
                for line in stream:
                    if line.startswith('MemTotal:'):
                        total = int(line.split()[1]) * 1024
                        break
        except (OSError, ValueError, IndexError):
            total = None
    return {'available_bytes': available, 'total_bytes': total,
            'percent_used': (None if not available or not total
                             else round(100 * (1 - available / total), 1))}


def ensure_memory(required_bytes, *, stage='生成', include_idle_models=True):
    """Free low-priority resources until ``required_bytes`` is available."""
    if type(required_bytes) is not int or required_bytes <= 0:
        raise ValueError('required_bytes must be a positive integer')
    available = available_memory_bytes()
    report = {'ok': True, 'available_bytes': available, 'required_bytes': required_bytes,
              'released': [], 'note': ''}
    if available is None:
        report['note'] = '无法读取系统可用内存；跳过预检查，失败时会自动清理。'
        return report
    attempts = 0
    while available < required_bytes and attempts < 4:
        attempts += 1
        released = release_low_priority_resources(include_idle_models=include_idle_models)
        new_available = available_memory_bytes() or available
        report['released'].extend(released)
        if new_available <= available and not released:
            break
        available = new_available
    report['available_bytes'] = available
    if available < required_bytes:
        report['ok'] = False
        raise MemoryError(
            f'{stage}需要约 {required_bytes / 1024 ** 3:.1f} GiB 可用内存，'
            f'当前只有 {available / 1024 ** 3:.1f} GiB。已尝试释放页面缓存、历史预览和空闲模型；'
            '请关闭其它应用、缩短单次台词或改用更小的模型后重试。')
    return report

