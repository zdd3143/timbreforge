"""Persistent state for heavy model tasks.

The queue is a small JSON document so a refresh, crash or CLI/UI hand-off can
show what happened. It intentionally stores no audio paths that could be
mistaken for durable results: results remain in their generation records, and
this file only tracks lifecycle, errors and cleanup.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import uuid
from datetime import datetime, timezone

from timbreforge.config import get_config

SCHEMA_VERSION = 1
STATES = ('waiting', 'queued', 'preparing', 'generating', 'saving', 'cleaning',
          'completed', 'failed', 'cancelled')
ACTIVE_STATES = frozenset({'waiting', 'queued', 'preparing', 'generating', 'saving', 'cleaning'})
TERMINAL_STATES = frozenset({'completed', 'failed', 'cancelled'})
ALLOWED = {
    'waiting': {'queued', 'cancelled', 'failed'},
    'queued': {'preparing', 'cancelled', 'failed'},
    'preparing': {'generating', 'cancelled', 'failed'},
    'generating': {'saving', 'cancelled', 'failed'},
    'saving': {'cleaning', 'failed', 'cancelled'},
    'cleaning': {'completed', 'failed', 'cancelled'},
    'completed': set(),
    'failed': set(),
    'cancelled': set(),
}
MAX_JOBS = 200
_UNSET = object()
_LOCK = threading.Lock()


class JobQueueError(ValueError):
    """Raised when a job record or transition is invalid."""


def _now():
    return datetime.now(timezone.utc).isoformat()


class JobQueue:
    """Validated, atomically written task records in one JSON document."""

    def __init__(self, root=None):
        self.root = Path(root if root is not None else Path(get_config().outputs_dir) / 'jobs').expanduser().resolve()

    @property
    def path(self):
        return self.root / 'jobs.json'

    # ---------------------------------------------------------------- storage
    def _read(self):
        path = self.path
        if path.is_symlink():
            raise JobQueueError('任务记录指向了不安全的位置。')
        if not path.is_file():
            return {'schema_version': SCHEMA_VERSION, 'jobs': []}
        if path.stat().st_size > 8 * 1024 * 1024:
            raise JobQueueError('任务记录异常过大，已拒绝读取。')
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, UnicodeError, ValueError) as exc:
            raise JobQueueError(f'任务记录损坏：{exc}') from exc
        if not isinstance(data, dict) or data.get('schema_version') != SCHEMA_VERSION:
            raise JobQueueError('任务记录版本不兼容。')
        jobs = data.get('jobs')
        if not isinstance(jobs, list):
            raise JobQueueError('任务记录结构无效。')
        return {'schema_version': SCHEMA_VERSION, 'jobs': jobs}

    def _write(self, data):
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name('.' + uuid.uuid4().hex + '.tmp')
        try:
            with temporary.open('x', encoding='utf-8') as stream:
                json.dump(data, stream, ensure_ascii=False, indent=2, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    # -------------------------------------------------------------- operations
    def create(self, kind, *, role_id='', payload=None, priority=0):
        if not isinstance(kind, str) or not 1 <= len(kind) <= 40:
            raise JobQueueError('任务类型必须是不超过 40 字的文本。')
        if not isinstance(role_id, str) or len(role_id) > 64:
            raise JobQueueError('角色 ID 无效。')
        if type(priority) is not int or not -10 <= priority <= 10:
            raise JobQueueError('任务优先级必须是 -10 到 10 的整数。')
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            raise JobQueueError('任务参数必须是对象。')
        try:
            json.dumps(payload, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise JobQueueError(f'任务参数必须是可序列化的 JSON：{exc}') from exc
        with _LOCK:
            data = self._read()
            now = _now()
            job = {
                'id': uuid.uuid4().hex,
                'schema_version': SCHEMA_VERSION,
                'kind': kind,
                'role_id': role_id,
                'priority': priority,
                'state': 'waiting',
                'message': '等待调度',
                'error': '',
                'cancel_requested': False,
                'payload': payload,
                'result': None,
                'created_at': now,
                'updated_at': now,
                'started_at': None,
                'finished_at': None,
            }
            data['jobs'].append(job)
            self._prune(data)
            self._write(data)
            return job

    def _prune(self, data):
        if len(data['jobs']) <= MAX_JOBS:
            return
        keep, terminal = [], []
        for job in data['jobs']:
            if isinstance(job, dict) and job.get('state') in TERMINAL_STATES:
                terminal.append(job)
            else:
                keep.append(job)
        terminal.sort(key=lambda item: item.get('updated_at', ''))
        keep.extend(terminal[-(MAX_JOBS - len(keep)):])
        data['jobs'] = keep

    def get(self, job_id):
        if not isinstance(job_id, str) or not job_id:
            raise JobQueueError('任务 ID 无效。')
        with _LOCK:
            for job in self._read()['jobs']:
                if isinstance(job, dict) and job.get('id') == job_id:
                    return job
        raise JobQueueError('找不到该任务记录。')

    def list(self, *, limit=50):
        if type(limit) is not int or not 1 <= limit <= 200:
            raise JobQueueError('limit 必须在 1–200 之间。')
        jobs = [job for job in self._read()['jobs'] if isinstance(job, dict)]
        jobs.sort(key=lambda item: item.get('created_at', ''), reverse=True)
        return jobs[:limit]

    def transition(self, job_id, state, *, message=None, error=None, result=_UNSET):
        if state not in STATES:
            raise JobQueueError(f'未知任务状态：{state!r}。')
        with _LOCK:
            data = self._read()
            job = None
            for item in data['jobs']:
                if isinstance(item, dict) and item.get('id') == job_id:
                    job = item
                    break
            if job is None:
                raise JobQueueError('找不到该任务记录。')
            current = job.get('state')
            if state == current:
                if message is not None:
                    job['message'] = str(message)[:500]
                    job['updated_at'] = _now()
                    self._write(data)
                return job
            if state not in ALLOWED.get(current, set()):
                raise JobQueueError(f'任务状态不能从 {current} 变为 {state}。')
            if message is not None:
                job['message'] = str(message)[:500]
            if error is not None:
                job['error'] = str(error)[:4000]
            if result is not _UNSET:
                job['result'] = result
            job['state'] = state
            job['updated_at'] = _now()
            if state in ('preparing', 'generating', 'saving', 'cleaning') and not job.get('started_at'):
                job['started_at'] = job['updated_at']
            if state in TERMINAL_STATES:
                job['finished_at'] = job['updated_at']
                if state == 'cancelled':
                    job['message'] = job.get('message') or '任务已取消'
            self._write(data)
            return job

    def request_cancel(self, job_id):
        with _LOCK:
            data = self._read()
            for job in data['jobs']:
                if not isinstance(job, dict) or job.get('id') != job_id:
                    continue
                if job.get('state') in TERMINAL_STATES:
                    return job
                job['cancel_requested'] = True
                job['message'] = '已请求取消，等待释放资源'
                job['updated_at'] = _now()
                self._write(data)
                return job
        raise JobQueueError('找不到该任务记录。')

    def cancel_requested(self, job_id):
        return bool(self.get(job_id).get('cancel_requested'))

    def recover_interrupted(self):
        """Mark jobs left running by a previous process as failed."""
        with _LOCK:
            data = self._read()
            recovered = []
            now = _now()
            for job in data['jobs']:
                if isinstance(job, dict) and job.get('state') in ACTIVE_STATES:
                    previous = job.get('state')
                    job['state'] = 'failed'
                    job['error'] = (f'应用上次退出时任务仍处于 {previous}；'
                                    '已停止模型进程并清理临时文件，请重新生成。')
                    job['message'] = '上次运行被中断'
                    job['updated_at'] = now
                    job['finished_at'] = now
                    recovered.append(job['id'])
            if recovered:
                self._write(data)
            return recovered

