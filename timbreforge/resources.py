"""Serialize local model workers across studio/CLI processes without importing torch."""
from __future__ import annotations

from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse
import uuid

from timbreforge.config import get_config

_LOCK = threading.Lock()


def is_memory_error(message):
    text = str(message).lower()
    return any(word in text for word in ('out of memory', 'outofmemory', 'memoryerror',
        'not enough memory', 'cannot allocate memory', '1455', '页面文件', '显存不足', '内存不足',
        'c0000017', '-1073741801'))


def _comfy(base, endpoint, data=None):
    request = urllib.request.Request(base + endpoint,
        data=json.dumps(data).encode('utf-8') if data is not None else None,
        headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request, timeout=3) as response:
        raw = response.read()
    return json.loads(raw) if raw else {}


def release_idle_models(base=None):
    """Release configured localhost Comfy caches, never interrupt jobs or kill apps."""
    base = (base or get_config().comfy_url).rstrip('/')
    if urlparse(base).hostname not in ('localhost', '127.0.0.1', '::1'):
        return '非本机 ComfyUI，未自动释放。'
    try:
        queue = _comfy(base, '/queue')
    except (OSError, ValueError):
        return 'ComfyUI 未运行，无驻留模型需要释放。'
    if queue.get('queue_running') or queue.get('queue_pending'):
        raise RuntimeError('ComfyUI 仍有任务运行或排队，请等待完成后重试；未中断现有任务。')
    _comfy(base, '/free', {'unload_models':True, 'free_memory':True})
    # A custom IndexTTS cache is outside Comfy's model manager. Execute its own
    # unload node through a real output, then clear executor references as well.
    schema = _comfy(base, '/object_info/IndexTTS25Unload')
    if schema.get('IndexTTS25Unload'):
        graph = {'1': {'class_type':'IndexTTS25Unload', 'inputs':{'unload':True}},
                 '2': {'class_type':'PreviewAny', 'inputs':{'source':['1',0]}}}
        # /free is asynchronous: it is processed before or just after the first
        # queued graph. A second barrier guarantees unload isn't satisfied from
        # an old node cache, even when /free was processed after the first graph.
        for _ in range(2):
            submitted = _comfy(base, '/prompt', {'prompt':graph, 'client_id':'timbreforge-release-' + uuid.uuid4().hex})
            prompt_id = submitted.get('prompt_id')
            if not prompt_id:
                raise RuntimeError('无法提交闲置 IndexTTS 模型释放任务。')
            deadline = time.monotonic() + 20
            while True:
                entry = _comfy(base, '/history/' + prompt_id).get(prompt_id)
                if entry:
                    if entry.get('status', {}).get('status_str') == 'error':
                        raise RuntimeError('IndexTTS 闲置模型释放失败，请查看 ComfyUI 日志。')
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError('等待闲置模型释放超时，请等待 ComfyUI 完成后重试。')
                time.sleep(.2)
    _comfy(base, '/free', {'unload_models':True, 'free_memory':True})
    return '已请求释放闲置 ComfyUI / IndexTTS 模型和执行缓存。'


@contextmanager
def model_slot(output_root=None, *, release_comfy=True, timeout=1800):
    """One active model per output workspace, including independent CLI processes.

    OS file locks are automatically released on process exit. The worker caller
    must wait for its child to exit before leaving this scope (subprocess.run does).
    """
    root = Path(output_root or get_config().outputs_dir)
    root.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    if not _LOCK.acquire(timeout=timeout):
        raise TimeoutError('等待其他模型任务释放内存超时。')
    try:
        with (root / '.model-resource.lock').open('a+b') as stream:
            if stream.seek(0, os.SEEK_END) == 0:
                stream.write(b'0')
                stream.flush()
            locked = False
            try:
                while not locked:
                    try:
                        stream.seek(0)
                        if os.name == 'nt':
                            import msvcrt
                            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                        else:
                            import fcntl
                            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        locked = True
                    except (BlockingIOError, PermissionError):
                        if time.monotonic() >= deadline:
                            raise TimeoutError('其他工作室或命令行模型仍在使用内存，请稍后重试。')
                        time.sleep(.1)
                gc.collect()
                if release_comfy:
                    release_idle_models()
                yield
            finally:
                if locked:
                    stream.seek(0)
                    if os.name == 'nt':
                        import msvcrt
                        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
                gc.collect()
    finally:
        _LOCK.release()
