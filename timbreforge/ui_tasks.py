"""Thin UI adapter over the serial heavy-task resource manager.

Tabs must never start a model while being built. They call :func:`run_heavy`
only in response to an explicit button click, which keeps one GPU task at a time
and gives every run a persisted job record plus guaranteed cleanup.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import threading

from timbreforge.config import get_config
from timbreforge.resource_manager import ResourceManager, memory_snapshot

_MANAGER = None
_MANAGER_LOCK = threading.Lock()


def get_manager():
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = ResourceManager()
        return _MANAGER


def run_heavy(kind, task, *, role_id='', payload=None, timeout=1800.0):
    """Run ``task(context)`` serially and return its serialized result.

    Raises ``RuntimeError`` with the persisted Chinese error when the task fails.
    """
    manager = get_manager()
    job = manager.submit(kind, task, role_id=role_id, payload=payload or {})
    finished = manager.wait(job['id'], timeout=timeout)
    if finished['state'] == 'completed':
        return finished.get('result')
    if finished['state'] == 'cancelled':
        raise RuntimeError('任务已取消；模型进程与临时文件已清理。')
    raise RuntimeError(finished.get('error') or '任务失败，已清理临时资源。')


def cancel_job(job_id):
    return get_manager().cancel(job_id)


def recent_jobs(limit=20):
    try:
        return get_manager().list(limit=limit)
    except (OSError, ValueError, TypeError):
        return []


def memory_summary():
    """Chinese one-line memory status for the settings page."""
    try:
        snapshot = memory_snapshot()
    except (OSError, AttributeError, ValueError):
        return '无法读取系统内存状态。'
    available = snapshot.get('available_bytes')
    total = snapshot.get('total_bytes')
    if not available:
        return '无法读取系统可用内存；生成失败时会自动清理临时资源。'
    text = f'可用内存 {available / 1024 ** 3:.1f} GiB'
    if total:
        text += f' / 共 {total / 1024 ** 3:.1f} GiB（已用 {snapshot.get("percent_used")}%）'
    return text


def job_rows(limit=20):
    rows = []
    for job in recent_jobs(limit):
        rows.append([job.get('kind', ''), job.get('state', ''), job.get('role_id', '')[:8],
                     (job.get('message') or '')[:60], (job.get('error') or '')[:120],
                     (job.get('created_at') or '')[:19]])
    return rows


def clear_old_previews(keep_per_role=1):
    """Delete disposable preview copies, keeping the newest one per character."""
    from timbreforge.audio_preview import cleanup_previews
    from timbreforge.characters import CharacterStore
    removed = 0
    for card in CharacterStore().list():
        directory = Path(card['reference_audio']).resolve().parent / 'audio' / 'previews'
        removed += cleanup_previews(directory, keep=keep_per_role)
    return removed


def clear_staging():
    from timbreforge.generation import cleanup_generation_staging
    return cleanup_generation_staging()


def clear_ephemeral_data(*, outputs_dir=None, characters_dir=None):
    """Remove all non-character records from one runtime output directory.

    Character cards live in ``characters_dir`` and are deliberately outside
    this operation.  The guard refuses to clean an output root that contains
    the character root, so a misconfigured TOML file cannot delete saved roles.
    The output root itself is retained and recreated on the next run.
    """
    cfg = get_config()
    output_root = Path(outputs_dir or cfg.outputs_dir).expanduser().resolve()
    character_root = Path(characters_dir or cfg.characters_dir).expanduser().resolve()
    try:
        character_is_inside = character_root == output_root or character_root.is_relative_to(output_root)
    except AttributeError:  # Python 3.11+ has is_relative_to; keep the guard explicit.
        character_is_inside = character_root == output_root or str(character_root).startswith(str(output_root) + '\\')
    if character_is_inside:
        raise RuntimeError(
            f'拒绝清理：角色库目录 {character_root} 位于输出目录 {output_root} 内。'
            '请先分离 [paths] outputs_dir 与 characters_dir。')
    if not output_root.exists():
        return {'removed': 0, 'failed': [], 'root': str(output_root)}
    removed = 0
    failed = []
    for child in list(output_root.iterdir()):
        try:
            if child.is_symlink() or child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink(missing_ok=True)
            removed += 1
        except (OSError, RuntimeError) as exc:
            failed.append({'path': str(child), 'error': str(exc)[:300]})
    return {'removed': removed, 'failed': failed, 'root': str(output_root)}


def shutdown_runtime(*, clear_records=True):
    """Stop heavy work, release models, then clear disposable runtime records."""
    manager = _MANAGER
    if manager is not None:
        manager.shutdown(wait=True, timeout=30.0)
    try:
        from timbreforge.resource_manager import release_low_priority_resources
        release_low_priority_resources(include_idle_models=True)
    except (OSError, RuntimeError, ValueError):
        pass
    if not clear_records:
        return {'removed': 0, 'failed': [], 'root': ''}
    return clear_ephemeral_data()


_REGISTERED = False


def register_default_cleanups():
    """Register low-priority releasable resources exactly once per process."""
    global _REGISTERED
    if _REGISTERED:
        return
    from timbreforge.resource_manager import register_cleanup
    register_cleanup('history-previews', 20, lambda: clear_old_previews(1))
    register_cleanup('generation-staging', 15, lambda: clear_staging())
    _REGISTERED = True

