"""Reusable Qwen Clone Prompt cache keyed by every input it depends on.

A cached prompt stores only Qwen's conditioning tensors. The card remains the
single owner of audio, transcript, model identity and edit history; the cache
metadata repeats their hashes so every reuse can prove it still matches. Any
change to audio, transcript, model version, clone mode or the audio edit chain
makes the entry stale and the caller falls back to an ordinary clone.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import uuid
from datetime import datetime, timezone

from timbreforge import backend_qwen
from timbreforge.characters import CharacterStore, audio_path
from timbreforge.config import get_config

SCHEMA_VERSION = 1
PROMPT_DIR = 'prompts'
METADATA_NAME = 'clone_prompt.json'
METADATA_FIELDS = frozenset({
    'schema_version', 'role_id', 'role_version', 'clone_mode', 'model_name',
    'model_revision', 'model_dir', 'model_config_sha256', 'qwen_package_version',
    'reference_audio', 'reference_audio_sha256', 'reference_text_sha256',
    'audio_edit_chain_sha256', 'prompt_file', 'prompt_sha256', 'prompt_bytes',
    'tensor_info', 'created_at', 'device', 'dtype',
})
_ID = re.compile(r'[0-9a-f]{32}\Z')
_HASH = re.compile(r'[0-9a-f]{64}\Z')


class ClonePromptError(ValueError):
    """Base class for clone-prompt cache problems."""


class ClonePromptMissing(ClonePromptError):
    """No usable cache entry exists for this character."""


class ClonePromptStale(ClonePromptError):
    """A cache entry exists but its recorded inputs no longer match the card."""

    def __init__(self, reasons, metadata=None):
        self.reasons = list(reasons)
        self.metadata = metadata
        super().__init__('；'.join(self.reasons) or 'Clone Prompt 已失效')


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _chain_digest(chain):
    """Stable digest over the edit history, including event timestamps."""
    payload = json.dumps(chain if isinstance(chain, list) else [], ensure_ascii=False,
                         sort_keys=True, separators=(',', ':'), allow_nan=False)
    return _sha256(payload.encode('utf-8'))


def _prompt_directory(card):
    return Path(card['reference_audio']).resolve().parent / PROMPT_DIR


def _model_config(card):
    cfg = get_config()
    model = Path(cfg.qwen_base_model).expanduser().resolve()
    config = model / 'config.json'
    if not config.is_file():
        return model, None
    return model, _sha256(config.read_bytes())


def current_fingerprint(card):
    """The exact identity inputs a cached prompt must match."""
    model_dir, config_hash = _model_config(card)
    audio = Path(audio_path(card))
    audio_hash = card.get('canonical_sha256') or card.get('reference_sha256')
    return {
        'role_id': card['id'],
        'clone_mode': card.get('clone_mode') or 'icl',
        'model_name': card.get('model_name') or '',
        'model_revision': card.get('model_revision') or '',
        'model_dir': str(model_dir),
        'model_config_sha256': config_hash,
        'qwen_package_version': backend_qwen.QWEN_PACKAGE_VERSION,
        'reference_audio': audio.resolve().relative_to(
            Path(card['reference_audio']).resolve().parent).as_posix(),
        'reference_audio_sha256': audio_hash,
        'reference_text_sha256': _sha256(card.get('reference_text', '').encode('utf-8')),
        'audio_edit_chain_sha256': _chain_digest(card.get('audio_edit_chain', [])),
    }


def _metadata_path(directory):
    return directory / METADATA_NAME


def _read_metadata(directory):
    path = _metadata_path(directory)
    if path.is_symlink() or not path.is_file():
        raise ClonePromptMissing('尚未创建 Clone Prompt 缓存。')
    if path.stat().st_size > 64 * 1024:
        raise ClonePromptMissing('Clone Prompt 元数据异常过大，已忽略。')
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ClonePromptMissing(f'Clone Prompt 元数据损坏：{exc}') from exc
    if not isinstance(data, dict) or set(data) != METADATA_FIELDS:
        raise ClonePromptMissing('Clone Prompt 元数据字段不完整，已忽略。')
    if type(data['schema_version']) is not int or data['schema_version'] != SCHEMA_VERSION:
        raise ClonePromptMissing('Clone Prompt 元数据版本不兼容，请重新创建。')
    if not isinstance(data.get('prompt_file'), str) or not re.fullmatch(
            r'clone-(icl|xvector)-[0-9a-f]{12}\.pt', data['prompt_file']):
        raise ClonePromptMissing('Clone Prompt 文件名字段不安全，已忽略。')
    for field in ('prompt_sha256', 'model_config_sha256', 'reference_audio_sha256',
                  'reference_text_sha256', 'audio_edit_chain_sha256'):
        value = data.get(field)
        if not isinstance(value, str) or not _HASH.fullmatch(value):
            raise ClonePromptMissing(f'Clone Prompt 元数据 {field} 无效，已忽略。')
    if not isinstance(data.get('role_id'), str) or not _ID.fullmatch(data['role_id']):
        raise ClonePromptMissing('Clone Prompt 角色 ID 无效，已忽略。')
    if data.get('clone_mode') not in ('icl', 'xvector'):
        raise ClonePromptMissing('Clone Prompt 克隆模式无效，已忽略。')
    if type(data.get('role_version')) is not int or data['role_version'] < 1:
        raise ClonePromptMissing('Clone Prompt 角色版本无效，已忽略。')
    if type(data.get('prompt_bytes')) is not int or not 0 < data['prompt_bytes'] <= 2 * 1024 ** 3:
        raise ClonePromptMissing('Clone Prompt 文件大小记录无效，已忽略。')
    for field in ('model_name', 'model_revision', 'model_dir', 'qwen_package_version',
                  'reference_audio', 'created_at', 'device', 'dtype'):
        if not isinstance(data.get(field), str):
            raise ClonePromptMissing(f'Clone Prompt 元数据 {field} 无效，已忽略。')
    if not isinstance(data.get('tensor_info'), dict):
        raise ClonePromptMissing('Clone Prompt 张量信息无效，已忽略。')
    try:
        created = datetime.fromisoformat(data['created_at'])
    except ValueError as exc:
        raise ClonePromptMissing('Clone Prompt 创建时间无效，已忽略。') from exc
    if created.tzinfo is None:
        raise ClonePromptMissing('Clone Prompt 创建时间缺少时区，已忽略。')
    return data


def _reasons(metadata, card):
    """Compare recorded inputs against the card; one message per difference."""
    current = current_fingerprint(card)
    reasons = []
    if metadata['role_id'] != current['role_id']:
        reasons.append('缓存的角色 ID 与当前角色卡不一致')
    if current['model_config_sha256'] is None:
        reasons.append('本地 Qwen Base 模型缺失或未配置，无法确认缓存是否仍然匹配')
    else:
        if metadata['model_dir'] != current['model_dir']:
            reasons.append('模型目录已变化')
        if metadata['model_config_sha256'] != current['model_config_sha256']:
            reasons.append('模型版本已变化')
    if (metadata['model_name'] != current['model_name']
            or metadata['model_revision'] != current['model_revision']):
        reasons.append('角色卡登记的模型版本已变化')
    if metadata['qwen_package_version'] != current['qwen_package_version']:
        reasons.append('qwen-tts 版本已变化')
    if metadata['clone_mode'] != current['clone_mode']:
        reasons.append('克隆模式已切换（ICL / x-vector-only）')
    if metadata['reference_audio'] != current['reference_audio']:
        reasons.append('参考音频来源已变化')
    if metadata['reference_audio_sha256'] != current['reference_audio_sha256']:
        reasons.append('参考音频已变化')
    if metadata['reference_text_sha256'] != current['reference_text_sha256']:
        reasons.append('参考音频逐字稿已变化')
    if metadata['audio_edit_chain_sha256'] != current['audio_edit_chain_sha256']:
        reasons.append('音频处理链已变化')
    return reasons


def validate_clone_prompt(role_id, *, store=None):
    """Return a structured report; never raises for an ordinary stale cache."""
    store = store or CharacterStore()
    card = store.load(role_id)
    if card.get('backend') != 'qwen3':
        return {'ok': False, 'status': 'unsupported', 'reasons': ['只有 Qwen3 角色支持 Clone Prompt 缓存'],
                'metadata': None, 'path': None, 'card': card}
    directory = _prompt_directory(card)
    try:
        metadata = _read_metadata(directory)
    except ClonePromptMissing as exc:
        return {'ok': False, 'status': 'missing', 'reasons': [str(exc)],
                'metadata': None, 'path': None, 'card': card}
    path = directory / metadata['prompt_file']
    if path.is_symlink() or not path.is_file():
        return {'ok': False, 'status': 'corrupt', 'reasons': ['Clone Prompt 文件缺失，请重新创建'],
                'metadata': metadata, 'path': str(path), 'card': card}
    digest = _sha256(path.read_bytes())
    if digest != metadata['prompt_sha256']:
        return {'ok': False, 'status': 'corrupt', 'reasons': ['Clone Prompt 文件哈希不匹配，可能已损坏'],
                'metadata': metadata, 'path': str(path), 'card': card}
    reasons = _reasons(metadata, card)
    if reasons:
        return {'ok': False, 'status': 'stale', 'reasons': reasons,
                'metadata': metadata, 'path': str(path), 'card': card}
    if card.get('clone_prompt') is None or Path(card['clone_prompt']).name != metadata['prompt_file']:
        # The cache file is intact but the card does not point at it: repair the
        # pointer on the next explicit create rather than treating it as valid.
        return {'ok': False, 'status': 'stale', 'reasons': ['角色卡没有指向这份 Clone Prompt'],
                'metadata': metadata, 'path': str(path), 'card': card}
    return {'ok': True, 'status': 'valid', 'reasons': [], 'metadata': metadata,
            'path': str(path), 'card': card}


def load_clone_prompt(role_id, *, store=None):
    """Return ``{path, metadata, mode, card}`` or raise Missing/Stale."""
    report = validate_clone_prompt(role_id, store=store)
    if report['ok']:
        return {'path': report['path'], 'metadata': report['metadata'],
                'mode': report['metadata']['clone_mode'], 'card': report['card'],
                'sha256': report['metadata']['prompt_sha256']}
    if report['status'] in ('missing', 'unsupported'):
        raise ClonePromptMissing('；'.join(report['reasons']))
    raise ClonePromptStale(report['reasons'], metadata=report.get('metadata'))


def _write_metadata(directory, metadata):
    temporary = directory / ('.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(metadata, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, _metadata_path(directory))
    finally:
        temporary.unlink(missing_ok=True)


def create_clone_prompt(role_id, *, store=None, force=False, device='cuda:0',
                        dtype='float16', cancel=None, progress=None):
    """Build the cache entry; returns ``{created, metadata, path}``.

    A valid entry is reused unless ``force`` is set. A stale or corrupt entry is
    removed before rebuilding so a failed rebuild can never leave a pointer to
    tensors that do not match the current card.
    """
    store = store or CharacterStore()
    card = store.load(role_id)
    if card.get('backend') != 'qwen3':
        raise ClonePromptError('只有 Qwen3 角色支持 Clone Prompt 缓存。')
    report = validate_clone_prompt(role_id, store=store)
    if report['ok'] and not force:
        return {'created': False, 'metadata': report['metadata'], 'path': report['path'],
                'card': card}
    if report['status'] in ('stale', 'corrupt', 'missing') and report['metadata'] is not None:
        # The entry cannot be reused; drop its files before rebuilding.
        _remove_files(_prompt_directory(card), keep=None)
    elif report['ok'] and force:
        _remove_files(_prompt_directory(card), keep=None)

    model_dir, config_hash = _model_config(card)
    if config_hash is None:
        raise FileNotFoundError(
            f'本地 Qwen Base 模型缺失：{model_dir}。请在设置中检查 [qwen] base_model，'
            '不会自动下载模型。')
    reference = Path(audio_path(card))
    if not reference.is_file():
        raise FileNotFoundError(f'参考音频不存在：{reference}')
    if not card.get('reference_text', '').strip():
        raise ClonePromptError('请先填写参考音频逐字稿，再创建 Clone Prompt。')
    if progress is not None:
        progress(0.1, desc='创建 Clone Prompt：准备模型')
    directory = _prompt_directory(card)
    if directory.is_symlink():
        raise ClonePromptError('角色 prompts 目录是符号链接，拒绝写入 Clone Prompt。')
    directory.mkdir(parents=True, exist_ok=True)
    if directory.resolve().parent != Path(card['reference_audio']).resolve().parent:
        raise ClonePromptError('角色 prompts 目录逃出了角色卡目录，拒绝写入。')
    result = backend_qwen.create_clone_prompt(
        reference, card['reference_text'], clone_mode=card.get('clone_mode') or 'icl',
        device=device, dtype=dtype, output_dir=directory, cancel=cancel)
    fingerprint = current_fingerprint(card)
    prompt_hash = result['prompt_sha256']
    final_name = f"clone-{fingerprint['clone_mode']}-{prompt_hash[:12]}.pt"
    final_path = directory / final_name
    source_path = Path(result['prompt_file'])
    if final_path.exists() and _sha256(final_path.read_bytes()) == prompt_hash:
        source_path.unlink(missing_ok=True)
    else:
        os.replace(source_path, final_path)
    metadata = {
        'schema_version': SCHEMA_VERSION,
        'role_id': card['id'],
        'role_version': card.get('version', 1),
        'clone_mode': fingerprint['clone_mode'],
        'model_name': fingerprint['model_name'],
        'model_revision': fingerprint['model_revision'],
        'model_dir': fingerprint['model_dir'],
        'model_config_sha256': result['model_config_sha256'],
        'qwen_package_version': result['qwen_package_version'],
        'reference_audio': fingerprint['reference_audio'],
        'reference_audio_sha256': fingerprint['reference_audio_sha256'],
        'reference_text_sha256': fingerprint['reference_text_sha256'],
        'audio_edit_chain_sha256': fingerprint['audio_edit_chain_sha256'],
        'prompt_file': final_name,
        'prompt_sha256': prompt_hash,
        'prompt_bytes': int(final_path.stat().st_size),
        'tensor_info': result.get('tensor_info') or {},
        'created_at': _now(),
        'device': result.get('device', device),
        'dtype': result.get('dtype', dtype),
    }
    if not isinstance(metadata['tensor_info'], dict) or not metadata['tensor_info']:
        final_path.unlink(missing_ok=True)
        raise ClonePromptError('模型没有返回可用的 Clone Prompt 张量信息。')
    if metadata['model_config_sha256'] != config_hash:
        final_path.unlink(missing_ok=True)
        raise ClonePromptError('模型配置在创建 Clone Prompt 期间发生变化，已放弃缓存。')
    _write_metadata(directory, metadata)
    _remove_files(directory, keep={final_name, METADATA_NAME})
    card = store.set_clone_prompt(role_id, f'{PROMPT_DIR}/{final_name}', metadata['clone_mode'])
    if progress is not None:
        progress(1.0, desc='Clone Prompt 已缓存')
    return {'created': True, 'metadata': metadata, 'path': str(final_path), 'card': card}


def _remove_files(directory, keep):
    """Delete only generated cache files inside the character's prompts dir."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    removed = []
    for item in directory.iterdir():
        if not item.is_file() or item.is_symlink():
            continue
        if item.name == METADATA_NAME:
            if keep is None:
                item.unlink(missing_ok=True)
                removed.append(item.name)
            continue
        if not re.fullmatch(r'(clone-[a-z]+-[0-9a-f]{12}\.pt|\.[0-9a-f]{32}\.tmp|'
                            r'[0-9a-f]{32}\.pt)', item.name):
            continue
        if keep is not None and item.name in keep:
            continue
        item.unlink(missing_ok=True)
        removed.append(item.name)
    return removed


def invalidate_clone_prompt(role_id, reason, *, store=None):
    """Delete cached tensors and clear the card pointer; safe to call twice."""
    store = store or CharacterStore()
    card = store.load(role_id)
    directory = _prompt_directory(card)
    removed = _remove_files(directory, keep=None)
    if card.get('clone_prompt') is None:
        return {'ok': True, 'removed': removed, 'card': card}
    card = store.clear_clone_prompt(role_id, reason)
    return {'ok': True, 'removed': removed, 'card': card}


def status_text(report):
    """Short Chinese status line for cards and pages."""
    status = report.get('status')
    if status == 'valid':
        created = (report.get('metadata') or {}).get('created_at', '')[:19].replace('T', ' ')
        return f'Clone Prompt 已缓存（{created}）'
    if status == 'missing':
        return '未缓存 Clone Prompt；首次生成会自动使用普通克隆流程。'
    if status == 'stale':
        return 'Clone Prompt 已失效：' + '；'.join(report.get('reasons', [])) + '。下次生成会回退到普通克隆。'
    if status == 'corrupt':
        return 'Clone Prompt 文件损坏：' + '；'.join(report.get('reasons', [])) + '。请重新创建。'
    if status == 'unsupported':
        return '该后端不支持 Clone Prompt 缓存。'
    return 'Clone Prompt 状态未知。'

