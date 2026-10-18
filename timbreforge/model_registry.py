"""Local model directory and version registry.

The registry never downloads weights. It records model directories the user
already has, fingerprints their configuration, and lets a character card switch
between versions with an explicit rollback trail. Removing an entry removes only
the registry record, never the files.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import uuid
from datetime import datetime, timezone

from timbreforge.characters import CharacterStore
from timbreforge.config import get_config

SCHEMA_VERSION = 1
ENTRY_FIELDS = frozenset({
    'id', 'name', 'kind', 'revision', 'path', 'config_sha256', 'bytes',
    'registered_at', 'notes', 'base_model',
})
KINDS = ('tts_base', 'tts_design', 'vc', 'lora', 'adapter', 'planner')
_ENTRY_ID = re.compile(r'[0-9a-f]{32}\Z')
_HASH = re.compile(r'[0-9a-f]{64}\Z')


class ModelRegistryError(ValueError):
    """Raised when model registration or activation is unsafe."""


def _now():
    return datetime.now(timezone.utc).isoformat()


def default_root():
    cfg = get_config()
    return Path(cfg.outputs_dir).expanduser().resolve().parent / 'models'


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_bytes(directory):
    total = 0
    for item in Path(directory).rglob('*'):
        if item.is_file() and not item.is_symlink():
            total += item.stat().st_size
    return total


class ModelRegistry:
    """JSON registry of local model directories and their versions."""

    def __init__(self, root=None):
        self.root = Path(root if root is not None else default_root()).expanduser().resolve()

    @property
    def path(self):
        return self.root / 'registry.json'

    def _read(self):
        path = self.path
        if path.is_symlink():
            raise ModelRegistryError('模型注册表指向不安全的位置。')
        if not path.is_file():
            return {'schema_version': SCHEMA_VERSION, 'entries': [], 'active': {}}
        if path.stat().st_size > 2 * 1024 * 1024:
            raise ModelRegistryError('模型注册表异常过大，已拒绝读取。')
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, UnicodeError, ValueError) as exc:
            raise ModelRegistryError(f'模型注册表损坏：{exc}') from exc
        if not isinstance(data, dict) or data.get('schema_version') != SCHEMA_VERSION:
            raise ModelRegistryError('模型注册表版本不兼容。')
        entries = data.get('entries')
        active = data.get('active', {})
        if not isinstance(entries, list) or not isinstance(active, dict):
            raise ModelRegistryError('模型注册表结构无效。')
        return {'schema_version': SCHEMA_VERSION, 'entries': entries, 'active': active}

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

    def _validate_entry(self, entry):
        if not isinstance(entry, dict) or set(entry) != ENTRY_FIELDS:
            raise ModelRegistryError('模型记录字段不完整。')
        if not isinstance(entry['id'], str) or not _ENTRY_ID.fullmatch(entry['id']):
            raise ModelRegistryError('模型记录 ID 无效。')
        for field in ('name', 'kind', 'revision', 'path', 'notes', 'base_model'):
            if not isinstance(entry[field], str):
                raise ModelRegistryError(f'模型记录 {field} 必须是文本。')
        if entry['kind'] not in KINDS:
            raise ModelRegistryError(f'模型类型必须是：{", ".join(KINDS)}。')
        if not entry['name'].strip() or len(entry['name']) > 120:
            raise ModelRegistryError('模型名称必须是不超过 120 字的非空文本。')
        if not entry['path'] or not Path(entry['path']).is_absolute():
            raise ModelRegistryError('模型路径必须是绝对路径。')
        if not isinstance(entry['config_sha256'], str) or not _HASH.fullmatch(entry['config_sha256']):
            raise ModelRegistryError('模型配置哈希无效。')
        if type(entry['bytes']) is not int or entry['bytes'] < 0:
            raise ModelRegistryError('模型体积记录无效。')
        try:
            registered = datetime.fromisoformat(entry['registered_at'])
        except (TypeError, ValueError) as exc:
            raise ModelRegistryError('模型登记时间无效。') from exc
        if registered.tzinfo is None:
            raise ModelRegistryError('模型登记时间缺少时区。')
        return entry

    def load_all(self):
        entries = []
        for entry in self._read()['entries']:
            try:
                entries.append(self._validate_entry(entry))
            except ModelRegistryError:
                continue
        return sorted(entries, key=lambda item: item['registered_at'], reverse=True)

    def get(self, entry_id):
        if not isinstance(entry_id, str) or not _ENTRY_ID.fullmatch(entry_id):
            raise ModelRegistryError('模型记录 ID 无效。')
        for entry in self._read()['entries']:
            if isinstance(entry, dict) and entry.get('id') == entry_id:
                return self._validate_entry(entry)
        raise ModelRegistryError('模型注册表中找不到该记录。')

    def register(self, name, path, revision='local', *, kind='tts_base', notes='',
                 base_model='', replace=False):
        """Verify and register an existing local model directory."""
        if not isinstance(name, str) or not name.strip() or len(name) > 120:
            raise ModelRegistryError('模型名称必须是不超过 120 字的非空文本。')
        if not isinstance(revision, str) or not revision.strip() or len(revision) > 120:
            raise ModelRegistryError('模型版本必须是不超过 120 字的非空文本。')
        if kind not in KINDS:
            raise ModelRegistryError(f'模型类型必须是：{", ".join(KINDS)}。')
        for field, value in (('notes', notes), ('base_model', base_model)):
            if not isinstance(value, str) or len(value) > 1000:
                raise ModelRegistryError(f'{field} 必须是不超过 1000 字的文本。')
        directory = Path(path).expanduser().resolve()
        if not directory.is_dir():
            raise ModelRegistryError(f'本地模型目录不存在：{directory}；本工具不会自动下载模型。')
        config = directory / 'config.json'
        if not config.is_file():
            raise ModelRegistryError(f'模型目录缺少 config.json：{directory}')
        if kind in ('tts_base', 'tts_design', 'vc', 'planner') and not any(directory.glob('*.safetensors')):
            raise ModelRegistryError(f'模型目录缺少 safetensors 权重：{directory}')
        config_hash = _sha256(config)
        entries = self._read()
        for index, entry in enumerate(entries['entries']):
            try:
                existing = self._validate_entry(entry)
            except ModelRegistryError:
                continue
            if existing['name'] == name.strip() and existing['revision'] == revision.strip():
                if existing['path'] == str(directory) and existing['config_sha256'] == config_hash:
                    return existing
                if not replace:
                    raise ModelRegistryError('同名称同版本的模型已登记且指向不同目录；请使用 replace。')
                entries['entries'][index] = dict(existing, path=str(directory),
                                                 config_sha256=config_hash,
                                                 bytes=_directory_bytes(directory),
                                                 notes=notes, base_model=base_model,
                                                 registered_at=_now())
                self._write(entries)
                return self._validate_entry(entries['entries'][index])
        record = {
            'id': uuid.uuid4().hex,
            'name': name.strip(),
            'kind': kind,
            'revision': revision.strip(),
            'path': str(directory),
            'config_sha256': config_hash,
            'bytes': _directory_bytes(directory),
            'registered_at': _now(),
            'notes': notes,
            'base_model': base_model,
        }
        entries['entries'].append(record)
        self._write(entries)
        return self._validate_entry(record)

    def verify(self, entry_id):
        """Re-check files and hashes without trusting the registry record."""
        entry = self.get(entry_id)
        directory = Path(entry['path'])
        config = directory / 'config.json'
        problems = []
        if not directory.is_dir():
            problems.append('模型目录不存在')
        elif not config.is_file():
            problems.append('config.json 缺失')
        elif _sha256(config) != entry['config_sha256']:
            problems.append('config.json 已变化（可能是另一个版本）')
        if entry['kind'] in ('tts_base', 'tts_design', 'vc', 'planner') and not any(
                directory.glob('*.safetensors')):
            problems.append('safetensors 权重缺失')
        return {'ok': not problems, 'problems': problems, 'entry': entry,
                'bytes': _directory_bytes(directory) if directory.is_dir() else 0}

    def remove(self, entry_id):
        """Remove a registry record only; the model files stay on disk."""
        entry = self.get(entry_id)
        data = self._read()
        data['entries'] = [item for item in data['entries']
                           if not (isinstance(item, dict) and item.get('id') == entry_id)]
        self._write(data)
        return entry

    def find(self, name=None, kind=None):
        entries = self.load_all()
        if name is not None:
            entries = [entry for entry in entries if entry['name'] == name]
        if kind is not None:
            entries = [entry for entry in entries if entry['kind'] == kind]
        return entries

    def activate(self, character_id, entry_id, *, store=None):
        """Point a card at a registered version; stale clone prompts are dropped."""
        store = store or CharacterStore()
        entry = self.get(entry_id)
        card = store.load(character_id)
        if (card.get('model_name') == entry['name']
                and card.get('model_revision') == entry['revision']):
            return {'changed': False, 'card': card, 'entry': entry}
        history = card.get('provenance', {}).get('model_activations', [])
        if not isinstance(history, list):
            history = []
        record = {'at': _now(), 'entry_id': entry_id,
                  'from': {'name': card.get('model_name', ''),
                           'revision': card.get('model_revision', '')},
                  'to': {'name': entry['name'], 'revision': entry['revision']}}
        card = store.update(
            character_id, model_name=entry['name'], model_revision=entry['revision'],
            provenance_patch={'model_activations': (history + [record])[-20:]},
            invalidate_reason=f"模型切换为 {entry['name']} @ {entry['revision']}")
        return {'changed': True, 'card': card, 'entry': entry, 'record': record}

    def rollback(self, character_id, *, store=None):
        """Restore the model recorded before the most recent activation."""
        store = store or CharacterStore()
        card = store.load(character_id)
        history = card.get('provenance', {}).get('model_activations', [])
        if not isinstance(history, list) or not history:
            raise ModelRegistryError('这张角色卡没有可回滚的模型切换记录。')
        previous = history[-1].get('from') if isinstance(history[-1], dict) else None
        if not isinstance(previous, dict):
            raise ModelRegistryError('模型切换记录损坏，无法回滚。')
        remaining = history[:-1]
        card = store.update(
            character_id,
            model_name=previous.get('name', ''),
            model_revision=previous.get('revision', ''),
            provenance_patch={'model_activations': remaining,
                              'model_rollback_at': _now()},
            invalidate_reason='模型已回滚到上一版本')
        return {'card': card, 'restored': previous}

    def status(self):
        entries = self.load_all()
        return {'entries': entries,
                'by_kind': {kind: [entry for entry in entries if entry['kind'] == kind]
                            for kind in KINDS},
                'root': str(self.root)}

