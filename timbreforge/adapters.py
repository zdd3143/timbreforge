"""Safe local registry for community PEFT/LoRA adapters.

The registry never executes files from an adapter package and never downloads
anything.  It validates a PEFT ``adapter_config.json`` plus safetensors, keeps
content hashes, and stores only an explicit binding on a character card.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import uuid
import zipfile

from timbreforge.characters import CharacterStore
from timbreforge.config import get_config

SCHEMA_VERSION = 1
ADAPTER_TYPES = ('peft_lora',)
_ID = re.compile(r'[0-9a-f]{32}\Z')
_HASH = re.compile(r'[0-9a-f]{64}\Z')
_MAX_ZIP_BYTES = 8 * 1024**3
_MAX_MANIFEST_BYTES = 2 * 1024**2


class AdapterError(ValueError):
    """Raised when an adapter is missing, unsafe or incompatible."""


def _now():
    return datetime.now(timezone.utc).isoformat()


def default_root():
    cfg = get_config()
    return Path(cfg.outputs_dir).expanduser().resolve().parent / 'models' / 'adapters'


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_hash(directory):
    digest = hashlib.sha256()
    files = []
    for path in Path(directory).rglob('*'):
        if path.is_symlink():
            raise AdapterError(f'适配器目录不允许符号链接：{path}')
        if path.is_file():
            relative = path.relative_to(directory).as_posix()
            files.append((relative, path))
    for relative, path in sorted(files):
        digest.update(relative.encode('utf-8'))
        digest.update(b'\\0')
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b''):
                digest.update(chunk)
    return digest.hexdigest(), sum(path.stat().st_size for _, path in files)


def _safe_directory(path):
    directory = Path(path).expanduser().resolve()
    if not directory.is_dir() or directory.is_symlink():
        raise AdapterError(f'适配器目录不存在或不安全：{directory}')
    return directory


def inspect_adapter(path):
    """Validate a PEFT LoRA folder and return its normalized metadata."""
    directory = _safe_directory(path)
    config_path = directory / 'adapter_config.json'
    weights = directory / 'adapter_model.safetensors'
    if not config_path.is_file() or config_path.stat().st_size > _MAX_MANIFEST_BYTES:
        raise AdapterError('适配器必须包含不超过 2 MiB 的 adapter_config.json。')
    if not weights.is_file():
        raise AdapterError('适配器必须包含 adapter_model.safetensors；拒绝加载 pickle 权重。')
    try:
        config = json.loads(config_path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, ValueError) as exc:
        raise AdapterError(f'adapter_config.json 无法读取：{exc}') from exc
    if not isinstance(config, dict):
        raise AdapterError('adapter_config.json 必须是 JSON 对象。')
    peft_type = str(config.get('peft_type', '')).upper()
    if peft_type != 'LORA':
        raise AdapterError(f'只支持 PEFT LoRA；当前类型为 {peft_type or "未知"}。')
    rank = config.get('r')
    alpha = config.get('lora_alpha')
    if type(rank) is not int or not 1 <= rank <= 1024:
        raise AdapterError('LoRA rank 必须是 1–1024 的整数。')
    if type(alpha) not in (int, float) or not 0 < float(alpha) <= 4096:
        raise AdapterError('LoRA alpha 必须是 0–4096 的数字。')
    target_modules = config.get('target_modules')
    if not isinstance(target_modules, (list, str)):
        raise AdapterError('adapter_config.json 缺少 target_modules。')
    artifact_hash, size = _artifact_hash(directory)
    return {
        'config': config,
        'config_path': str(config_path),
        'weights_path': str(weights),
        'config_sha256': _sha256(config_path),
        'artifact_sha256': artifact_hash,
        'bytes': size,
        'base_model': str(config.get('base_model_name_or_path', '') or ''),
        'rank': rank,
        'alpha': float(alpha),
        'target_modules': target_modules,
    }


class AdapterRegistry:
    """Content-addressed metadata for local community adapters."""

    def __init__(self, root=None):
        self.root = Path(root if root is not None else default_root()).expanduser().resolve()

    @property
    def path(self):
        return self.root / 'registry.json'

    def _read(self):
        if not self.path.is_file():
            return {'schema_version': SCHEMA_VERSION, 'entries': []}
        if self.path.is_symlink() or self.path.stat().st_size > _MAX_MANIFEST_BYTES:
            raise AdapterError('适配器注册表不安全或过大。')
        try:
            data = json.loads(self.path.read_text(encoding='utf-8'))
        except (OSError, UnicodeError, ValueError) as exc:
            raise AdapterError(f'适配器注册表损坏：{exc}') from exc
        if not isinstance(data, dict) or data.get('schema_version') != SCHEMA_VERSION:
            raise AdapterError('适配器注册表版本不兼容。')
        if not isinstance(data.get('entries'), list):
            raise AdapterError('适配器注册表结构无效。')
        return data

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

    @staticmethod
    def _validate_entry(entry):
        if not isinstance(entry, dict):
            raise AdapterError('适配器记录必须是对象。')
        required = {'id', 'name', 'path', 'adapter_type', 'base_model', 'base_revision',
                    'config_sha256', 'artifact_sha256', 'bytes', 'default_scale',
                    'author', 'source_url', 'license', 'notes', 'registered_at',
                    'rank', 'alpha', 'target_modules'}
        if set(entry) != required or not _ID.fullmatch(str(entry.get('id', ''))):
            raise AdapterError('适配器记录字段不完整。')
        if entry['adapter_type'] not in ADAPTER_TYPES:
            raise AdapterError('适配器类型不受支持。')
        if not isinstance(entry['path'], str) or not Path(entry['path']).is_absolute():
            raise AdapterError('适配器路径必须是绝对路径。')
        for key in ('config_sha256', 'artifact_sha256'):
            if not _HASH.fullmatch(str(entry[key])):
                raise AdapterError(f'{key} 无效。')
        if type(entry['bytes']) is not int or entry['bytes'] < 0:
            raise AdapterError('适配器体积无效。')
        if type(entry['default_scale']) not in (int, float) or not 0 <= float(entry['default_scale']) <= 2:
            raise AdapterError('LoRA scale 必须在 0–2 之间。')
        for key in ('name', 'base_model', 'base_revision', 'author', 'source_url', 'license', 'notes'):
            if not isinstance(entry[key], str) or len(entry[key]) > 2000:
                raise AdapterError(f'{key} 必须是文本。')
        if type(entry['rank']) is not int or type(entry['alpha']) not in (int, float):
            raise AdapterError('LoRA 参数无效。')
        if not isinstance(entry['target_modules'], (list, str)):
            raise AdapterError('target_modules 无效。')
        try:
            parsed = datetime.fromisoformat(entry['registered_at'])
        except (TypeError, ValueError) as exc:
            raise AdapterError('适配器登记时间无效。') from exc
        if parsed.tzinfo is None:
            raise AdapterError('适配器登记时间缺少时区。')
        return entry

    def list(self):
        result = []
        for entry in self._read()['entries']:
            try:
                result.append(self._validate_entry(entry))
            except AdapterError:
                continue
        return sorted(result, key=lambda item: item['registered_at'], reverse=True)

    def get(self, adapter_id):
        if not isinstance(adapter_id, str) or not _ID.fullmatch(adapter_id):
            raise AdapterError('适配器 ID 无效。')
        for entry in self._read()['entries']:
            if isinstance(entry, dict) and entry.get('id') == adapter_id:
                return self._validate_entry(entry)
        raise AdapterError('找不到该社区适配器。')

    def register(self, path, *, name='', author='', source_url='', license='未知',
                 notes='', base_revision='', default_scale=0.3):
        info = inspect_adapter(path)
        directory = Path(info['config_path']).parent
        if not isinstance(default_scale, (int, float)) or not 0 <= float(default_scale) <= 2:
            raise AdapterError('LoRA scale 必须在 0–2 之间。')
        data = self._read()
        for existing in data['entries']:
            if (isinstance(existing, dict) and existing.get('path') == str(directory)
                    and existing.get('artifact_sha256') == info['artifact_sha256']):
                return self._validate_entry(existing)
        entry = {
            'id': uuid.uuid4().hex,
            'name': (name or directory.name)[:200],
            'path': str(directory),
            'adapter_type': 'peft_lora',
            'base_model': info['base_model'],
            'base_revision': str(base_revision or info['config'].get('revision', '') or ''),
            'config_sha256': info['config_sha256'],
            'artifact_sha256': info['artifact_sha256'],
            'bytes': info['bytes'],
            'default_scale': float(default_scale),
            'author': str(author or '')[:500],
            'source_url': str(source_url or '')[:2000],
            'license': str(license or '未知')[:200],
            'notes': str(notes or '')[:2000],
            'registered_at': _now(),
            'rank': info['rank'],
            'alpha': info['alpha'],
            'target_modules': info['target_modules'],
        }
        data['entries'].append(entry)
        self._write(data)
        return self._validate_entry(entry)

    def import_zip(self, archive, **metadata):
        archive = Path(archive).expanduser().resolve()
        if not archive.is_file() or archive.stat().st_size > _MAX_ZIP_BYTES:
            raise AdapterError('适配器 ZIP 不存在或超过 8 GiB。')
        target = self.root / ('.import-' + uuid.uuid4().hex)
        target.mkdir(parents=True, exist_ok=False)
        try:
            with zipfile.ZipFile(archive) as bundle:
                for member in bundle.infolist():
                    name = Path(member.filename)
                    if name.is_absolute() or '..' in name.parts or member.is_dir():
                        if member.is_dir():
                            continue
                        raise AdapterError('ZIP 含有不安全路径。')
                    destination = (target / name).resolve()
                    if target not in destination.parents:
                        raise AdapterError('ZIP 路径越界。')
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(member) as source, destination.open('xb') as sink:
                        shutil.copyfileobj(source, sink)
            candidates = [target]
            nested = [p for p in target.iterdir() if p.is_dir()]
            if not (target / 'adapter_config.json').is_file() and len(nested) == 1:
                candidates = [nested[0]]
            entry = self.register(candidates[0], **metadata)
            return entry
        except Exception:
            shutil.rmtree(target, ignore_errors=True)
            raise

    def verify(self, adapter_id):
        entry = self.get(adapter_id)
        try:
            info = inspect_adapter(entry['path'])
        except AdapterError as exc:
            return {'ok': False, 'problems': [str(exc)], 'entry': entry}
        problems = []
        if info['config_sha256'] != entry['config_sha256']:
            problems.append('adapter_config.json 已变化。')
        if info['artifact_sha256'] != entry['artifact_sha256']:
            problems.append('适配器权重或文件内容已变化。')
        return {'ok': not problems, 'problems': problems, 'entry': entry, 'info': info}

    def remove(self, adapter_id):
        entry = self.get(adapter_id)
        data = self._read()
        data['entries'] = [item for item in data['entries']
                           if not (isinstance(item, dict) and item.get('id') == adapter_id)]
        self._write(data)
        return entry

    def bind_role(self, role_id, adapter_id, *, scale=None, store=None):
        store = store or CharacterStore()
        entry = self.get(adapter_id)
        report = self.verify(adapter_id)
        if not report['ok']:
            raise AdapterError('适配器校验失败：' + '；'.join(report['problems']))
        card = store.load(role_id)
        if card.get('backend') != 'qwen3':
            raise AdapterError('社区 LoRA 当前只支持 Qwen3 角色。')
        base = entry['base_model'].lower()
        card_model = str(card.get('model_name', '')).lower()
        if base and card_model and base.split('/')[-1] not in card_model and card_model not in base:
            raise AdapterError(f'适配器基础模型 {entry["base_model"]} 与角色模型 {card.get("model_name")} 不匹配。')
        value = entry['default_scale'] if scale is None else scale
        if not isinstance(value, (int, float)) or not 0 <= float(value) <= 2:
            raise AdapterError('LoRA scale 必须在 0–2 之间。')
        binding = {'id': entry['id'], 'scale': float(value),
                   'artifact_sha256': entry['artifact_sha256'],
                   'config_sha256': entry['config_sha256'],
                   'name': entry['name'], 'base_model': entry['base_model']}
        return store.update(role_id, provenance_patch={'adapter': binding},
                            invalidate_reason=f'已绑定社区适配器 {entry["name"]}')

    def unbind_role(self, role_id, *, store=None):
        store = store or CharacterStore()
        card = store.load(role_id)
        if not (card.get('provenance') or {}).get('adapter'):
            return card
        return store.update(role_id, provenance_patch={'adapter': None},
                            invalidate_reason='已解除社区适配器绑定')

    def resolve_binding(self, card):
        binding = (card.get('provenance') or {}).get('adapter')
        if binding is None:
            return None
        if not isinstance(binding, dict) or not isinstance(binding.get('id'), str):
            raise AdapterError('角色卡中的社区适配器绑定已损坏。')
        entry = self.get(binding['id'])
        report = self.verify(entry['id'])
        if not report['ok']:
            raise AdapterError('社区适配器已变化：' + '；'.join(report['problems']))
        if binding.get('artifact_sha256') != entry['artifact_sha256']:
            raise AdapterError('角色卡绑定的适配器哈希与当前文件不一致。')
        return {'entry': entry, 'path': entry['path'],
                'scale': float(binding.get('scale', entry['default_scale'])),
                'artifact_sha256': entry['artifact_sha256']}


__all__ = ['AdapterError', 'AdapterRegistry', 'ADAPTER_TYPES', 'inspect_adapter']

