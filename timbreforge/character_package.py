"""Portable character packages: duplicate, export and re-import cards safely.

A package is a ZIP containing the original recording, the current canonical
derivative and (when the size budget allows) older derived WAVs, plus the card
manifest and a small ``package.json``. Clone prompts are deliberately excluded:
they are model- and host-specific tensors that must be rebuilt locally.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone

from timbreforge.characters import (CharacterStore, _migrate_v1, _validate_audio,
                                    _validate_metadata, _write_manifest)
from timbreforge.voice_recipe import empty_recipe

FORMAT_VERSION = 1
DEFAULT_BUDGET = 64 * 1024 * 1024
MAX_PACKAGE_BYTES = 512 * 1024 * 1024
MAX_MEMBER_BYTES = 128 * 1024 * 1024
_REFERENCE = re.compile(r'reference\.[A-Za-z0-9]{1,10}\Z')
_AUDIO = re.compile(r'audio/[A-Za-z0-9._-]{1,120}\.wav\Z')
_ALLOWED = re.compile(r'(manifest\.json|package\.json|README\.txt|'
                      r'reference\.[A-Za-z0-9]{1,10}|audio/[A-Za-z0-9._-]{1,120}\.wav)\Z')


def _now():
    return datetime.now(timezone.utc).isoformat()


def _safe_member(name):
    if not isinstance(name, str) or '\\' in name or name.startswith('/'):
        return False
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or len(path.parts) > 2:
        return False
    return bool(_ALLOWED.fullmatch(name))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _card_directory(card):
    return Path(card['reference_audio']).resolve().parent


def _derived_files(card):
    """Existing derived WAVs referenced by the edit history, newest first."""
    directory = _card_directory(card)
    seen, files = set(), []
    for entry in reversed(card.get('audio_edit_chain', [])):
        output = entry.get('output_audio')
        if not isinstance(output, str) or not _AUDIO.fullmatch(output):
            continue
        path = directory / output
        if output in seen or not path.is_file() or path.is_symlink():
            continue
        seen.add(output)
        files.append(path)
    return files


def export_card(character_id, destination, *, store=None, budget=DEFAULT_BUDGET,
                overwrite=False):
    """Write a self-contained package; never includes clone prompts."""
    store = store or CharacterStore()
    card = store.load(character_id)
    directory = _card_directory(card)
    destination = Path(destination).expanduser().resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(f'导出文件已存在：{destination}')
    if not isinstance(budget, int) or not 1 * 1024 * 1024 <= budget <= MAX_PACKAGE_BYTES:
        raise ValueError('budget 必须在 1 MiB–512 MiB 之间。')

    manifest_path = directory / 'character.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 2:
        raise ValueError('只能导出 schema 2 角色卡；请先打开一次角色库完成迁移。')
    manifest = dict(manifest, clone_prompt=None)

    reference = directory / manifest['reference_audio']
    canonical = directory / manifest['canonical_audio'] if manifest['canonical_audio'] else None
    history = _derived_files(card)
    included, omitted, total = [], [], 0
    planned = [(reference, manifest['reference_audio'])]
    if canonical is not None and canonical.is_file():
        planned.append((canonical, manifest['canonical_audio']))
    for path, name in planned:
        size = path.stat().st_size
        total += size
        included.append((path, name))
    if total > budget:
        raise ValueError('原始音频与当前标准音频已超过导出大小上限。')
    for path in history:
        name = path.resolve().relative_to(directory).as_posix()
        if any(name == item[1] for item in included):
            continue
        size = path.stat().st_size
        if total + size > budget:
            omitted.append(name)
            continue
        total += size
        included.append((path, name))

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f'.export-{uuid.uuid4().hex}.tmp'
    package = {
        'format_version': FORMAT_VERSION,
        'application': 'TimbreForge',
        'exported_at': _now(),
        'source_id': card['id'],
        'name': card['name'],
        'description': card['description'],
        'includes_derived_audio': bool(canonical is not None),
        'included_audio': [name for _path, name in included],
        'omitted_history': omitted,
        'clone_prompt': 'not included; regenerate locally after import',
    }
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6) as archive:
            archive.writestr('manifest.json',
                             json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
            archive.writestr('package.json',
                             json.dumps(package, ensure_ascii=False, indent=2, sort_keys=True))
            archive.writestr('README.txt', _readme(card, omitted))
            for path, name in included:
                archive.write(path, name)
        if temporary.stat().st_size > MAX_PACKAGE_BYTES:
            raise ValueError('导出包超过 512 MiB 上限。')
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {'path': str(destination), 'bytes': destination.stat().st_size,
            'included_audio': package['included_audio'], 'omitted_history': omitted,
            'source_id': card['id']}


def _readme(card, omitted):
    lines = [f"TimbreForge 角色包：{card['name']}", '',
             '该包只包含角色音频与元数据，不包含模型权重或克隆缓存。',
             '重新导入后需要在「音频修整」或「角色库」重新创建 Clone Prompt。', '']
    if omitted:
        lines.append('为控制体积，以下历史派生音频未包含在包内（处理记录仍保留）：')
        lines.extend(f'- {name}' for name in omitted)
    return '\n'.join(lines) + '\n'


def _read_manifest(archive):
    try:
        data = json.loads(archive.read('manifest.json').decode('utf-8'))
    except (KeyError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f'角色包缺少可读的 manifest.json：{exc}') from exc
    if not isinstance(data, dict):
        raise ValueError('角色包 manifest.json 必须是对象。')
    version = data.get('schema_version')
    if version == 1:
        data = _migrate_v1(data)
    elif version != 2:
        raise ValueError('角色包角色卡版本不受支持。')
    return data


def import_card(package, *, store=None, name=None):
    """Import a package as a new card; the package itself is never modified."""
    store = store or CharacterStore()
    package = Path(package).expanduser().resolve()
    if not package.is_file():
        raise FileNotFoundError(f'角色包不存在：{package}')
    store.root.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        try:
            archive = zipfile.ZipFile(package)
        except zipfile.BadZipFile as exc:
            raise ValueError(f'角色包不是有效的 ZIP 文件：{exc}') from exc
        with archive:
            members = archive.infolist()
            if not members:
                raise ValueError('角色包为空。')
            total = 0
            for member in members:
                if member.is_dir():
                    continue
                if not _safe_member(member.filename):
                    raise ValueError(f'角色包含不安全或多余的文件：{member.filename}')
                if member.file_size > MAX_MEMBER_BYTES:
                    raise ValueError('角色包内单个文件超过 128 MiB。')
                total += member.file_size
                if total > MAX_PACKAGE_BYTES:
                    raise ValueError('角色包解压后超过 512 MiB。')
            manifest = _read_manifest(archive)
            manifest = _rewrite_manifest(manifest, name=name)
            temp_path = Path(tempfile.mkdtemp(prefix='.character-import-', dir=store.root))
            for member in members:
                if member.is_dir() or member.filename in {'manifest.json', 'package.json', 'README.txt'}:
                    continue
                archive.extract(member, temp_path)
        reference = manifest['reference_audio']
        if not _REFERENCE.fullmatch(reference):
            raise ValueError('角色包引用音频文件名不安全。')
        reference_path = temp_path / reference
        if not reference_path.is_file():
            raise ValueError('角色包缺少原始参考音频。')
        _validate_audio(reference_path)
        if hashlib.sha256(reference_path.read_bytes()).hexdigest() != manifest['reference_sha256']:
            raise ValueError('角色包原始音频哈希不匹配，可能已被修改。')
        if manifest['canonical_audio']:
            canonical = temp_path / manifest['canonical_audio']
            if not canonical.is_file():
                # The export budget may have omitted derived audio; fall back to
                # the untouched original instead of inventing a canonical take.
                manifest['canonical_audio'] = None
                manifest['canonical_sha256'] = None
            elif hashlib.sha256(canonical.read_bytes()).hexdigest() != manifest['canonical_sha256']:
                raise ValueError('角色包标准音频哈希不匹配。')
        manifest = _validate_metadata(manifest, manifest['id'], temp_path)
        _write_manifest(temp_path, manifest)
        target = store.root / manifest['id']
        if target.exists() or target.is_symlink():
            raise FileExistsError('导入生成的角色 ID 已存在，请重试。')
        os.rename(temp_path, target)
        temp_path = None
        return store.load(manifest['id'])
    finally:
        if temp_path is not None and temp_path.exists():
            shutil.rmtree(temp_path, ignore_errors=True)


def _rewrite_manifest(manifest, *, name):
    """Assign a new identity and drop derived, host-specific pointers."""
    if not isinstance(manifest.get('id'), str) or not re.fullmatch(r'[0-9a-f]{32}', manifest['id']):
        raise ValueError('角色包 ID 无效。')
    original = manifest['id']
    new_id = uuid.uuid4().hex
    now = _now()
    manifest = dict(manifest, id=new_id, created_at=now, updated_at=now,
                    clone_prompt=None, version=1)
    if name is not None:
        if not isinstance(name, str) or not name.strip() or len(name) > 120:
            raise ValueError('导入角色名必须是不超过 120 字的非空文本。')
        manifest['name'] = name.strip()
    provenance = manifest.get('provenance')
    manifest['provenance'] = dict(provenance if isinstance(provenance, dict) else {},
                                  imported_from_package=original,
                                  imported_at=now)
    if not isinstance(manifest.get('voice_recipe'), dict):
        manifest['voice_recipe'] = empty_recipe()
    manifest['source'] = (manifest.get('source') or '') + '/imported'
    return manifest


def duplicate_card(character_id, *, store=None, name=None):
    """Copy a card byte-for-byte (audio and history) under a new identity."""
    store = store or CharacterStore()
    card = store.load(character_id)
    original_dir = _card_directory(card)
    store.root.mkdir(parents=True, exist_ok=True)
    temp_path = Path(tempfile.mkdtemp(prefix='.character-copy-', dir=store.root))
    try:
        (temp_path).rmdir()  # copytree needs to create the final component
        shutil.copytree(original_dir, temp_path, symlinks=False)
        # Clone prompts are host/model specific; drop previews and prompts.
        shutil.rmtree(temp_path / 'prompts', ignore_errors=True)
        shutil.rmtree(temp_path / 'audio' / 'previews', ignore_errors=True)
        manifest_path = temp_path / 'character.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('schema_version') == 1:
            manifest = _migrate_v1(manifest)
        new_id = uuid.uuid4().hex
        now = _now()
        manifest.update(id=new_id, clone_prompt=None, version=1,
                        created_at=now, updated_at=now,
                        name=(name.strip() if isinstance(name, str) and name.strip()
                              else card['name'] + '（副本）'),
                        provenance=dict(manifest.get('provenance') or {},
                                        duplicated_from=card['id'], duplicated_at=now))
        manifest = _validate_metadata(manifest, new_id, temp_path)
        _write_manifest(temp_path, manifest)
        target = store.root / new_id
        if target.exists() or target.is_symlink():
            raise FileExistsError('复制生成的角色 ID 已存在，请重试。')
        os.rename(temp_path, target)
        temp_path = None
        return store.load(new_id)
    finally:
        if temp_path is not None and temp_path.exists():
            shutil.rmtree(temp_path, ignore_errors=True)

