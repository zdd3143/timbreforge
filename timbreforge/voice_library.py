"""Character groups that can own several saved voices.

A group is only an index: every voice points at a validated character card in
:mod:`timbreforge.characters`, which remains the single owner of the reference
audio, transcript and generation provenance. Deleting or corrupting a card is
therefore visible here instead of silently inventing a voice.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import uuid

from timbreforge.characters import CharacterStore, _validate_json_value
from timbreforge.config import get_config

SCHEMA_VERSION = 1
INDEX_NAME = '_voice-library.json'
_ID = re.compile(r'[0-9a-f]{32}\Z')
MAX_NAME = 120
MAX_TEXT = 2000


def _identifier(value, label):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(f'{label} 无效。')
    return value


def _text(value, label, limit):
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError(f'{label} 必须是不超过 {limit} 字的文本。')
    return value.strip()


def _timestamp(value):
    if not isinstance(value, str):
        raise ValueError('声线库时间戳无效。')
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError('声线库时间戳无效。') from exc
    if parsed.tzinfo is None:
        raise ValueError('声线库时间戳缺少时区。')
    return value


def _now():
    return datetime.now(timezone.utc).isoformat()


class VoiceLibrary:
    """Local index of characters and the voices saved for each of them."""

    def __init__(self, root=None):
        self.root = Path(root if root is not None else get_config().characters_dir).expanduser().resolve()

    @property
    def path(self):
        return self.root / INDEX_NAME

    # ---------------------------------------------------------------- storage
    def _read(self):
        path = self.path
        if path.is_symlink():
            raise ValueError('声线库索引指向了不安全的位置。')
        if not path.is_file():
            return dict(schema_version=SCHEMA_VERSION, characters=[])
        if path.stat().st_size > 4 * 1024 * 1024:
            raise ValueError('声线库索引异常过大，已拒绝读取。')
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            raise ValueError('声线库索引损坏，无法读取。') from exc
        if not isinstance(data, dict) or data.get('schema_version') != SCHEMA_VERSION:
            raise ValueError('声线库索引版本不兼容。')
        characters = data.get('characters')
        if not isinstance(characters, list):
            raise ValueError('声线库索引结构无效。')
        return dict(schema_version=SCHEMA_VERSION, characters=characters)

    def _write(self, data):
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name('.' + uuid.uuid4().hex + '.tmp')
        try:
            with temporary.open('x', encoding='utf-8') as stream:
                json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    # ------------------------------------------------------------ validation
    def _valid_group(self, entry):
        if not isinstance(entry, dict):
            raise ValueError('声线库角色记录无效。')
        group = dict(id=_identifier(entry.get('id'), '声线库角色 ID'),
                     name=_text(entry.get('name'), '角色名', MAX_NAME),
                     description=_text(entry.get('description', ''), '角色描述', MAX_TEXT),
                     created_at=_timestamp(entry.get('created_at')))
        if not group['name']:
            raise ValueError('声线库角色名不能为空。')
        voices = entry.get('voices')
        if not isinstance(voices, list):
            raise ValueError('声线库声线列表无效。')
        group['voices'] = []
        seen = set()
        for voice in voices:
            if not isinstance(voice, dict):
                raise ValueError('声线库声线条目无效。')
            card = _identifier(voice.get('character_id'), '声线卡 ID')
            if card in seen:
                raise ValueError('声线库存在重复的声线卡。')
            seen.add(card)
            origin = voice.get('origin', {})
            if not isinstance(origin, dict):
                raise ValueError('声线来源记录无效。')
            _validate_json_value(origin, 'origin')
            group['voices'].append(dict(
                character_id=card,
                label=_text(voice.get('label', ''), '声线备注', MAX_NAME),
                description=_text(voice.get('description', ''), '声线描述', MAX_TEXT),
                preview_text=_text(voice.get('preview_text', ''), '试听台词', 500),
                created_at=_timestamp(voice.get('created_at')), origin=origin))
        return group

    def load_all(self):
        """Validated groups, newest first. Corrupt individual entries are skipped."""
        groups = []
        seen = set()
        for entry in self._read()['characters']:
            try:
                group = self._valid_group(entry)
            except (ValueError, RecursionError):
                continue
            if group['id'] in seen:
                continue
            seen.add(group['id'])
            groups.append(group)
        return sorted(groups, key=lambda item: item['created_at'], reverse=True)

    def get(self, character_id):
        _identifier(character_id, '声线库角色 ID')
        for group in self.load_all():
            if group['id'] == character_id:
                return group
        return None

    def _locate(self, data, character_id):
        for index, entry in enumerate(data['characters']):
            if isinstance(entry, dict) and entry.get('id') == character_id:
                return index
        raise ValueError('声线库中找不到该角色。')

    # ----------------------------------------------------------- operations
    def create_character(self, name, description=''):
        name = _text(name, '角色名', MAX_NAME)
        if not name:
            raise ValueError('请填写角色名。')
        description = _text(description, '角色描述', MAX_TEXT)
        data = self._read()
        group = dict(id=uuid.uuid4().hex, name=name, description=description,
                     created_at=_now(), voices=[])
        data['characters'].append(group)
        self._write(data)
        return self._valid_group(group)

    def add_voice(self, character_id, card_id, *, label='', description='', preview_text='', origin=None):
        _identifier(character_id, '声线库角色 ID')
        _identifier(card_id, '声线卡 ID')
        # The card owns the audio; a dangling reference would be a fake voice entry.
        card = CharacterStore(self.root).load(card_id)
        label = _text(label, '声线备注', MAX_NAME)
        description = _text(description, '声线描述', MAX_TEXT)
        preview_text = _text(preview_text, '试听台词', 500)
        payload = dict(character_id=card_id, label=label, description=description,
                       preview_text=preview_text, created_at=_now(),
                       origin=dict(origin or {}, card_backend=card.get('backend')))
        _validate_json_value(payload, 'voice')
        data = self._read()
        group = self._valid_group(data['characters'][self._locate(data, character_id)])
        if any(voice['character_id'] == card_id for voice in group['voices']):
            raise ValueError('这条声线已经在该角色名下。')
        data['characters'][self._locate(data, character_id)]['voices'] = (
            list(group['voices']) + [payload])
        self._write(data)
        return payload

    def refresh_voice(self, character_id, card_id, *, label=None, description=None):
        data = self._read()
        group = self._valid_group(data['characters'][self._locate(data, character_id)])
        for voice in group['voices']:
            if voice['character_id'] == card_id:
                if label is not None:
                    voice['label'] = _text(label, '声线备注', MAX_NAME)
                if description is not None:
                    voice['description'] = _text(description, '声线描述', MAX_TEXT)
                data['characters'][self._locate(data, character_id)]['voices'] = group['voices']
                self._write(data)
                return voice
        raise ValueError('该角色名下没有这条声线。')

    def remove_voice(self, character_id, card_id):
        """Detach a voice; the character card itself is never deleted here."""
        data = self._read()
        index = self._locate(data, character_id)
        group = self._valid_group(data['characters'][index])
        remaining = [voice for voice in group['voices'] if voice['character_id'] != card_id]
        if len(remaining) == len(group['voices']):
            raise ValueError('该角色名下没有这条声线。')
        if remaining:
            data['characters'][index]['voices'] = remaining
        else:
            data['characters'].pop(index)
        self._write(data)

    def remove_card_references(self, card_id):
        """Remove every library voice that points at a deleted character card."""
        _identifier(card_id, '声线卡 ID')
        data = self._read()
        changed = False
        groups = []
        for entry in data['characters']:
            if not isinstance(entry, dict):
                groups.append(entry)
                continue
            voices = entry.get('voices')
            if not isinstance(voices, list):
                groups.append(entry)
                continue
            remaining = [voice for voice in voices
                         if not isinstance(voice, dict) or voice.get('character_id') != card_id]
            if len(remaining) != len(voices):
                changed = True
                entry = dict(entry, voices=remaining)
            if not remaining and voices:
                continue
            groups.append(entry)
        if changed:
            data['characters'] = groups
            self._write(data)


def library_view(root=None):
    """Library contents for the UI, with cards that no longer load marked."""
    library = VoiceLibrary(root)
    store = CharacterStore(library.root)
    try:
        groups = library.load_all()
    except (OSError, ValueError) as exc:
        return dict(ok=False, error=str(exc), characters=[])
    result = []
    for group in groups:
        voices = []
        for voice in group['voices']:
            try:
                card = store.load(voice['character_id'])
            except (OSError, ValueError):
                continue
            voices.append(dict(voice, card=card))
        result.append(dict(group, voices=voices))
    return dict(ok=True, error='', characters=result)
