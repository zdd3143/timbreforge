"""Validated, serializable schema for non-destructive audio edit chains.

A chain is an ordered list of operations plus the hashes of its input and
output. Nothing here touches audio files: this module only answers "is this
recipe complete, in range and reproducible?" so the pipeline and the character
manifest can share one definition.
"""
from __future__ import annotations

import hashlib
import json
import math
import re

SCHEMA_VERSION = 1
MAX_OPERATIONS = 32
MAX_EQ_BANDS = 8
HASH_RE = re.compile(r'[0-9a-f]{64}\Z')
EVENT_NAMES = ('revert_to_original', 'promote_candidate', 'duplicated', 'imported')

# name -> {param: (kind, minimum, maximum, default)}
# "number" params accept int or float but are always stored as float; the few
# sample/block sizes are integer-only so a chain cannot silently mean 3.5 ms.
_SPECS = {
    'trim_silence': {
        'threshold_db': ('number', -90.0, -10.0, -42.0),
        'keep_leading_ms': ('number', 0.0, 500.0, 50.0),
        'keep_trailing_ms': ('number', 0.0, 500.0, 80.0),
    },
    'denoise': {
        'strength': ('number', 0.0, 1.0, 0.35),
        'window_ms': ('integer', 16, 64, 32),
        'noise_seconds': ('number', 0.05, 2.0, 0.35),
    },
    'de_click': {
        'threshold': ('number', 2.0, 20.0, 5.5),
        'window': ('integer', 3, 15, 5),
    },
    'normalize': {
        'target_lufs': ('number', -40.0, -6.0, -16.0),
        'ceiling_dbfs': ('number', -6.0, 0.0, -1.0),
        # Prevent a quiet/noisy reference from being amplified until its noise
        # floor becomes the dominant sound after denoising.
        'max_gain_db': ('number', 0.0, 24.0, 8.0),
    },
    'compress': {
        'threshold_db': ('number', -60.0, 0.0, -18.0),
        'ratio': ('number', 1.0, 20.0, 2.0),
        'attack_ms': ('number', 0.1, 200.0, 10.0),
        'release_ms': ('number', 5.0, 2000.0, 120.0),
        'makeup_db': ('number', -12.0, 24.0, 0.0),
    },
}
EQ_BAND_TYPES = ('highpass', 'lowpass', 'peaking', 'lowshelf', 'highshelf')
EQ_BAND_SPECS = {
    'freq': ('number', 20.0, 20000.0, 200.0),
    'gain_db': ('number', -18.0, 18.0, 0.0),
    'q': ('number', 0.1, 10.0, 0.707),
}
OPERATION_NAMES = tuple(_SPECS) + ('eq',)
# Reasonable ordering inside a preset; explicit chains may use any order.
SUGGESTED_ORDER = ('trim_silence', 'de_click', 'denoise', 'eq', 'compress', 'normalize')

_LABELS = {
    'trim_silence': '去除首尾静音',
    'denoise': '基础降噪',
    'de_click': '去除点击声',
    'normalize': '响度标准化',
    'compress': '动态压缩',
    'eq': '均衡器',
}


class ChainError(ValueError):
    """Raised when an edit chain cannot be validated or safely interpreted."""


def _number(value, path, low, high, *, integer=False):
    if type(value) is bool or not isinstance(value, (int, float)):
        raise ChainError(f'{path} 必须是数字。')
    if integer and type(value) is not int:
        raise ChainError(f'{path} 必须是整数。')
    if not math.isfinite(float(value)):
        raise ChainError(f'{path} 必须是有限数字。')
    if not low <= float(value) <= high:
        raise ChainError(f'{path} 超出允许范围 {low}–{high}。')
    return int(value) if integer else float(value)


def validate_operation(operation):
    """Return a normalized copy of one operation; reject unknown keys."""
    if not isinstance(operation, dict):
        raise ChainError('音频处理步骤必须是对象。')
    name = operation.get('name')
    if name not in OPERATION_NAMES:
        raise ChainError(f'不支持的音频处理步骤：{name!r}。')
    if name == 'eq':
        bands = operation.get('bands')
        if not isinstance(bands, list) or not 1 <= len(bands) <= MAX_EQ_BANDS:
            raise ChainError(f'eq.bands 必须包含 1–{MAX_EQ_BANDS} 个频段。')
        normalized_bands = []
        for index, band in enumerate(bands):
            path = f'eq.bands[{index}]'
            if not isinstance(band, dict) or set(band) != {'type', 'freq', 'gain_db', 'q'}:
                raise ChainError(f'{path} 必须且只能包含 type/freq/gain_db/q。')
            if band['type'] not in EQ_BAND_TYPES:
                raise ChainError(f'{path}.type 不支持：{band["type"]!r}。')
            normalized_bands.append(dict(
                type=band['type'],
                freq=_number(band['freq'], path + '.freq',
                             EQ_BAND_SPECS['freq'][1], EQ_BAND_SPECS['freq'][2]),
                gain_db=_number(band['gain_db'], path + '.gain_db',
                                EQ_BAND_SPECS['gain_db'][1], EQ_BAND_SPECS['gain_db'][2]),
                q=_number(band['q'], path + '.q',
                          EQ_BAND_SPECS['q'][1], EQ_BAND_SPECS['q'][2]),
            ))
        allowed = {'name', 'bands'}
        unknown = set(operation) - allowed
        if unknown:
            raise ChainError(f'eq 包含未知参数：{sorted(unknown)}。')
        return dict(name='eq', bands=normalized_bands)

    spec = _SPECS[name]
    unknown = set(operation) - set(spec) - {'name'}
    if unknown:
        raise ChainError(f'{name} 包含未知参数：{sorted(unknown)}。')
    normalized = dict(name=name)
    for key, (kind, low, high, default) in spec.items():
        value = operation.get(key, default)
        normalized[key] = _number(value, f'{name}.{key}', low, high, integer=(kind == 'integer'))
    if name == 'de_click' and normalized['window'] % 2 == 0:
        # An even median window cannot be centred on a sample, so the detected
        # click would always be shifted by half a sample.
        raise ChainError('de_click.window 必须是奇数。')
    return normalized


def validate_operations(operations):
    """Validate an ordered operation list and return its normalization."""
    if not isinstance(operations, list) or len(operations) > MAX_OPERATIONS:
        raise ChainError(f'operations 必须是不超过 {MAX_OPERATIONS} 步的列表。')
    normalized, seen_single = [], set()
    for index, operation in enumerate(operations):
        item = validate_operation(operation)
        # trim/denoise/de_click/normalize are single-instance so the chain is
        # predictable; eq and compress may legitimately appear twice.
        if item['name'] in {'trim_silence', 'denoise', 'de_click', 'normalize'}:
            if item['name'] in seen_single:
                raise ChainError(f'{item["name"]} 在一条处理链中只能出现一次。')
            seen_single.add(item['name'])
        normalized.append(item)
    return normalized


def chain_digest(operations):
    """Stable SHA-256 of a normalized chain; used to fingerprint clone prompts."""
    normalized = validate_operations(operations)
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def _hash(value, path, *, optional=False):
    if value is None and optional:
        return None
    if not isinstance(value, str) or not HASH_RE.fullmatch(value):
        raise ChainError(f'{path} 必须是 64 位小写十六进制 SHA-256。')
    return value


def _text(value, path, limit=512):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ChainError(f'{path} 必须是不超过 {limit} 字的非空文本。')
    return value


def _time(value, path):
    from datetime import datetime
    if not isinstance(value, str):
        raise ChainError(f'{path} 必须是 ISO 时间字符串。')
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ChainError(f'{path} 不是有效的 ISO 时间。') from exc
    if parsed.tzinfo is None:
        raise ChainError(f'{path} 必须包含时区。')
    return value


def make_edit_document(*, source_audio, source_hash, operations, output_audio, output_hash,
                       preset='', sample_rate=0, channels=0, duration_in=0.0,
                       duration_out=0.0, created_at=None, notes=None):
    """Build a validated edit document. The pipeline fills measured durations."""
    from datetime import datetime, timezone
    document = {
        'schema_version': SCHEMA_VERSION,
        'source_audio': _text(source_audio, 'source_audio'),
        'source_hash': _hash(source_hash, 'source_hash'),
        'operations': validate_operations(operations),
        'output_audio': _text(output_audio, 'output_audio'),
        'output_hash': _hash(output_hash, 'output_hash'),
        'preset': _text(preset, 'preset', 64) if preset else '',
        'sample_rate': int(sample_rate) if sample_rate else 0,
        'channels': int(channels) if channels else 0,
        'duration_in': _number(duration_in, 'duration_in', 0.0, 3600.0),
        'duration_out': _number(duration_out, 'duration_out', 0.0, 3600.0),
        'created_at': _time(created_at or datetime.now(timezone.utc).isoformat(), 'created_at'),
        'notes': list(notes or []),
    }
    for note in document['notes']:
        if not isinstance(note, str) or len(note) > 500:
            raise ChainError('notes 必须是不超过 500 字的文本列表。')
    return validate_document(document)


def validate_document(document):
    """Validate a stored edit document; reject missing or extra fields."""
    if not isinstance(document, dict):
        raise ChainError('处理记录必须是对象。')
    fields = {'schema_version', 'source_audio', 'source_hash', 'operations',
              'output_audio', 'output_hash', 'preset', 'sample_rate', 'channels',
              'duration_in', 'duration_out', 'created_at', 'notes'}
    if set(document) != fields:
        raise ChainError('处理记录字段与 schema 不符。')
    if type(document['schema_version']) is not int or document['schema_version'] != SCHEMA_VERSION:
        raise ChainError('处理记录版本不受支持。')
    normalized = dict(document)
    normalized['source_audio'] = _text(document['source_audio'], 'source_audio')
    normalized['source_hash'] = _hash(document['source_hash'], 'source_hash')
    normalized['operations'] = validate_operations(document['operations'])
    normalized['output_audio'] = _text(document['output_audio'], 'output_audio')
    normalized['output_hash'] = _hash(document['output_hash'], 'output_hash')
    preset = document['preset']
    if preset != '' and (not isinstance(preset, str) or len(preset) > 64):
        raise ChainError('preset 必须是不超过 64 字的文本。')
    for key in ('sample_rate', 'channels'):
        if type(document[key]) is not int or not 0 <= document[key] <= 384000:
            raise ChainError(f'{key} 无效。')
    for key in ('duration_in', 'duration_out'):
        normalized[key] = _number(document[key], key, 0.0, 3600.0)
    normalized['created_at'] = _time(document['created_at'], 'created_at')
    notes = document['notes']
    if not isinstance(notes, list) or len(notes) > 20:
        raise ChainError('notes 必须是不超过 20 条文本的列表。')
    for note in notes:
        if not isinstance(note, str) or len(note) > 500:
            raise ChainError('notes 必须是不超过 500 字的文本列表。')
    return normalized


def make_event(*, event, at=None, source_audio=None, source_hash=None,
               output_audio=None, output_hash=None, note=''):
    """Build a validated history event (revert/promote/duplicate/import).

    Events record *why* the current canonical audio changed without pretending
    to be a reproducible operation chain.
    """
    from datetime import datetime, timezone
    if event not in EVENT_NAMES:
        raise ChainError(f'不支持的处理历史事件：{event!r}。')
    document = {
        'schema_version': SCHEMA_VERSION,
        'event': event,
        'at': _time(at or datetime.now(timezone.utc).isoformat(), 'at'),
        'source_audio': _text(source_audio, 'source_audio') if source_audio else None,
        'source_hash': _hash(source_hash, 'source_hash', optional=True),
        'output_audio': _text(output_audio, 'output_audio') if output_audio else None,
        'output_hash': _hash(output_hash, 'output_hash', optional=True),
        'note': note if isinstance(note, str) and len(note) <= 500 else '',
    }
    return validate_event(document)


def validate_event(document):
    if not isinstance(document, dict):
        raise ChainError('处理历史事件必须是对象。')
    fields = {'schema_version', 'event', 'at', 'source_audio', 'source_hash',
              'output_audio', 'output_hash', 'note'}
    if set(document) != fields:
        raise ChainError('处理历史事件字段与 schema 不符。')
    if type(document['schema_version']) is not int or document['schema_version'] != SCHEMA_VERSION:
        raise ChainError('处理历史事件版本不受支持。')
    if document['event'] not in EVENT_NAMES:
        raise ChainError('处理历史事件类型不受支持。')
    return {
        'schema_version': SCHEMA_VERSION,
        'event': document['event'],
        'at': _time(document['at'], 'at'),
        'source_audio': (_text(document['source_audio'], 'source_audio')
                         if document['source_audio'] is not None else None),
        'source_hash': _hash(document['source_hash'], 'source_hash', optional=True),
        'output_audio': (_text(document['output_audio'], 'output_audio')
                         if document['output_audio'] is not None else None),
        'output_hash': _hash(document['output_hash'], 'output_hash', optional=True),
        'note': document['note'] if isinstance(document['note'], str) and len(document['note']) <= 500 else '',
    }


def validate_chain_entry(entry):
    """Validate one entry of ``audio_edit_chain`` (edit document or event)."""
    if not isinstance(entry, dict):
        raise ChainError('处理链条目必须是对象。')
    if 'event' in entry:
        return validate_event(entry)
    return validate_document(entry)


def describe_operations(operations):
    """Short Chinese summary used by the UI and by saved provenance."""
    parts = []
    for operation in validate_operations(operations):
        name = operation['name']
        label = _LABELS[name]
        if name == 'trim_silence':
            label += f"（阈值 {operation['threshold_db']:g} dB）"
        elif name == 'denoise':
            label += f"（强度 {operation['strength']:g}）"
        elif name == 'de_click':
            label += f"（阈值 {operation['threshold']:g}）"
        elif name == 'normalize':
            label += f"（{operation['target_lufs']:g} LUFS）"
        elif name == 'compress':
            label += f"（{operation['ratio']:g}:1）"
        elif name == 'eq':
            label += f"（{len(operation['bands'])} 个频段）"
        parts.append(label)
    return parts or ['无处理，直接使用原始音频']


def summarize_chain(chain):
    """Aggregate counts for status text; never raises on damaged history."""
    operations, documents, events, labels = 0, 0, 0, []
    for entry in chain if isinstance(chain, list) else []:
        try:
            item = validate_chain_entry(entry)
        except ChainError:
            continue
        if 'event' in item:
            events += 1
        else:
            documents += 1
            operations += len(item['operations'])
            # A history entry is described on its own: the same single-instance
            # operation may legitimately appear in several successive edits.
            labels.extend(describe_operations(item['operations']))
    return {'documents': documents, 'events': events, 'operations': operations,
            'labels': labels or ['无处理，直接使用原始音频']}

