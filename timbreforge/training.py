"""Offline, opt-in LoRA training preparation and registry hand-off.

This is an advanced, fully offline feature. It prepares a licensed dataset, creates
a reproducible training *job*, optionally invokes an **external** trainer the user
already installed, records evaluation evidence, and registers the resulting
checkpoint as a local LoRA model. It never imports PyTorch, never downloads a model
or dependency, never re-encodes source recordings, and never overwrites the Base
model, an existing dataset, or an existing checkpoint.

Design notes
------------
* Dataset clips are validated with SoundFile and copied byte-for-byte. The SNR
  figure is an *auxiliary estimate*: frame RMS over 20 ms frames (speech frames
  are those above -42 dBFS) compared with the 10th-percentile frame RMS mapped to
  dB. It is not a calibrated measurement and must not be treated as one.
* The VRAM figure from :func:`estimate_vram_gb` is a documented heuristic:
  ``2.2 x parameter_billions + 1.5`` GB for LoRA-style training. A 0.6 B model
  therefore estimates ~2.8 GB. It is an estimate, not a guarantee.
* A trainer is configured with the environment variable
  ``TIMBREFORGE_TRAINING_COMMAND`` as a JSON string array containing ``{request}``
  and ``{result}`` placeholders. Without it (and without an injected test runner)
  :func:`run_training` raises :class:`TrainingUnavailable` and runs nothing.
* The trainer writes into the staging directory passed in the request. On success
  the staging directory is published with an atomic rename; a previous checkpoint
  is first moved aside to ``checkpoints.previous-<timestamp>`` so it can be
  recovered. Failed runs leave the existing checkpoint untouched.
* Registration goes through :class:`timbreforge.model_registry.ModelRegistry`
  with kind ``lora``. Registering may fail when the checkpoint is not a valid
  local model directory (for example when it has no ``config.json``); that error
  is propagated instead of being hidden.
* No real training loop lives here: an external trainer must provide the actual
  LoRA implementation and enough GPU memory. Everything else is validation,
  bookkeeping, and atomic file handling.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import random
import re
import shutil
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

from timbreforge.characters import CharacterStore
from timbreforge.model_registry import ModelRegistry

DATASET_SCHEMA = 1
JOB_SCHEMA = 1

__all__ = [
    'DATASET_SCHEMA', 'JOB_SCHEMA', 'TrainingError', 'TrainingUnavailable',
    'create_training_job', 'dataset_summary', 'estimate_vram_gb', 'load_dataset',
    'prepare_dataset', 'record_evaluation', 'register_trained_model',
    'rollback_model', 'run_training',
]

_TRAINER_ENV = 'TIMBREFORGE_TRAINING_COMMAND'
_SPLITS = ('train', 'valid')
_JOB_STATUSES = ('prepared', 'completed', 'failed')

_MIN_SAMPLE_RATE = 8000
_MAX_SAMPLE_RATE = 192000
_MAX_TRANSCRIPT_CHARS = 5000
_MAX_EVALUATION_NOTES = 1000
_MIN_VOICED_RATIO = 0.25
_MIN_SNR_DB = 12.0
_SILENCE_DBFS = -42.0
_CLIP_THRESHOLD = 0.999
_MAX_CLIP_FRACTION = 0.001
_FRAME_SECONDS = 0.02

_VRAM_GB_PER_BILLION = 2.2
_VRAM_OVERHEAD_GB = 1.5

_ID_RE = re.compile(r'[0-9a-f]{32}\Z')
_HASH_RE = re.compile(r'[0-9a-f]{64}\Z')
_CLIP_PATH_RE = re.compile(r'clips/[0-9a-f]{16}(?:\.[A-Za-z0-9]{1,10})\Z')
_SUFFIX_RE = re.compile(r'\.[A-Za-z0-9]{1,10}\Z')

_SETTINGS_FIELDS = frozenset({
    'target_sample_rate', 'min_seconds', 'max_seconds', 'train_ratio', 'seed',
})
_COUNT_FIELDS = frozenset({'train', 'valid', 'total'})
_ITEM_FIELDS = frozenset({
    'audio', 'sha256', 'transcript', 'split', 'duration', 'sample_rate',
    'channels', 'snr_db', 'voiced_ratio',
})
_REJECTED_FIELDS = frozenset({'audio', 'reason'})
_DATASET_FIELDS = frozenset({
    'schema_version', 'created_at', 'counts', 'total_duration', 'settings',
    'items', 'rejected', 'license_confirmed', 'warnings',
})
# Convenience aliases consumed by the advanced-training UI; optional in files so
# a hand-written, spec-shaped dataset.json remains loadable.
_DATASET_ALIAS_FIELDS = frozenset({'train_count', 'valid_count'})

_DEFAULT_PARAMS = {
    'rank': 16,
    'alpha': 32,
    'epochs': 5,
    'learning_rate': 1e-4,
    'batch_size': 1,
    'gradient_accumulation': 8,
    'save_every': 1,
    'seed': 42,
}
_PARAM_SPECS = {
    'rank': (int, 4, 256),
    'alpha': (int, 4, 512),
    'epochs': (int, 1, 50),
    'learning_rate': (float, 1e-6, 1e-2),
    'batch_size': (int, 1, 32),
    'gradient_accumulation': (int, 1, 64),
    'save_every': (int, 1, 20),
    'seed': (int, 0, 0x7FFFFFFFFFFFFFFF),
}

_JOB_REQUIRED_FIELDS = frozenset({
    'schema_version', 'id', 'name', 'status', 'created_at', 'dataset',
    'base_model', 'params', 'vram_estimate_gb', 'trainer', 'log_file',
    'checkpoint_dir', 'evaluation', 'registered_model', 'role_id', 'warnings',
})
_JOB_OPTIONAL_FIELDS = frozenset({'finished_at', 'error', 'last_run', 'path'})
_TRAINER_FIELDS = frozenset({'configured', 'command', 'returncode'})
_LAST_RUN_FIELDS = frozenset({
    'started_at', 'finished_at', 'request_file', 'result_file', 'staging_dir',
    'command', 'returncode', 'previous_checkpoint', 'result_checkpoint_dir',
})


class TrainingError(RuntimeError):
    """Raised when training preparation, execution, or registration is unsafe."""


class TrainingUnavailable(TrainingError):
    """Raised when no external trainer is configured, so nothing is trained."""


# --------------------------------------------------------------------- helpers
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp(value, label):
    if not isinstance(value, str) or not value.strip():
        raise TrainingError(f'{label} 必须是带时区的 ISO 时间文本。')
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise TrainingError(f'{label} 必须是带时区的 ISO 时间文本。') from exc
    if parsed.tzinfo is None:
        raise TrainingError(f'{label} 必须包含时区。')
    return parsed


def _number(value, label, *, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingError(f'{label} 必须是数字。')
    result = float(value)
    if not math.isfinite(result):
        raise TrainingError(f'{label} 必须是有限数字。')
    if minimum is not None and result < minimum:
        raise TrainingError(f'{label} 不能小于 {minimum:g}。')
    if maximum is not None and result > maximum:
        raise TrainingError(f'{label} 不能大于 {maximum:g}。')
    return result


def _integer(value, label, *, minimum=None, maximum=None):
    if type(value) is not int:
        raise TrainingError(f'{label} 必须是整数。')
    if minimum is not None and value < minimum:
        raise TrainingError(f'{label} 不能小于 {minimum}。')
    if maximum is not None and value > maximum:
        raise TrainingError(f'{label} 不能大于 {maximum}。')
    return value


def _path_argument(value, label):
    """Return an expanded Path, rejecting empty or non-path arguments."""
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise TrainingError(f'{label} 必须是路径。') from exc
    if not isinstance(raw, str) or not raw.strip():
        raise TrainingError(f'{label} 必须是非空路径。')
    return Path(raw).expanduser()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path, data) -> None:
    """Publish UTF-8 JSON through a fsynced sibling file plus ``os.replace``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}-{uuid.uuid4().hex}.tmp')
    try:
        with temporary.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(data, stream, ensure_ascii=False, allow_nan=False,
                      indent=2, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_jsonl(path, rows) -> None:
    """Write one JSON object per line; the file exists even when ``rows`` is empty."""
    path = Path(path)
    temporary = path.with_name(f'.{path.name}-{uuid.uuid4().hex}.tmp')
    try:
        with temporary.open('x', encoding='utf-8', newline='\n') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False,
                                        sort_keys=True))
                stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _append_log(path: Path, text: str) -> None:
    """Best-effort progress line; failing to log must not crash a run."""
    try:
        with Path(path).open('a', encoding='utf-8', newline='\n') as stream:
            stream.write(text)
    except OSError:
        pass


def _remove_empty_dir(path: Path) -> None:
    try:
        if path.is_dir() and not path.is_symlink() and not any(path.iterdir()):
            path.rmdir()
    except OSError:
        pass


def _cleanup_temp(path: Path | None, prefix: str) -> None:
    if path is None:
        return
    try:
        if path.name.startswith(prefix) and path.exists() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)
    except OSError:
        pass


def _validate_role_id(value, label='role_id'):
    if value is None:
        return None
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise TrainingError(f'{label} 必须是 32 位小写十六进制角色 ID。')
    return value


def _validate_role_reference(value, label='role_id'):
    """The job's optional role hint is stored only; it is never used as a path."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 120:
        raise TrainingError(f'{label} 必须是不超过 120 字的非空文本或 None。')
    return value.strip()


def _validate_name(name) -> str:
    if not isinstance(name, str) or not name.strip() or len(name.strip()) > 120:
        raise TrainingError('训练任务名称必须是不超过 120 字的非空文本。')
    return name.strip()


def _validate_params(params) -> dict:
    """Return a canonical parameter dict, rejecting unknown keys and bad ranges."""
    if not isinstance(params, dict):
        raise TrainingError('params 必须是字典。')
    unknown = sorted(set(params) - set(_PARAM_SPECS))
    if unknown:
        raise TrainingError(
            'params 包含未知参数：' + '、'.join(unknown)
            + '；允许的参数为：' + '、'.join(_PARAM_SPECS) + '。')
    result = dict(_DEFAULT_PARAMS)
    for key, (kind, low, high) in _PARAM_SPECS.items():
        if key not in params:
            continue
        value = params[key]
        if kind is int:
            value = _integer(value, f'params.{key}', minimum=low, maximum=high)
        else:
            value = _number(value, f'params.{key}', minimum=low, maximum=high)
        result[key] = value
    return result


def _validate_stored_params(params) -> dict:
    if not isinstance(params, dict) or set(params) != set(_PARAM_SPECS):
        raise TrainingError('job.json 中的 params 字段不完整或包含未知参数。')
    return _validate_params(params)


def _validate_counts(counts, label='counts') -> dict:
    if not isinstance(counts, dict) or set(counts) != _COUNT_FIELDS:
        raise TrainingError(f'{label} 必须且只能包含 train/valid/total。')
    for key in _COUNT_FIELDS:
        _integer(counts[key], f'{label}.{key}', minimum=0)
    if counts['train'] + counts['valid'] != counts['total']:
        raise TrainingError(f'{label} 的 train + valid 必须等于 total。')
    return counts


def _validate_settings(settings) -> dict:
    if not isinstance(settings, dict) or set(settings) != _SETTINGS_FIELDS:
        raise TrainingError('settings 字段不完整或包含未知字段。')
    target = _integer(settings['target_sample_rate'], 'settings.target_sample_rate',
                      minimum=_MIN_SAMPLE_RATE, maximum=_MAX_SAMPLE_RATE)
    minimum = _number(settings['min_seconds'], 'settings.min_seconds', minimum=0.0)
    maximum = _number(settings['max_seconds'], 'settings.max_seconds', minimum=0.0)
    if not minimum < maximum:
        raise TrainingError('settings.min_seconds 必须小于 settings.max_seconds。')
    ratio = _number(settings['train_ratio'], 'settings.train_ratio',
                    minimum=0.0, maximum=1.0)
    if not 0.0 < ratio < 1.0:
        raise TrainingError('settings.train_ratio 必须在 0 和 1 之间（不含端点）。')
    seed = _integer(settings['seed'], 'settings.seed', minimum=0,
                    maximum=0x7FFFFFFFFFFFFFFF)
    return {
        'target_sample_rate': target,
        'min_seconds': minimum,
        'max_seconds': maximum,
        'train_ratio': ratio,
        'seed': seed,
    }


def _normalize_trainer_command(value, label='训练命令'):
    """Validate a command array of nonempty strings and return it (or ``None``)."""
    if value is None:
        return None
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise TrainingError(f'{label} 必须是字符串数组。')
    if not value:
        return None
    if len(value) > 64:
        raise TrainingError(f'{label} 元素过多。')
    parts = []
    for part in value:
        if not isinstance(part, str) or not part.strip():
            raise TrainingError(f'{label} 的每个元素都必须是非空字符串。')
        if len(part) > 4000:
            raise TrainingError(f'{label} 的元素过长。')
        parts.append(part)
    return parts


def _read_trainer_command():
    """Read ``TIMBREFORGE_TRAINING_COMMAND`` without executing anything."""
    raw = os.environ.get(_TRAINER_ENV)
    if raw is None or not raw.strip():
        return None, []
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise TrainingError(
            f'环境变量 {_TRAINER_ENV} 必须是 JSON 字符串数组，例如：'
            '["python","train_lora.py","--request","{request}",'
            '"--result","{result}"]。') from exc
    if not isinstance(parsed, list):
        raise TrainingError(f'环境变量 {_TRAINER_ENV} 必须是 JSON 字符串数组。')
    if not parsed:
        return None, [f'环境变量 {_TRAINER_ENV} 是空数组，训练器仍未配置。']
    command = _normalize_trainer_command(parsed, f'环境变量 {_TRAINER_ENV}')
    warnings = []
    joined = ' '.join(command)
    if '{request}' not in joined or '{result}' not in joined:
        warnings.append(
            f'环境变量 {_TRAINER_ENV} 中的命令缺少 {{request}} 或 {{result}} 占位符；'
            '默认运行器无法把请求和结果路径传给训练器。')
    return command, warnings


def _require_placeholders(command) -> None:
    joined = ' '.join(command)
    if '{request}' not in joined or '{result}' not in joined:
        raise TrainingError(
            '默认运行器要求训练命令中包含 {request} 和 {result} 占位符，'
            '以便传入请求 JSON 和结果 JSON 的路径。')


def _format_command(command, request_path, result_path) -> list:
    formatted = []
    for part in command:
        if '{' in part or '}' in part:
            try:
                part = part.format(request=str(request_path), result=str(result_path))
            except (KeyError, IndexError, ValueError, AttributeError) as exc:
                raise TrainingError(
                    '训练命令中的花括号占位符无效；只支持 {request} 和 {result}。') from exc
        formatted.append(part)
    return formatted


def _validate_timeout(timeout) -> float:
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise TrainingError('timeout 必须是正数秒。')
    value = float(timeout)
    if not math.isfinite(value) or value <= 0:
        raise TrainingError('timeout 必须是正数秒。')
    return value


# ------------------------------------------------------------ dataset building
class _ClipRejected(Exception):
    """Internal: one item failed audio validation and is skipped."""


def _normalize_item(item, index):
    if isinstance(item, dict):
        if set(item) != {'audio', 'transcript'}:
            raise TrainingError(
                f'第 {index + 1} 项必须是 {{"audio": 路径, "transcript": 文本}}，'
                '不能包含其它字段。')
        audio, transcript = item['audio'], item['transcript']
    elif isinstance(item, (tuple, list)):
        if len(item) != 2:
            raise TrainingError(f'第 {index + 1} 项必须是 (audio, transcript) 二元组。')
        audio, transcript = item
    else:
        raise TrainingError(
            f'第 {index + 1} 项必须是 {{"audio": 路径, "transcript": 文本}} 或二元组。')
    if isinstance(audio, bool):
        raise TrainingError(f'第 {index + 1} 项的 audio 必须是路径。')
    try:
        audio_path = os.fspath(audio)
    except TypeError as exc:
        raise TrainingError(f'第 {index + 1} 项的 audio 必须是路径。') from exc
    if not isinstance(audio_path, str) or not audio_path.strip():
        raise TrainingError(f'第 {index + 1} 项的 audio 必须是非空路径。')
    if not isinstance(transcript, str):
        raise TrainingError(f'第 {index + 1} 项的 transcript 必须是文本。')
    if not transcript.strip():
        raise TrainingError(f'第 {index + 1} 项的 transcript 不能为空。')
    if len(transcript) > _MAX_TRANSCRIPT_CHARS:
        raise TrainingError(
            f'第 {index + 1} 项的 transcript 超过 {_MAX_TRANSCRIPT_CHARS} 字。')
    return Path(audio_path).expanduser(), transcript.strip()


def _frame_statistics(mono, sample_rate):
    """Return (voiced_ratio, snr_db) using the documented auxiliary heuristic."""
    frame_length = max(1, int(round(_FRAME_SECONDS * sample_rate)))
    usable = (len(mono) // frame_length) * frame_length
    if usable <= 0:
        raise _ClipRejected('音频过短，无法按 20 毫秒帧估算语音比例。')
    frames = mono[:usable].reshape(-1, frame_length)
    frame_rms = np.sqrt(np.mean(np.square(frames), axis=1))
    threshold = 10 ** (_SILENCE_DBFS / 20.0)
    voiced = frame_rms > threshold
    voiced_ratio = float(np.count_nonzero(voiced)) / frame_rms.size
    if voiced_ratio < _MIN_VOICED_RATIO:
        raise _ClipRejected(
            f'静音过多：有效语音帧仅占 {voiced_ratio * 100:.1f}%，低于 '
            f'{_MIN_VOICED_RATIO * 100:.0f}%。')
    speech_rms = float(np.mean(frame_rms[voiced])) if voiced.any() else 0.0
    noise_rms = float(np.percentile(frame_rms, 10.0))
    denom = max(noise_rms, 1e-9)
    snr_db = 20.0 * math.log10(max(speech_rms, 1e-9) / denom)
    snr_db = min(snr_db, 120.0)
    if snr_db < _MIN_SNR_DB:
        raise _ClipRejected(
            f'信噪比过低（约 {snr_db:.1f} dB，低于 {_MIN_SNR_DB:.0f} dB）。')
    return voiced_ratio, snr_db


def _inspect_clip(path: Path, min_seconds: float, max_seconds: float):
    """Decode one source recording and return measured fields or reject it."""
    try:
        with sf.SoundFile(str(path)) as recording:
            frames = recording.frames
            channels = recording.channels
            sample_rate = recording.samplerate
            if frames <= 0 or channels <= 0 or sample_rate <= 0:
                raise _ClipRejected('文件为空或不包含音频采样。')
            if not _MIN_SAMPLE_RATE <= sample_rate <= _MAX_SAMPLE_RATE:
                raise _ClipRejected(
                    f'采样率 {sample_rate} Hz 不在 {_MIN_SAMPLE_RATE}–'
                    f'{_MAX_SAMPLE_RATE} Hz 允许范围内。')
            duration = frames / sample_rate
            if duration < min_seconds or duration > max_seconds:
                raise _ClipRejected(
                    f'时长 {duration:.2f} 秒不在 {min_seconds:g}–{max_seconds:g} 秒范围内。')
            mono = np.empty(frames, dtype=np.float64)
            offset = 0
            peak = 0.0
            clipped = 0
            for block in recording.blocks(blocksize=65536, dtype='float64',
                                          always_2d=True):
                if not np.isfinite(block).all():
                    raise _ClipRejected('采样包含非有限数值（NaN 或 Inf）。')
                if block.size:
                    peak = max(peak, float(np.max(np.abs(block))))
                    clipped += int(np.count_nonzero(np.abs(block) >= _CLIP_THRESHOLD))
                count = len(block)
                mono[offset:offset + count] = block.mean(axis=1)
                offset += count
            if offset != frames:
                raise _ClipRejected('文件不完整或读取长度与声明不符。')
    except _ClipRejected:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise _ClipRejected(f'无法读取音频文件：{exc}') from exc
    if peak <= 1e-8:
        raise _ClipRejected('音频近乎静音（峰值接近 0）。')
    clip_fraction = clipped / float(frames * channels)
    if clip_fraction >= _MAX_CLIP_FRACTION:
        raise _ClipRejected(
            f'削波过多：{clip_fraction * 100:.2f}% 的采样达到或超过 '
            f'{_CLIP_THRESHOLD:g}。')
    voiced_ratio, snr_db = _frame_statistics(mono, sample_rate)
    suffix = path.suffix if _SUFFIX_RE.fullmatch(path.suffix or '') else '.audio'
    return {
        'duration': round(duration, 6),
        'sample_rate': int(sample_rate),
        'channels': int(channels),
        'snr_db': round(float(snr_db), 3),
        'voiced_ratio': round(float(voiced_ratio), 4),
        'suffix': suffix,
    }


def _split_items(accepted, train_ratio: float, seed: int):
    """Deterministic sha-sorted shuffle; >=2 clips always leave one for valid."""
    ordered = sorted(accepted, key=lambda record: record['sha256'])
    shuffled = list(ordered)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    warnings = []
    if total == 1:
        train_count = 1
        warnings.append('只有 1 条录音通过校验，valid 集为空；请至少准备 2 条录音以获得验证集。')
    else:
        train_count = int(round(total * train_ratio))
        train_count = max(1, min(total - 1, train_count))
    train_shas = {record['sha256'] for record in shuffled[:train_count]}
    return ordered, train_shas, warnings


def _build_manifest(accepted, ordered, train_shas, rejected, settings, warnings):
    items = []
    for record in ordered:
        split = 'train' if record['sha256'] in train_shas else 'valid'
        items.append({
            'audio': f"clips/{record['filename']}",
            'sha256': record['sha256'],
            'transcript': record['transcript'],
            'split': split,
            'duration': record['duration'],
            'sample_rate': record['sample_rate'],
            'channels': record['channels'],
            'snr_db': record['snr_db'],
            'voiced_ratio': record['voiced_ratio'],
        })
    counts = {
        'train': sum(1 for item in items if item['split'] == 'train'),
        'valid': sum(1 for item in items if item['split'] == 'valid'),
        'total': len(items),
    }
    total_duration = round(sum(item['duration'] for item in items), 6)
    return {
        'schema_version': DATASET_SCHEMA,
        'created_at': _now(),
        'counts': counts,
        'train_count': counts['train'],
        'valid_count': counts['valid'],
        'total_duration': total_duration,
        'settings': settings,
        'items': items,
        'rejected': rejected,
        'license_confirmed': True,
        'warnings': warnings,
    }


def prepare_dataset(items, output_dir, *, license_confirmed=False,
                    target_sample_rate=24000, min_seconds=1.5, max_seconds=15.0,
                    train_ratio=0.9, seed=42) -> dict:
    """Validate licensed recordings and publish an immutable dataset directory.

    Each item is a dict ``{'audio': path, 'transcript': str}`` or a
    ``(path, transcript)`` tuple. Audio is decoded only for measurement; accepted
    files are then copied byte-for-byte into ``clips/<sha256[:16]><suffix>`` and
    the source recordings are never modified. Bad audio is skipped and listed in
    ``rejected`` with a Simplified-Chinese reason.

    The SNR field is an auxiliary estimate (see the module docstring), not a
    calibrated measurement. The dataset is built in a temporary sibling directory
    and published with ``os.rename``; an existing output directory is refused.
    ``train.jsonl`` and ``valid.jsonl`` repeat the manifest records, and their
    ``audio`` values are relative to the published dataset directory.
    """
    if license_confirmed is not True:
        raise TrainingError(
            '必须确认所有录音都有合法使用权，才能准备训练数据集；'
            '未确认时不会读取、复制或写入任何文件。')
    if not isinstance(items, (list, tuple)) or not items:
        raise TrainingError(
            'items 必须是非空列表，每项为 {"audio": 路径, "transcript": 文本} 或二元组。')
    settings = _validate_settings({
        'target_sample_rate': target_sample_rate,
        'min_seconds': min_seconds,
        'max_seconds': max_seconds,
        'train_ratio': train_ratio,
        'seed': seed,
    })

    raw_output = _path_argument(output_dir, 'output_dir')
    if raw_output.exists() or raw_output.is_symlink():
        raise TrainingError(f'输出目录已存在，拒绝覆盖：{raw_output}')
    output = raw_output.resolve()
    normalized = []
    for index, item in enumerate(items):
        normalized.append(_normalize_item(item, index))

    seen: dict[str, Path] = {}
    candidates = []
    rejected = []
    for source, transcript in normalized:
        if not source.is_file():
            rejected.append({'audio': str(source),
                             'reason': '录音文件不存在或不是普通文件。'})
            continue
        try:
            digest = _sha256_file(source)
        except OSError as exc:
            rejected.append({'audio': str(source),
                             'reason': f'无法读取录音文件：{exc}'})
            continue
        if digest in seen:
            raise TrainingError(
                f'检测到重复的录音内容（SHA-256 相同）：{seen[digest]} 与 {source}。')
        seen[digest] = source
        candidates.append((source, transcript, digest))

    accepted = []
    for source, transcript, digest in candidates:
        try:
            fields = _inspect_clip(source, settings['min_seconds'],
                                   settings['max_seconds'])
        except _ClipRejected as exc:
            rejected.append({'audio': str(source), 'reason': str(exc)})
            continue
        accepted.append({
            'source': source,
            'sha256': digest,
            'transcript': transcript,
            'filename': digest[:16] + fields['suffix'],
            'duration': fields['duration'],
            'sample_rate': fields['sample_rate'],
            'channels': fields['channels'],
            'snr_db': fields['snr_db'],
            'voiced_ratio': fields['voiced_ratio'],
        })

    if not accepted:
        reasons = '；'.join(entry['reason'] for entry in rejected[:5]) or '没有输入录音'
        raise TrainingError(
            f'没有可用的录音片段，未创建数据集。前几条原因：{reasons}')

    ordered, train_shas, split_warnings = _split_items(
        accepted, settings['train_ratio'], settings['seed'])

    warnings = list(split_warnings)
    if rejected:
        warnings.append(f'有 {len(rejected)} 条录音未通过校验，原因见 rejected 字段。')
    if any(record['sample_rate'] != settings['target_sample_rate']
           for record in accepted):
        warnings.append(
            '部分录音的采样率与目标采样率不同；片段按原始字节复制、未重采样，'
            '训练器需自行处理。')

    manifest = _build_manifest(accepted, ordered, train_shas, rejected,
                               settings, warnings)
    _validate_dataset_manifest(manifest)

    try:
        output.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise TrainingError(f'无法创建数据集父目录：{exc}') from exc

    temporary = None
    try:
        temporary = Path(tempfile.mkdtemp(prefix='.dataset-tmp-', dir=str(output.parent)))
        clips = temporary / 'clips'
        clips.mkdir()
        for record in accepted:
            destination = clips / record['filename']
            try:
                shutil.copyfile(record['source'], destination)
            except OSError as exc:
                raise TrainingError(
                    f'复制录音失败：{record["source"]}：{exc}') from exc
            copied = _sha256_file(destination)
            try:
                original = _sha256_file(record['source'])
            except OSError as exc:
                raise TrainingError(
                    f'无法再次读取源录音：{record["source"]}：{exc}') from exc
            if copied != record['sha256'] or original != record['sha256']:
                raise TrainingError(
                    f'源录音在准备数据集的过程中发生变化：{record["source"]}；'
                    '已放弃发布数据集。')
        _write_jsonl(temporary / 'train.jsonl',
                     [item for item in manifest['items'] if item['split'] == 'train'])
        _write_jsonl(temporary / 'valid.jsonl',
                     [item for item in manifest['items'] if item['split'] == 'valid'])
        _write_json_atomic(temporary / 'dataset.json', manifest)
        if output.exists() or output.is_symlink():
            raise TrainingError(f'输出目录已存在，拒绝覆盖：{output}')
        os.rename(temporary, output)
        temporary = None
    finally:
        _cleanup_temp(temporary, '.dataset-tmp-')
    return dataset_summary(output)


# ------------------------------------------------------------ dataset reading
def _validate_item(item, settings, index):
    if not isinstance(item, dict) or set(item) != _ITEM_FIELDS:
        raise TrainingError(f'dataset.json 第 {index + 1} 条记录字段不完整或包含未知字段。')
    audio = item['audio']
    if not isinstance(audio, str) or not _CLIP_PATH_RE.fullmatch(audio):
        raise TrainingError(f'dataset.json 第 {index + 1} 条记录的 audio 路径不合法。')
    digest = item['sha256']
    if not isinstance(digest, str) or not _HASH_RE.fullmatch(digest):
        raise TrainingError(f'dataset.json 第 {index + 1} 条记录的 sha256 不合法。')
    if audio != f'clips/{digest[:16]}{Path(audio).suffix}':
        raise TrainingError(f'dataset.json 第 {index + 1} 条记录的文件名与哈希不一致。')
    transcript = item['transcript']
    if (not isinstance(transcript, str) or not transcript.strip()
            or len(transcript) > _MAX_TRANSCRIPT_CHARS):
        raise TrainingError(f'dataset.json 第 {index + 1} 条记录的 transcript 不合法。')
    if item['split'] not in _SPLITS:
        raise TrainingError(f'dataset.json 第 {index + 1} 条记录的 split 不合法。')
    duration = _number(item['duration'], f'dataset.json 第 {index + 1} 条时长',
                       minimum=0.0)
    if not settings['min_seconds'] - 1e-9 <= duration <= settings['max_seconds'] + 1e-9:
        raise TrainingError(f'dataset.json 第 {index + 1} 条时长超出数据集设置范围。')
    _integer(item['sample_rate'], f'dataset.json 第 {index + 1} 条采样率',
             minimum=_MIN_SAMPLE_RATE, maximum=_MAX_SAMPLE_RATE)
    _integer(item['channels'], f'dataset.json 第 {index + 1} 条通道数', minimum=1)
    _number(item['snr_db'], f'dataset.json 第 {index + 1} 条信噪比')
    voiced = _number(item['voiced_ratio'], f'dataset.json 第 {index + 1} 条语音比例',
                     minimum=0.0, maximum=1.0)
    return voiced


def _validate_dataset_manifest(manifest) -> dict:
    if not isinstance(manifest, dict):
        raise TrainingError('dataset.json 必须是 JSON 对象。')
    if (set(manifest) - _DATASET_FIELDS - _DATASET_ALIAS_FIELDS
            or _DATASET_FIELDS - set(manifest)):
        raise TrainingError('dataset.json 字段不完整或包含未知字段。')
    if type(manifest['schema_version']) is not int or manifest['schema_version'] != DATASET_SCHEMA:
        raise TrainingError('不支持的 dataset.json schema_version。')
    _timestamp(manifest['created_at'], 'dataset.created_at')
    settings = _validate_settings(manifest['settings'])
    counts = _validate_counts(manifest['counts'], 'dataset.counts')
    for field, expected in (('train_count', counts['train']),
                            ('valid_count', counts['valid'])):
        if field in manifest:
            _integer(manifest[field], f'dataset.{field}', minimum=0)
            if manifest[field] != expected:
                raise TrainingError(f'dataset.{field} 与 counts 不一致。')
    items = manifest['items']
    if not isinstance(items, list) or not items:
        raise TrainingError('dataset.json 必须包含至少一条音频记录。')
    if counts['total'] != len(items):
        raise TrainingError('dataset.counts.total 与 items 数量不一致。')
    if counts['train'] < 1 or counts['valid'] < 0:
        raise TrainingError('dataset.json 的 train/valid 数量不合法。')
    if counts['total'] >= 2 and counts['valid'] < 1:
        raise TrainingError('包含 2 条及以上录音时，valid 集不能为空。')
    split_counts = {'train': 0, 'valid': 0}
    seen = set()
    duration_sum = 0.0
    for index, item in enumerate(items):
        _validate_item(item, settings, index)
        split_counts[item['split']] += 1
        if item['sha256'] in seen:
            raise TrainingError('dataset.json 中存在重复的 sha256。')
        seen.add(item['sha256'])
        duration_sum += float(item['duration'])
    if (split_counts['train'] != counts['train']
            or split_counts['valid'] != counts['valid']):
        raise TrainingError('dataset.counts 与 items 的 split 分布不一致。')
    total_duration = _number(manifest['total_duration'], 'dataset.total_duration',
                             minimum=0.0)
    if abs(total_duration - duration_sum) > max(0.05, 1e-6 * len(items)):
        raise TrainingError('dataset.total_duration 与各片段时长之和不一致。')
    rejected = manifest['rejected']
    if not isinstance(rejected, list):
        raise TrainingError('dataset.rejected 必须是列表。')
    for index, entry in enumerate(rejected):
        if not isinstance(entry, dict) or set(entry) != _REJECTED_FIELDS:
            raise TrainingError(f'dataset.rejected 第 {index + 1} 项字段不完整。')
        for key in ('audio', 'reason'):
            if not isinstance(entry[key], str) or not entry[key].strip():
                raise TrainingError(f'dataset.rejected 第 {index + 1} 项的 {key} 不合法。')
    if manifest['license_confirmed'] is not True:
        raise TrainingError('dataset.json 未记录授权确认，拒绝使用该数据集。')
    warnings = manifest['warnings']
    if not isinstance(warnings, list) or len(warnings) > 64:
        raise TrainingError('dataset.warnings 必须是列表。')
    for warning in warnings:
        if not isinstance(warning, str) or len(warning) > 1000:
            raise TrainingError('dataset.warnings 中的每条内容都必须是不超过 1000 字的文本。')
    return manifest


def _read_manifest(dataset_dir):
    root = _path_argument(dataset_dir, 'dataset_dir')
    if not root.is_dir():
        raise TrainingError(f'数据集目录不存在：{root}')
    root = root.resolve()
    path = root / 'dataset.json'
    if path.is_symlink() or not path.is_file():
        raise TrainingError(f'数据集缺少 dataset.json：{root}')
    if path.stat().st_size > 8 * 1024 * 1024:
        raise TrainingError('dataset.json 异常过大，已拒绝读取。')
    try:
        manifest = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, ValueError) as exc:
        raise TrainingError(f'dataset.json 无法解析：{exc}') from exc
    return root, _validate_dataset_manifest(manifest)


def _summary(manifest, root: Path) -> dict:
    items = manifest['items']
    return {
        'manifest': manifest,
        'train': [item for item in items if item['split'] == 'train'],
        'valid': [item for item in items if item['split'] == 'valid'],
        'path': str(root),
        'counts': dict(manifest['counts']),
        'total_duration': manifest['total_duration'],
        'rejected': [dict(entry) for entry in manifest['rejected']],
        'rejected_count': len(manifest['rejected']),
    }


def _read_jsonl(path: Path, label: str):
    if path.is_symlink() or not path.is_file():
        raise TrainingError(f'数据集缺少 {label}。')
    if path.stat().st_size > 64 * 1024 * 1024:
        raise TrainingError(f'{label} 异常过大，已拒绝读取。')
    try:
        text = path.read_text(encoding='utf-8')
    except (OSError, UnicodeError) as exc:
        raise TrainingError(f'{label} 无法读取：{exc}') from exc
    entries = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise TrainingError(f'{label} 第 {number} 行为空，数据集文件可能已损坏。')
        try:
            entry = json.loads(line)
        except ValueError as exc:
            raise TrainingError(f'{label} 第 {number} 行不是合法 JSON。') from exc
        if not isinstance(entry, dict):
            raise TrainingError(f'{label} 第 {number} 行必须是 JSON 对象。')
        entries.append(entry)
    return entries


def _check_split_entries(entries, items, split, label):
    expected = {item['sha256']: item for item in items if item['split'] == split}
    if len(entries) != len(expected):
        raise TrainingError(f'{label} 的行数与 dataset.json 的 {split} 数量不一致。')
    seen = set()
    for entry in entries:
        digest = entry.get('sha256')
        if not isinstance(digest, str) or digest not in expected:
            raise TrainingError(f'{label} 中存在 dataset.json 未记录的片段。')
        if digest in seen:
            raise TrainingError(f'{label} 中存在重复片段。')
        seen.add(digest)
        if entry != expected[digest]:
            raise TrainingError(f'{label} 与 dataset.json 中的记录不一致。')


def load_dataset(dataset_dir) -> dict:
    """Validate and load a dataset, verifying every clip's SHA-256.

    Returns ``{'manifest', 'train', 'valid', 'path', 'counts',
    'total_duration', 'rejected', 'rejected_count'}``. The train/valid lists are
    the manifest records for each split, and the JSONL files must match them
    exactly. Clip bytes are hashed; decoding is left to the trainer.
    """
    root, manifest = _read_manifest(dataset_dir)
    clips = root / 'clips'
    if clips.is_symlink() or not clips.is_dir():
        raise TrainingError('数据集缺少 clips 目录或该目录不安全。')
    for item in manifest['items']:
        candidate = root / item['audio']
        if candidate.is_symlink() or not candidate.is_file():
            raise TrainingError(f'数据集缺少音频片段：{item["audio"]}')
        if candidate.resolve().parent != clips.resolve():
            raise TrainingError(f'音频片段逃逸出 clips 目录：{item["audio"]}')
        if _sha256_file(candidate) != item['sha256']:
            raise TrainingError(f'音频片段哈希不匹配，可能已被修改：{item["audio"]}')
    _check_split_entries(_read_jsonl(root / 'train.jsonl', 'train.jsonl'),
                         manifest['items'], 'train', 'train.jsonl')
    _check_split_entries(_read_jsonl(root / 'valid.jsonl', 'valid.jsonl'),
                         manifest['items'], 'valid', 'valid.jsonl')
    return _summary(manifest, root)


def dataset_summary(dataset_dir) -> dict:
    """Return counts, durations, and rejections without hashing or decoding audio.

    Only the manifest structure and clip existence are checked; use
    :func:`load_dataset` when byte-level integrity must be re-verified.
    """
    root, manifest = _read_manifest(dataset_dir)
    for item in manifest['items']:
        candidate = root / item['audio']
        if candidate.is_symlink() or not candidate.is_file():
            raise TrainingError(f'数据集缺少音频片段：{item["audio"]}')
    return _summary(manifest, root)


# ------------------------------------------------------------------ job setup
def _validate_base_model(base_model):
    path = _path_argument(base_model, 'base_model')
    if not path.is_dir():
        raise TrainingError(
            f'本地 Base 模型目录不存在：{path}；本工具不会下载模型。')
    path = path.resolve()
    config = path / 'config.json'
    if config.is_symlink() or not config.is_file():
        raise TrainingError(f'Base 模型目录缺少 config.json：{path}')
    total = 0
    try:
        for item in path.rglob('*.safetensors'):
            if item.is_file() and not item.is_symlink():
                total += item.stat().st_size
    except OSError as exc:
        raise TrainingError(f'无法读取 Base 模型目录：{exc}') from exc
    return path, total


def estimate_vram_gb(parameter_billions) -> float:
    """Heuristic LoRA-training VRAM estimate in GB, rounded to one decimal.

    Formula: ``2.2 x parameter_billions + 1.5`` GB. For example a 0.6 B model
    estimates ~2.8 GB. This is a documented estimate for planning only, not a
    guarantee that training will fit or run.
    """
    if isinstance(parameter_billions, bool) or not isinstance(
            parameter_billions, (int, float)):
        raise TrainingError('参数量必须是数字，单位为十亿参数（B）。')
    value = float(parameter_billions)
    if not math.isfinite(value) or value < 0:
        raise TrainingError('参数量必须是非负有限数字，单位为十亿参数（B）。')
    return round(_VRAM_GB_PER_BILLION * value + _VRAM_OVERHEAD_GB, 1)


def _job_warnings(base_bytes: int, vram: float, command_warnings) -> list:
    warnings = [
        '离线高级功能：准备与训练过程不会联网，也不会下载模型或依赖。',
        '训练需要显存；显存不足时训练会失败，请降低 batch_size 或使用更大的显卡。',
        '训练只发布新的 LoRA 检查点，不覆盖 Base 模型，也不会复制或修改它。',
        '只支持有合法授权的录音；本工具无法核实授权的真实性。',
        f'显存估算 {vram:.1f} GB 为启发式值（2.2 × 参数量(GB) + 1.5 GB），'
        '是不能替代实测的估算，不保证一定能运行。',
    ]
    if base_bytes == 0:
        warnings.append(
            'Base 模型目录中没有 *.safetensors 权重，参数量按 0 估算，显存数字仅供参考。')
    warnings.extend(command_warnings)
    return warnings


def create_training_job(dataset_dir, *, base_model, output_dir, name, params=None,
                        registry=None, role_id=None, store=None) -> dict:
    """Create a reproducible training job without training or downloading anything.

    The dataset is re-verified with :func:`load_dataset`, the Base model must be an
    existing local directory containing ``config.json``, and parameters are
    strictly validated. The trainer command is recorded from
    ``TIMBREFORGE_TRAINING_COMMAND`` (JSON string array) when set; it is never
    executed here. ``registry`` and ``store`` are accepted for callers that share
    them, but this function does not modify either.
    """
    name = _validate_name(name)
    params = _validate_params({} if params is None else params)
    role_id = _validate_role_reference(role_id)
    dataset = load_dataset(dataset_dir)
    base_path, base_bytes = _validate_base_model(base_model)
    parameter_billions = round(base_bytes / 2 / 1_000_000_000, 6)
    vram = estimate_vram_gb(parameter_billions)
    command, command_warnings = _read_trainer_command()

    raw_output = _path_argument(output_dir, 'output_dir')
    if raw_output.exists() and not raw_output.is_dir():
        raise TrainingError(f'输出目录不是目录：{raw_output}')
    output = raw_output.resolve()
    try:
        output.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise TrainingError(f'无法创建训练任务输出目录：{exc}') from exc

    job_id = uuid.uuid4().hex
    job_dir = output / job_id
    if job_dir.exists() or job_dir.is_symlink():
        raise TrainingError(f'训练任务目录已存在，拒绝覆盖：{job_dir}')

    job = {
        'schema_version': JOB_SCHEMA,
        'id': job_id,
        'name': name,
        'path': str(job_dir),
        'status': 'prepared',
        'created_at': _now(),
        'dataset': {
            'path': dataset['path'],
            'counts': dict(dataset['counts']),
            'clips': dataset['counts']['total'],
            'total_duration': dataset['total_duration'],
        },
        'base_model': {
            'path': str(base_path),
            'parameter_billions': parameter_billions,
            'safetensors_bytes': base_bytes,
        },
        'params': params,
        'vram_estimate_gb': vram,
        'trainer': {'configured': command is not None, 'command': command},
        'log_file': str(job_dir / 'train.log'),
        'checkpoint_dir': str(job_dir / 'checkpoints'),
        'evaluation': None,
        'registered_model': None,
        'role_id': role_id,
        'warnings': _job_warnings(base_bytes, vram, command_warnings),
    }

    temporary = None
    try:
        temporary = Path(tempfile.mkdtemp(prefix='.job-tmp-', dir=str(output)))
        _write_json_atomic(temporary / 'job.json', job)
        if job_dir.exists() or job_dir.is_symlink():
            raise TrainingError(f'训练任务目录已存在，拒绝覆盖：{job_dir}')
        os.rename(temporary, job_dir)
        temporary = None
    finally:
        _cleanup_temp(temporary, '.job-tmp-')
    _root, stored = _load_job(job_dir)
    return stored


# ------------------------------------------------------------- job validation
def _require_absolute_path(value, label) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise TrainingError(f'{label} 必须是非空路径文本。')
    path = Path(value)
    if not path.is_absolute():
        raise TrainingError(f'{label} 必须是绝对路径。')
    return path


def _validate_trainer_record(trainer):
    if not isinstance(trainer, dict) or set(trainer) - _TRAINER_FIELDS:
        raise TrainingError('job.json 中的 trainer 字段不合法。')
    if 'configured' not in trainer or not isinstance(trainer['configured'], bool):
        raise TrainingError('job.json 中的 trainer.configured 必须是布尔值。')
    if 'command' not in trainer:
        raise TrainingError('job.json 中的 trainer 缺少 command 字段。')
    command = trainer['command']
    if command is not None:
        command = _normalize_trainer_command(command, 'job.json trainer.command')
    if 'returncode' in trainer and trainer['returncode'] is not None:
        _integer(trainer['returncode'], 'job.json trainer.returncode')
    return trainer


def _validate_last_run(last_run):
    if not isinstance(last_run, dict) or set(last_run) - _LAST_RUN_FIELDS:
        raise TrainingError('job.json 中的 last_run 字段不合法。')
    for key in ('started_at', 'finished_at'):
        if key in last_run:
            _timestamp(last_run[key], f'job.json last_run.{key}')
    for key in ('request_file', 'result_file', 'staging_dir',
                'previous_checkpoint', 'result_checkpoint_dir'):
        if key in last_run and last_run[key] is not None:
            if not isinstance(last_run[key], str) or not last_run[key].strip():
                raise TrainingError(f'job.json last_run.{key} 必须是非空路径文本。')
    if 'command' in last_run and last_run['command'] is not None:
        _normalize_trainer_command(last_run['command'], 'job.json last_run.command')
    if 'returncode' in last_run and last_run['returncode'] is not None:
        _integer(last_run['returncode'], 'job.json last_run.returncode')


def _validate_job(job, root: Path):
    if not isinstance(job, dict) or set(job) - (_JOB_REQUIRED_FIELDS | _JOB_OPTIONAL_FIELDS):
        raise TrainingError('job.json 字段不完整或包含未知字段。')
    missing = _JOB_REQUIRED_FIELDS - set(job)
    if missing:
        raise TrainingError('job.json 缺少必要字段：' + '、'.join(sorted(missing)) + '。')
    if type(job['schema_version']) is not int or job['schema_version'] != JOB_SCHEMA:
        raise TrainingError('不支持的 job.json schema_version。')
    if not isinstance(job['id'], str) or not _ID_RE.fullmatch(job['id']):
        raise TrainingError('job.json 中的任务 ID 不合法。')
    _validate_name(job['name'])
    if job['status'] not in _JOB_STATUSES:
        raise TrainingError('job.json 中的任务状态不合法。')
    _timestamp(job['created_at'], 'job.created_at')
    if 'path' in job:
        if _require_absolute_path(job['path'], 'job.json path').resolve() != root:
            raise TrainingError('job.json 中的任务路径与所在目录不一致，已拒绝。')

    dataset = job['dataset']
    if (not isinstance(dataset, dict)
            or set(dataset) - {'path', 'counts', 'total_duration', 'clips'}
            or {'path', 'counts', 'total_duration'} - set(dataset)):
        raise TrainingError('job.json 中的 dataset 字段不合法。')
    _require_absolute_path(dataset['path'], 'job.json dataset.path')
    _validate_counts(dataset['counts'], 'job.json dataset.counts')
    _number(dataset['total_duration'], 'job.json dataset.total_duration', minimum=0.0)
    if 'clips' in dataset:
        _integer(dataset['clips'], 'job.json dataset.clips', minimum=0)
        if dataset['clips'] != dataset['counts']['total']:
            raise TrainingError('job.json dataset.clips 与 counts.total 不一致。')

    base = job['base_model']
    if (not isinstance(base, dict)
            or set(base) != {'path', 'parameter_billions', 'safetensors_bytes'}):
        raise TrainingError('job.json 中的 base_model 字段不合法。')
    _require_absolute_path(base['path'], 'job.json base_model.path')
    _number(base['parameter_billions'], 'job.json base_model.parameter_billions',
            minimum=0.0)
    _integer(base['safetensors_bytes'], 'job.json base_model.safetensors_bytes',
             minimum=0)

    _validate_stored_params(job['params'])
    _number(job['vram_estimate_gb'], 'job.json vram_estimate_gb', minimum=0.0)
    _validate_trainer_record(job['trainer'])

    log_path = _require_absolute_path(job['log_file'], 'job.json log_file')
    checkpoint_path = _require_absolute_path(job['checkpoint_dir'],
                                             'job.json checkpoint_dir')
    if log_path.resolve() != (root / 'train.log').resolve():
        raise TrainingError('job.json 中的日志路径与任务目录不一致，已拒绝。')
    if checkpoint_path.resolve() != (root / 'checkpoints').resolve():
        raise TrainingError('job.json 中的检查点路径与任务目录不一致，已拒绝。')

    evaluation = job['evaluation']
    if evaluation is not None:
        if (not isinstance(evaluation, dict)
                or set(evaluation) != {'path', 'created_at', 'metrics'}):
            raise TrainingError('job.json 中的 evaluation 字段不合法。')
        _require_absolute_path(evaluation['path'], 'job.json evaluation.path')
        _timestamp(evaluation['created_at'], 'job.json evaluation.created_at')
        _normalize_metrics(evaluation['metrics'])

    registered = job['registered_model']
    if registered is not None:
        if (not isinstance(registered, dict)
                or set(registered) != {'id', 'name', 'revision', 'registered_at'}):
            raise TrainingError('job.json 中的 registered_model 字段不合法。')
        if not isinstance(registered['id'], str) or not _ID_RE.fullmatch(registered['id']):
            raise TrainingError('job.json 中的 registered_model.id 不合法。')
        _validate_name(registered['name'])
        if (not isinstance(registered['revision'], str)
                or not registered['revision'].strip()
                or len(registered['revision']) > 120):
            raise TrainingError('job.json 中的 registered_model.revision 不合法。')
        _timestamp(registered['registered_at'], 'job.json registered_model.registered_at')

    _validate_role_reference(job['role_id'])
    warnings = job['warnings']
    if not isinstance(warnings, list) or len(warnings) > 64:
        raise TrainingError('job.json 中的 warnings 必须是列表。')
    for warning in warnings:
        if not isinstance(warning, str) or len(warning) > 1000:
            raise TrainingError('job.json 中的每条 warning 都必须是不超过 1000 字的文本。')
    if 'finished_at' in job and job['finished_at'] is not None:
        _timestamp(job['finished_at'], 'job.finished_at')
    if 'error' in job and job['error'] is not None:
        if not isinstance(job['error'], str) or not job['error'].strip():
            raise TrainingError('job.json 中的 error 必须是非空文本。')
    if 'last_run' in job:
        _validate_last_run(job['last_run'])
    return job


def _load_job(job_dir):
    root = _path_argument(job_dir, 'job_dir')
    if not root.is_dir():
        raise TrainingError(f'训练任务目录不存在：{root}')
    root = root.resolve()
    path = root / 'job.json'
    if path.is_symlink() or not path.is_file():
        raise TrainingError(f'训练任务缺少 job.json：{root}')
    if path.stat().st_size > 2 * 1024 * 1024:
        raise TrainingError('job.json 异常过大，已拒绝读取。')
    try:
        job = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, ValueError) as exc:
        raise TrainingError(f'job.json 无法解析：{exc}') from exc
    return root, _validate_job(job, root)


# -------------------------------------------------------------- training run
def _training_unavailable_message() -> str:
    return (
        '未配置训练器，没有运行任何训练。请把环境变量 '
        + _TRAINER_ENV
        + ' 设为 JSON 字符串数组，并在命令中包含 {request} 和 {result} 占位符，例如：'
        '["python","train_lora.py","--request","{request}","--result","{result}"]。'
        '本功能完全离线，不会下载模型或依赖；若显存不足，训练同样会失败。')


def _default_runner(command, request_path, result_path, log_path, env, timeout):
    """Run the external trainer, streaming stdout/stderr into ``train.log``."""
    formatted = _format_command(command, request_path, result_path)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('ab') as log:
        try:
            completed = subprocess.run(
                formatted, cwd=str(log_path.parent), env=env, timeout=timeout,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                check=False)
        except FileNotFoundError as exc:
            raise TrainingError(f'无法启动训练器命令：{exc}') from exc
    return int(completed.returncode)


def _run_injected(runner, command, request_path, result_path, log_path, env, timeout):
    args = (command, request_path, result_path, log_path, env, timeout)
    try:
        signature = inspect.signature(runner)
    except (TypeError, ValueError):
        signature = None
    if signature is None:
        value = runner(*args)
    else:
        parameters = list(signature.parameters.values())
        if any(parameter.kind == parameter.VAR_POSITIONAL for parameter in parameters):
            value = runner(*args)
        else:
            positional = [parameter for parameter in parameters
                          if parameter.kind in (parameter.POSITIONAL_ONLY,
                                                parameter.POSITIONAL_OR_KEYWORD)]
            value = runner(*args[:min(6, len(positional))])
    if value is None:
        return 0
    if type(value) is int:
        return value
    if isinstance(value, dict):
        returncode = value.get('returncode')
        if type(returncode) is int:
            return returncode
        raise TrainingError('注入的 runner 返回的字典必须包含整数 returncode。')
    returncode = getattr(value, 'returncode', None)
    if type(returncode) is int:
        return returncode
    raise TrainingError('注入的 runner 必须返回 None、整数退出码或带 returncode 的对象。')


def _read_training_result(result_path: Path):
    if result_path.is_symlink() or not result_path.is_file():
        raise TrainingError('训练器没有写出 result.json，无法确认训练成功。')
    if result_path.stat().st_size > 2 * 1024 * 1024:
        raise TrainingError('训练结果文件异常过大，已拒绝读取。')
    try:
        result = json.loads(result_path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, ValueError) as exc:
        raise TrainingError(f'训练结果不是合法 JSON：{exc}') from exc
    if not isinstance(result, dict):
        raise TrainingError('训练结果必须是 JSON 对象。')
    if result.get('status') != 'ok':
        raise TrainingError(
            f'训练器报告的 status 为 {result.get("status")!r}，不是 "ok"。')
    checkpoint = result.get('checkpoint_dir')
    if not isinstance(checkpoint, str) or not checkpoint.strip():
        raise TrainingError('训练结果缺少 checkpoint_dir。')
    checkpoint_path = Path(checkpoint).expanduser()
    if checkpoint_path.is_symlink() or not checkpoint_path.is_dir():
        raise TrainingError('训练结果中的 checkpoint_dir 不存在或不是目录。')
    return result, checkpoint_path.resolve()


def _validate_staging(staging: Path) -> None:
    if staging.is_symlink() or not staging.is_dir():
        raise TrainingError('训练暂存目录不存在或不是目录，未发布检查点。')
    found = False
    for item in staging.rglob('*'):
        if item.is_symlink():
            raise TrainingError('训练暂存目录包含符号链接，拒绝发布检查点。')
        if item.is_file() and item.stat().st_size > 0:
            found = True
    if not found:
        raise TrainingError('训练器没有在暂存目录中写出任何非空检查点文件。')


def _publish_checkpoint(staging: Path, checkpoints: Path):
    if checkpoints.is_symlink():
        raise TrainingError('检查点路径是符号链接，拒绝发布。')
    if checkpoints.exists() and not checkpoints.is_dir():
        raise TrainingError('检查点路径已存在且不是目录，拒绝覆盖。')
    previous = None
    if checkpoints.exists():
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        previous = checkpoints.with_name(f'checkpoints.previous-{stamp}')
        if previous.exists() or previous.is_symlink():
            previous = checkpoints.with_name(
                f'checkpoints.previous-{stamp}-{uuid.uuid4().hex[:8]}')
        os.replace(checkpoints, previous)
    try:
        os.replace(staging, checkpoints)
    except OSError:
        if previous is not None and not checkpoints.exists():
            try:
                os.replace(previous, checkpoints)
            except OSError:
                pass
        raise
    return previous


def _job_export(job: dict) -> dict:
    """Return a job dict with all fields at top level and a nested ``job`` view.

    The nested key lets the advanced-training UI keep a stable ``result['job']``
    shape while callers can still read ``result['status']`` directly. The nested
    value is the stored job dict (without this extra key), so it stays serializable.
    """
    response = dict(job)
    response['job'] = job
    return response


def run_training(job_dir, *, trainer_command=None, timeout=1800, runner=None) -> dict:
    """Run the configured external trainer for one prepared or failed job.

    The request JSON (job id, dataset JSONL paths, Base model, checkpoint output,
    staging directory, params, seed) is written into the job directory. The
    trainer must write ``result.json`` with ``status: "ok"`` and a valid
    ``checkpoint_dir``. Its staging output is validated and published atomically;
    an old checkpoint is moved to ``checkpoints.previous-<timestamp>`` first.
    On failure or timeout the job is marked ``failed`` and existing checkpoints
    stay untouched. ``runner`` is an injection seam for tests: it is called with
    ``(command, request_path, result_path, log_path[, env, timeout])`` and returns
    ``None``, an integer return code, a dict with ``returncode``, or an object
    with a ``.returncode`` attribute.
    """
    root, job = _load_job(job_dir)
    if job['status'] not in ('prepared', 'failed'):
        raise TrainingError(
            f'任务当前状态为 {job["status"]}，只有 prepared 或 failed 状态可以运行训练。')
    timeout = _validate_timeout(timeout)

    command = None
    if trainer_command is not None:
        command = _normalize_trainer_command(trainer_command, 'trainer_command')
    elif job['trainer'].get('command'):
        command = list(job['trainer']['command'])
    else:
        command, _unused = _read_trainer_command()
    if command is None and runner is None:
        raise TrainingUnavailable(_training_unavailable_message())
    if command is not None and runner is None:
        _require_placeholders(command)
    if command is None:
        command = []

    dataset_dir = Path(job['dataset']['path'])
    if not dataset_dir.is_dir():
        raise TrainingError(f'训练任务引用的数据集目录不存在：{dataset_dir}')
    train_jsonl = dataset_dir / 'train.jsonl'
    valid_jsonl = dataset_dir / 'valid.jsonl'
    for label, path in (('train.jsonl', train_jsonl), ('valid.jsonl', valid_jsonl)):
        if path.is_symlink() or not path.is_file():
            raise TrainingError(f'训练任务引用的 {label} 不存在：{path}')
    base_path = Path(job['base_model']['path'])
    if base_path.is_symlink() or not base_path.is_dir():
        raise TrainingError(f'训练任务引用的 Base 模型目录不存在：{base_path}')
    if not (base_path / 'config.json').is_file():
        raise TrainingError(f'训练任务引用的 Base 模型缺少 config.json：{base_path}')

    checkpoints = root / 'checkpoints'
    log_path = root / 'train.log'
    request_path = root / 'request.json'
    result_path = root / 'result.json'
    staging = root / f'checkpoints.staging-{uuid.uuid4().hex[:12]}'
    try:
        staging.mkdir()
    except OSError as exc:
        raise TrainingError(f'无法创建训练暂存目录：{exc}') from exc

    request = {
        'schema_version': 1,
        'job_id': job['id'],
        'id': job['id'],
        'job_name': job['name'],
        'dataset': {
            'path': str(dataset_dir),
            'train_jsonl': str(train_jsonl),
            'valid_jsonl': str(valid_jsonl),
            'train': str(train_jsonl),
            'valid': str(valid_jsonl),
            'counts': dict(job['dataset']['counts']),
        },
        'dataset_dir': str(dataset_dir),
        'train_jsonl': str(train_jsonl),
        'valid_jsonl': str(valid_jsonl),
        'train': str(train_jsonl),
        'valid': str(valid_jsonl),
        'base_model': str(base_path),
        'output_dir': str(checkpoints),
        'checkpoint_dir': str(checkpoints),
        'staging_dir': str(staging),
        'params': dict(job['params']),
        'seed': job['params']['seed'],
        'log_file': str(log_path),
        'result_file': str(result_path),
    }
    try:
        _write_json_atomic(request_path, request)
    except Exception:
        _remove_empty_dir(staging)
        raise
    if result_path.exists():
        if result_path.is_dir() or result_path.is_symlink():
            _remove_empty_dir(staging)
            raise TrainingError('训练任务目录中的 result.json 未通过安全检查。')
        result_path.unlink()

    env = dict(os.environ)
    env['HF_HUB_OFFLINE'] = '1'
    started = _now()
    _append_log(log_path, f'[{started}] 开始训练任务 {job["id"]}。\n')

    error = None
    returncode = None
    previous = None
    result_checkpoint = None
    try:
        if runner is None:
            returncode = _default_runner(command, request_path, result_path,
                                         log_path, env, timeout)
        else:
            returncode = _run_injected(runner, command, request_path, result_path,
                                       log_path, env, timeout)
    except subprocess.TimeoutExpired:
        error = (f'训练在 {timeout:g} 秒后超时；已停止等待，未发布新检查点。'
                 f'日志：{log_path}')
    except Exception as exc:  # noqa: BLE001 - report any trainer failure as failed status
        error = f'训练器执行失败：{exc}；未发布新检查点。日志：{log_path}'

    if error is None and returncode != 0:
        error = f'训练器返回非零退出码 {returncode}；未发布新检查点。日志：{log_path}'
    if error is None:
        try:
            _result, result_checkpoint = _read_training_result(result_path)
            _validate_staging(staging)
            previous = _publish_checkpoint(staging, checkpoints)
        except TrainingError as exc:
            error = f'{exc} 日志：{log_path}'
        except OSError as exc:
            error = f'发布检查点失败：{exc} 日志：{log_path}'

    finished = _now()
    if error is not None:
        _remove_empty_dir(staging)
        job['status'] = 'failed'
        job['error'] = error
    else:
        job['status'] = 'completed'
        job.pop('error', None)
    job['finished_at'] = finished
    job['checkpoint_dir'] = str(checkpoints)
    job['log_file'] = str(log_path)
    # Keep the command recorded when the job was created; record the command that
    # was actually used (including a per-run override) in last_run instead.
    trainer_record = {
        'configured': job['trainer']['configured'],
        'command': job['trainer']['command'],
    }
    if isinstance(returncode, int) and not isinstance(returncode, bool):
        trainer_record['returncode'] = returncode
    job['trainer'] = trainer_record
    job['last_run'] = {
        'started_at': started,
        'finished_at': finished,
        'request_file': str(request_path),
        'result_file': str(result_path),
        'staging_dir': str(staging),
        'command': list(command) if command else None,
        'returncode': (returncode if isinstance(returncode, int)
                       and not isinstance(returncode, bool) else None),
        'previous_checkpoint': str(previous) if previous is not None else None,
        'result_checkpoint_dir': (str(result_checkpoint)
                                  if result_checkpoint is not None else None),
    }
    _append_log(log_path, f'[{finished}] 训练结束：{job["status"]}。'
                          + (f' {error}' if error else '') + '\n')
    _write_json_atomic(root / 'job.json', job)
    return _job_export(job)


# ---------------------------------------------------------------- evaluation
_METRIC_FIELDS = frozenset({'cer', 'wer', 'speaker_similarity', 'mos', 'notes'})


def _normalize_metrics(metrics):
    if not isinstance(metrics, dict):
        raise TrainingError('metrics 必须是字典。')
    unknown = sorted(set(metrics) - _METRIC_FIELDS)
    if unknown:
        raise TrainingError(
            'metrics 包含未知指标：' + '、'.join(unknown)
            + '；允许的指标为：cer、wer、speaker_similarity、mos、notes。')
    normalized = {}
    for key in ('cer', 'wer'):
        if key in metrics:
            normalized[key] = _number(metrics[key], f'metrics.{key}', minimum=0.0)
    if 'speaker_similarity' in metrics:
        normalized['speaker_similarity'] = _number(
            metrics['speaker_similarity'], 'metrics.speaker_similarity',
            minimum=-1.0, maximum=1.0)
    if 'mos' in metrics:
        normalized['mos'] = _number(metrics['mos'], 'metrics.mos',
                                    minimum=0.0, maximum=5.0)
    if 'notes' in metrics:
        notes = metrics['notes']
        if not isinstance(notes, str) or len(notes) > _MAX_EVALUATION_NOTES:
            raise TrainingError(
                f'metrics.notes 必须是不超过 {_MAX_EVALUATION_NOTES} 字的文本。')
        normalized['notes'] = notes
    if not any(key in normalized for key in ('cer', 'wer', 'speaker_similarity', 'mos')):
        raise TrainingError(
            '评估至少需要一个数值指标（cer、wer、speaker_similarity 或 mos）。')
    return normalized


def record_evaluation(job_dir, metrics) -> dict:
    """Record validated evaluation metrics for a completed training job.

    At least one numeric metric (``cer``, ``wer``, ``speaker_similarity`` or
    ``mos``) is required; ``notes`` is optional supplementary text.
    """
    root, job = _load_job(job_dir)
    if job['status'] != 'completed':
        raise TrainingError('只有已完成（completed）的训练任务才能记录评估结果。')
    normalized = _normalize_metrics(metrics)
    evaluation = {
        'schema_version': 1,
        'job_id': job['id'],
        'created_at': _now(),
        'metrics': normalized,
    }
    path = root / 'evaluation.json'
    _write_json_atomic(path, evaluation)
    job['evaluation'] = {
        'path': str(path),
        'created_at': evaluation['created_at'],
        'metrics': normalized,
    }
    _write_json_atomic(root / 'job.json', job)
    return {'job': job, 'evaluation': evaluation}


def _evaluation_metrics_from_job(root: Path, job) -> dict:
    pointer = job.get('evaluation')
    if not pointer:
        raise TrainingError('登记模型前必须先记录评估结果（record_evaluation）。')
    path_value = pointer.get('path') if isinstance(pointer, dict) else None
    if not isinstance(path_value, str):
        raise TrainingError('job.json 中的评估记录不完整，请重新记录评估。')
    path = Path(path_value)
    if path.resolve() != (root / 'evaluation.json').resolve():
        raise TrainingError('评估文件路径与任务目录不一致，已拒绝登记。')
    if path.is_symlink() or not path.is_file():
        raise TrainingError('评估文件缺失，请重新记录评估后再登记模型。')
    try:
        evaluation = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, ValueError) as exc:
        raise TrainingError(f'评估文件无法解析：{exc}') from exc
    if (not isinstance(evaluation, dict)
            or set(evaluation) != {'schema_version', 'job_id', 'created_at', 'metrics'}):
        raise TrainingError('评估文件结构不合法，请重新记录评估。')
    if (type(evaluation['schema_version']) is not int
            or evaluation['schema_version'] != 1):
        raise TrainingError('不支持的评估文件 schema_version。')
    if evaluation['job_id'] != job['id']:
        raise TrainingError('评估文件与训练任务不匹配，已拒绝登记。')
    _timestamp(evaluation['created_at'], 'evaluation.created_at')
    return _normalize_metrics(evaluation['metrics'])


def _evaluation_notes(job, metrics) -> str:
    labels = {
        'cer': 'CER', 'wer': 'WER',
        'speaker_similarity': '说话人相似度', 'mos': 'MOS',
    }
    parts = []
    for key in ('cer', 'wer', 'speaker_similarity', 'mos'):
        if key in metrics:
            parts.append(f'{labels[key]}={metrics[key]:.4g}')
    if metrics.get('notes'):
        parts.append(f'备注：{str(metrics["notes"])[:200]}')
    text = (f'LoRA 训练任务 {job["id"]}（离线微调；仅登记检查点，不覆盖 Base）。')
    if parts:
        text += '评估：' + '；'.join(parts) + '。'
    text += '指标为辅助证据，不能替代试听确认。'
    return text[:1000]


# ---------------------------------------------------------------- registration
def register_trained_model(job_dir, *, registry=None, bind_role_id=None, store=None,
                           confirmed=False) -> dict:
    """Register a completed job's checkpoint as a local LoRA model.

    An evaluation and ``confirmed is True`` are required. The checkpoint directory
    must already be a valid local model directory for :class:`ModelRegistry`
    (``config.json`` is required); registry validation errors are propagated.
    With ``bind_role_id`` the model is also activated on that character card,
    which invalidates its cached clone prompt. Without a role id no card is
    touched. The Base model is only referenced, never copied or modified.
    """
    if confirmed is not True:
        raise TrainingError(
            '登记训练模型前必须明确确认：评估结果已复核，且该模型的使用范围已得到授权。')
    bind_role_id = _validate_role_id(bind_role_id, 'bind_role_id')
    root, job = _load_job(job_dir)
    if job['status'] != 'completed':
        raise TrainingError('只有已完成（completed）的训练任务才能登记模型。')
    metrics = _evaluation_metrics_from_job(root, job)
    checkpoints = Path(job['checkpoint_dir'])
    if checkpoints.is_symlink() or not checkpoints.is_dir():
        raise TrainingError(f'训练检查点目录不存在或不是目录：{checkpoints}')

    registry = registry if registry is not None else ModelRegistry()
    revision = f'job-{job["id"][:12]}'
    entry = registry.register(
        job['name'], str(checkpoints), revision, kind='lora',
        notes=_evaluation_notes(job, metrics),
        base_model=job['base_model']['path'])
    job['registered_model'] = {
        'id': entry['id'],
        'name': entry['name'],
        'revision': entry['revision'],
        'registered_at': _now(),
    }
    _write_json_atomic(root / 'job.json', job)

    card = None
    if bind_role_id is not None:
        store = store if store is not None else CharacterStore()
        result = registry.activate(bind_role_id, entry['id'], store=store)
        card = result.get('card') if isinstance(result, dict) else None
    return {'job': job, 'card': card, 'model': entry, 'entry': entry}


def rollback_model(role_id, *, registry=None, store=None) -> dict:
    """Restore the model previously recorded on the card (registry rollback).

    This delegates to :meth:`ModelRegistry.rollback`, so the card's last model
    activation is undone and its cached clone prompt is invalidated.
    """
    registry = registry if registry is not None else ModelRegistry()
    return registry.rollback(role_id, store=store)

