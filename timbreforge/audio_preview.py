"""Compressed audition copies for derived reference audio.

The canonical reference stays an uncompressed WAV for model input; previews are
small, disposable compressed copies used only for browser playback. They are
never fed to a model and can be regenerated or deleted at any time.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import uuid

import numpy as np
import soundfile as sf

MAX_PREVIEW_SECONDS = 60.0
PREVIEW_PREFIX = 'preview-'


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def build_preview(source, dest_dir, *, max_seconds=MAX_PREVIEW_SECONDS):
    """Write a compressed preview next to the character's derived audio.

    OGG/Vorbis is preferred; when the local libsndfile cannot encode it the
    function falls back to FLAC, and finally to 16-bit WAV only if neither
    compressed codec is available (reported in ``format``).
    """
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'待预览音频不存在：{path}')
    try:
        info = sf.info(str(path))
    except (OSError, RuntimeError) as exc:
        raise ValueError(f'无法读取待预览音频：{exc}') from exc
    if info.frames <= 0 or info.samplerate <= 0:
        raise ValueError('待预览音频为空。')
    if not 0 < float(max_seconds) <= MAX_PREVIEW_SECONDS:
        raise ValueError(f'max_seconds 必须在 0–{MAX_PREVIEW_SECONDS:g} 之间。')
    frames = min(info.frames, int(round(max_seconds * info.samplerate)))
    samples, sample_rate = sf.read(str(path), start=0, stop=frames,
                                   dtype='float32', always_2d=True)
    if not samples.size or not np.isfinite(samples).all():
        raise ValueError('预览音频包含无效采样。')

    directory = Path(dest_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    preview_id = _digest(path)[:16]
    attempts = [('OGG', 'VORBIS', '.ogg'), ('FLAC', 'PCM_16', '.flac'),
                ('WAV', 'PCM_16', '.wav')]
    errors = []
    for container, subtype, suffix in attempts:
        destination = directory / f'{PREVIEW_PREFIX}{preview_id}{suffix}'
        temporary = directory / f'.preview-{uuid.uuid4().hex}.tmp'
        try:
            with sf.SoundFile(str(temporary), mode='x', samplerate=sample_rate,
                              channels=samples.shape[1], format=container,
                              subtype=subtype) as stream:
                stream.write(samples)
            check = sf.info(str(temporary))
            if check.frames != frames or check.samplerate != sample_rate:
                raise ValueError('预览音频写入尺寸不一致。')
            temporary.replace(destination)
            return {
                'path': str(destination),
                'url': destination.as_posix(),
                'format': container,
                'subtype': subtype,
                'bytes': destination.stat().st_size,
                'duration': frames / sample_rate,
                'truncated': frames < info.frames,
                'source_sha256': _digest(path),
            }
        except (OSError, RuntimeError, ValueError) as exc:
            errors.append(f'{container}: {exc}')
            temporary.unlink(missing_ok=True)
    raise RuntimeError('无法生成压缩预览：' + '；'.join(errors))


def cleanup_previews(directory, *, keep=0):
    """Delete generated previews, optionally keeping the newest ``keep`` files."""
    folder = Path(directory).expanduser().resolve()
    if not folder.is_dir():
        return 0
    files = sorted((item for item in folder.iterdir()
                    if item.is_file() and item.name.startswith(PREVIEW_PREFIX)),
                   key=lambda item: item.stat().st_mtime, reverse=True)
    removed = 0
    for item in files[max(0, int(keep)):]:
        try:
            item.unlink()
            removed += 1
        except OSError:
            continue
    return removed


def preview_path(card_dir):
    """Return the newest preview inside a character directory, if any."""
    folder = Path(card_dir) / 'audio' / 'previews'
    if not folder.is_dir():
        return None
    files = sorted((item for item in folder.iterdir()
                    if item.is_file() and item.name.startswith(PREVIEW_PREFIX)),
                   key=lambda item: item.stat().st_mtime, reverse=True)
    return str(files[0]) if files else None

