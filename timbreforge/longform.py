"""Long-form TTS segmentation, quality checks and concatenation.

The module is intentionally dependency-light: NumPy and SoundFile only. It never
imports a model or a network client. Generation is injected as a callable, so
tests and the UI can supply the real backend while this module stays testable.

Nothing here rewrites a source recording. :func:`concatenate` always writes a
brand-new FLOAT WAV through a temporary file and ``os.replace``; an existing
derived output is only replaced with an explicit ``overwrite=True``.

Measurements are documented auxiliary estimates for human review. They are not
ASR, speaker identity, MOS or automatic ranking.
"""
from __future__ import annotations

import json
import math
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

from timbreforge.audio_edit_pipeline import measure_loudness

MAX_TEXT_CHARS = 20000
MIN_SEGMENT_CHARS = 20
MAX_SEGMENT_CHARS = 500
_MAX_ROLE_ID_CHARS = 200
_MAX_SEED = 0x7FFFFFFFFFFFFFFF
_SILENCE_DB = -42.0
_FRAME_SECONDS = 0.02
_CLIP_THRESHOLD = 0.999
_EPS = 1e-12
_TERMINATORS = frozenset("。！？!?；;\n")
_SOFT_BREAKS = frozenset("，,、：:")


# --------------------------------------------------------------------------
# Text segmentation
# --------------------------------------------------------------------------
def _validate_source_text(text):
    """Reject missing, blank or over-long source text with actionable messages."""
    if not isinstance(text, str):
        raise ValueError("待合成文本必须是字符串。")
    if not text.strip():
        raise ValueError("待合成文本不能为空，也不能只包含空白字符。")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(
            f"待合成文本长度为 {len(text)} 个字符，超过上限 {MAX_TEXT_CHARS}；"
            "请删减内容或拆分为多次长文本任务。")
    return text


def _validate_max_chars(max_chars):
    """Return ``max_chars`` as an int, enforcing the documented 20–500 range."""
    if isinstance(max_chars, bool) or not isinstance(max_chars, int):
        raise ValueError("max_chars 必须是 20–500 之间的整数。")
    if not MIN_SEGMENT_CHARS <= max_chars <= MAX_SEGMENT_CHARS:
        raise ValueError(
            f"max_chars 必须在 {MIN_SEGMENT_CHARS}–{MAX_SEGMENT_CHARS} 之间，"
            f"当前为 {max_chars}。")
    return max_chars


def _sentence_atoms(text):
    """Split into sentence-sized atoms, keeping every terminator in place."""
    atoms = []
    current = []
    for character in text:
        current.append(character)
        if character in _TERMINATORS:
            atoms.append("".join(current))
            current = []
    if current:
        atoms.append("".join(current))
    return atoms


def _clamp_atom(atom, max_chars):
    """Split one over-long sentence at commas, then by hard slices."""
    if len(atom) <= max_chars:
        return [atom]
    parts = []
    current = []
    for character in atom:
        current.append(character)
        if character in _SOFT_BREAKS:
            parts.append("".join(current))
            current = []
    if current:
        parts.append("".join(current))
    pieces = []
    for part in parts:
        if len(part) <= max_chars:
            pieces.append(part)
            continue
        for start in range(0, len(part), max_chars):
            pieces.append(part[start:start + max_chars])
    merged = []
    for piece in pieces:
        if merged and len(merged[-1]) + len(piece) <= max_chars:
            merged[-1] += piece
        else:
            merged.append(piece)
    return merged


def split_text(text, *, max_chars=120) -> list[str]:
    """Split text into sentence-sized segments without losing a single character.

    Paragraphs end at newlines and sentences end at ``。！？!?；;``; both stay at
    the end of the segment they terminate. Sentences longer than ``max_chars``
    are first split after Chinese/Latin commas, then hard-sliced. Whitespace is
    preserved: ``"".join(split_text(text)) == text`` for nonempty text.

    A segment is never longer than ``max_chars``: over-long sentences are split
    at soft breaks and then hard-sliced, and trailing whitespace is rebalanced
    rather than dropped or turned into a whitespace-only segment.
    """
    _validate_source_text(text)
    max_chars = _validate_max_chars(max_chars)
    pieces = []
    for atom in _sentence_atoms(text):
        pieces.extend(_clamp_atom(atom, max_chars))

    # Whitespace-only pieces are attached to the preceding segment when they
    # fit. Sentences are never merged: each non-whitespace piece starts a new
    # segment (or several, if pending whitespace pushes it over the bound).
    result = []
    pending = ""
    for piece in pieces:
        if not piece.strip():
            if result and len(result[-1]) + len(pending) + len(piece) <= max_chars:
                result[-1] += pending + piece
                pending = ""
            else:
                pending += piece
            continue
        combined = pending + piece
        pending = ""
        for start in range(0, len(combined), max_chars):
            result.append(combined[start:start + max_chars])
    if pending:
        if result and len(result[-1]) + len(pending) <= max_chars:
            result[-1] += pending
        elif result and len(result[-1]) > 1 and len(pending) + 1 <= max_chars:
            # Rebalance the final character so the whitespace stays inside the
            # bound without dropping it or creating a whitespace-only segment.
            last_character = result[-1][-1]
            result[-1] = result[-1][:-1]
            result.append(last_character + pending)
        else:
            for start in range(0, len(pending), max_chars):
                result.append(pending[start:start + max_chars])

    assert "".join(result) == text, "分段结果必须完整保留原文"
    return result


# --------------------------------------------------------------------------
# Segmentation coverage checks
# --------------------------------------------------------------------------
def _compact(text):
    """Remove every whitespace character so spacing never affects coverage."""
    return "".join(text.split())


def _excerpt(text, limit=36):
    """Shorten a fragment for a readable Chinese problem message."""
    return text if len(text) <= limit else text[:limit] + "…"


def check_segments(source_text, segments) -> dict:
    """Verify that segments cover the source text in order without invention.

    Whitespace is ignored on both sides. Coverage is measured from substring
    positions in the compacted source, so repeated or reordered segments are
    reported instead of being silently accepted. A segment that does not occur
    in the source at all is treated as hallucinated text.
    """
    _validate_source_text(source_text)
    if not isinstance(segments, (list, tuple)):
        raise ValueError("segments 必须是字符串列表。")
    for index, segment in enumerate(segments, 1):
        if not isinstance(segment, str):
            raise ValueError(f"第 {index} 个片段必须是字符串。")
    compact_source = _compact(source_text)
    if not compact_source:
        raise ValueError("原文去掉空白后为空，无法检查分段覆盖率。")

    covered = [False] * len(compact_source)
    problems = []
    missing_positions = []
    seen = {}
    duplicates = []
    cursor = 0

    for index, raw_segment in enumerate(segments, 1):
        segment = _compact(raw_segment)
        if not segment:
            continue  # Whitespace-only segment: nothing to cover.
        seen[segment] = seen.get(segment, 0) + 1
        position = compact_source.find(segment, cursor)
        if position >= 0:
            cursor = position + len(segment)
        else:
            position = compact_source.find(segment)
            if position < 0:
                problems.append(
                    f"第 {index} 个片段在原文中找不到，可能是凭空添加的内容："
                    f"{_excerpt(segment)}")
                continue
            problems.append(
                f"第 {index} 个片段的顺序与原文不一致，可能分段顺序被打乱："
                f"{_excerpt(segment)}")
        for offset in range(position, position + len(segment)):
            covered[offset] = True
        if seen[segment] == 2:
            duplicates.append(segment)

    if not segments:
        problems.append("segments 为空，没有覆盖任何原文。")

    start = None
    for offset, flag in enumerate(covered + [True]):
        if not flag and start is None:
            start = offset
        elif flag and start is not None:
            missing_positions.append((start, offset))
            start = None
    missing_fragments = []
    for start, end in missing_positions:
        fragment = compact_source[start:end].strip()
        if fragment:
            missing_fragments.append(fragment)
    missing_chars = sum(end - start for start, end in missing_positions)
    if missing_chars:
        problems.append(
            f"原文有 {missing_chars} 个字符没有被任何片段覆盖："
            f"{'；'.join(_excerpt(item) for item in missing_fragments[:5])}")
    if duplicates:
        problems.append(
            "以下片段重复出现，可能造成重复合成：" +
            "；".join(_excerpt(item) for item in duplicates[:5]))

    coverage = sum(covered) / len(compact_source)
    return {
        "ok": not problems,
        "missing": missing_fragments,
        "duplicates": duplicates,
        "coverage": float(min(max(coverage, 0.0), 1.0)),
        "segment_count": len(segments),
        "problems": problems,
    }


# --------------------------------------------------------------------------
# Audio analysis
# --------------------------------------------------------------------------
def _validate_audio_path(value, label):
    if isinstance(value, bool) or not isinstance(value, (str, Path)):
        raise ValueError(f"{label} 必须是音频文件路径。")
    if not str(value).strip():
        raise ValueError(f"{label} 不能是空路径。")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"{label} 不存在或不是文件：{path}")
    return path


def _read_channels(path, label):
    """Decode a WAV into float64 ``(frames, channels)`` without modifying it."""
    try:
        with sf.SoundFile(str(path)) as recording:
            if recording.frames <= 0 or recording.channels <= 0:
                raise ValueError(f"{label} 不包含任何音频采样。")
            if recording.samplerate <= 0:
                raise ValueError(f"{label} 的采样率无效。")
            sample_rate = int(recording.samplerate)
            data = recording.read(dtype="float64", always_2d=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"{label} 不是可读取的音频文件：{exc}") from exc
    if not data.size:
        raise ValueError(f"{label} 不包含任何音频采样。")
    if not np.isfinite(data).all():
        raise ValueError(f"{label} 包含非有限数值（NaN 或无穷大）。")
    return data, sample_rate


def _frame_rms(signal, frame):
    """Non-overlapping frame RMS; the final partial frame uses its real length."""
    full = len(signal) // frame
    remainder = len(signal) - full * frame
    pieces = []
    if full:
        pieces.append(np.sqrt(np.mean(
            signal[:full * frame].reshape(full, frame) ** 2, axis=1)))
    if remainder:
        tail = signal[full * frame:]
        pieces.append(np.array([np.sqrt(np.mean(tail ** 2))]))
    if not pieces:  # pragma: no cover - empty audio is rejected earlier
        return np.zeros(0)
    return np.concatenate(pieces)


def _silence_profile(mono, sample_rate):
    """Return ``(leading, trailing, voiced)`` seconds using 20 ms frames."""
    frame = max(1, int(round(_FRAME_SECONDS * sample_rate)))
    rms = _frame_rms(mono, frame)
    threshold = 10 ** (_SILENCE_DB / 20.0)
    active = rms > threshold
    duration = len(mono) / sample_rate
    if not active.any():
        return duration, duration, 0.0
    first = int(np.argmax(active))
    last = len(active) - int(np.argmax(active[::-1]))
    start_sample = min(len(mono), first * frame)
    end_sample = min(len(mono), last * frame)
    if len(mono) % frame and last == len(rms):
        end_sample = len(mono)  # The active frame is the shorter final one.
    leading = start_sample / sample_rate
    trailing = max(0.0, (len(mono) - end_sample) / sample_rate)
    voiced = max(0.0, (end_sample - start_sample) / sample_rate)
    return leading, trailing, min(voiced, duration)


def _spectral_centroid(mono, sample_rate):
    """Average magnitude-weighted spectral centroid over fixed 40 ms windows."""
    if mono.size < 16:
        return 0.0
    window = max(32, int(round(0.04 * sample_rate)))
    if len(mono) < window:
        window = len(mono)
    window = max(16, window)
    hop = max(1, window // 2)
    taper = np.hanning(window)
    frequencies = np.fft.rfftfreq(window, 1.0 / sample_rate)
    weighted = 0.0
    total_mass = 0.0
    for start in range(0, len(mono) - window + 1, hop):
        frame = mono[start:start + window] * taper
        magnitude = np.abs(np.fft.rfft(frame))
        mass = float(magnitude.sum())
        if mass <= _EPS:
            continue
        weighted += float(np.dot(magnitude, frequencies))
        total_mass += mass
    if total_mass <= _EPS:
        # A very short or mostly silent segment: fall back to the whole signal.
        taper = np.hanning(len(mono))
        frequencies = np.fft.rfftfreq(len(mono), 1.0 / sample_rate)
        magnitude = np.abs(np.fft.rfft(mono * taper))
        total_mass = float(magnitude.sum())
        if total_mass <= _EPS:
            return 0.0
        return float(np.dot(magnitude, frequencies) / total_mass)
    return float(weighted / total_mass)


def _click_count(mono):
    """Count click events whose second difference exceeds a robust bound.

    A click is a contiguous run of samples above the bound, so one impulse is
    reported once instead of once per affected sample. The bound combines the
    median, a MAD-derived sigma and a fraction of the 99th percentile. The
    percentile term keeps a legitimately bright, stable tone from being mistaken
    for a click train, while the small absolute floor still detects a click in
    digital silence.
    """
    if len(mono) < 3:
        return 0
    second_difference = np.abs(np.diff(mono, n=2))
    median = float(np.median(second_difference))
    deviation = np.abs(second_difference - median)
    sigma = 1.4826 * float(np.median(deviation))
    high = float(np.percentile(second_difference, 99.0))
    scale = max(sigma, 0.25 * high, 0.005)
    threshold = median + 8.0 * scale
    above = second_difference > threshold
    if not above.any():
        return 0
    starts = above & np.concatenate(([True], ~above[:-1]))
    return int(np.count_nonzero(starts))


def _median_or_none(values):
    finite = [float(value) for value in values
              if value is not None and math.isfinite(float(value))]
    if not finite:
        return None
    return float(np.median(finite))


def _count_chars(text):
    """Characters that carry speech content: everything except whitespace."""
    return len("".join(text.split()))


def analyze_segments(audio_paths, texts) -> dict:
    """Measure per-segment duration, rate, loudness, timbre and click artefacts.

    Each file is decoded once as float64 and mixed down to mono for analysis;
    the file itself is never rewritten. Median-based flags (sudden acceleration,
    volume jump, timbre drift) need at least two segments; a single segment is
    still measured and checked for clicks.
    """
    if not isinstance(audio_paths, (list, tuple)):
        raise ValueError("audio_paths 必须是列表或元组。")
    if not isinstance(texts, (list, tuple)):
        raise ValueError("texts 必须是列表或元组。")
    if len(audio_paths) != len(texts):
        raise ValueError(
            f"音频数量（{len(audio_paths)}）与文本数量（{len(texts)}）不一致，"
            "无法逐段分析。")
    if not audio_paths:
        raise ValueError("至少需要一个音频片段才能分析。")
    for index, text in enumerate(texts, 1):
        if not isinstance(text, str):
            raise ValueError(f"第 {index} 段文本必须是字符串。")

    segments = []
    for index, (audio_path, text) in enumerate(zip(audio_paths, texts), 1):
        label = f"第 {index} 个音频片段"
        path = _validate_audio_path(audio_path, label)
        data, sample_rate = _read_channels(path, label)
        mono = data.mean(axis=1)
        duration = len(mono) / sample_rate
        peak = float(np.max(np.abs(mono))) if mono.size else 0.0
        peak_dbfs = 20.0 * math.log10(max(peak, _EPS))
        leading, trailing, voiced = _silence_profile(mono, sample_rate)
        loudness = measure_loudness(mono, sample_rate)
        if not math.isfinite(loudness):
            loudness = None
        characters = _count_chars(text)
        chars_per_second = (characters / voiced) if voiced > 0 else 0.0
        centroid = _spectral_centroid(mono, sample_rate)
        clicks = _click_count(mono)
        segment_problems = []
        if voiced <= 0:
            segment_problems.append(
                f"第 {index} 段没有检测到有效语音（全部低于 {_SILENCE_DB:g} dBFS）。")
        if loudness is None:
            segment_problems.append(f"第 {index} 段无法测量响度（可能近乎静音）。")
        segments.append({
            "index": index - 1,
            "path": str(path),
            "text": text,
            "duration": float(duration),
            "sample_rate": int(sample_rate),
            "channels": int(data.shape[1]),
            "chars_per_second": float(chars_per_second),
            "loudness_lufs": loudness,
            "peak_dbfs": float(peak_dbfs),
            "leading_silence": float(leading),
            "trailing_silence": float(trailing),
            "voiced_seconds": float(voiced),
            "clipped_samples": int(np.count_nonzero(np.abs(data) >= _CLIP_THRESHOLD)),
            "spectral_centroid_hz": float(centroid),
            "click_count": int(clicks),
            "flags": {
                "sudden_acceleration": False,
                "volume_jump": False,
                "timbre_drift": False,
                "clicks": bool(clicks > 0),
            },
        })
        segments[-1]["problems"] = segment_problems

    problems = []
    for segment in segments:
        problems.extend(segment["problems"])

    # Median-based flags only make sense with something to compare against.
    if len(segments) > 1:
        median_rate = _median_or_none(item["chars_per_second"] for item in segments)
        median_loudness = _median_or_none(item["loudness_lufs"] for item in segments)
        median_centroid = _median_or_none(item["spectral_centroid_hz"] for item in segments)
        for segment in segments:
            index = segment["index"] + 1
            rate = segment["chars_per_second"]
            if (median_rate is not None and median_rate > 0
                    and rate > 1.6 * median_rate):
                segment["flags"]["sudden_acceleration"] = True
                problems.append(
                    f"第 {index} 段语速突然加快：{rate:.2f} 字/秒，"
                    f"是中位数 {median_rate:.2f} 字/秒的 {rate / median_rate:.2f} 倍。")
            loudness = segment["loudness_lufs"]
            if (loudness is not None and median_loudness is not None
                    and abs(loudness - median_loudness) > 6.0):
                segment["flags"]["volume_jump"] = True
                problems.append(
                    f"第 {index} 段响度跳变：{loudness:.1f} LUFS，"
                    f"与中位数 {median_loudness:.1f} LUFS 相差 "
                    f"{abs(loudness - median_loudness):.1f} dB。")
            centroid = segment["spectral_centroid_hz"]
            if (centroid > 0 and median_centroid is not None
                    and median_centroid > 0):
                ratio = max(centroid / median_centroid, median_centroid / centroid)
                if ratio > 1.35:
                    segment["flags"]["timbre_drift"] = True
                    problems.append(
                        f"第 {index} 段音色漂移：频谱质心 {centroid:.0f} Hz，"
                        f"与中位数 {median_centroid:.0f} Hz 相差 {ratio:.2f} 倍。")
    for segment in segments:
        if segment["flags"]["clicks"]:
            problems.append(
                f"第 {segment['index'] + 1} 段检测到 {segment['click_count']} "
                "处疑似点击声，建议重合成或清理。")

    summary = {
        "segment_count": len(segments),
        "total_duration": float(sum(item["duration"] for item in segments)),
        "median_duration": _median_or_none(item["duration"] for item in segments),
        "median_chars_per_second": _median_or_none(
            item["chars_per_second"] for item in segments),
        "median_loudness_lufs": _median_or_none(
            item["loudness_lufs"] for item in segments),
        "median_peak_dbfs": _median_or_none(
            item["peak_dbfs"] for item in segments),
        "median_spectral_centroid_hz": _median_or_none(
            item["spectral_centroid_hz"] for item in segments),
        "median_click_count": _median_or_none(
            item["click_count"] for item in segments),
        "clipped_samples": int(sum(item["clipped_samples"] for item in segments)),
        "flagged_segments": int(sum(
            1 for item in segments if any(item["flags"].values()))),
    }
    return {
        "ok": not problems,
        "segments": segments,
        "problems": problems,
        "summary": summary,
    }


# --------------------------------------------------------------------------
# Concatenation
# --------------------------------------------------------------------------
def _validate_milliseconds(value, label, maximum):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} 必须是 0–{maximum:g} 之间的数字。")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= maximum:
        raise ValueError(f"{label} 必须在 0–{maximum:g} 之间，当前为 {value!r}。")
    return number


def _validate_output_path(value, label="输出路径"):
    if isinstance(value, bool) or not isinstance(value, (str, Path)):
        raise ValueError(f"{label} 必须是文件路径字符串。")
    if not str(value).strip():
        raise ValueError(f"{label} 不能是空路径。")
    path = Path(value).expanduser().resolve()
    if path.exists() and path.is_dir():
        raise ValueError(f"{label} 不能是已存在的目录：{path}")
    return path


def _apply_fades(data, fade_samples, *, fade_in, fade_out):
    """Apply raised-cosine fades in place on a copy (a fade, not an overlap)."""
    result = np.array(data, dtype=np.float64, copy=True)
    count = min(int(fade_samples), len(result))
    if count <= 0:
        return result
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, count))
    if fade_in:
        result[:count] *= ramp[:, None]
    if fade_out:
        result[-count:] *= ramp[::-1, None]
    return result


def _write_wav_atomic(samples, sample_rate, destination, *, overwrite):
    """Publish a FLOAT WAV via a temporary sibling file and ``os.replace``."""
    destination = Path(destination)
    if destination.parent.exists() and not destination.parent.is_dir():
        raise ValueError(f"输出目录不可用（已存在同名文件）：{destination.parent}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and destination.exists():
        raise FileExistsError(f"输出音频已存在，拒绝覆盖：{destination}")
    temporary = destination.parent / f".longform-{uuid.uuid4().hex}.tmp"
    try:
        with sf.SoundFile(str(temporary), mode="x", samplerate=int(sample_rate),
                          channels=int(samples.shape[1]), format="WAV",
                          subtype="FLOAT") as stream:
            stream.write(samples.astype(np.float32))
        # Verify what will be published before the rename.
        with sf.SoundFile(str(temporary)) as check:
            if check.frames != samples.shape[0] or check.channels != samples.shape[1]:
                raise RuntimeError("写入后的拼接音频尺寸不一致，已放弃发布。")
        if not overwrite and destination.exists():
            raise FileExistsError(f"输出音频已存在，拒绝覆盖：{destination}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def concatenate(audio_paths, output_path, *, crossfade_ms=25, gap_ms=80,
                overwrite=False) -> dict:
    """Join decoded segments with silence gaps and short boundary fades.

    ``crossfade_ms`` is a raised-cosine **fade** at each gap boundary (tail of
    the previous segment, head of the next one), not a true overlap, so the
    total duration is exactly ``sum(segment durations) + gaps``. All inputs must
    share one sample rate and channel count. The result is a new FLOAT WAV; the
    source files are never touched and an existing output needs
    ``overwrite=True``.
    """
    if not isinstance(audio_paths, (list, tuple)):
        raise ValueError("audio_paths 必须是列表或元组。")
    if not audio_paths:
        raise ValueError("至少需要一个输入音频才能拼接。")
    if not isinstance(overwrite, bool):
        raise ValueError("overwrite 必须是布尔值。")
    fade_ms = _validate_milliseconds(crossfade_ms, "crossfade_ms", 200.0)
    gap_ms = _validate_milliseconds(gap_ms, "gap_ms", 2000.0)
    destination = _validate_output_path(output_path)

    decoded = []
    inputs = set()
    for index, item in enumerate(audio_paths, 1):
        label = f"第 {index} 个输入音频"
        path = _validate_audio_path(item, label)
        inputs.add(path)
        data, sample_rate = _read_channels(path, label)
        decoded.append((path, data, sample_rate))

    if destination in inputs:
        raise ValueError("输出路径不能与任何输入音频相同，以免覆盖源音频。")
    if not overwrite and destination.exists():
        raise FileExistsError(f"输出音频已存在，拒绝覆盖：{destination}")

    sample_rate = decoded[0][2]
    channels = decoded[0][1].shape[1]
    for index, (path, data, rate) in enumerate(decoded, 1):
        if rate != sample_rate:
            raise ValueError(
                f"片段采样率不一致：第 {index} 个片段为 {rate} Hz，"
                f"而第 1 个片段为 {sample_rate} Hz；请先统一采样率。")
        if data.shape[1] != channels:
            raise ValueError(
                f"片段通道数不一致：第 {index} 个片段为 {data.shape[1]} 声道，"
                f"而第 1 个片段为 {channels} 声道。")

    fade_samples = int(round(fade_ms * sample_rate / 1000.0))
    gap_samples = int(round(gap_ms * sample_rate / 1000.0))
    total = sum(len(data) for _path, data, _rate in decoded)
    total += gap_samples * (len(decoded) - 1)
    output = np.zeros((total, channels), dtype=np.float64)
    cursor = 0
    for index, (_path, data, _rate) in enumerate(decoded):
        piece = _apply_fades(
            data, fade_samples,
            fade_in=index > 0, fade_out=index < len(decoded) - 1)
        output[cursor:cursor + len(piece)] = piece
        cursor += len(piece)
        if index < len(decoded) - 1:
            cursor += gap_samples

    _write_wav_atomic(output, sample_rate, destination, overwrite=overwrite)
    return {
        "path": str(destination),
        "duration": float(len(output) / sample_rate),
        "segments": len(decoded),
        "sample_rate": int(sample_rate),
        "crossfade_ms": fade_ms,
        "gap_ms": gap_ms,
    }


# --------------------------------------------------------------------------
# Long-form generation orchestration
# --------------------------------------------------------------------------
def _validate_seed(seed):
    if seed is None:
        return None
    if isinstance(seed, bool) or type(seed) is not int:
        raise ValueError("seed 必须是 0–2^63-1 之间的整数或 None。")
    if not 0 <= seed <= _MAX_SEED:
        raise ValueError("seed 必须在 0–2^63-1 之间。")
    return seed


def _validate_directory(value, label):
    if isinstance(value, bool) or not isinstance(value, (str, Path)):
        raise ValueError(f"{label} 必须是目录路径字符串。")
    if not str(value).strip():
        raise ValueError(f"{label} 不能是空路径。")
    path = Path(value).expanduser().resolve()
    if path.exists() and not path.is_dir():
        raise ValueError(f"{label} 已存在但不是目录：{path}")
    return path


def _default_generate(role_id, text, *, params=None, seed=None, device="cuda:0",
                      dtype="float16", cancel=None, store=None):
    """Lazily use the project generation backend; never imported until needed."""
    try:
        from timbreforge.generation import generate_for_character
    except ImportError as exc:
        raise RuntimeError(
            "未找到生成后端 timbreforge.generation.generate_for_character；"
            "请传入 generate 可调用对象或安装完整后端。") from exc
    kwargs = dict(params=params, seed=seed, device=device, dtype=dtype,
                  cancel=cancel)
    if store is not None:
        kwargs["store"] = store
    return generate_for_character(role_id, text, **kwargs)


def _json_safe(value):
    """Convert generator metadata into JSON-safe values without silently lying."""
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        return str(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _write_json_atomic(destination, payload, *, overwrite):
    """Publish UTF-8 JSON through a temporary sibling file and ``os.replace``."""
    destination = Path(destination)
    if destination.parent.exists() and not destination.parent.is_dir():
        raise ValueError(f"报告目录不可用（已存在同名文件）：{destination.parent}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and destination.exists():
        raise FileExistsError(f"报告文件已存在，拒绝覆盖：{destination}")
    temporary = destination.parent / f".longform-report-{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            try:
                json.dump(payload, stream, ensure_ascii=False, allow_nan=False,
                          indent=2)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"报告包含无法序列化的内容：{exc}") from exc
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if not overwrite and destination.exists():
            raise FileExistsError(f"报告文件已存在，拒绝覆盖：{destination}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _unique_strings(items):
    result = []
    seen = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def generate_longform(role_id, text, *, store=None, output_dir=None,
                      max_chars=120, crossfade_ms=25, progress=None, cancel=None,
                      generate=None, params=None, seed=None, device="cuda:0",
                      dtype="float16") -> dict:
    """Split, synthesize each segment, then check, analyse and concatenate.

    ``generate`` is called once per segment as
    ``generate(role_id, segment, params=..., seed=..., device=..., dtype=...,
    cancel=...)`` and must return a dict with a ``file`` key. The default
    implementation lazily imports the project backend, so this module never
    imports a model at import time; ``store`` is forwarded to that default
    backend only, never to an injected callable.

    No output is published when generation fails or cancellation is requested;
    the WAV and its JSON sidecar are only written after every segment exists and
    the analysis finished.
    """
    if not isinstance(role_id, str) or not role_id.strip():
        raise ValueError("role_id 必须是非空字符串。")
    if len(role_id) > _MAX_ROLE_ID_CHARS:
        raise ValueError(f"role_id 过长（最多 {_MAX_ROLE_ID_CHARS} 个字符）。")
    segments = split_text(text, max_chars=max_chars)
    if params is not None and not isinstance(params, dict):
        raise ValueError("params 必须是字典或 None。")
    if progress is not None and not callable(progress):
        raise ValueError("progress 必须是可调用对象或 None。")
    if cancel is not None and not callable(getattr(cancel, "is_set", None)):
        raise ValueError("cancel 必须提供 is_set() 方法或为 None。")
    if generate is not None and not callable(generate):
        raise ValueError("generate 必须是可调用对象或 None。")
    if not isinstance(device, str) or not device.strip():
        raise ValueError("device 必须是非空字符串。")
    if not isinstance(dtype, str) or not dtype.strip():
        raise ValueError("dtype 必须是非空字符串。")
    seed = _validate_seed(seed)
    if seed is not None and seed + len(segments) - 1 > _MAX_SEED:
        raise ValueError("seed 与片段数量相加后超出 2^63-1，请换一个更小的 seed。")
    fade_ms = _validate_milliseconds(crossfade_ms, "crossfade_ms", 200.0)
    # The public helper signature has no gap_ms parameter; long-form narration
    # uses the documented default gap so every segment breathes the same way.
    gap_ms = 80.0
    destination_dir = (_validate_directory(output_dir, "output_dir")
                       if output_dir is not None else None)

    if generate is None:
        backend_store = store

        def _generate_with_store(role_id, segment_text, *, params=None, seed=None,
                                 device="cuda:0", dtype="float16", cancel=None):
            return _default_generate(
                role_id, segment_text, params=params, seed=seed, device=device,
                dtype=dtype, cancel=cancel, store=backend_store)

        generate = _generate_with_store

    segment_seeds = ([seed + index for index in range(len(segments))]
                     if seed is not None else [None] * len(segments))
    generated = []
    for index, segment in enumerate(segments):
        if cancel is not None and cancel.is_set():
            raise RuntimeError("任务已取消")
        result = generate(
            role_id, segment, params=params, seed=segment_seeds[index],
            device=device, dtype=dtype, cancel=cancel)
        if not isinstance(result, dict):
            raise ValueError(f"第 {index + 1} 段生成结果必须是字典。")
        file_value = result.get("file")
        if isinstance(file_value, bool) or not isinstance(file_value, (str, Path)):
            raise ValueError(f"第 {index + 1} 段生成结果缺少音频文件路径（file）。")
        file_path = Path(file_value).expanduser().resolve()
        if not file_path.is_file():
            raise ValueError(
                f"第 {index + 1} 段生成结果中的音频文件不存在：{file_path}")
        metadata_value = result.get("metadata_file")
        metadata_file = None
        if metadata_value is not None:
            if isinstance(metadata_value, bool) or not isinstance(
                    metadata_value, (str, Path)):
                raise ValueError(f"第 {index + 1} 段的 metadata_file 必须是路径或 None。")
            metadata_path = Path(metadata_value).expanduser().resolve()
            if not metadata_path.is_file():
                raise ValueError(
                    f"第 {index + 1} 段的 metadata_file 不存在：{metadata_path}")
            metadata_file = str(metadata_path)
        extras = {
            key: _json_safe(value)
            for key, value in result.items()
            if key not in {"file", "metadata_file"}
        }
        generated.append({
            "index": index,
            "role_id": role_id,
            "text": segment,
            "params": _json_safe(params) if params is not None else None,
            "seed": segment_seeds[index],
            "file": str(file_path),
            "metadata_file": metadata_file,
            "metadata": extras,
        })
        if progress is not None:
            progress((index + 1) / len(segments),
                     desc=f"已完成第 {index + 1}/{len(segments)} 段")

    if cancel is not None and cancel.is_set():
        raise RuntimeError("任务已取消")

    files = [item["file"] for item in generated]
    check = check_segments(text, segments)
    analysis = analyze_segments(files, segments)

    base_dir = destination_dir if destination_dir is not None \
        else Path(files[0]).parent / "longform"
    output_path = base_dir / f"longform-{uuid.uuid4().hex}.wav"
    audio = concatenate(files, output_path, crossfade_ms=fade_ms,
                        gap_ms=gap_ms, overwrite=False)

    report_path = output_path.with_suffix(".json")
    created_at = datetime.now(timezone.utc).isoformat()
    report = {
        "schema_version": 1,
        "kind": "timbreforge.longform",
        "role_id": role_id,
        "created_at": created_at,
        "text": text,
        "split": {
            "max_chars": int(max_chars),
            "segment_count": len(segments),
        },
        "segments": generated,
        "segment_texts": list(segments),
        "files": files,
        "generation": {
            "params": _json_safe(params) if params is not None else None,
            "seed": seed,
            "device": device,
            "dtype": dtype,
            "files": files,
        },
        "audio": {
            "path": audio["path"],
            "duration": audio["duration"],
            "sample_rate": audio["sample_rate"],
            "crossfade_ms": audio["crossfade_ms"],
            "gap_ms": audio["gap_ms"],
        },
        "check": check,
        "analysis": analysis,
    }
    try:
        _write_json_atomic(report_path, report, overwrite=False)
    except Exception:
        # The sidecar is part of the deliverable: never leave half a result.
        output_path.unlink(missing_ok=True)
        raise

    warnings = _unique_strings(list(check["problems"]) + list(analysis["problems"]))
    return {
        "ok": True,
        "file": str(output_path),
        "report_file": str(report_path),
        "segments": [
            {"index": item["index"], "text": item["text"], "file": item["file"]}
            for item in generated
        ],
        "analysis": analysis,
        "warnings": warnings,
    }

