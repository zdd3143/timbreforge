"""Fixed evaluation set and auxiliary metrics for generated candidate audio.

The metrics here are deliberately independent and transparent:

* signal measurements come from NumPy + SoundFile;
* character/word error rates need a **user supplied** transcript, because this
  project does not run ASR;
* speaker similarity is opt-in and delegates to the existing CAM++ scorer;
* timbre drift compares spectral centroids, which is a coarse brightness proxy,
  not speaker identity.

Nothing is ranked or selected automatically. Chinese text is evaluated with CER;
WER is only reported for whitespace-separated (Latin style) text.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf

from timbreforge.audio_edit_pipeline import measure_loudness

TEST_CATEGORIES = (
    "low_emotion_short",
    "question",
    "command",
    "long_sentence",
    "numbers_and_names",
    "high_emotion",
    "cold_low_energy",
    "phonetic_coverage",
)

_TEST_SET = (
    {
        "category": "low_emotion_short",
        "label": "低情绪短句",
        "text": "好的，我知道了。",
    },
    {
        "category": "question",
        "label": "疑问句",
        "text": "你确定明天下午三点能准时到会议室吗？",
    },
    {
        "category": "command",
        "label": "命令句",
        "text": "请立即关闭设备，然后回到安全区域等待通知。",
    },
    {
        "category": "long_sentence",
        "label": "长句",
        "text": "在这次测试中，我们需要确保每一段语音的语速、响度和音色都保持稳定，"
                "避免出现明显的跳变或断续。",
    },
    {
        "category": "numbers_and_names",
        "label": "数字与专有名词",
        "text": "订单 2024 年 8 月 15 日由王小明签收，金额为 3860.5 元。",
    },
    {
        "category": "high_emotion",
        "label": "高情绪",
        "text": "太好了！我们终于成功了，这真是难以置信的突破！",
    },
    {
        "category": "cold_low_energy",
        "label": "冰冷低能量",
        "text": "随你吧，我没有任何意见。",
    },
    {
        "category": "phonetic_coverage",
        "label": "音素覆盖",
        "text": "春风吹过江面，碧波荡漾；小桥旁的石阶上，有人轻声念着古老的歌谣。",
    },
)

_SILENCE_DB = -42.0
_FRAME_SECONDS = 0.02
_CLIP_THRESHOLD = 0.999
_EPS = 1e-12
_MAX_TRANSCRIPT_CHARS = 5000
_FAST_RATE = 8.5
_SLOW_RATE = 2.0
_SIGNAL_METRICS = (
    "duration",
    "peak_dbfs",
    "rms_dbfs",
    "loudness_lufs",
    "leading_silence",
    "trailing_silence",
    "voiced_seconds",
    "clipped_samples",
    "near_full_scale_fraction",
)
_ENTRY_METRICS = {
    "cer": "cer",
    "wer": "wer",
    "speaker_similarity": "speaker_similarity",
    "drift": "timbre_drift",
}
_AVERAGE_METRICS = _SIGNAL_METRICS + (
    "cer",
    "wer",
    "speaker_similarity",
    "drift",
)
_KEY_METRICS = ("cer", "drift", "speaker_similarity", "clipped_samples")
_WORST_DIRECTION = {
    "cer": "max",
    "drift": "max",
    "speaker_similarity": "min",
    "clipped_samples": "max",
}
_SUMMARY_NOTE = (
    "汇总只是辅助统计：指标是估算值，既不构成自动排名，也不能替代人工听感。"
)
_AUXILIARY_WARNING = (
    "以下指标是辅助参考信号，不是自动排名、自动挑选或最终听感结论。"
)


# --------------------------------------------------------------------------
# Fixed evaluation set
# --------------------------------------------------------------------------
def fixed_test_set() -> list[dict]:
    """Return a fresh copy of the fixed Simplified-Chinese evaluation set."""
    return [dict(item) for item in _TEST_SET]


# --------------------------------------------------------------------------
# Audio decoding and shared signal helpers
# --------------------------------------------------------------------------
def _validate_audio_path(value, label="音频路径"):
    if isinstance(value, bool) or not isinstance(value, (str, Path)):
        raise ValueError(f"{label} 必须是音频文件路径。")
    if not str(value).strip():
        raise ValueError(f"{label} 不能是空路径。")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"{label} 不存在或不是文件：{path}")
    return path


def _read_audio(audio_path):
    """Decode any supported audio into float64 ``(frames, channels)``."""
    path = _validate_audio_path(audio_path)
    try:
        with sf.SoundFile(str(path)) as recording:
            if recording.frames <= 0 or recording.channels <= 0:
                raise ValueError(f"音频文件不包含任何采样：{path}")
            if recording.samplerate <= 0:
                raise ValueError(f"音频文件的采样率无效：{path}")
            sample_rate = int(recording.samplerate)
            data = recording.read(dtype="float64", always_2d=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"无法读取音频文件 {path}：{exc}") from exc
    if not data.size:
        raise ValueError(f"音频文件不包含任何采样：{path}")
    if not np.isfinite(data).all():
        raise ValueError(f"音频文件包含非有限数值（NaN 或无穷大）：{path}")
    return path, data, sample_rate


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
    active = rms > 10 ** (_SILENCE_DB / 20.0)
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


def _mean_spectral_centroid(mono, sample_rate):
    """Average magnitude-weighted spectral centroid over fixed 40 ms windows."""
    if mono.size < 16:
        raise ValueError("音频过短，无法计算频谱质心。")
    window = max(32, int(round(0.04 * sample_rate)))
    window = min(window, len(mono))
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
        taper = np.hanning(len(mono))
        frequencies = np.fft.rfftfreq(len(mono), 1.0 / sample_rate)
        magnitude = np.abs(np.fft.rfft(mono * taper))
        total_mass = float(magnitude.sum())
        if total_mass <= _EPS:
            raise ValueError("音频近乎静音，无法计算频谱质心。")
        return float(np.dot(magnitude, frequencies) / total_mass)
    return float(weighted / total_mass)


def _levenshtein(expected, observed):
    """Unicode-aware edit distance over code points (punctuation included)."""
    if len(expected) < len(observed):
        expected, observed = observed, expected
    previous = list(range(len(observed) + 1))
    for row, left in enumerate(expected, 1):
        current = [row]
        for column, right in enumerate(observed, 1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (left != right),
            ))
        previous = current
    return previous[-1]


def _validate_transcript(value, label):
    if not isinstance(value, str):
        raise ValueError(f"{label}必须是字符串。")
    if len(value) > _MAX_TRANSCRIPT_CHARS:
        raise ValueError(
            f"{label}长度为 {len(value)} 个字符，超过 {_MAX_TRANSCRIPT_CHARS} 的上限。")
    return value


# --------------------------------------------------------------------------
# Signal metrics
# --------------------------------------------------------------------------
def measure_signal(audio_path) -> dict:
    """Measure signal-only attributes; no transcript is used or guessed.

    ``loudness_lufs`` is ``None`` when loudness cannot be measured (digital
    silence). Silence boundaries use 20 ms frames below -42 dBFS, and clipping
    counts samples at or above 0.999 of full scale.
    """
    path, data, sample_rate = _read_audio(audio_path)
    mono = data.mean(axis=1)
    duration = len(mono) / sample_rate
    peak = float(np.max(np.abs(data)))
    rms = float(np.sqrt(np.mean(np.square(data))))
    leading, trailing, voiced = _silence_profile(mono, sample_rate)
    loudness = measure_loudness(mono, sample_rate)
    if not math.isfinite(loudness):
        loudness = None
    clipped = int(np.count_nonzero(np.abs(data) >= _CLIP_THRESHOLD))
    return {
        "duration": float(duration),
        "sample_rate": int(sample_rate),
        "channels": int(data.shape[1]),
        "peak_dbfs": float(20.0 * math.log10(max(peak, _EPS))),
        "rms_dbfs": float(20.0 * math.log10(max(rms, _EPS))),
        "loudness_lufs": loudness,
        "leading_silence": float(leading),
        "trailing_silence": float(trailing),
        "voiced_seconds": float(voiced),
        "clipped_samples": clipped,
        "near_full_scale_fraction": float(np.mean(np.abs(data) >= _CLIP_THRESHOLD)),
    }


# --------------------------------------------------------------------------
# Text metrics
# --------------------------------------------------------------------------
def character_error_rate(expected, observed) -> dict:
    """Unicode Levenshtein distance divided by the expected character count.

    Punctuation and spacing are kept as ordinary characters; this is the
    appropriate metric for Chinese, where whitespace tokenisation is meaningless.
    """
    _validate_transcript(expected, "期望文本")
    _validate_transcript(observed, "识别文本")
    if not expected:
        raise ValueError("期望文本不能为空，无法计算字符错误率。")
    distance = _levenshtein(expected, observed)
    return {
        "status": "measured",
        "cer": float(distance / len(expected)),
        "distance": int(distance),
        "expected_length": int(len(expected)),
    }


def word_error_rate(expected, observed) -> dict:
    """Whitespace-token Levenshtein divided by the expected token count.

    Chinese has no reliable whitespace tokenisation, so a string without any
    whitespace returns ``not_applicable`` instead of a fabricated WER.
    """
    _validate_transcript(expected, "期望文本")
    _validate_transcript(observed, "识别文本")
    if not expected.strip():
        raise ValueError("期望文本不能为空，无法计算词错误率。")
    if not any(character.isspace() for character in expected):
        return {
            "status": "not_applicable",
            "wer": None,
            "reason": "期望文本不含空格分词，中文请使用字符错误率（CER）。",
        }
    expected_tokens = expected.split()
    observed_tokens = observed.split()
    if not expected_tokens:  # pragma: no cover - guarded by the strip check
        raise ValueError("期望文本没有可比较的词。")
    distance = _levenshtein(expected_tokens, observed_tokens)
    return {
        "status": "measured",
        "wer": float(distance / len(expected_tokens)),
        "distance": int(distance),
        "expected_length": int(len(expected_tokens)),
    }


# --------------------------------------------------------------------------
# Timbre drift
# --------------------------------------------------------------------------
def timbre_drift(audio_path, reference_audio) -> dict:
    """Compare average spectral centroids as a coarse brightness drift proxy."""
    path, data, sample_rate = _read_audio(audio_path)
    reference_path, reference_data, reference_rate = _read_audio(reference_audio)
    mono = data.mean(axis=1)
    reference_mono = reference_data.mean(axis=1)
    if float(np.max(np.abs(mono))) <= 1e-6:
        raise ValueError(f"候选音频近乎静音，无法计算音色漂移：{path}")
    if float(np.max(np.abs(reference_mono))) <= 1e-6:
        raise ValueError(f"参考音频近乎静音，无法计算音色漂移：{reference_path}")
    centroid = _mean_spectral_centroid(mono, sample_rate)
    reference_centroid = _mean_spectral_centroid(reference_mono, reference_rate)
    if reference_centroid <= _EPS:
        raise ValueError("参考音频的频谱质心为零，无法计算音色漂移。")
    ratio = centroid / reference_centroid
    return {
        "status": "measured",
        "centroid_hz": float(centroid),
        "reference_centroid_hz": float(reference_centroid),
        "ratio": float(ratio),
        "drift": float(abs(ratio - 1.0)),
    }


# --------------------------------------------------------------------------
# Candidate evaluation
# --------------------------------------------------------------------------
def _validate_rating(rating):
    if rating is None:
        return None
    if isinstance(rating, bool) or not isinstance(rating, int):
        raise ValueError("rating 必须是 1–5 的整数或 None。")
    if not 1 <= rating <= 5:
        raise ValueError("rating 必须在 1–5 之间（含 1 和 5）。")
    return rating


def _quality_checks(signal, expected_text, cer_entry):
    """Build Chinese human-review hints; they never block or rank a candidate."""
    checks = []
    if signal["clipped_samples"] > 0:
        checks.append(
            f"检测到 {signal['clipped_samples']} 个接近满刻度的采样点，"
            "可能存在削波失真。")
    if signal["peak_dbfs"] > -0.5:
        checks.append(
            f"峰值 {signal['peak_dbfs']:.1f} dBFS 已接近 0 dBFS，注意削波风险。")
    if signal["peak_dbfs"] < -30.0:
        checks.append(
            f"峰值偏低（{signal['peak_dbfs']:.1f} dBFS），可能过于安静。")
    if signal["loudness_lufs"] is None:
        checks.append("无法测量整体响度，音频可能近乎静音。")
    elif signal["loudness_lufs"] < -35.0:
        checks.append(
            f"整体响度偏低（{signal['loudness_lufs']:.1f} LUFS）。")
    characters = len("".join(expected_text.split()))
    if characters and signal["voiced_seconds"] > 0:
        rate = characters / signal["voiced_seconds"]
        if rate > _FAST_RATE:
            checks.append(
                f"语速明显偏快：约 {rate:.1f} 字/秒，请确认是否漏字或吃字。")
        elif rate < _SLOW_RATE:
            checks.append(
                f"语速明显偏慢：约 {rate:.1f} 字/秒，请确认是否拖沓。")
    if signal["leading_silence"] > 2.0:
        checks.append(
            f"开头静音约 {signal['leading_silence']:.1f} 秒，可能影响衔接。")
    if signal["trailing_silence"] > 3.0:
        checks.append(
            f"结尾静音约 {signal['trailing_silence']:.1f} 秒，可能影响衔接。")
    if cer_entry.get("status") == "measured" and cer_entry["value"] > 0.2:
        checks.append(
            f"字符错误率偏高（{cer_entry['value']:.2f}），建议人工复听核对。")
    return checks


def evaluate_candidate(audio_path, expected_text, *, observed_text=None,
                       reference_audio=None, rating=None, run_speaker=False) -> dict:
    """Measure one candidate; accuracy stays unverified without a transcript.

    ``observed_text`` is a user-provided transcript, never an ASR result. Without
    it the CER entry is ``unverified`` with ``value=None``; this function never
    invents accuracy. Speaker similarity is measured only for
    ``run_speaker=True`` and marked unavailable (with the error) on failure.
    """
    if not isinstance(expected_text, str) or not expected_text.strip():
        raise ValueError("expected_text 必须是非空字符串。")
    _validate_transcript(expected_text, "期望文本")
    if observed_text is not None:
        _validate_transcript(observed_text, "识别文本")
    if not isinstance(run_speaker, bool):
        raise ValueError("run_speaker 必须是布尔值。")
    rating = _validate_rating(rating)
    path = _validate_audio_path(audio_path)
    reference_path = None
    if reference_audio is not None:
        reference_path = _validate_audio_path(reference_audio, "参考音频")

    signal = measure_signal(path)
    metrics = {
        "signal": signal,
        "cer": {"status": "unverified", "value": None},
        "wer": {"status": "unverified", "value": None},
        "speaker_similarity": {"status": "skipped", "value": None},
        "timbre_drift": {"status": "not_requested", "value": None},
    }

    if observed_text is None:
        metrics["cer"] = {
            "status": "unverified",
            "value": None,
            "reason": "未提供人工听写文本，内容准确度未经验证。",
        }
        metrics["wer"] = {
            "status": "unverified",
            "value": None,
            "reason": "未提供人工听写文本，未计算词错误率。",
        }
    else:
        cer = character_error_rate(expected_text, observed_text)
        metrics["cer"] = {
            "status": "measured",
            "value": float(cer["cer"]),
            "distance": int(cer["distance"]),
            "expected_length": int(cer["expected_length"]),
            "method": "Unicode Levenshtein / 期望字符数（含标点）",
        }
        wer = word_error_rate(expected_text, observed_text)
        if wer["status"] == "measured":
            metrics["wer"] = {
                "status": "measured",
                "value": float(wer["wer"]),
                "distance": int(wer["distance"]),
                "expected_length": int(wer["expected_length"]),
                "method": "空格分词 Levenshtein / 期望词数",
            }
        else:
            metrics["wer"] = {
                "status": "not_applicable",
                "value": None,
                "reason": wer["reason"],
            }

    if run_speaker:
        if reference_path is None:
            raise ValueError("启用说话人相似度时必须提供参考音频（reference_audio）。")
        try:
            from timbreforge.score import score_one
            similarity = float(score_one(str(reference_path), str(path)))
            if not math.isfinite(similarity) or not -1.0 <= similarity <= 1.0:
                raise ValueError("打分器返回的余弦值不在 [-1, 1] 内。")
            metrics["speaker_similarity"] = {
                "status": "measured",
                "value": similarity,
                "method": "CAM++ 余弦相似度（可选）",
            }
        except Exception as exc:  # noqa: BLE001 - optional external scorer
            metrics["speaker_similarity"] = {
                "status": "unavailable",
                "value": None,
                "error": str(exc),
            }

    if reference_path is not None:
        try:
            drift = timbre_drift(path, reference_path)
            metrics["timbre_drift"] = {
                "status": "measured",
                "value": float(drift["drift"]),
                "ratio": float(drift["ratio"]),
                "centroid_hz": float(drift["centroid_hz"]),
                "reference_centroid_hz": float(drift["reference_centroid_hz"]),
                "method": "平均频谱质心比值（粗略亮度漂移，不是说话人身份）",
            }
        except ValueError as exc:
            metrics["timbre_drift"] = {
                "status": "unavailable",
                "value": None,
                "error": str(exc),
            }

    warnings = [_AUXILIARY_WARNING]
    if observed_text is None:
        warnings.append(
            "未提供人工听写文本：字符/词错误率为“未验证”，不会凭空给出准确率。")
    return {
        "metrics": metrics,
        "checks": _quality_checks(signal, expected_text, metrics["cer"]),
        "file": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "expected_text": expected_text,
        "observed_text": observed_text,
        "rating": rating,
        "warnings": warnings,
    }


# --------------------------------------------------------------------------
# Summary and report persistence
# --------------------------------------------------------------------------
def _metric_value(result, name):
    """Extract one finite numeric metric, or ``None`` when it is not measured."""
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        return None
    if name in _SIGNAL_METRICS:
        signal = metrics.get("signal")
        value = signal.get(name) if isinstance(signal, dict) else None
    else:
        entry = metrics.get(_ENTRY_METRICS[name])
        if isinstance(entry, dict):
            value = entry.get("value")
        elif isinstance(entry, (int, float)) and not isinstance(entry, bool):
            # Tolerate a compact report that stores the number directly.
            value = entry
        else:
            value = None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def summarize(results) -> dict:
    """Average measured metrics and point out the worst candidate per key axis.

    Unverified/unavailable metrics are skipped and named in ``unmeasured``. The
    summary is descriptive only: ``worst`` is not a recommendation to select or
    discard a candidate.
    """
    if not isinstance(results, (list, tuple)):
        raise ValueError("results 必须是评测结果列表。")
    for index, result in enumerate(results, 1):
        if not isinstance(result, dict):
            raise ValueError(f"第 {index} 条评测结果必须是字典。")

    averages = {}
    unmeasured = []
    for name in _AVERAGE_METRICS:
        values = [value for value in
                  (_metric_value(result, name) for result in results)
                  if value is not None]
        if values:
            averages[name] = float(sum(values) / len(values))
        else:
            unmeasured.append(name)

    worst = {}
    for name in _KEY_METRICS:
        measured = []
        for index, result in enumerate(results):
            value = _metric_value(result, name)
            if value is not None:
                measured.append((index, result, value))
        if not measured:
            worst[name] = None
            continue
        direction = _WORST_DIRECTION[name]
        chosen = measured[0]
        for item in measured[1:]:
            if direction == "max" and item[2] > chosen[2]:
                chosen = item
            elif direction == "min" and item[2] < chosen[2]:
                chosen = item
        worst[name] = {
            "value": chosen[2],
            "file": chosen[1].get("file"),
            "category": chosen[1].get("category"),
            "index": chosen[0],
        }

    return {
        "count": len(results),
        "averages": averages,
        "worst": worst,
        "unmeasured": unmeasured,
        "note": _SUMMARY_NOTE,
    }


def save_report(report, path, *, overwrite=False) -> str:
    """Atomically write a UTF-8 JSON report; never overwrite unless asked."""
    if not isinstance(report, dict):
        raise ValueError("report 必须是字典。")
    if not isinstance(overwrite, bool):
        raise ValueError("overwrite 必须是布尔值。")
    if isinstance(path, bool) or not isinstance(path, (str, Path)):
        raise ValueError("path 必须是文件路径字符串。")
    if not str(path).strip():
        raise ValueError("path 不能是空路径。")
    destination = Path(path).expanduser().resolve()
    if destination.exists():
        if destination.is_dir():
            raise ValueError(f"报告路径不能是已存在的目录：{destination}")
        if not overwrite:
            raise FileExistsError(f"报告文件已存在，拒绝覆盖：{destination}")
    if destination.parent.exists() and not destination.parent.is_dir():
        raise ValueError(f"报告目录不可用（已存在同名文件）：{destination.parent}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".report-{uuid.uuid4().hex}.json.tmp"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            try:
                json.dump(report, stream, ensure_ascii=False, allow_nan=False,
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
    return str(destination)

