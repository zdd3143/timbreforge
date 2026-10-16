"""Conservative reference preparation; preserve native sample rate and timbre.

Energy trimming is a heuristic, NOT a speech/speaker detector. No denoising,
pitch shift or forced loudness boost. Cache identity covers source content and
all processing settings, so changed audio cannot reuse stale conditioning.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf


def prepare_reference(source: str | Path, max_seconds: float | None = 6.0,
                      cache_dir: str | Path | None = None,
                      target_peak: float | None = None,
                      max_gain_db: float = 6.0) -> Path:
    source = Path(source).resolve()
    if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("max_seconds must be positive and finite, or None")
    if target_peak is not None and not 0 < target_peak < 1:
        raise ValueError("target_peak must be between zero and one")
    if not math.isfinite(max_gain_db) or max_gain_db < 0:
        raise ValueError("max_gain_db must be nonnegative and finite")
    raw = source.read_bytes()
    settings = dict(version=1, max_seconds=max_seconds, target_peak=target_peak,
                    max_gain_db=max_gain_db)
    identity = hashlib.sha256(raw + json.dumps(settings, sort_keys=True).encode()).hexdigest()
    if cache_dir is None:
        from timbreforge.config import get_config
        cache_dir = get_config().outputs_dir / 'cache' / 'prepared'
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / f"ref_{identity[:24]}.wav"
    meta = dest.with_suffix(".json")
    if dest.is_file() and meta.is_file():
        return dest
    x, sr = sf.read(source, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if not x.size or not np.isfinite(x).all():
        raise ValueError("Reference is empty or contains nonfinite samples")
    peak = float(np.max(np.abs(x)))
    if peak < 1e-6:
        raise ValueError("Reference contains no usable signal")
    # Remove only leading/trailing low-energy frames, leaving 100 ms context.
    hop = max(1, int(sr * 0.02))
    pad = (-len(x)) % hop
    frames = np.pad(x, (0, pad)).reshape(-1, hop)
    energy = np.mean(frames.astype(np.float64) ** 2, axis=1)
    threshold = max(float(np.max(energy)) * 0.001, 1e-10)
    active = np.flatnonzero(energy > threshold)
    if not active.size:
        raise ValueError("Reference contains no usable sustained signal")
    margin = int(0.1 * sr)
    start = max(0, int(active[0]) * hop - margin)
    end = min(len(x), (int(active[-1]) + 1) * hop + margin)
    if max_seconds is not None:
        count = max(1, int(max_seconds * sr))
        if end - start > count:
            # Select a contiguous high-energy window, never concatenate fragments.
            span = max(1, count // hop)
            sums = np.convolve(energy, np.ones(span), mode="valid")
            lo = (start + hop - 1) // hop
            hi = min(len(sums) - 1, max(lo, (end - count) // hop))
            frame = lo + int(np.argmax(sums[lo:hi + 1])) if hi >= lo else lo
            start = min(max(frame * hop, start), end - count)
            end = start + count
    y = x[start:end]
    gain = 1.0
    if target_peak is not None:
        gain = min(target_peak / float(np.max(np.abs(y))), 10 ** (max_gain_db / 20))
        y = y * gain
    # Attenuate overload, but never clip or normalize a quiet signal by default.
    actual_peak = float(np.max(np.abs(y)))
    if actual_peak >= 1:
        attenuation = 0.99 / actual_peak
        y = y * attenuation
        gain *= attenuation
    sf.write(dest, y, sr, subtype="FLOAT")
    meta.write_text(json.dumps(dict(source=str(source), source_sha256=hashlib.sha256(raw).hexdigest(),
                                   sample_rate=sr, start_seconds=start / sr,
                                   duration_seconds=len(y) / sr, gain=gain,
                                   settings=settings), ensure_ascii=False, indent=2), encoding="utf-8")
    return dest
