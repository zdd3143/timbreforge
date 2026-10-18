"""Non-destructive reference-audio conditioning.

Every function here is deterministic, dependency-light (NumPy + SoundFile) and
works on decoded samples only. The pipeline never rewrites its input: callers
write a *new* derived WAV, keep the source bytes intact, and record the chain
and both hashes in the character manifest.

Measurements are documented auxiliary estimates: integrated loudness follows
ITU-R BS.1770 gating, while EQ, denoise and compression are simple, auditable
DSP written for short TTS reference clips, not mastering-grade processors.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import uuid

import numpy as np
import soundfile as sf

from timbreforge.audio_edit_schema import (
    ChainError,
    describe_operations,
    make_edit_document,
    validate_operations,
)

MAX_SECONDS = 300.0
MAX_SAMPLES = 192000 * int(MAX_SECONDS)
_EPS = 1e-12


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_audio(path):
    """Decode a mono/multichannel recording into float64 ``(frames, channels)``."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f'音频文件不存在：{source}')
    try:
        with sf.SoundFile(str(source)) as recording:
            if recording.frames <= 0 or recording.channels <= 0:
                raise ChainError('音频文件不包含任何采样。')
            if not 8000 <= recording.samplerate <= 192000:
                raise ChainError('音频采样率超出 8–192 kHz 的允许范围。')
            if recording.channels > 32:
                raise ChainError('音频通道数过多。')
            if recording.frames > MAX_SAMPLES:
                raise ChainError(f'音频超过 {MAX_SECONDS:.0f} 秒的上限。')
            samples = recording.read(dtype='float64', always_2d=True)
            sample_rate = recording.samplerate
    except (OSError, RuntimeError) as exc:
        raise ChainError(f'无法读取音频文件：{exc}') from exc
    if not samples.size or not np.isfinite(samples).all():
        raise ChainError('音频采样为空或包含非有限数值。')
    if np.max(np.abs(samples)) <= 1e-8:
        raise ChainError('音频近乎静音，无法处理。')
    return samples, sample_rate


def _rms_db(x):
    rms = float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0
    return 20.0 * math.log10(max(rms, _EPS))


def measure_peak_dbfs(x):
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    return 20.0 * math.log10(max(peak, _EPS))


def count_clipped(x, threshold=0.999):
    """Number of samples at or above the near-full-scale threshold."""
    return int(np.count_nonzero(np.abs(x) >= threshold))


def _k_weighting_coefficients(sample_rate):
    """BS.1770 K-weighting as cascaded RBJ biquads at any sample rate."""
    # Stage 1: high shelf (+4 dB above ~1.5 kHz).
    f0, gain, q = 1681.974450955533, 4.0, 0.7071752369554196
    w0 = 2 * math.pi * f0 / sample_rate
    cos_w0, sin_w0 = math.cos(w0), math.sin(w0)
    alpha = sin_w0 / (2 * q)
    a = 10 ** (gain / 40)
    sqrt_a = math.sqrt(a)
    shelf = (
        [a * ((a + 1) + (a - 1) * cos_w0 + 2 * sqrt_a * alpha),
         -2 * a * ((a - 1) + (a + 1) * cos_w0),
         a * ((a + 1) + (a - 1) * cos_w0 - 2 * sqrt_a * alpha),
         (a + 1) - (a - 1) * cos_w0 + 2 * sqrt_a * alpha,
         2 * ((a - 1) - (a + 1) * cos_w0),
         (a + 1) - (a - 1) * cos_w0 - 2 * sqrt_a * alpha],
        ['highshelf'],
    )
    # Stage 2: RLB high-pass (~38 Hz).
    f0, q = 38.13547087602444, 0.5003270373238773
    w0 = 2 * math.pi * f0 / sample_rate
    cos_w0, sin_w0 = math.cos(w0), math.sin(w0)
    alpha = sin_w0 / (2 * q)
    highpass = (
        [(1 + cos_w0) / 2, -(1 + cos_w0), (1 + cos_w0) / 2,
         1 + alpha, -2 * cos_w0, 1 - alpha],
        ['highpass'],
    )
    return [shelf, highpass]


def _biquad(kind, freq, gain_db, q, sample_rate):
    """RBJ audio-EQ-cookbook biquad; returns normalized ``(b, a)``."""
    freq = min(max(freq, 10.0), sample_rate * 0.49)
    w0 = 2 * math.pi * freq / sample_rate
    cos_w0, sin_w0 = math.cos(w0), math.sin(w0)
    if kind == 'highpass':
        alpha = sin_w0 / (2 * q)
        b = [(1 + cos_w0) / 2, -(1 + cos_w0), (1 + cos_w0) / 2]
        a = [1 + alpha, -2 * cos_w0, 1 - alpha]
    elif kind == 'lowpass':
        alpha = sin_w0 / (2 * q)
        b = [(1 - cos_w0) / 2, 1 - cos_w0, (1 - cos_w0) / 2]
        a = [1 + alpha, -2 * cos_w0, 1 - alpha]
    elif kind == 'peaking':
        alpha = sin_w0 / (2 * q)
        amplitude = 10 ** (gain_db / 40)
        b = [1 + alpha * amplitude, -2 * cos_w0, 1 - alpha * amplitude]
        a = [1 + alpha / amplitude, -2 * cos_w0, 1 - alpha / amplitude]
    elif kind == 'lowshelf':
        alpha = sin_w0 / 2 * math.sqrt(2)
        amplitude = 10 ** (gain_db / 40)
        sqrt_a = math.sqrt(amplitude)
        b = [amplitude * ((amplitude + 1) - (amplitude - 1) * cos_w0 + 2 * sqrt_a * alpha),
             2 * amplitude * ((amplitude - 1) - (amplitude + 1) * cos_w0),
             amplitude * ((amplitude + 1) - (amplitude - 1) * cos_w0 - 2 * sqrt_a * alpha)]
        a = [(amplitude + 1) + (amplitude - 1) * cos_w0 + 2 * sqrt_a * alpha,
             -2 * ((amplitude - 1) + (amplitude + 1) * cos_w0),
             (amplitude + 1) + (amplitude - 1) * cos_w0 - 2 * sqrt_a * alpha]
    else:  # highshelf
        alpha = sin_w0 / 2 * math.sqrt(2)
        amplitude = 10 ** (gain_db / 40)
        sqrt_a = math.sqrt(amplitude)
        b = [amplitude * ((amplitude + 1) + (amplitude - 1) * cos_w0 + 2 * sqrt_a * alpha),
             -2 * amplitude * ((amplitude - 1) + (amplitude + 1) * cos_w0),
             amplitude * ((amplitude + 1) + (amplitude - 1) * cos_w0 - 2 * sqrt_a * alpha)]
        a = [(amplitude + 1) - (amplitude - 1) * cos_w0 + 2 * sqrt_a * alpha,
             2 * ((amplitude - 1) - (amplitude + 1) * cos_w0),
             (amplitude + 1) - (amplitude - 1) * cos_w0 - 2 * sqrt_a * alpha]
    a0 = a[0]
    return [value / a0 for value in b], [value / a0 for value in a]


def _biquad_response(b, a, freqs, sample_rate):
    z = np.exp(-1j * 2 * np.pi * freqs / sample_rate)
    numerator = b[0] + b[1] * z + b[2] * z * z
    denominator = a[0] + a[1] * z + a[2] * z * z
    return numerator / denominator


def _fft_channel(x, response):
    """Apply a frequency response with zero padding to avoid wrap-around."""
    length = len(x)
    if length == 0:
        return x
    size = 1 << (2 * length - 1).bit_length()
    spectrum = np.fft.rfft(x, size)
    return np.fft.irfft(spectrum * response[:spectrum.size], size)[:length]


def measure_loudness(x, sample_rate):
    """Integrated loudness in LUFS with BS.1770 absolute + relative gating."""
    if x.ndim == 1:
        x = x[:, None]
    weighted = np.empty_like(x, dtype=np.float64)
    for channel in range(x.shape[1]):
        y = x[:, channel]
        for coefficients, _kind in _k_weighting_coefficients(sample_rate):
            b, a = coefficients[:3], coefficients[3:]
            size = 1 << (2 * len(y) - 1).bit_length()
            spectrum = np.fft.rfft(y, size)
            freqs = np.fft.rfftfreq(size, 1 / sample_rate)
            y = np.fft.irfft(spectrum * _biquad_response(b, a, freqs, sample_rate), size)[:len(y)]
        weighted[:, channel] = y
    block = max(1, int(round(0.4 * sample_rate)))
    step = max(1, int(round(0.1 * sample_rate)))
    if len(weighted) < block:
        energy = float(np.mean(np.sum(np.square(weighted), axis=1)))
        if energy < 1e-12:
            return -math.inf
        return -0.691 + 10 * math.log10(max(energy, _EPS))
    starts = range(0, len(weighted) - block + 1, step)
    energies = np.array([float(np.mean(np.sum(np.square(weighted[start:start + block]), axis=1)))
                         for start in starts])
    if float(np.max(energies)) < 1e-12:
        return -math.inf
    loudness = -0.691 + 10 * np.log10(np.maximum(energies, _EPS))
    gated = loudness[loudness >= -70.0]
    if not gated.size:
        return -math.inf
    relative = -0.691 + 10 * math.log10(max(float(np.mean(10 ** ((gated + 0.691) / 10))), _EPS)) - 10.0
    gated = gated[gated >= relative]
    if not gated.size:
        return -math.inf
    return -0.691 + 10 * math.log10(max(float(np.mean(10 ** ((gated + 0.691) / 10))), _EPS))


def _trim_silence(x, sample_rate, threshold_db=-42.0, keep_leading_ms=50.0, keep_trailing_ms=80.0):
    envelope = np.max(np.abs(x), axis=1)
    frame = max(1, int(round(0.02 * sample_rate)))
    hop = max(1, frame // 2)
    if len(envelope) <= frame:
        return x, ['音频过短，未执行静音裁剪。']
    starts = np.arange(0, len(envelope) - frame + 1, hop)
    energy = np.array([float(np.sqrt(np.mean(np.square(envelope[s:s + frame])))) for s in starts])
    # Compare frame RMS against the absolute threshold, never against the peak:
    # a loud click must not make the whole take look non-silent.
    threshold = 10 ** (threshold_db / 20.0)
    active = np.flatnonzero(energy > threshold)
    if not active.size:
        raise ChainError('整段音频都低于静音阈值，去静音会删除全部内容。')
    first = int(starts[active[0]])
    last = int(starts[active[-1]] + frame)
    first = max(0, first - int(round(keep_leading_ms * sample_rate / 1000.0)))
    last = min(len(x), last + int(round(keep_trailing_ms * sample_rate / 1000.0)))
    if last - first < int(0.05 * sample_rate):
        return x, ['有效语音不足 50 毫秒，未执行静音裁剪。']
    note = f'去静音：保留 {first / sample_rate:.2f}s–{last / sample_rate:.2f}s。'
    return x[first:last], [note]


def _de_click(x, sample_rate, threshold=5.5, window=5):
    if window % 2 == 0:
        raise ChainError('de_click.window 必须是奇数。')
    pad = window // 2
    clicked_total = 0
    result = x.copy()
    block = 200_000
    for start in range(0, len(x), block):
        original = x[start:start + block]
        # Edge padding keeps every real sample aligned with one median window.
        chunk = np.pad(original, ((pad, pad), (0, 0)), mode='edge')
        windows = np.lib.stride_tricks.sliding_window_view(chunk, window, axis=0)
        median = np.median(windows, axis=-1)
        deviation = np.abs(original - median)
        sigma = 1.4826 * float(np.median(np.abs(deviation - np.median(deviation))))
        sigma = max(sigma, 1e-6)
        # Reject the ordinary waveform shape: only deviations well above the
        # locally robust spread are treated as clicks.
        mask = deviation > threshold * sigma
        cleaned = original.copy()
        indices = np.flatnonzero(mask.any(axis=1))
        if indices.size:
            cleaned[indices] = median[indices]
            clicked_total += int(indices.size)
        result[start:start + block] = cleaned
    note = f'去点击声：修正约 {clicked_total} 个采样点。' if clicked_total else '去点击声：未发现明显点击。'
    return result, [note]


def _denoise(x, sample_rate, strength=0.35, window_ms=32, noise_seconds=0.35):
    if strength <= 0:
        return x, ['降噪强度为 0，未处理。']
    window = max(16, int(round(sample_rate * window_ms / 1000.0)))
    hop = max(1, window // 2)
    if len(x) < window * 2:
        return x, ['音频过短，未执行降噪。']
    taper = np.hanning(window)
    # A short symmetric pad lets overlap-add reconstruct the first and last
    # speech frames without the boundary attenuation of the previous chain.
    pad = window // 2
    result = np.empty_like(x)
    for channel in range(x.shape[1]):
        signal = x[:, channel]
        padded = np.pad(signal, (pad, pad), mode='reflect')
        frames = np.lib.stride_tricks.sliding_window_view(padded, window)[::hop]
        spectrum = np.fft.rfft(frames * taper, axis=1)
        magnitude = np.abs(spectrum)
        # Estimate the stationary noise from the quietest frames.  The old
        # implementation took the *minimum* of two estimates and then only
        # subtracted a fraction of it; on real recordings this left most of the
        # broadband hiss in place, which normalization subsequently amplified.
        frame_energy = np.sqrt(np.mean(np.square(frames), axis=1))
        frame_count = max(2, min(len(frame_energy),
                                 int(round(noise_seconds * sample_rate / hop))))
        quiet = np.argsort(frame_energy)[:frame_count]
        # Use a robust median PSD from the quietest frames.  Estimating power
        # before taking the median is less sensitive to one accidental breath
        # than subtracting a raw magnitude spectrum.
        noise_power = np.median(np.square(magnitude[quiet]), axis=0)
        signal_power = np.square(magnitude)
        # Oversubtraction is bounded.  A residual floor and temporal smoothing
        # reduce the musical-noise chirps produced by hard spectral gates.
        alpha = 0.8 + 2.2 * float(strength)
        snr = np.maximum(signal_power - alpha * noise_power, 0.0)
        wiener = snr / (signal_power + alpha * noise_power + _EPS)
        floor = 0.10 - 0.05 * float(strength)
        gain = floor + (1.0 - floor) * wiener
        if gain.shape[1] > 4:
            gain_pad = np.pad(gain, ((0, 0), (1, 1)), mode='edge')
            gain = 0.25 * gain_pad[:, :-2] + 0.5 * gain_pad[:, 1:-1] + 0.25 * gain_pad[:, 2:]
        if gain.shape[0] > 2:
            smoothed = gain.copy()
            smoothed[1:-1] = 0.25 * gain[:-2] + 0.5 * gain[1:-1] + 0.25 * gain[2:]
            gain = smoothed
        restored = np.fft.irfft(spectrum * gain, n=window, axis=1) * taper
        summed = np.zeros(len(padded))
        weights = np.zeros(len(padded))
        for index in range(restored.shape[0]):
            begin = index * hop
            summed[begin:begin + window] += restored[index]
            weights[begin:begin + window] += taper * taper
        reconstructed = summed / np.maximum(weights, 1e-8)
        result[:, channel] = reconstructed[pad:pad + len(signal)]
    note = (f'保守频谱降噪：强度 {strength:g}，使用低能量帧中位噪声谱、'
            f'受限 Wiener 增益和时频平滑（估计 {noise_seconds:g}s）。')
    return result, [note]


def _normalize(x, sample_rate, target_lufs=-16.0, ceiling_dbfs=-1.0, max_gain_db=8.0):
    measured = measure_loudness(x, sample_rate)
    if not math.isfinite(measured):
        raise ChainError('音频能量过低，无法测量响度。')
    requested_gain = target_lufs - measured
    gain_db = min(requested_gain, float(max_gain_db))
    gain_limited = gain_db < requested_gain
    peak_after = measure_peak_dbfs(x) + gain_db
    limited = False
    if peak_after > ceiling_dbfs:
        gain_db = ceiling_dbfs - measure_peak_dbfs(x)
        limited = True
    output = x * (10 ** (gain_db / 20.0))
    note = f'响度标准化：{measured:.1f} → {measure_loudness(output, sample_rate):.1f} LUFS。'
    if gain_limited:
        note += f' 输入过低，仅提升 {gain_db:.1f} dB（上限 {max_gain_db:.1f} dB），避免把底噪一起放大。'
    if limited:
        note += ' 受峰值上限限制，未完全达到目标响度。'
    return output, [note]


def _eq(x, sample_rate, bands):
    size = 1 << (2 * len(x) - 1).bit_length()
    freqs = np.fft.rfftfreq(size, 1 / sample_rate)
    response = np.ones(freqs.size, dtype=np.complex128)
    for band in bands:
        b, a = _biquad(band['type'], band['freq'], band['gain_db'], band['q'], sample_rate)
        response *= _biquad_response(b, a, freqs, sample_rate)
    result = np.stack([_fft_channel(x[:, channel], response) for channel in range(x.shape[1])], axis=1)
    labels = '、'.join(f"{band['type']} {band['freq']:g}Hz {band['gain_db']:+g}dB" for band in bands)
    return result, [f'EQ：{labels}。']


def _compress(x, sample_rate, threshold_db=-18.0, ratio=2.0, attack_ms=10.0,
              release_ms=120.0, makeup_db=0.0):
    if ratio <= 1.0 and makeup_db == 0.0:
        return x, ['压缩比 1:1 且无补偿增益，未处理。']
    block = max(1, int(round(0.001 * sample_rate)))
    frames = int(math.ceil(len(x) / block))
    padded = np.zeros((frames * block, x.shape[1]))
    padded[:len(x)] = x
    peaks = np.max(np.abs(padded.reshape(frames, block, x.shape[1])), axis=(1, 2))
    level = 20 * np.log10(np.maximum(peaks, 1e-9))
    reduction = np.where(level > threshold_db, (level - threshold_db) * (1 - 1 / ratio), 0.0)
    reduction -= makeup_db
    block_seconds = block / sample_rate
    attack = math.exp(-block_seconds / max(attack_ms / 1000.0, 1e-6))
    release = math.exp(-block_seconds / max(release_ms / 1000.0, 1e-6))
    smoothed = np.empty_like(reduction)
    envelope = 0.0
    for index, value in enumerate(reduction):
        coefficient = attack if value > envelope else release
        envelope = coefficient * envelope + (1 - coefficient) * value
        smoothed[index] = envelope
    positions = np.arange(len(x)) / block
    gain = 10 ** (-np.interp(positions, np.arange(frames), smoothed) / 20.0)
    result = x * gain[:, None]
    note = f'动态压缩：阈值 {threshold_db:g} dB，{ratio:g}:1，补偿 {makeup_db:+g} dB。'
    return result, [note]


def apply_operations(x, sample_rate, operations):
    """Apply a validated chain left-to-right; returns ``(samples, notes)``."""
    value = np.asarray(x, dtype=np.float64)
    if value.ndim != 2:
        raise ChainError('内部音频采样必须是二维数组（采样 × 通道）。')
    notes = []
    for operation in validate_operations(operations):
        name = operation['name']
        params = {key: item for key, item in operation.items() if key != 'name'}
        if name == 'trim_silence':
            value, extra = _trim_silence(value, sample_rate, **params)
        elif name == 'de_click':
            value, extra = _de_click(value, sample_rate, **params)
        elif name == 'denoise':
            value, extra = _denoise(value, sample_rate, **params)
        elif name == 'normalize':
            value, extra = _normalize(value, sample_rate, **params)
        elif name == 'eq':
            value, extra = _eq(value, sample_rate, operation['bands'])
        elif name == 'compress':
            value, extra = _compress(value, sample_rate, **params)
        else:  # pragma: no cover - validate_operations already rejected it
            raise ChainError(f'未实现的处理步骤：{name}')
        notes.extend(extra)
        if not np.isfinite(value).all():
            raise ChainError(f'处理步骤 {name} 产生了非有限采样。')
    peak = float(np.max(np.abs(value))) if value.size else 0.0
    if peak > 1.0:
        # A derived reference must never clip; scale the whole signal instead of
        # hard-limiting individual samples so the waveform shape is preserved.
        attenuation = 1.0 / peak
        value = value * attenuation
        notes.append(f'输出峰值 {20 * math.log10(peak):.1f} dBFS 超过 0 dBFS，整体衰减了 {-20 * math.log10(attenuation):.1f} dB。')
    return value, notes


def render_chain(source_path, operations, *, dest_dir, output_name, source_name,
                 preset='', output_label=None):
    """Render a derived WAV and return its edit document and measurements.

    ``source_name``/``output_label`` are metadata labels relative to the owning
    character directory; ``output_name`` must be a plain filename used on disk.
    The source file is hashed before and after the render to prove it was not
    modified.
    """
    source = Path(source_path).expanduser().resolve()
    destination_dir = Path(dest_dir).expanduser().resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    name = str(output_name)
    if not name or Path(name).name != name or name in {'.', '..'}:
        raise ChainError('输出文件名必须是不含目录的普通文件名。')
    destination = destination_dir / name
    if destination.exists():
        raise FileExistsError(f'派生音频已存在，拒绝覆盖：{destination}')

    source_hash = _sha256(source)
    samples, sample_rate = read_audio(source)
    processed, notes = apply_operations(samples, sample_rate, operations)
    if not processed.size:
        raise ChainError('处理结果为空，已放弃写入。')

    temporary = destination_dir / f'.render-{uuid.uuid4().hex}.tmp'
    try:
        with sf.SoundFile(str(temporary), mode='x', samplerate=sample_rate,
                          channels=processed.shape[1], format='WAV', subtype='FLOAT') as stream:
            stream.write(processed.astype(np.float32))
        output_hash = _sha256(temporary)
        # Re-read exactly what will be published before renaming it into place.
        check, check_rate = read_audio(temporary)
        if check_rate != sample_rate or check.shape != processed.shape:
            raise ChainError('写入后的派生音频尺寸不一致。')
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)

    if _sha256(source) != source_hash:
        destination.unlink(missing_ok=True)
        raise ChainError('原始音频在处理过程中被修改；已删除派生文件并放弃发布。')
    if _sha256(destination) != output_hash:
        destination.unlink(missing_ok=True)
        raise ChainError('派生音频哈希校验失败；已删除文件并放弃发布。')

    document = make_edit_document(
        source_audio=str(source_name), source_hash=source_hash,
        operations=operations, output_audio=str(output_label or output_name),
        output_hash=output_hash, preset=preset, sample_rate=sample_rate,
        channels=processed.shape[1],
        duration_in=len(samples) / sample_rate, duration_out=len(processed) / sample_rate,
        notes=notes)
    return {
        'document': document,
        'output_path': str(destination),
        'output_name': name,
        'output_hash': output_hash,
        'source_hash': source_hash,
        'sample_rate': sample_rate,
        'channels': processed.shape[1],
        'duration': len(processed) / sample_rate,
        'metrics': {
            'lufs_in': measure_loudness(samples, sample_rate),
            'lufs_out': measure_loudness(processed, sample_rate),
            'peak_dbfs_in': measure_peak_dbfs(samples),
            'peak_dbfs_out': measure_peak_dbfs(processed),
            'clipped_samples_out': count_clipped(processed),
        },
        'notes': notes,
        'summary': '；'.join(describe_operations(operations)),
    }

