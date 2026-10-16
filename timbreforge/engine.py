"""TimbreForge —— numpy 合成引擎（P0-7）

离线渲染优先，保证：确定性可复现、CPU 上足够快、无第三方依赖（只要 numpy）。

信号链：
    振荡器组(unison 失谐 + FM) + 噪声
      -> 时变状态变量低通滤波(SVF)
      -> 幅度包络(指数 ADSR) x 滤波包络
      -> 软饱和(温暖)
      -> 合唱 -> 延迟 -> 混响
      -> 峰值归一化 + 限幅
"""
from __future__ import annotations

import numpy as np

from .patch import Patch, clamp

SR = 48000.0


# ---------------------------------------------------------------- 工具

def midi_to_hz(note: float) -> float:
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def env_exp(n: int, attack: float, decay: float, sustain: float,
            release: float, sr: float) -> np.ndarray:
    """指数 ADSR，长度恰好 n。"""
    a = max(int(attack * sr), 1)
    d = max(int(decay * sr), 1)
    r = max(int(release * sr), 1)
    s = clamp(sustain)
    body = max(n - r, 1)
    if a + d > body:                       # 空间不够，按比例压缩
        scale = body / (a + d)
        a = max(int(a * scale), 1)
        d = max(int(d * scale), 1)
    sus = max(body - a - d, 0)

    # 指数曲线：用 1-exp 得到自然的斜率
    ta = np.linspace(0.0, 1.0, a, endpoint=False)
    e_att = 1.0 - np.exp(-4.0 * ta)
    e_att /= e_att[-1] if e_att[-1] > 1e-9 else 1.0
    td = np.linspace(0.0, 1.0, d, endpoint=False)
    e_dec = s + (1.0 - s) * np.exp(-4.0 * td)
    e_sus = np.full(sus, s)
    e_rel = s * np.exp(-4.0 * np.linspace(0.0, 1.0, r))
    env = np.concatenate([e_att, e_dec, e_sus, e_rel])
    if env.size < n:
        env = np.pad(env, (0, n - env.size))
    env = env[:n]
    # 强制首尾归零，避免爆音
    k = min(32, env.size)
    env[:k] *= np.linspace(0.0, 1.0, k)
    env[-k:] *= np.linspace(1.0, 0.0, k)
    return env


def _saw(phase: np.ndarray) -> np.ndarray:
    return 2.0 * (phase - np.floor(phase + 0.5))


def _square(phase: np.ndarray) -> np.ndarray:
    return np.where((phase % 1.0) < 0.5, 1.0, -1.0)


def _triangle(phase: np.ndarray) -> np.ndarray:
    p = phase % 1.0
    return 4.0 * np.abs(p - 0.5) - 1.0


def polyblep(phase: np.ndarray, dt: float) -> np.ndarray:
    """PolyBLEP 修正，降低锯齿/方波的混叠。"""
    p = phase % 1.0
    out = np.zeros_like(p)
    m = p < dt
    out[m] = (p[m] / dt) ** 2 - 2 * (p[m] / dt) + 1
    m2 = p > 1 - dt
    t = (p[m2] - 1) / dt
    out[m2] = t ** 2 + 2 * t + 1
    return out


def oscillator(waveform: str, freq: float, n: int, sr: float,
               phase0: float = 0.0, fm_signal: np.ndarray | None = None) -> np.ndarray:
    """生成一个振荡器（支持 FM）。"""
    t = np.arange(n, dtype=np.float64)
    if fm_signal is not None:
        # 相位调制
        phase = phase0 + np.cumsum((freq + fm_signal * freq) / sr)
    else:
        phase = phase0 + freq * t / sr
    dt = freq / sr
    if waveform == "sine":
        return np.sin(2 * np.pi * phase)
    if waveform == "saw":
        return _saw(phase) - polyblep(phase, dt)
    if waveform == "square":
        return _square(phase) + polyblep(phase, dt) - polyblep(phase + 0.5, dt)
    if waveform == "triangle":
        return _triangle(phase)
    if waveform == "fm":
        mod = np.sin(2 * np.pi * phase * 2.0)
        return np.sin(2 * np.pi * phase + mod * 1.0)
    return np.sin(2 * np.pi * phase)


def svf_lowpass(x: np.ndarray, cutoff: np.ndarray | float, q: float,
                sr: float) -> np.ndarray:
    """时变状态变量低通（TPT/ZDF 拓扑，**无条件稳定**，适合时变截止频率）。

    这里刻意不用 Chamberlin 形式：后者在 f 接近 1 时会发散，
    而我们的截止频率会被滤波包络放大到接近 Nyquist，实测会溢出。
    """
    n = x.size
    if np.isscalar(cutoff):
        fc = np.full(n, float(cutoff))
    else:
        fc = np.asarray(cutoff, dtype=np.float64)
        if fc.size != n:
            fc = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, fc.size), fc)
        else:
            fc = fc.copy()
    # g = tan(pi*fc/sr)，钳到 (0, 0.999) 保证稳定
    fc = np.clip(fc, 20.0, sr * 0.45)
    g = np.tan(np.pi * fc / sr)
    g = np.clip(g, 1e-6, 0.999)
    R = 1.0 / (2.0 * max(q, 0.35))          # 阻尼，越大越不共振
    y = np.empty(n)
    ic1 = ic2 = 0.0
    for i in range(n):
        gi = g[i]
        denom = 1.0 + gi * (gi + R)
        v3 = x[i] - ic2
        v1 = (gi * v3 + ic1) / denom
        v2 = ic2 + gi * v1
        ic1 = 2.0 * v1 - ic1
        ic2 = 2.0 * v2 - ic2
        y[i] = v2
        if not np.isfinite(y[i]):           # 兜底：任何数值意外都不让音频炸掉
            y[i] = 0.0
            ic1 = ic2 = 0.0
    return y


def svf_bandpass(x: np.ndarray, center: float, bandwidth: float,
                 sr: float) -> np.ndarray:
    """定中心频率的带通（TPT/ZDF，用带通输出）。

    人声共振峰就用它：中心频率 = 共振峰频率，带宽 = 共振峰带宽。

    ⚠️ 这里踩过一个关键坑：TPT/ZDF 的阻尼 R 必须由**品质因数**换算，
        标准关系是 Q = fc / bw，R = 1/(2Q)。
        早期误写成 R = bw/fc，对 730Hz/90Hz 得到的是极宽的滤波器
        （正确的 Q 约 8，错的写法相当于 Q 约 0.12），
        导致谱上根本没有共振峰结构、LPC 也估不出来。
    """
    n = x.size
    fc = float(np.clip(center, 20.0, sr * 0.45))
    bw = float(max(bandwidth, 20.0))
    g = float(np.clip(np.tan(np.pi * fc / sr), 1e-6, 0.999))
    q = float(np.clip(fc / bw, 0.5, 60.0))       # 品质因数
    R = 1.0 / (2.0 * q)                          # 阻尼
    y = np.empty(n)
    ic1 = ic2 = 0.0
    denom = 1.0 + g * (g + R)
    for i in range(n):
        v3 = x[i] - ic2
        v1 = (g * v3 + ic1) / denom
        v2 = ic2 + g * v1
        ic1 = 2.0 * v1 - ic1
        ic2 = 2.0 * v2 - ic2
        y[i] = v1                          # 带通输出
        if not np.isfinite(y[i]):
            y[i] = 0.0
            ic1 = ic2 = 0.0
    return y


def reverb_simple(x: np.ndarray, mix: float, sr: float, seed: int = 0) -> np.ndarray:
    """Schroeder 式混响：4 梳状 + 2 全通，够用且便宜。"""
    if mix <= 1e-4:
        return x
    combs = [(int(0.0297 * sr), 0.805), (int(0.0371 * sr), 0.793),
             (int(0.0411 * sr), 0.786), (int(0.0437 * sr), 0.777)]
    allpass = [(int(0.005 * sr), 0.7), (int(0.0017 * sr), 0.7)]
    acc = np.zeros_like(x)
    for delay, g in combs:
        buf = np.zeros(delay)
        idx = 0
        out = np.empty_like(x)
        for i in range(x.size):
            y = buf[idx]
            out[i] = y
            buf[idx] = x[i] + y * g
            idx += 1
            if idx == delay:
                idx = 0
        acc += out
    acc /= len(combs)
    for delay, g in allpass:
        buf = np.zeros(delay)
        idx = 0
        out = np.empty_like(acc)
        for i in range(acc.size):
            bufout = buf[idx]
            y = -g * acc[i] + bufout
            buf[idx] = acc[i] + g * y
            out[i] = y
            idx += 1
            if idx == delay:
                idx = 0
        acc = out
    return (1.0 - mix) * x + mix * acc


def chorus(x: np.ndarray, mix: float, sr: float, rate: float = 0.6) -> np.ndarray:
    """两路调制延迟，做出宽度与厚度。"""
    if mix <= 1e-4:
        return x
    n = x.size
    t = np.arange(n) / sr
    out = np.zeros_like(x)
    for k, (delay_ms, depth_ms, phase) in enumerate([(18.0, 4.0, 0.0), (26.0, 6.0, 0.5)]):
        d = (delay_ms + depth_ms * np.sin(2 * np.pi * rate * t + phase * np.pi)) * sr / 1000.0
        idx = np.arange(n) - d
        i0 = np.clip(np.floor(idx).astype(np.int64), 0, n - 1)
        i1 = np.clip(i0 + 1, 0, n - 1)
        frac = np.clip(idx - i0, 0.0, 1.0)
        out += x[i0] * (1 - frac) + x[i1] * frac
    out /= 2.0
    return (1.0 - mix) * x + mix * out


def delay_simple(x: np.ndarray, mix: float, sr: float,
                 time_s: float = 0.25, feedback: float = 0.35) -> np.ndarray:
    if mix <= 1e-4:
        return x
    d = max(int(time_s * sr), 1)
    out = x.copy()
    for i in range(d, x.size):
        out[i] += out[i - d] * feedback
    return (1.0 - mix) * x + mix * out


def apply_effects(body: np.ndarray, patch, sr: float) -> np.ndarray:
    """共享效果链（音色与人声两条路径都用）。"""
    body = chorus(body, patch.chorus_mix, sr)
    body = delay_simple(body, patch.delay_mix, sr,
                        time_s=60.0 / 90.0 * 0.5, feedback=0.32)
    return reverb_simple(body, patch.reverb_mix, sr, seed=patch.seed)


def finalize(body: np.ndarray, patch) -> np.ndarray:
    """共享收尾：去 NaN -> 归一化 -> 软限幅。"""
    body = np.nan_to_num(body, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(body)))
    if peak > 1e-9:
        body = body / peak * 0.9 * clamp(patch.gain / 0.75, 0.2, 1.2)
    body = np.tanh(body * 1.05) * 0.95
    return body.astype(np.float32)


# ---------------------------------------------------------------- 主渲染

def render(patch: Patch, sr: float = SR, duration: float | None = None) -> np.ndarray:
    """把一个 Patch 渲染成音频（float32，单声道，峰值约 0.9）。"""
    dur = duration if duration is not None else (patch.duration or 2.0)
    # 固定尾巴长度：缓冲长度**不能**依赖 seed 抖动出来的 release，
    # 否则同 seed 复现以外的场景会出现长度不一致（P0 验收踩过这个坑）
    tail = 0.30
    n = int((dur + tail) * sr)
    if n < 64:
        n = 64

    # ---- 振荡器组 ----
    voices = max(int(patch.unison_voices), 1)
    osc = np.zeros(n)
    base = patch.frequency
    rng = np.random.default_rng(int(patch.seed) & 0xFFFFFFFF)
    for v in range(voices):
        if voices == 1:
            detune = 0.0
        else:
            spread = (v / (voices - 1)) * 2.0 - 1.0
            detune = spread * patch.unison_detune
        f = base * (1.0 + detune)
        phase0 = float(rng.random()) if patch.unison_voices > 1 else 0.0

        if patch.fm_index > 1e-4:
            # FM：调制器为整数比（inver-synth 经验，保证音乐性）
            t = np.arange(n) / sr
            mod = np.sin(2 * np.pi * f * patch.fm_ratio * t)
            fm_sig = mod * patch.fm_index / max(patch.fm_ratio, 1.0)
            s = oscillator(patch.waveform, f, n, sr, phase0, fm_signal=fm_sig)
        else:
            s = oscillator(patch.waveform, f, n, sr, phase0)
        osc += s
    osc /= voices

    if patch.noise_level > 1e-4:
        osc = osc * (1.0 - patch.noise_level) + rng.standard_normal(n) * patch.noise_level * 0.4

    # ---- 滤波（包络 + LFO 驱动截止频率）----
    amp_env = env_exp(n, patch.amp_attack, patch.amp_decay,
                      patch.amp_sustain, patch.amp_release, sr)
    t = np.arange(n) / sr
    # 滤波包络：用快起慢落的另一条包络
    fenv = env_exp(n, max(patch.amp_attack, 0.001), max(patch.amp_decay * 0.8, 0.02),
                   clamp(patch.amp_sustain * 0.6), patch.amp_release * 0.9, sr)
    cutoff = patch.filter_cutoff * (1.0 + patch.filter_env_depth * fenv)
    if patch.lfo_depth > 1e-4:
        lfo = np.sin(2 * np.pi * patch.lfo_rate * t)
        cutoff = cutoff * np.exp(lfo * patch.lfo_depth * 0.8)
    cutoff = np.clip(cutoff, 25.0, sr * 0.45)
    body = svf_lowpass(osc, cutoff, patch.filter_resonance, sr)

    # ---- 驱动 + 幅度包络 ----
    if patch.filter_drive > 1e-4:
        body = np.tanh(body * (1.0 + patch.filter_drive * 3.0))
    body *= amp_env

    # ---- 温暖：磁带式软饱和 ----
    if patch.saturation > 1e-4:
        k = 1.0 + patch.saturation * 4.0
        body = np.tanh(body * k) / np.tanh(k)

    # ---- 效果链 ----
    body = apply_effects(body, patch, sr)

    # ---- 归一化 + 限幅 ----
    return finalize(body, patch)


def render_voice_patch(patch, sr: float = SR) -> np.ndarray:
    """人声（元音/共振峰）渲染：把 Patch 转成声门源 + 共振峰参数。

    与 render() 的区别：这里不做振荡器/滤波包络，而是
    「声门脉冲 + 3 个共振峰带通 + 气声 + 辅音」。
    """
    from .voice import render_voice, GENDER

    if getattr(patch, "formant_scale", 0.0) <= 0.0:
        patch.formant_scale = GENDER.get(patch.gender, GENDER["female"])["scale"]
    dur = patch.duration or 1.6
    body = render_voice(
        f0=patch.frequency or 220.0,
        vowel=patch.vowel,
        duration=dur,
        gender=patch.gender,
        breathiness=patch.breathiness,
        vibrato_depth=patch.vibrato_depth,
        vibrato_hz=patch.vibrato_rate,
        attack=patch.amp_attack,
        release=patch.amp_release,
        consonant=patch.consonant,
        brightness=getattr(patch, "_voice_brightness", 0.5),
        nasal=patch.nasal,
        seed=patch.seed,
        sr=sr,
        formant_scale=patch.formant_scale,
    ).astype(np.float64)
    body = apply_effects(body, patch, sr)
    return finalize(body, patch)


def render_patch(patch, sr: float = SR, duration: float | None = None) -> np.ndarray:
    """统一入口：按 patch.mode 分派到音色或人声引擎。"""
    if getattr(patch, "mode", "synth") == "voice":
        return render_voice_patch(patch, sr=sr)
    return render(patch, sr=sr, duration=duration)


def render_preview(patch: Patch, sr: float = SR, seconds: float = 2.5) -> np.ndarray:
    """短预览，用于快速试听。"""
    return render(patch, sr=sr, duration=seconds)
