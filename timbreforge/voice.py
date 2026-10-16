"""TimbreForge —— 人声（元音/共振峰）合成引擎

与抽象"音色"的区别：
  * 抽象音色 = 振荡器 + 滤波 + 包络
  * 人声       = **声门脉冲源**（带颤音/抖动）+ **3 个共振峰带通**（决定元音）
                + 气声噪声 + 辅音瞬态

只要共振峰对，人耳就会听成对应的元音 —— 这是"人声"能被认出来的关键。
"""
from __future__ import annotations

import math

import numpy as np

from .engine import env_exp, svf_bandpass, SR

# ---------------------------------------------------------------- 元音共振峰表
# (F1, F2, F3) Hz + 带宽 (B1, B2, B3) Hz —— 典型成年男性声道值
VOWEL_FORMANTS: dict[str, tuple[tuple[float, float, float],
                                tuple[float, float, float]]] = {
    "a": ((730, 1090, 2440), (90, 110, 140)),   # 啊 —— 最开阔
    "e": ((530, 1840, 2480), (80, 100, 140)),   # 诶 —— 偏亮
    "i": ((270, 2290, 3010), (70, 100, 160)),   # 衣 —— 最亮最窄
    "o": ((570, 840, 2410), (90, 100, 140)),    # 噢 —— 圆润
    "u": ((300, 870, 2240), (80, 90, 140)),     # 乌 —— 最暗最厚
    "v": ((440, 1020, 2240), (80, 100, 140)),   # 过渡/中性（ə）
    "n": ((280, 1100, 2300), (100, 120, 160)),  # 鼻音感（m/n）
}

# 中文音色词 -> (元音, 说明)
VOICE_CHARACTER = {
    "温暖": "u", "厚实": "u", "厚": "o", "饱满": "o", "深沉": "u",
    "明亮": "i", "尖锐": "i", "清亮": "i", "通透": "e",
    "空灵": "v", "飘渺": "v", "梦幻": "v", "柔和": "o",
}

# 性别 -> 基频范围与共振峰缩放
GENDER = {
    "female": {"f0": 220.0, "scale": 1.17},
    "male":   {"f0": 120.0, "scale": 0.89},
    "child":  {"f0": 300.0, "scale": 1.35},
}


def _glottal_source(n: int, f0: float, sr: float, seed: int,
                    jitter: float = 0.004, vibrato_hz: float = 5.2,
                    vibrato_depth: float = 0.012) -> np.ndarray:
    """声门脉冲源：谐波堆叠 + 颤音 + 微抖动。

    用谐波相加而不是单脉冲串，天然抗混叠。
    关键：普通 1/k^1.2 的谐波堆叠到 100 次谐波仍然很亮，
    会让"气声"完全体现不出来（实测加气声反而使高频占比下降），
    所以这里给声门谱加上 -12dB/oct 的谱倾斜，让高次谐波自然衰减。
    """
    t = np.arange(n) / sr
    rng = np.random.default_rng(seed & 0xFFFFFFFF)
    # 颤音（延迟起振，更自然）
    onset = np.clip(t / 0.35, 0.0, 1.0)
    freq = f0 * (1.0 + vibrato_depth * onset * np.sin(2 * np.pi * vibrato_hz * t))
    # 缓慢的随机音高漂移（jitter）
    if jitter > 0:
        freq = freq * (1.0 + jitter * np.sin(2 * np.pi * 0.7 * t + rng.uniform(0, 6.28)))
    # 相位积分
    phase = np.cumsum(freq) / sr
    # 谐波堆叠：每倍频程 -12dB（声门谱典型值）
    out = np.zeros(n)
    k = 1
    while f0 * k < sr * 0.45 and k <= 120:
        amp = 1.0 / (k ** 2.0)          # -12 dB/oct
        out += amp * np.sin(2 * np.pi * k * phase + rng.uniform(0, 6.28))
        k += 1
    return out / (np.max(np.abs(out)) + 1e-9)


def _apply_formants(src: np.ndarray, formants, bandwidths,
                    scale: float, sr: float, gains=(1.0, 0.7, 0.4)) -> np.ndarray:
    """三个并联带通叠加成元音。"""
    y = np.zeros_like(src)
    for (fc, bw, g) in zip(formants, bandwidths, gains):
        y += g * svf_bandpass(src, fc * scale, bw * scale, sr)
    return y


def _nasal(src: np.ndarray, sr: float, amount: float) -> np.ndarray:
    """鼻音化：加一个 250Hz 附近的低共振 + 整体低通一点。"""
    if amount <= 1e-4:
        return src
    low = svf_bandpass(src, 280.0, 120.0, sr)
    return src + amount * 0.9 * low


def render_voice(f0: float = 220.0, vowel: str = "a", duration: float = 1.6,
                 gender: str = "female", breathiness: float = 0.2,
                 vibrato_depth: float = 0.012, vibrato_hz: float = 5.2,
                 attack: float = 0.06, release: float = 0.25,
                 consonant: str = "none", brightness: float = 0.5,
                 nasal: float = 0.0, seed: int = 0, sr: float = SR,
                 formant_scale: float | None = None) -> np.ndarray:
    """合成一个元音（可带辅音起音）。

    consonant: none | s（嘶声）| t（爆破）| m（鼻音起音）
    """
    g = GENDER.get(gender, GENDER["female"])
    scale = formant_scale if formant_scale is not None else g["scale"]
    fmts, bws = VOWEL_FORMANTS.get(vowel, VOWEL_FORMANTS["a"])

    n = int((duration + release + 0.25) * sr)
    src = _glottal_source(n, f0, sr, seed, vibrato_hz=vibrato_hz,
                          vibrato_depth=vibrato_depth)

    # 气声：把一部分声门源换成噪声。
    # 注意：两部分必须**分别归一化**再混合，否则整体归一化会把噪声压掉，
    # 导致"加气声反而更不噪"（实测踩过这个坑）。
    if breathiness > 1e-4:
        rng = np.random.default_rng((seed + 977) & 0xFFFFFFFF)
        noise = rng.standard_normal(n)
        # 噪声也要经过共振峰，才有"气声说这个元音"的感觉
        noise = _apply_formants(noise, fmts, bws, scale, sr,
                                gains=(0.8, 0.6, 0.5))
        noise = noise / (np.max(np.abs(noise)) + 1e-9)
        src = src / (np.max(np.abs(src)) + 1e-9)
        src = (1.0 - breathiness) * src + breathiness * noise * 1.6
        src = src / (np.max(np.abs(src)) + 1e-9)

    body = _apply_formants(src, fmts, bws, scale, sr)
    body = _nasal(body, sr, nasal)

    # 亮度：轻度高频塑形（用带通补一点 3~4kHz 的"临场感"）
    if brightness > 0.5:
        pres = svf_bandpass(body, 3500.0, 1800.0, sr)
        body = body + (brightness - 0.5) * 1.2 * pres
    elif brightness < 0.5:
        # 变暗：减一点高频
        high = svf_bandpass(body, 4000.0, 2500.0, sr)
        body = body - (0.5 - brightness) * 0.9 * high

    # 包络
    env = env_exp(n, attack, max(duration * 0.25, 0.05), 0.85, release, sr)
    body *= env

    # 辅音起音
    body = _add_consonant(body, consonant, sr, seed)

    # 归一化
    body = np.nan_to_num(body)
    peak = float(np.max(np.abs(body)))
    if peak > 1e-9:
        body = body / peak * 0.9
    return (np.tanh(body * 1.02) * 0.95).astype(np.float32)


def _add_consonant(x: np.ndarray, kind: str, sr: float, seed: int) -> np.ndarray:
    """在起音处叠加辅音瞬态。"""
    if kind in ("none", "", None):
        return x
    rng = np.random.default_rng((seed + 31) & 0xFFFFFFFF)
    n = x.size
    y = x.copy()
    if kind == "s":            # 嘶声：高频噪声 120ms
        dur = int(0.12 * sr)
        noise = rng.standard_normal(dur)
        noise = svf_bandpass(noise, 6500.0, 3000.0, sr)
        env = np.exp(-np.linspace(0, 3, dur))
        y[:dur] += noise * env * 0.35
    elif kind == "t":          # 爆破：极短宽带脉冲 15ms
        dur = int(0.015 * sr)
        y[:dur] += rng.standard_normal(dur) * 0.5
    elif kind == "m":          # 鼻音起音：低频哼鸣 150ms
        dur = int(0.15 * sr)
        t = np.arange(dur) / sr
        hum = np.sin(2 * np.pi * 110 * t) * 0.4
        hum = svf_bandpass(hum, 280.0, 150.0, sr)
        y[:dur] += hum * np.exp(-np.linspace(0, 2, dur))
    return y


def profile_hint(vowel: str) -> str:
    """元音听感提示，给旁白用。"""
    hints = {
        "a": "开阔的『啊』", "e": "偏亮的『诶』", "i": "明亮的『衣』",
        "o": "圆润的『噢』", "u": "低沉厚实的『乌』", "v": "中性过渡元音",
        "n": "带鼻音感",
    }
    return hints.get(vowel, vowel)
