"""TimbreForge —— 参数模式层（P0-5）

设计要点：
  * 双层参数：6 个「宏 macro」（给自然语言用）+ 底层合成参数（给引擎用）
  * 宏 → 底层 的展开是**确定性闭式映射**，保证单调、可控、可复现
  * 随机性全部由 seed 驱动：同 seed 同描述 → 逐样本一致的音频
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

# ---------------------------------------------------------------- 工具

def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def exp_map(t: float, lo: float, hi: float) -> float:
    """把 0..1 指数映射到 lo..hi（适合频率、时间这类量）。"""
    t = clamp(t)
    return lo * ((hi / lo) ** t)


# ---------------------------------------------------------------- 宏定义

MACRO_NAMES = (
    "brightness",        # 亮度：滤波截止 + 滤波包络深度 + 波形谐波量
    "warmth",            # 温暖：饱和染色 + 失谐 + 高频衰减
    "movement",          # 流动：LFO 深度/速率 + 滤波缓慢漂移
    "attack_sharpness",  # 起音锐度：包络 attack/decay
    "space",             # 空间：混响/合唱/延迟干湿
    "grit",              # 粗糙：FM 指数 + 噪声 + 波形折返感
)

DEFAULT_MACROS = {
    "brightness": 0.45,
    "warmth": 0.50,
    "movement": 0.25,
    "attack_sharpness": 0.60,
    "space": 0.30,
    "grit": 0.20,
}

# 乐器族先验：决定"这是贝斯/铺底/拨弦/主音"
FAMILIES = {
    "bass":   {"octave": -2, "brightness": 0.35, "attack_sharpness": 0.70, "warmth": 0.65,
               "space": 0.15, "sustain": 0.75, "waveform": "saw", "duration": 1.6},
    "pad":    {"octave": 0,  "brightness": 0.45, "attack_sharpness": 0.15, "warmth": 0.55,
               "space": 0.65, "sustain": 0.85, "waveform": "saw", "duration": 3.0},
    "pluck":  {"octave": 0,  "brightness": 0.65, "attack_sharpness": 0.95, "warmth": 0.35,
               "space": 0.30, "sustain": 0.00, "waveform": "triangle", "duration": 1.2},
    "lead":   {"octave": 0,  "brightness": 0.65, "attack_sharpness": 0.80, "warmth": 0.45,
               "space": 0.25, "sustain": 0.65, "waveform": "saw", "duration": 1.5},
    "keys":   {"octave": -1, "brightness": 0.50, "attack_sharpness": 0.60, "warmth": 0.60,
               "space": 0.35, "sustain": 0.45, "waveform": "triangle", "duration": 2.0},
    "bell":   {"octave": 1,  "brightness": 0.80, "attack_sharpness": 0.90, "warmth": 0.20,
               "space": 0.45, "sustain": 0.00, "waveform": "fm", "duration": 2.0},
}


@dataclass
class Patch:
    """一个完整音色：宏 + 展开后的底层参数。"""
    name: str = "untitled"
    family: str = "keys"
    seed: int = 0
    root_note: int = 48          # MIDI 音符号（C3）
    duration: float = 2.0
    macros: dict = field(default_factory=lambda: dict(DEFAULT_MACROS))

    # ---- 展开后的底层参数（由 expand() 填充）----
    frequency: float = 0.0
    waveform: str = "saw"
    unison_voices: int = 1
    unison_detune: float = 0.0
    fm_ratio: float = 2.0
    fm_index: float = 0.0
    noise_level: float = 0.0

    filter_cutoff: float = 2000.0
    filter_resonance: float = 0.7
    filter_drive: float = 0.0
    filter_env_depth: float = 0.0

    amp_attack: float = 0.01
    amp_decay: float = 0.2
    amp_sustain: float = 0.7
    amp_release: float = 0.2

    lfo_rate: float = 5.0
    lfo_depth: float = 0.0
    lfo_target: str = "filter_cutoff"

    chorus_mix: float = 0.0
    reverb_mix: float = 0.0
    delay_mix: float = 0.0
    saturation: float = 0.0
    gain: float = 0.8
    source_words: list = field(default_factory=list)

    # ---- 人声（元音/共振峰）模式专用 ----
    mode: str = "synth"          # "synth"（抽象音色）| "voice"（人声元音）
    vowel: str = "a"             # a/e/i/o/u/v(中性)/n(鼻音)
    gender: str = "female"       # female / male / child
    breathiness: float = 0.18    # 气声比例
    vibrato_depth: float = 0.012
    vibrato_rate: float = 5.2
    consonant: str = "none"      # none / s / t / m
    nasal: float = 0.0
    formant_scale: float = 0.0   # 0 表示按性别自动

    # ---------------------------------------------------------------- 展开
    def expand(self) -> "Patch":
        """把 6 个宏确定性地展开成底层参数（保持单调性）。"""
        m = {k: clamp(v) for k, v in self.macros.items()}
        fam = FAMILIES.get(self.family, FAMILIES["keys"])

        # 频率
        self.frequency = 440.0 * (2.0 ** ((self.root_note + 12 * fam["octave"] - 69) / 12.0))

        # 亮度 -> 截止频率（指数映射，保证"亮度升则质心升"）
        self.filter_cutoff = exp_map(m["brightness"], 180.0, 12000.0)
        # 共振：亮度偏高时略降，避免刺耳
        self.filter_resonance = lerp(1.4, 0.6, m["brightness"])

        # 温暖 -> 饱和 + 失谐 + 轻微高频衰减
        self.saturation = lerp(0.0, 0.85, m["warmth"])
        self.unison_detune = lerp(0.0, 0.02, m["warmth"]) * (1.0 - 0.5 * m["grit"])
        self.unison_voices = 1 + int(round(lerp(0.0, 4.0, m["warmth"])))

        # 粗糙 -> FM 指数 + 噪声 + 滤波驱动
        self.fm_index = lerp(0.0, 8.0, m["grit"] ** 1.5)
        self.noise_level = lerp(0.0, 0.35, m["grit"] ** 2.0)
        self.filter_drive = lerp(0.0, 0.7, m["grit"] * 0.6 + m["warmth"] * 0.4)
        # inver-synth 经验：整数比保证各参数点都有谐波边带，避免出噪声
        self.fm_ratio = 2.0 if m["grit"] < 0.55 else 3.0

        # 起音锐度 -> ADSR
        self.amp_attack = exp_map(1.0 - m["attack_sharpness"], 0.001, 1.2)
        self.amp_decay = exp_map(1.0 - m["attack_sharpness"], 0.03, 1.0)
        self.amp_sustain = clamp(fam["sustain"] * lerp(1.15, 0.7, m["attack_sharpness"]))
        self.amp_release = exp_map(0.15 + 0.85 * m["space"], 0.05, 2.0)
        self.filter_env_depth = lerp(0.0, 1.4, m["brightness"] * (0.4 + 0.6 * m["attack_sharpness"]))

        # 流动 -> LFO
        self.lfo_rate = exp_map(1.0 - m["movement"], 0.15, 9.0)
        self.lfo_depth = lerp(0.0, 0.45, m["movement"])

        # 空间 -> 效果
        self.chorus_mix = lerp(0.0, 0.45, m["space"] * 0.8)
        self.reverb_mix = lerp(0.0, 0.55, m["space"])
        self.delay_mix = lerp(0.0, 0.3, max(0.0, m["space"] - 0.5) * 2.0)

        # 波形与时长来自乐器族
        self.waveform = fam["waveform"]
        if self.duration <= 0:
            # 量化到 0.1s：保证渲染长度只由乐器族决定，
            # 不受 seed 抖动影响（否则同 seed 也可能长度不同）
            self.duration = round(fam["duration"] * 10) / 10.0

        # 增益：多声部/高共鸣时补偿
        comp = 1.0 / (1.0 + 0.25 * (self.unison_voices - 1))
        self.gain = clamp(0.75 * comp, 0.15, 0.95)
        return self

    def apply_voice_macros(self) -> "Patch":
        """人声模式下，把 6 个宏映射到人声参数（共振峰/气声/颤音）。"""
        m = {k: clamp(v) for k, v in self.macros.items()}
        # 亮度 -> 高频临场感 + 轻微共振峰上移
        self._voice_brightness = m["brightness"]
        # 温暖 -> 共振峰整体下移（更厚）
        self.formant_scale = lerp(1.12, 0.92, m["warmth"])
        # 粗糙 -> 气声 + 基频抖动
        self.breathiness = clamp(0.06 + 0.62 * m["grit"])
        # 流动 -> 颤音深度
        self.vibrato_depth = lerp(0.002, 0.045, m["movement"])
        # 空间 -> 混响（复用宏）
        self.reverb_mix = lerp(0.0, 0.45, m["space"])
        self.chorus_mix = lerp(0.0, 0.25, m["space"] * 0.6)
        self.delay_mix = 0.0
        self.duration = self.duration or 1.6
        return self

    # ---------------------------------------------------------------- 复现
    def signature(self) -> str:
        """用于去重/收藏的短签名。"""
        return (f"{self.family}|{self.root_note}|"
                + ",".join(f"{k}={self.macros[k]:.3f}" for k in MACRO_NAMES)
                + f"|seed={self.seed}")

    def describe(self) -> str:
        """人类可读的宏画像（一行）。"""
        bars = " ".join(f"{k[:4]}={self.macros[k]:.2f}" for k in MACRO_NAMES)
        return f"[{self.family}] {self.name}  {bars}"


def apply_deltas(patch: Patch, deltas: dict, strength: float = 1.0) -> Patch:
    """把词库给出的增量叠加到宏上（带强度缩放，自动裁剪到 0..1）。"""
    for k, v in deltas.items():
        if k in patch.macros:
            patch.macros[k] = clamp(patch.macros[k] + v * strength)
    return patch


def randomize(patch: Patch, amount: float = 0.12, rng=None) -> Patch:
    """按 seed 在宏空间里做小幅扰动，产生『同族不同味』的音色。"""
    import random as _random
    rng = rng or _random.Random(patch.seed)
    for k in MACRO_NAMES:
        patch.macros[k] = clamp(patch.macros[k] + rng.uniform(-amount, amount))
    return patch


def to_dict(patch: Patch) -> dict:
    return asdict(patch)
