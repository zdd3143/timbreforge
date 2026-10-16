"""TimbreForge —— 声线库（VoiceBank）

数据来源：HuggingFace 数据集 `TTS-AGI/moss-character-reference-voices`（CC-BY-4.0）
  * **1336 个合成角色声线**，每个带一条 MP3 参考音频
  * 每条都由模型自动标注：性别、年龄、音区(register)、
    音色六轴(metallic/throat_guttural/falsetto/chest_voice/roughness/brightness 0~5)、
    独特特征、情绪范围、选角建议、标签、音质评分

核心思路（取代原先"用音频变换硬造声线"的做法）：
    中文描述 -> 声线维度 -> 在**真实存在的 1336 个角色声线**里检索最匹配的一条
这样声线来自数据集，而不是被我们改造出来的。

许可提醒：数据集为 CC-BY-4.0，且标注为"研究用途、全合成音、
无真人声音"；商用前请自行确认。
"""
from __future__ import annotations

import json
import os
import random
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DATA_DIR = Path("data_ref/moss_char")
AUDIO_CACHE = Path("data_ref/moss_char/audio")
WAV_CACHE = Path("data_ref/moss_char/wav")
HF_BASE = "https://huggingface.co/datasets/TTS-AGI/moss-character-reference-voices/resolve/main"

# ---------------------------------------------------------------- 维度锚点
# 把英文标签/音区关键词映射到「声线维度」。这是中文检索的桥梁。
TAG_ANCHORS: dict[str, dict[str, float]] = {
    # 亮度
    "bright": {"bright": +1.0}, "clear": {"bright": +0.7},
    "polished": {"bright": +0.5}, "crisp": {"bright": +0.8},
    "dark": {"bright": -1.0}, "muffled": {"bright": -0.9},
    "somber": {"bright": -0.6}, "denasal": {"bright": +0.6},
    "thin": {"bright": +0.5, "depth": -0.5},
    "airy": {"breath": +0.7, "bright": +0.3},
    # 深度 / 厚度
    "deep": {"depth": +1.0}, "bass": {"depth": +1.0},
    "baritone": {"depth": +0.8}, "bass-baritone": {"depth": +0.9},
    "resonant": {"depth": +0.5}, "chest": {"depth": +0.6},
    "high-pitched": {"depth": -0.9}, "soprano": {"depth": -0.8},
    "tenor": {"depth": -0.3}, "alto": {"depth": +0.2},
    # 粗糙 / 沙哑
    "raspy": {"rough": +1.0}, "gravelly": {"rough": +1.0},
    "gritty": {"rough": +0.9}, "weathered": {"rough": +0.8},
    "rough": {"rough": +1.0}, "husky": {"rough": +0.8},
    "smooth": {"rough": -1.0}, "velvety": {"rough": -0.9},
    "silky": {"rough": -0.9},
    # 气声
    "breathy": {"breath": +1.0}, "whispery": {"breath": +1.0},
    "whisper": {"breath": +1.0}, "intimate": {"breath": +0.5},
    "soft": {"breath": +0.4},
    # 空间感 / 空灵
    "ethereal": {"airy": +1.0}, "atmospheric": {"airy": +0.8},
    "otherworldly": {"airy": +0.9}, "celestial": {"airy": +0.9},
    "dreamy": {"airy": +0.8}, "ghostly": {"airy": +0.7},
    # 年龄
    "youthful": {"age": -0.8}, "young": {"age": -0.9},
    "child": {"age": -1.0}, "teen": {"age": -0.8},
    "mature": {"age": +0.5}, "elderly": {"age": +1.0},
    "ancient": {"age": +0.9}, "old": {"age": +0.9},
    "weary": {"age": +0.6}, "wise": {"age": +0.5},
    # 性别
    "feminine": {"gender_f": +1.0}, "female": {"gender_f": +1.0},
    "masculine": {"gender_m": +1.0}, "male": {"gender_m": +1.0},
    "androgynous": {"gender_n": +1.0},
    # 金属 / 非人
    "metallic": {"metallic": +1.0}, "robotic": {"metallic": +0.9, "nonhuman": +1.0},
    "synthetic": {"metallic": +0.6, "nonhuman": +0.6},
    "ai": {"nonhuman": +0.8}, "alien": {"nonhuman": +1.0},
    "monster": {"nonhuman": +1.0, "rough": +0.6},
    # 语速 / 节奏（用于 rate 参考）
    "slow": {"rate": -0.7}, "measured": {"rate": -0.5},
    "energetic": {"rate": +0.8}, "rapid": {"rate": +0.8},
    "monotone": {"rate": -0.3}, "staccato": {"rate": +0.5},
}

REGISTER_ANCHORS = {
    "bass": {"depth": +1.0}, "baritone": {"depth": +0.7},
    "tenor": {"depth": -0.3}, "alto": {"depth": +0.1},
    "mezzo": {"depth": -0.2}, "soprano": {"depth": -0.8},
    "contralto": {"depth": +0.3},
    "chest": {"depth": +0.5}, "head": {"depth": -0.4},
    "nasal": {"bright": +0.4}, "breathy": {"breath": +0.8},
    "raspy": {"rough": +0.9}, "gravelly": {"rough": +0.9},
    "fry": {"rough": +0.7}, "resonant": {"depth": +0.4},
    "bright": {"bright": +0.7}, "dark": {"bright": -0.8},
    "warm": {"rough": -0.3, "depth": +0.3},
    "thin": {"depth": -0.5}, "deep": {"depth": +0.9},
    "low": {"depth": +0.5}, "high": {"depth": -0.6},
    "airy": {"breath": +0.6},
}

AXES = ("bright", "depth", "rough", "breath", "airy", "metallic",
        "nonhuman", "age", "rate", "gender_f", "gender_m", "gender_n")


def _age_value(age_text: str) -> float:
    """把 '20s to early 30s' / 'Late 50s to 70s' 这类文本变成 0~1 年龄感。

    ⚠️ 踩过的坑：最初用 `\\b(\\d{2})\\b` 匹配，结果 "70s"/"80s" 全部匹配失败——
    因为 `\\b` 要求数字后是非单词字符，而 "70s" 的 0 后面紧跟字母 s。
    实测库里 244 种年龄取值几乎全被解析成 0.5，导致年龄维度完全失效。
    正确做法是允许后缀字母：`(\\d{2})s?`。
    """
    t = str(age_text).lower().strip()
    if not t:
        return 0.5

    # 先处理无数字的描述
    if "toddler" in t or "baby" in t or "infant" in t:
        return 0.02
    if "child" in t:
        return 0.06
    if "teen" in t and not re.search(r"\d", t):
        return 0.15
    if "middle-aged" in t and not re.search(r"\d", t):
        return 0.60
    if "mature" in t and not re.search(r"\d", t):
        return 0.65

    nums = [int(x) for x in re.findall(r"(\d{2})s?", t)]
    if not nums:
        return 0.5
    mid = sum(nums) / len(nums)
    if "late" in t:
        mid += 3
    if "early" in t:
        mid -= 3
    # 10 岁 -> 0，80 岁 -> 1
    return float(min(max((mid - 10.0) / 70.0, 0.0), 1.0))


@dataclass
class VoiceEntry:
    """数据集里的一条声线。"""
    cid: str
    name: str = ""
    tagline: str = ""
    gender: str = ""
    age: str = ""
    register: str = ""
    timbre_profile: dict = field(default_factory=dict)
    distinctive_features: str = ""
    emotional_range: str = ""
    casting: dict = field(default_factory=dict)
    tags: list = field(default_factory=list)
    audio: str = ""
    scores: dict = field(default_factory=dict)
    search_text: str = ""
    axes: dict = field(default_factory=dict)
    concept_hits: dict = field(default_factory=dict)

    def compute_concept_hits(self) -> dict:
        """对 search_text + tagline + features 做概念关键词匹配（0~1 归一）。"""
        blob = " ".join([self.search_text or "", self.tagline or "",
                         self.distinctive_features or "",
                         self.emotional_range or "", self.register or "",
                         " ".join(self.tags or [])]).lower()
        hits = {}
        for concept, kws in CONCEPT_KEYWORDS.items():
            total_w = sum(kws.values())
            got = 0.0
            for kw, w in kws.items():
                if text_matches(blob, kw):
                    got += w
            hits[concept] = got / total_w if total_w > 0 else 0.0
        self.concept_hits = hits
        return hits

    @property
    def duration(self) -> float:
        try:
            return float(self.scores.get("dur", 0.0))
        except Exception:
            return 0.0

    def local_audio(self) -> Path:
        return AUDIO_CACHE / f"{self.cid}.mp3"

    def summary(self, cn: bool = True) -> str:
        tp = self.timbre_profile or {}
        ax = self.axes or {}
        lines = [
            f"**{self.name}**　`{self.cid}`",
            f"　{self.tagline}",
            f"　性别 `{self.gender}`　年龄 `{self.age}`　时长 `{self.duration:.1f}s`",
            f"　音区：{self.register}",
            f"　音色六轴：金属{_v(tp.get('metallic'))} 喉音{_v(tp.get('throat_guttural'))} "
            f"假声{_v(tp.get('falsetto'))} 胸声{_v(tp.get('chest_voice'))} "
            f"粗糙{_v(tp.get('roughness'))} 亮度{_v(tp.get('brightness'))}（0-5）",
            f"　检索维度：亮度{ax.get('bright',0):+.2f} 厚度{ax.get('depth',0):+.2f} "
            f"粗糙{ax.get('rough',0):+.2f} 气声{ax.get('breath',0):+.2f} "
            f"空灵{ax.get('airy',0):+.2f} 年龄感{ax.get('age',0):.2f}",
        ]
        if self.tags:
            lines.append(f"　标签：{'、'.join(self.tags[:12])}")
        return "\n".join(lines)


def _v(x) -> str:
    return "—" if x is None else str(x)


def compute_axes(entry: VoiceEntry) -> dict:
    """由标签 + 音区文本 + 音色六轴算出可比较的维度向量。"""
    ax = {k: 0.0 for k in AXES}
    tl = [str(t).lower() for t in (entry.tags or [])]
    for t in tl:
        for key, delta in TAG_ANCHORS.get(t, {}).items():
            ax[key] = ax.get(key, 0.0) + delta
    reg = str(entry.register or "").lower()
    for key, delta in REGISTER_ANCHORS.items():
        if key in reg:
            for k2, v2 in delta.items():
                ax[k2] = ax.get(k2, 0.0) + v2 * 0.6      # 音区证据略弱于标签

    # 用六轴原始分数补充/校正
    tp = entry.timbre_profile or {}
    try:
        ax["bright"] += (float(tp.get("brightness", 0)) - 2.5) * 0.30
        ax["rough"] += (float(tp.get("roughness", 0)) - 1.5) * 0.40
        ax["metallic"] += (float(tp.get("metallic", 0)) - 0.8) * 0.35
        ax["depth"] += (float(tp.get("chest_voice", 0)) - 2.0) * 0.25
        ax["breath"] += float(tp.get("falsetto", 0)) * 0.12
    except Exception:
        pass

    # 性别与年龄（性别优先用显式字段，更可靠）
    g = str(entry.gender or "").lower()
    if "female" in g or "feminine" in g:
        ax["gender_f"] = 1.0
    elif "male" in g or "masculine" in g:
        ax["gender_m"] = 1.0
    elif "androgynous" in g:
        ax["gender_n"] = 1.0
    elif g in ("non-human", "nonhuman"):
        ax["nonhuman"] += 1.0

    ax["age"] = _age_value(entry.age)

    # 归一化：用 tanh 软压缩而不是硬裁剪。
    # 硬裁剪会让 37% 的声线在 depth 轴上全部等于 +1.00（实测 498/1336），
    # 导致"低沉男声"与"普通男声"分数完全相同、无法区分。
    # tanh(x/3) 保留排序且几乎不产生饱和点。
    for k in ("bright", "depth", "rough", "breath", "airy", "metallic",
              "nonhuman", "rate"):
        ax[k] = float(np.tanh(ax[k] / 3.0))
    return ax


class VoiceBank:
    """声线库：加载 + 检索。"""

    def __init__(self, data_dir: str | Path = DATA_DIR):
        self.dir = Path(data_dir)
        meta_path = self.dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"缺少声线库元数据 {meta_path}，请先运行 tools/fetch_voicebank.py")
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
        self.entries: list[VoiceEntry] = []
        for r in raw:
            e = VoiceEntry(
                cid=str(r.get("cid", "")),
                name=str(r.get("name", "")),
                tagline=str(r.get("tagline", "")),
                gender=str(r.get("gender", "") or ""),
                age=str(r.get("age", "") or ""),
                register=str(r.get("register", "") or ""),
                timbre_profile=r.get("timbre_profile") or {},
                distinctive_features=str(r.get("distinctive_features", "") or ""),
                emotional_range=str(r.get("emotional_range", "") or ""),
                casting=r.get("casting") or {},
                tags=[str(t) for t in (r.get("tags") or [])],
                audio=str(r.get("audio", "") or ""),
                scores=r.get("scores") or {},
                search_text=str(r.get("search_text", "") or ""),
            )
            e.axes = compute_axes(e)
            e.compute_concept_hits()
            self.entries.append(e)
        self.by_cid = {e.cid: e for e in self.entries}

    def __len__(self) -> int:
        return len(self.entries)

    # ---------------------------------------------------------------- 检索
    def search(self, want: dict, top_k: int = 10,
               seed: int | None = None,
               want_concepts: tuple[str, ...] = ()) -> list[tuple[VoiceEntry, float]]:
        """按声线维度需求检索。

        want 可包含：gender("female"/"male"/None)、age(0~1)、
                    bright/depth/rough/breath/airy/metallic/nonhuman(-1~1)、rate
        want_concepts：概念键（见 CONCEPT_KEYWORDS），会按 search_text 命中加权
        seed 不为 None 时，在得分接近的候选里做**确定性**抽样，
        以实现"同描述换种子出同族不同声线"。
        """
        scored = []
        for e in self.entries:
            s = 0.0
            wsum = 0.0
            ax = e.axes
            for k, target in want.items():
                if k in ("gender", "age", "seed"):
                    continue
                if k not in ax:
                    continue
                w = 1.0
                if k in ("rough", "breath", "depth", "bright"):
                    w = 1.4
                s += w * (1.0 - abs(ax[k] - float(target)))
                wsum += w

            # 性别匹配
            g = want.get("gender")
            if g == "female":
                s += 1.6 * ax.get("gender_f", 0.0)
                s -= 1.0 * ax.get("gender_m", 0.0)
                wsum += 1.6
            elif g == "male":
                s += 1.6 * ax.get("gender_m", 0.0)
                s -= 1.0 * ax.get("gender_f", 0.0)
                wsum += 1.6
            elif g == "neutral":
                s += 1.2 * ax.get("gender_n", 0.0)
                wsum += 1.2

            # 年龄
            if want.get("age") is not None:
                s += 1.2 * (1.0 - abs(ax.get("age", 0.5) - float(want["age"])))
                wsum += 1.2

            # 概念关键词命中：对完整的 search_text（约 1400 字/条）做匹配。
            # 这是提升"气质类"描述命中率的关键——只靠稀疏的 12 个标签
            # 无法区分"睿智"和"疲惫"这类近似词（实测踩过）。
            ch = e.concept_hits or {}
            for c in want_concepts:
                s += 1.5 * float(ch.get(c, 0.0))
                wsum += 1.5

            # 音质微调（有评分就更稳）
            try:
                s += 0.35 * float(e.scores.get("voice", 0.5))
            except Exception:
                pass

            # 参考时长偏好：IndexTTS 这类零样本克隆通常希望 5~10 秒参考。
            # 数据集里时长跨度很大（实测 2.0~13.8s），太短的参考会降低克隆质量，
            # 因此对 6~12 秒给满权重，偏离则线性衰减。
            d = e.duration
            if d > 0:
                if 6.0 <= d <= 12.0:
                    s += 0.30
                elif d < 6.0:
                    s += 0.30 * max(0.0, (d - 2.0) / 4.0)
                else:
                    s += 0.30 * max(0.0, 1.0 - (d - 12.0) / 12.0)
                wsum += 0.30

            # ---- 可用性偏好：避免"有气无力"的声线 ----
            # 实测：The Fading Matriarch 这类"缺胸腔共鸣、以气流声为主"的声线
            # 克隆出来又轻又颤（输出 RMS 0.18，而正常参考能到 0.40）。
            # 这不是检索错误（它确实"沙哑气声"），而是**可用性**问题：
            # 克隆参考需要胸腔共鸣与稳定声源。
            # 做法：对"高气声 + 低厚度 + 低亮度"的组合施加惩罚，
            # 但只在用户没有明确要求气声时才罚，且惩罚可被 want 抵消。
            if "breath" not in want and "airy" not in want:
                frailty = (max(0.0, ax.get("breath", 0.0)) * 0.5
                           + max(0.0, -ax.get("depth", 0.0)) * 0.3
                           + max(0.0, -ax.get("bright", 0.0)) * 0.2)
                s -= 0.8 * float(min(frailty, 1.0))
                wsum += 0.8

            # 真实感（数据集的 genu，越高越像真人录音）
            try:
                g = float(e.scores.get("genu", 0.0))
                s += 0.25 * max(0.0, min(g / 2.0, 1.0))
                wsum += 0.25
            except Exception:
                pass

            norm = s / wsum if wsum > 0 else 0.0
            scored.append((e, norm))

        scored.sort(key=lambda t: -t[1])
        if seed is None or top_k <= 0:
            return scored[:top_k]

        # 确定性抽样：从头部候选里按 seed 挑一个（保证可复现）
        head = scored[: max(top_k * 3, 12)]
        rng = random.Random(int(seed))
        weights = [max(t[1] - head[-1][1] + 1e-6, 1e-6) ** 2 for t in head]
        total = sum(weights)
        r = rng.random() * total
        acc = 0.0
        for (e, sc), w in zip(head, weights):
            acc += w
            if r <= acc:
                chosen = (e, sc)
                break
        else:
            chosen = head[0]
        rest = [t for t in scored[:top_k + 1] if t[0].cid != chosen[0].cid]
        return [chosen] + rest[:top_k - 1]

    # ---------------------------------------------------------------- 音频
    def fetch_audio(self, entry: VoiceEntry, force: bool = False) -> Path:
        """下载（并缓存）该声线的参考音频。"""
        AUDIO_CACHE.mkdir(parents=True, exist_ok=True)
        dest = entry.local_audio()
        if dest.exists() and dest.stat().st_size > 2048 and not force:
            return dest
        remote = entry.audio or f"audio/{entry.cid}.mp3"
        url = f"{HF_BASE}/{remote}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        return dest

    def stats(self) -> dict:
        from collections import Counter
        return {
            "总数": len(self.entries),
            "性别": dict(Counter(e.gender for e in self.entries)),
            "有音频": sum(1 for e in self.entries if e.audio),
        }

    # ---------------------------------------------------------------- 转 WAV
    def to_wav(self, entry: VoiceEntry, force: bool = False,
               sr: int = 48000, target_peak: float | None = None,
               max_seconds: float | None = 6.0) -> Path:
        """Prepare a content-addressed reference at its native sample rate.

        Six seconds is only a configurable starting point, not a proven optimum.
        Default amplitude is preserved: peak gain is a playback/preprocessing
        choice, not a fix for weak prosody. Cropping is contiguous and logged.
        """
        from timbreforge.reference import prepare_reference
        mp3 = self.fetch_audio(entry, force=force)
        # Preserve native rate: upsampling does not create missing detail. `sr`
        # remains in the signature for compatibility, not as a quality control.
        return prepare_reference(mp3, max_seconds=max_seconds,
                                 cache_dir=WAV_CACHE, target_peak=target_peak)

    def get(self, cid: str) -> VoiceEntry | None:
        """按 cid 取声线。"""
        return self.by_cid.get(cid)


# ---------------------------------------------------------------- 便捷入口
_BANK: VoiceBank | None = None


def get_bank(reload: bool = False) -> VoiceBank:
    global _BANK
    if _BANK is None or reload:
        _BANK = VoiceBank()
    return _BANK


# ---------------------------------------------------------------- 概念关键词
# 中文概念 -> 英文关键词（权重）。
# ⚠️ 全部用**实测词频**筛过：例如"甜美"对应的 sweet 在 1336 条里只出现 8 次、
#    lively 出现 0 次，说明这个数据集里几乎没有"甜美活泼"型声线——
#    检索不到不是算法错，而是数据里没有。保留下来的都是高频可用词。
CONCEPT_KEYWORDS: dict[str, dict[str, float]] = {
    "bright": {
        "bright": 1.0, "clear": 0.8, "crisp": 0.9, "crystalline": 0.9,
        "bell-like": 0.8, "ringing": 0.7, "polished": 0.6, "shrill": 0.7,
        "silvery": 0.8, "sparkling": 0.6,
    },
    "dark": {
        "dark": 1.0, "muffled": 0.9, "somber": 0.8, "shadowy": 0.7,
        "low-pass": 0.7, "murky": 0.6, "dusky": 0.7,
    },
    "depth": {
        "deep": 1.0, "bass": 1.0, "baritone": 0.9, "resonant": 0.7,
        "chest": 0.6, "booming": 0.8, "low": 0.6, "sonorous": 0.8,
        "rich": 0.5,
    },
    "high": {
        "high-pitched": 1.0, "soprano": 0.9, "thin": 0.7, "piercing": 0.8,
        "falsetto": 0.9, "head-voice": 0.8, "reedy": 0.6,
    },
    "rough": {
        "raspy": 1.0, "gravelly": 1.0, "gritty": 0.9, "weathered": 0.8,
        "rough": 0.9, "hoarse": 0.9, "textured": 0.6, "rasping": 0.9,
        "fry": 0.7, "abrasive": 0.8, "raucous": 0.8, "grating": 0.8,
    },
    "breath": {
        "breathy": 1.0, "airy": 0.8, "whispery": 1.0, "whisper": 0.9,
        "aspirated": 0.9, "hushed": 0.9, "wispy": 0.9, "exhaled": 0.6,
        "breathiness": 0.8,
    },
    "airy": {
        "ethereal": 1.0, "otherworldly": 0.9, "celestial": 0.9,
        "ghostly": 0.8, "atmospheric": 0.8, "haunting": 0.6,
        "shimmering": 0.7, "floating": 0.6, "transcendent": 0.7,
    },
    "agey": {
        "elderly": 1.0, "aged": 0.9, "ancient": 0.9, "old": 0.9,
        "weary": 0.6, "wise": 0.7, "sage": 0.7, "venerable": 0.9,
        "fading": 0.6, "weathered": 0.6,
    },
    "young": {
        "youthful": 1.0, "young": 1.0, "ingenue": 0.9, "girlish": 0.9,
        "boyish": 0.9, "teen": 0.9, "fresh": 0.6, "naive": 0.6,
    },
    "sweet": {                      # 实测库内稀少，保留以便命中那 8 条
        "sweet": 1.0, "honeyed": 1.0, "dulcet": 1.0, "sugary": 0.9,
        "melodic": 0.6, "singsong": 0.7,
    },
    "lively": {                     # lively 实测 0 次，故用相近的高频词
        "energetic": 1.0, "upbeat": 0.9, "playful": 0.9, "spirited": 0.8,
        "animated": 0.8, "bubbly": 0.9, "quick": 0.5,
    },
    "calm": {
        "calm": 1.0, "soothing": 0.9, "serene": 0.9, "steady": 0.8,
        "composed": 0.8, "measured": 0.7, "grounded": 0.7, "gentle": 0.7,
        "tranquil": 0.9,
    },
    "wise": {
        "wise": 1.0, "sage": 1.0, "scholarly": 0.9, "knowing": 0.8,
        "intelligent": 0.7, "erudite": 0.9, "professorial": 0.9,
        "authoritative": 0.6,
    },
    "magnetic": {
        "velvety": 1.0, "warm": 0.8, "rich": 0.7, "smooth": 0.7,
        "mellow": 0.8, "smoky": 0.8, "chocolate": 0.9, "suave": 0.8,
    },
    "metallic": {
        "metallic": 1.0, "robotic": 0.9, "synthetic": 0.7, "digital": 0.7,
        "machine": 0.8, "android": 0.9, "cybernetic": 0.9, "tinny": 0.8,
        "processed": 0.6,
    },
    "eerie": {
        "eerie": 1.0, "ominous": 1.0, "sinister": 0.9, "creepy": 0.9,
        "menacing": 0.9, "unsettling": 0.8, "haunting": 0.8, "dread": 0.8,
    },
    "elegant": {
        "elegant": 1.0, "refined": 0.9, "sophisticated": 0.9,
        "graceful": 0.9, "dignified": 0.8, "aristocratic": 0.9,
        "poised": 0.8,
    },
}

# 中文概念词 -> CONCEPT_KEYWORDS 的键（一个中文词可指向多个概念）
CN_CONCEPTS: dict[str, tuple[str, ...]] = {
    "明亮": ("bright",), "清亮": ("bright",), "清脆": ("bright",),
    "嘹亮": ("bright", "depth"), "通透": ("bright",),
    "暗": ("dark",), "暗哑": ("dark",), "闷": ("dark",), "低沉": ("depth",),
    "浑厚": ("depth",), "厚重": ("depth",), "深沉": ("depth",),
    "磁性": ("magnetic",), "醇厚": ("magnetic",), "温暖": ("magnetic",),
    "高音": ("high",), "尖细": ("high",), "尖锐": ("high", "rough"),
    "沙哑": ("rough",), "嘶哑": ("rough",), "粗糙": ("rough",),
    "粗犷": ("rough",), "苍老": ("rough", "agey"),
    "气声": ("breath",), "虚": ("breath",), "虚弱": ("breath",),
    "空灵": ("airy",), "飘渺": ("airy",), "梦幻": ("airy",), "仙": ("airy",),
    "老者": ("agey",), "爷爷": ("agey",), "奶奶": ("agey",), "睿智": ("wise",),
    "少年": ("young",), "正太": ("young",), "少女": ("young",),
    "萝莉": ("young",), "童声": ("young",), "儿童": ("young",), "稚嫩": ("young",),
    "甜美": ("sweet",), "甜": ("sweet",), "活泼": ("lively",),
    "慵懒": ("calm",), "沉稳": ("calm", "wise"), "温柔": ("calm",),
    "平静": ("calm",), "冷漠": ("calm",), "优雅": ("elegant",),
    "阴森": ("eerie",), "幽怨": ("eerie",),
    "金属": ("metallic",), "机器人": ("metallic",), "机械": ("metallic",),
    "非人": ("metallic",), "外星": ("metallic",), "电子": ("metallic",),
    "紧张": ("eerie",), "恐怖": ("eerie",),
}


def text_matches(text: str, keyword: str) -> bool:
    """关键词是否出现在文本里（用词边界，避免 sweet 命中 sweatshirt 之类）。"""
    if "-" in keyword:
        return keyword in text
    return re.search(rf"\b{re.escape(keyword)}", text) is not None


def concept_targets_deep(text: str) -> dict[str, float]:
    """中文概念 -> 概念维度的目标值（0~1）。"""
    from .voicelib import LEXICON as _LEX
    _ = _LEX
    out: dict[str, float] = {}
    for cn, concepts in CN_CONCEPTS.items():
        if cn in text:
            for c in concepts:
                out[c] = max(out.get(c, 0.0), 1.0)
    return out


def find_voice(prompt: str, seed: int = 0, top_k: int = 8,
               age: float | None = None, gender: str | None = None,
               overrides: dict | None = None) -> tuple[VoiceEntry, list]:
    """由中文描述找最匹配的声线（可复现）。"""
    from .voicelib import parse_voice_text

    parsed = parse_voice_text(prompt)
    want: dict = {}
    # 声线维度：由中文词库的整理参数换算。
    # 注意「沙哑(rough)」与「气声(breath)」必须分开：
    # 词库为简化合成把"沙哑"同时写进了 breath，但在数据集里
    # raspy/gravelly(粗糙) 与 breathy/whispery(气声) 是两类完全不同的声线。
    # 若不解耦，"冰冷沙哑的成熟女声"会命中 whisper 型声线（实测踩过）。
    rough_level = max(0.0, parsed["breath"])
    want["rough"] = max(-1.0, min(1.0, rough_level / 0.45))
    want["breath"] = max(-1.0, min(1.0, rough_level / 0.9))
    want["bright"] = max(-1.0, min(1.0, parsed["tilt"] / 0.35))
    want["depth"] = max(-1.0, min(1.0, -parsed["formant"] / 0.30))
    if abs(parsed["rate"] - 1.0) > 0.02:
        want["rate"] = max(-1.0, min(1.0, (parsed["rate"] - 1.0) / 0.35))

    # 概念维度：由中文概念词直接得到（空灵/金属/非人等）
    want_concepts: list[str] = []
    for cn, concepts in CN_CONCEPTS.items():
        if cn in prompt:
            for c in concepts:
                if c not in want_concepts:
                    want_concepts.append(c)
    # 概念的**方向**：用声线轴做粗定位，靠 search_text 命中做精排
    if "airy" in want_concepts:
        want["airy"] = 0.8
    if "metallic" in want_concepts:
        want["metallic"] = 0.6
    if "agey" in want_concepts:
        want["depth"] = max(want.get("depth", 0.0), 0.2)

    g = gender or parsed.get("gender")
    if g:
        want["gender"] = g
    a = age
    if a is None and parsed.get("age"):
        a = 0.5 + parsed["age"] * 0.45
    if a is not None:
        want["age"] = min(max(a, 0.0), 1.0)
    if overrides:
        want.update({k: v for k, v in overrides.items() if v is not None})

    bank = get_bank()
    ranked = bank.search(want, top_k=top_k, seed=seed,
                         want_concepts=tuple(want_concepts))
    if not ranked:
        raise RuntimeError("声线库为空")
    return ranked[0][0], ranked
