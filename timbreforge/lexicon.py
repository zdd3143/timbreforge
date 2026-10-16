"""TimbreForge —— 中文音色词库层（P0-6）

中文音色描述 → 6 个宏的增量 + 乐器族先验。
支持：
  * 程度副词（"非常""有点""稍微"）调制强度
  * 否定词（"不太亮""不厚"）反转方向
  * 乐器族关键词（贝斯/铺底/拨弦/主音/钢琴/铃）
"""
from __future__ import annotations

from .patch import FAMILIES, MACRO_NAMES, Patch, clamp

# ---------------------------------------------------------------- 词库
# 每个词给出一组宏增量（正数=增强该听感）
LEXICON: dict[str, dict[str, float]] = {
    # ── 亮度 ──────────────────────────────────────────────
    "明亮": {"brightness": +0.30},
    "亮":   {"brightness": +0.22},
    "通透": {"brightness": +0.22, "grit": -0.10},
    "清亮": {"brightness": +0.28, "warmth": -0.12},
    "清脆": {"brightness": +0.25, "attack_sharpness": +0.25},
    "暗淡": {"brightness": -0.28},
    "闷":   {"brightness": -0.30, "warmth": +0.10},
    "暗":   {"brightness": -0.22},
    "柔":   {"brightness": -0.15, "attack_sharpness": -0.20},
    "柔和": {"brightness": -0.15, "attack_sharpness": -0.25, "grit": -0.15},

    # ── 温暖 / 厚度 ───────────────────────────────────────
    "温暖": {"warmth": +0.32, "brightness": -0.10},
    "暖":   {"warmth": +0.25, "brightness": -0.08},
    "厚实": {"warmth": +0.28, "brightness": -0.12},
    "厚":   {"warmth": +0.22},
    "肥厚": {"warmth": +0.30, "brightness": -0.18},
    "饱满": {"warmth": +0.20, "brightness": +0.08},
    "单薄": {"warmth": -0.25, "brightness": +0.10},
    "冰冷": {"warmth": -0.32, "brightness": +0.22},
    "冷":   {"warmth": -0.25, "brightness": +0.15},
    "冷冽": {"warmth": -0.35, "brightness": +0.25, "grit": +0.10},
    "复古": {"warmth": +0.22, "grit": +0.12, "brightness": -0.10},
    "模拟": {"warmth": +0.25, "brightness": -0.10},
    "数字": {"warmth": -0.20, "brightness": +0.15},

    # ── 粗糙 / 攻击性 ─────────────────────────────────────
    "尖锐": {"grit": +0.20, "brightness": +0.22, "attack_sharpness": +0.20},
    "刺耳": {"grit": +0.30, "brightness": +0.25},
    "粗糙": {"grit": +0.30, "warmth": -0.10},
    "脏":   {"grit": +0.32, "warmth": +0.05},
    "失真": {"grit": +0.30, "warmth": +0.10},
    "金属": {"grit": +0.28, "brightness": +0.25, "attack_sharpness": +0.10},
    "沙哑": {"grit": +0.25, "warmth": +0.10},
    "干净": {"grit": -0.28, "warmth": -0.05},
    "纯净": {"grit": -0.30, "brightness": +0.08},

    # ── 起音 / 时间形态 ───────────────────────────────────
    "拨弦": {"attack_sharpness": +0.35, "space": -0.10},
    "打击": {"attack_sharpness": +0.38, "grit": +0.10},
    "短促": {"attack_sharpness": +0.30},
    "绵长": {"attack_sharpness": -0.28, "space": +0.12},
    "缓慢": {"attack_sharpness": -0.30},
    "渐入": {"attack_sharpness": -0.32},
    "持续": {"attack_sharpness": +0.15},

    # ── 空间 / 流动 ───────────────────────────────────────
    "空灵": {"space": +0.35, "movement": +0.18, "brightness": +0.10},
    "飘渺": {"space": +0.38, "movement": +0.22},
    "梦幻": {"space": +0.32, "movement": +0.20, "brightness": +0.08},
    "宽阔": {"space": +0.30, "warmth": +0.10},
    "干":   {"space": -0.30},
    "干涩": {"space": -0.32, "warmth": -0.10},
    "紧绷": {"space": -0.25, "attack_sharpness": +0.20},
    "流动": {"movement": +0.30},
    "律动": {"movement": +0.28, "attack_sharpness": +0.15},
    "颤抖": {"movement": +0.35, "grit": +0.08},
    "稳定": {"movement": -0.25},
    "静止": {"movement": -0.30},

    # ── 音区 / 用途 ───────────────────────────────────────
    "低音": {"brightness": -0.15, "warmth": +0.12},
    "低": {"brightness": -0.08},
    "高音": {"brightness": +0.20, "warmth": -0.12},
    "高": {"brightness": +0.10},
    "中音": {"brightness": 0.0},
    "深沉": {"brightness": -0.25, "warmth": +0.18},
    "厚重": {"warmth": +0.28, "brightness": -0.20},
    "尖锐高音": {"brightness": +0.30, "grit": +0.18},
}

# 人声关键词：出现即切到 voice 模式
VOICE_TRIGGER_WORDS = (
    "人声", "女声", "男声", "童声", "嗓音", "唱", "歌声", "吟唱", "哼唱",
    "元音", "vocal", "voice", "阿卡贝拉", "合唱",
)

# 性别 -> 说话人设定
GENDER_WORDS = {
    "女声": "female", "女": "female", "女音": "female", "女嗓": "female",
    "男声": "male", "男": "male", "男音": "male", "男嗓": "male",
    "童声": "child", "小孩": "child", "儿童": "child", "童音": "child",
    "低沉男声": "male", "低音男声": "male", "高音女声": "female",
}

# 元音关键词 -> 共振峰元音
VOWEL_WORDS = {
    "啊": "a", "阿": "a", "开口": "a",
    "诶": "e", "欸": "e",
    "衣": "i", "伊": "i", "闭口": "i",
    "噢": "o", "哦": "o", "欧": "o",
    "乌": "u", "呜": "u", "唔": "u",
}

# 嗓音特质 -> 宏增量（复用 6 个宏）
VOICE_TRAITS = {
    "气声": {"grit": +0.30, "space": +0.10},
    "沙哑": {"grit": +0.45},
    "嘶哑": {"grit": +0.45},
    "磁性": {"warmth": +0.30, "grit": +0.12},
    "醇厚": {"warmth": +0.35},
    "清亮": {"brightness": +0.30, "grit": -0.15},
    "甜美": {"brightness": +0.20, "warmth": +0.15, "grit": -0.10},
    "温柔": {"warmth": +0.25, "grit": -0.15, "movement": +0.05},
    "苍老": {"grit": +0.35, "warmth": +0.10},
    "空灵": {"space": +0.35, "movement": +0.20},
    "幽怨": {"warmth": +0.10, "movement": +0.15},
    "有力": {"attack_sharpness": +0.25, "brightness": +0.10},
    "虚弱": {"attack_sharpness": -0.20, "grit": +0.20},
    "颤抖": {"movement": +0.40},
    "平稳": {"movement": -0.25},
}

# 鼻音 / 辅音
NASAL_WORDS = {"鼻音": 0.5, "哼": 0.6, "哼鸣": 0.7, "吟": 0.3, "叹": 0.2}
CONSONANT_WORDS = {"嘶": "s", "齿音": "s", "爆破": "t", "吐字": "t", "哼鸣": "m"}

# 乐器族关键词 → FAMILIES 里的键
FAMILY_WORDS = {
    "贝斯": "bass", "低音贝斯": "bass", "bass": "bass", "低频": "bass", "低音": "bass",
    "铺底": "pad", "pad": "pad", "氛围": "pad", "背景": "pad", "pad音色": "pad",
    "拨弦": "pluck", "弹拨": "pluck", "pluck": "pluck", "琶音": "pluck",
    "主音": "lead", "lead": "lead", "独奏": "lead", "旋律": "lead",
    "钢琴": "keys", "键盘": "keys", "电钢": "keys", "keys": "keys",
    "铃": "bell", "钟": "bell", "铃铛": "bell", "bell": "bell",
}

# 否定词与程度副词
NEGATIONS = ("不", "没", "无", "非", "勿", "别")
DEGREE = {
    "非常": 1.7, "极其": 1.9, "特别": 1.6, "很": 1.5, "超": 1.6,
    "相当": 1.4, "比较": 1.15, "稍微": 0.6, "有点": 0.65, "略微": 0.55,
    "一点点": 0.45, "稍": 0.6, "略": 0.6, "挺": 1.3,
}


def _find_degree(text: str, pos: int) -> float:
    """看某个词前面是否有程度副词。"""
    head = text[max(0, pos - 3):pos]
    for w, s in DEGREE.items():
        if head.endswith(w):
            return s
    return 1.0


def _find_negation(text: str, pos: int) -> bool:
    """看某个词前面 1~2 个字是否有否定词。"""
    head = text[max(0, pos - 2):pos]
    return any(head.endswith(n) for n in NEGATIONS)


def parse_description(text: str) -> tuple[dict, str | None, list[str], list[str], dict]:
    """解析中文描述。

    返回 (宏增量, 乐器族, 命中词, 被否定的词, 人声信息)
    人声信息: {"voice": bool, "gender": str|None, "vowel": str|None,
              "nasal": float, "consonant": str}
    """
    deltas = {k: 0.0 for k in MACRO_NAMES}
    hits: list[str] = []
    negated: list[str] = []
    family = None
    voice = {"voice": False, "gender": None, "vowel": None,
             "nasal": 0.0, "consonant": "none"}

    # 人声触发
    for w in VOICE_TRIGGER_WORDS:
        if w in text:
            voice["voice"] = True
            hits.append(f"{w}→人声")
            break

    # 性别（长词优先）
    for w in sorted(GENDER_WORDS, key=len, reverse=True):
        if w in text:
            voice["gender"] = GENDER_WORDS[w]
            voice["voice"] = True
            hits.append(f"{w}→{GENDER_WORDS[w]}")
            break

    # 元音
    for w in sorted(VOWEL_WORDS, key=len, reverse=True):
        if w in text:
            voice["vowel"] = VOWEL_WORDS[w]
            voice["voice"] = True
            hits.append(f"{w}→元音{voice['vowel']}")
            break

    # 鼻音 / 辅音
    for w, amt in NASAL_WORDS.items():
        if w in text:
            voice["nasal"] = max(voice["nasal"], amt)
            voice["voice"] = True
            hits.append(f"{w}→鼻音{amt}")
    for w, cons in CONSONANT_WORDS.items():
        if w in text:
            voice["consonant"] = cons
            hits.append(f"{w}→辅音{cons}")
            break

    # 嗓音特质（并入宏增量）
    for w, d in VOICE_TRAITS.items():
        if w in text:
            voice["voice"] = True
            hits.append(w)
            for k, v in d.items():
                deltas[k] += v

    # 乐器族（优先匹配更长的词）
    for word in sorted(FAMILY_WORDS, key=len, reverse=True):
        if word in text:
            family = FAMILY_WORDS[word]
            hits.append(f"{word}→{family}")
            break

    # 听感词（长词优先，避免"暖"抢了"温暖"）
    consumed = [False] * len(text)
    for word in sorted(LEXICON, key=len, reverse=True):
        start = 0
        while True:
            pos = text.find(word, start)
            if pos < 0:
                break
            if any(consumed[pos:pos + len(word)]):
                start = pos + 1
                continue
            for i in range(pos, min(pos + len(word), len(consumed))):
                consumed[i] = True
            strength = _find_degree(text, pos)
            if _find_negation(text, pos):
                strength = -strength
                negated.append(word)
            else:
                hits.append(word)
            for k, v in LEXICON[word].items():
                deltas[k] += v * strength
            start = pos + len(word)

    return deltas, family, hits, negated, voice


def build_patch(text: str, seed: int = 0, jitter: float = 0.12,
                base_family: str | None = None,
                force_voice: bool = False,
                vowel: str | None = None,
                gender: str | None = None) -> Patch:
    """把一句中文描述变成 Patch。

    jitter 控制随机扰动幅度：同描述 + 不同 seed → 同族不同味。
    force_voice/vowel/gender 允许命令行显式覆盖。
    """
    import random as _random
    deltas, family, hits, negated, voice = parse_description(text)
    is_voice = force_voice or voice["voice"]
    family = family or base_family or ("voice" if is_voice else "keys")

    p = Patch(name=text.strip() or "untitled", family=family, seed=seed,
              duration=0.0, source_words=hits + [f"!{w}" for w in negated])

    # 乐器族先验作为起点
    fam = FAMILIES.get(family, FAMILIES["keys"])
    for k, v in fam.items():
        if k in p.macros:
            p.macros[k] = v

    # 叠加词库增量
    for k, v in deltas.items():
        p.macros[k] = clamp(p.macros[k] + v)

    # seed 扰动
    rng = _random.Random(seed)
    for k in MACRO_NAMES:
        p.macros[k] = clamp(p.macros[k] + rng.uniform(-jitter, jitter))

    p.expand()

    # ---- 人声模式 ----
    if is_voice:
        from .voice import VOICE_CHARACTER
        p.mode = "voice"
        p.family = "voice"
        p.gender = gender or voice["gender"] or "female"
        # 元音优先级：命令行 > 描述里的元音 > 由听感词推断
        if vowel:
            p.vowel = vowel
        elif voice["vowel"]:
            p.vowel = voice["vowel"]
        else:
            p.vowel = "a"
            for w, v in VOICE_CHARACTER.items():
                if w in text:
                    p.vowel = v
                    break
        p.nasal = voice["nasal"]
        p.consonant = voice["consonant"]
        from .voice import GENDER as VG
        p.root_note = 57 if p.gender == "female" else (45 if p.gender == "male" else 60)
        p.frequency = VG.get(p.gender, VG["female"])["f0"]
        p.amp_attack = 0.05
        p.amp_release = 0.30
        p.apply_voice_macros()
    return p


def explain(text: str) -> str:
    """给用户看：这句话是怎么被理解的。"""
    deltas, family, hits, negated, voice = parse_description(text)
    lines = [f'描述: "{text}"']
    if voice["voice"]:
        from .voice import profile_hint
        lines.append(f"  模式: 人声（元音/共振峰）")
        lines.append(f"  性别: {voice['gender'] or '未指明（默认女声）'}")
        v = voice["vowel"]
        lines.append(f"  元音: {v + '（' + profile_hint(v) + '）' if v else '由听感词推断'}")
        if voice["nasal"] > 0:
            lines.append(f"  鼻音量: {voice['nasal']}")
        if voice["consonant"] != "none":
            lines.append(f"  辅音起音: {voice['consonant']}")
    else:
        lines.append(f"  乐器族: {family or '未指明（默认 keys）'}")
    lines.append(f"  命中词: {'、'.join(hits) if hits else '（无）'}")
    if negated:
        lines.append(f"  否定词: {'、'.join(negated)}")
    nz = {k: v for k, v in deltas.items() if abs(v) > 1e-6}
    if nz:
        lines.append("  宏增量: " + "  ".join(f"{k}{v:+.2f}" for k, v in nz.items()))
    return "\n".join(lines)
