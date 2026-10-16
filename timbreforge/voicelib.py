"""TimbreForge —— 中文角色声线描述词典

把自然语言映射到**声线参数**（音高偏移/共振峰/明亮度/气声/语速/年龄感）
以及**情感向量**（IndexTTS 的 8 维）。

这是"AI 生图式提示词"在语音上的对应物：
    生图：提示词 + 种子 + 参数  ->  图像
    声线：声线描述 + 台词 + 种子 + 参数  ->  语音
"""
from __future__ import annotations

# ---------------------------------------------------------------- 声线维度
# 每个词给出一组参数增量。音高用半音，因子类用比例。
LEXICON: dict[str, dict] = {
    # ── 性别 / 音区 ─────────────────────────────────────
    "女声": {"gender": "female", "pitch": +2.0},
    "女":   {"gender": "female", "pitch": +2.0},
    "少女": {"gender": "female", "pitch": +4.5, "formant": +0.06, "age": -0.4},
    "萝莉": {"gender": "female", "pitch": +7.0, "formant": +0.12, "age": -0.6},
    "御姐": {"gender": "female", "pitch": -1.0, "formant": -0.04, "age": +0.15},
    "大妈": {"gender": "female", "pitch": -3.0, "formant": -0.05, "age": +0.6},
    "男声": {"gender": "male", "pitch": -7.0, "formant": -0.10},
    "男":   {"gender": "male", "pitch": -7.0, "formant": -0.10},
    "少年": {"gender": "male", "pitch": -3.0, "formant": +0.02, "age": -0.35},
    "正太": {"gender": "male", "pitch": -1.0, "formant": +0.10, "age": -0.6},
    "大叔": {"gender": "male", "pitch": -9.0, "formant": -0.14, "age": +0.5},
    "老者": {"gender": "male", "pitch": -8.0, "formant": -0.10, "age": +0.85},
    "爷爷": {"gender": "male", "pitch": -9.0, "formant": -0.12, "age": +0.9},
    "奶奶": {"gender": "female", "pitch": -5.0, "formant": -0.06, "age": +0.85},
    "童声": {"pitch": +6.0, "formant": +0.14, "age": -0.6},
    "儿童": {"pitch": +6.0, "formant": +0.14, "age": -0.6},
    "低音": {"pitch": -3.0},
    "高音": {"pitch": +3.0},

    # ── 音质 / 音色 ─────────────────────────────────────
    "沙哑": {"breath": +0.28, "tilt": -0.08},
    "嘶哑": {"breath": +0.32, "tilt": -0.05},
    "磁性": {"pitch": -2.0, "formant": -0.05, "tilt": -0.10},
    "醇厚": {"formant": -0.08, "tilt": -0.12},
    "浑厚": {"formant": -0.10, "tilt": -0.12},
    "低沉": {"pitch": -3.5, "tilt": -0.10},
    "清亮": {"tilt": +0.15, "formant": +0.03},
    "清脆": {"tilt": +0.20},
    "甜美": {"pitch": +2.5, "tilt": +0.10},
    "圆润": {"formant": -0.03, "tilt": -0.04},
    "尖细": {"formant": +0.12, "tilt": +0.15},
    "明亮": {"tilt": +0.18},
    "暗哑": {"tilt": -0.18},
    "空灵": {"breath": +0.12, "tilt": +0.06},
    "气声": {"breath": +0.35},
    "虚":   {"breath": +0.28, "tilt": -0.05},
    "结实": {"breath": -0.15, "tilt": -0.03},
    "温柔": {"tilt": -0.05, "breath": +0.06},
    "冷":   {"tilt": -0.08, "breath": +0.05},
    "冷冽": {"tilt": -0.12, "breath": +0.06},
    "温暖": {"tilt": -0.05, "formant": -0.03},
    "活泼": {"rate": +1.12, "pitch": +1.5},
    "慵懒": {"rate": -0.88, "pitch": -1.0, "breath": +0.08},
    "沉稳": {"rate": -0.93, "tilt": -0.06},
    "急促": {"rate": +1.20},
    "缓慢": {"rate": -0.85},
    "苍老": {"age": +0.5, "breath": +0.12, "tilt": -0.08},
    "稚嫩": {"age": -0.4, "pitch": +3.5, "formant": +0.08},

    # ── 情绪（映射到 IndexTTS 情感向量）───────────────────
    "开心": {"emo": {"happy": 0.8}},
    "高兴": {"emo": {"happy": 0.8}},
    "快乐": {"emo": {"happy": 0.75}},
    "兴奋": {"emo": {"happy": 0.95, "surprised": 0.35}},
    "生气": {"emo": {"angry": 0.85}},
    "愤怒": {"emo": {"angry": 0.95}},
    "难过": {"emo": {"sad": 0.8, "melancholic": 0.6}},
    "悲伤": {"emo": {"sad": 0.85, "melancholic": 0.7}},
    "忧郁": {"emo": {"melancholic": 0.8}},
    "恐惧": {"emo": {"fearful": 0.8}},
    "害怕": {"emo": {"fearful": 0.8}},
    "厌恶": {"emo": {"disgusted": 0.8}},
    "惊讶": {"emo": {"surprised": 0.85}},
    "平静": {"emo": {"calm": 0.8}},
    "冷漠": {"emo": {"calm": 0.6, "disgusted": 0.15}},
    "亲切": {"emo": {"happy": 0.35, "calm": 0.35}},
}

NEGATIONS = ("不", "没", "无", "非", "别")
DEGREE = {
    "非常": 1.7, "极其": 1.9, "特别": 1.6, "很": 1.5, "超": 1.6,
    "相当": 1.4, "比较": 1.15, "稍微": 0.6, "有点": 0.65, "略微": 0.55,
    "稍": 0.6, "略": 0.6, "挺": 1.3,
}

# 声线参数的合法范围（防止"极其沙哑"这类描述把参数推到不可用区）
RANGES = {
    "pitch": (-14.0, 14.0),      # 半音
    "formant": (-0.30, 0.30),    # 比例偏移
    "tilt": (-0.35, 0.35),
    "breath": (0.0, 0.65),
    "rate": (0.75, 1.35),
    "age": (-1.0, 1.0),
}


def _find_degree(text: str, pos: int) -> float:
    head = text[max(0, pos - 3):pos]
    for w, s in DEGREE.items():
        if head.endswith(w):
            return s
    return 1.0


def _find_negation(text: str, pos: int) -> bool:
    head = text[max(0, pos - 2):pos]
    return any(head.endswith(n) for n in NEGATIONS)


def clamp_params(p: dict) -> dict:
    out = {}
    for k, v in p.items():
        if k in RANGES:
            lo, hi = RANGES[k]
            out[k] = float(min(max(v, lo), hi))
        elif k != "emo":
            out[k] = v
    return out


def parse_voice_text(text: str) -> dict:
    """解析声线描述。

    返回:
      {
        "pitch": 半音, "formant": 比例偏移, "tilt": , "breath": , "rate": ,
        "age": , "gender": "female"/"male"/None,
        "emotion": {"happy": 0.8, ...},     # IndexTTS 情感向量
        "hits": [...], "negated": [...],
      }
    """
    params = {"pitch": 0.0, "formant": 0.0, "tilt": 0.0,
              "breath": 0.0, "rate": 1.0, "age": 0.0, "gender": None}
    emotion: dict[str, float] = {}
    hits: list[str] = []
    negated: list[str] = []

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
            neg = _find_negation(text, pos)
            if neg:
                strength = -strength
                negated.append(word)
            else:
                hits.append(word)
            for k, v in LEXICON[word].items():
                if k == "emo":
                    for ek, ev in v.items():
                        emotion[ek] = max(emotion.get(ek, 0.0),
                                          ev * (0.5 if neg else abs(strength)))
                elif k == "gender":
                    if not neg:
                        params["gender"] = v
                elif k == "rate":
                    # 语速是乘性参数，基准 1.0
                    base = params["rate"]
                    params["rate"] = base * (v ** (strength if not neg else -strength))
                else:
                    params[k] = params.get(k, 0.0) + v * strength
            start = pos + len(word)

    return {
        **clamp_params(params),
        "emotion": emotion,
        "hits": hits,
        "negated": negated,
    }


# ---------------------------------------------------------------- 年龄 -> 声线
def age_to_params(age: float) -> dict:
    """年龄感（0=婴儿 0.5=青年 1=老年）-> 声线参数。"""
    a = min(max(age, 0.0), 1.0)
    # 青年（0.5）附近不调整
    d = a - 0.5
    return {
        "pitch": -4.0 * d,           # 越老越低
        "formant": -0.06 * d,        # 越老腔体感越大
        "breath": 0.22 * max(d, 0.0),  # 越老越气
        "tilt": -0.10 * d,
        "rate": 1.0 - 0.10 * d,
    }


def describe_params(p: dict) -> str:
    """把参数翻译成人话，便于核对。"""
    parts = []
    if p.get("gender"):
        parts.append("女声" if p["gender"] == "female" else "男声")
    pit = p.get("pitch", 0.0)
    if abs(pit) >= 0.5:
        parts.append(f"音高{pit:+.1f}半音({'偏高' if pit > 0 else '偏低'})")
    fm = p.get("formant", 0.0)
    if abs(fm) >= 0.02:
        parts.append(f"腔体{'偏小/尖细' if fm > 0 else '偏大/浑厚'}{fm:+.2f}")
    tl = p.get("tilt", 0.0)
    if abs(tl) >= 0.03:
        parts.append(f"音色{'明亮' if tl > 0 else '暗哑'}{tl:+.2f}")
    br = p.get("breath", 0.0)
    if br >= 0.05:
        parts.append(f"气声{br:.2f}")
    rt = p.get("rate", 1.0)
    if abs(rt - 1.0) >= 0.03:
        parts.append(f"语速{rt:.2f}x")
    emo = p.get("emotion") or {}
    if emo:
        parts.append("情绪:" + "、".join(f"{k}={v:.2f}" for k, v in emo.items()))
    return "  ".join(parts) if parts else "（无明显特征）"
