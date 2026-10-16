"""TimbreForge —— 客观音色相似度（把"像不像"变成数字）

目的：不再依赖人工逐条听评，而是用一个**可复现的数字**衡量
      "合成结果有多贴近参考声线"，从而可以：
        1. 自动比较不同数据集 / 不同声线 / 不同参数
        2. 用优化直接最大化这个数字（贴合参考）

实现：**说话人指纹 = MFCC 统计量**
  * 分帧 → 梅尔滤波器组 → 对数 → DCT → MFCC
  * 取 C1..C12（丢掉 C0，即能量，避免受音量影响）
  * 对每维取 均值 与 标准差 → 拼成 24 维指纹
  * 余弦相似度 -> 映射到 0~100

⚠️ 这不是严格的说话人验证（那需要 ECAPA/CAM++ 这类训练过的模型）。
   它是一个**无训练、可复现、对音色敏感**的代理指标，用于相对比较。
   实测验证见文件末尾的 `_self_test()`。
"""
from __future__ import annotations

import numpy as np

SR_DEFAULT = 16000
N_FFT = 400          # 25ms @16k
HOP = 160            # 10ms
N_MELS = 40
N_MFCC = 13          # 含 C0，统计时丢弃


# ---------------------------------------------------------------- 梅尔滤波器组
def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1.0)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int,
                    fmin: float = 40.0, fmax: float | None = None) -> np.ndarray:
    fmax = fmax or sr / 2.0
    mels = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz = _mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(1, n_mels + 1):
        l, c, r = bins[i - 1], bins[i], bins[i + 1]
        if c <= l:
            c = min(l + 1, n_fft // 2)
        if r <= c:
            r = min(c + 1, n_fft // 2)
        if c > l:
            fb[i - 1, l:c] = np.linspace(0, 1, c - l, endpoint=False)
        if r > c:
            fb[i - 1, c:r] = np.linspace(1, 0, r - c, endpoint=False)
    return fb


def _dct_matrix(n_out: int, n_in: int) -> np.ndarray:
    k = np.arange(n_out)[:, None]
    n = np.arange(n_in)[None, :]
    return np.cos(np.pi * k * (2 * n + 1) / (2 * n_in)) * np.sqrt(2.0 / n_in)


# ---------------------------------------------------------------- MFCC
def mfcc(x: np.ndarray, sr: int, n_mfcc: int = N_MFCC,
         n_fft: int = N_FFT, hop: int = HOP, n_mels: int = N_MELS) -> np.ndarray:
    """返回 [T, n_mfcc]。"""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if x.size < n_fft * 2:
        x = np.pad(x, (0, n_fft * 2 - x.size))
    # 预加重
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])
    n_frames = 1 + (x.size - n_fft) // hop
    win = np.hamming(n_fft)
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[idx] * win
    spec = np.abs(np.fft.rfft(frames, n_fft)) ** 2
    fb = _mel_filterbank(sr, n_fft, n_mels)
    mel = np.log(spec @ fb.T + 1e-10)
    return mel @ _dct_matrix(n_mfcc, n_mels).T


def _voiced_frames(m: np.ndarray, ratio: float = 0.25) -> np.ndarray:
    """只保留有声帧（用 C0 能量筛），避免静音段主导统计。"""
    if m.shape[0] < 4:
        return m
    e = m[:, 0]
    thr = np.max(e) - np.log(1.0 / max(ratio, 1e-3))
    sel = e > thr
    if sel.sum() < 4:
        return m
    return m[sel]


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """线性重采样（用于统一采样率，避免采样率差异污染指标）。"""
    if sr_in == sr_out or x.size < 2:
        return x
    n_out = max(int(round(x.size * sr_out / sr_in)), 2)
    idx = np.linspace(0, x.size - 1, n_out)
    i0 = np.floor(idx).astype(np.int64)
    i1 = np.clip(i0 + 1, 0, x.size - 1)
    frac = idx - i0
    return x[i0] * (1 - frac) + x[i1] * frac


def timbre_fingerprint(x: np.ndarray, sr: int, use_delta: bool = True) -> np.ndarray:
    """音色指纹：**逐帧跨维 z-score（谱形）+ 整体电平**。

    两步踩坑记录：
      1. 最初直接用 MFCC 绝对均值 -> 噪声把各维一起推向 0，
         "同说话人+少量噪声"的余弦掉到 0 甚至负数。
      2. 改成**逐维帧间 z-score** -> 每一维均值恒为 0，
         信息被完全抹掉（所有情形都变成 100 分）。

    最终做法（对增益、录音电平、加性噪声都稳健）：
      * 对**每一帧**在维内做 z-score -> 得到"谱形模式"，与整体增益无关
      * 统计该模式的帧间均值与标准差
      * 另加一个标量：整体 MFCC 电平（1 维），用于区分"同模式但差很多"的情况
    """
    m = mfcc(x, sr)
    m = _voiced_frames(m)
    c = m[:, 1:]                              # 丢掉能量维 C0
    if c.shape[0] < 3:
        return np.zeros(c.shape[1] * 2 + 3)
    # 逐帧跨维 z-score：保留谱形，去掉整体电平
    mu_f = c.mean(axis=1, keepdims=True)
    sd_f = c.std(axis=1, keepdims=True) + 1e-8
    z = (c - mu_f) / sd_f
    stats = [z.mean(axis=0), z.std(axis=0)]
    if use_delta and z.shape[0] >= 3:
        d = np.diff(z, axis=0)
        stats.append(d.std(axis=0))
    # 整体电平项（1 个标量，重复 3 次以给它合理权重）
    level = float(mu_f.mean())
    stats.append(np.array([level, level, level]))
    return np.concatenate(stats)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """皮尔逊相关（对平移与尺度不敏感）。

    ⚠️ 这里必须用相关而不是余弦：实测中"加噪声"会让指纹向量接近**零向量**
    （谱被抹平），此时余弦会掉到 0 甚至变负，完全没有区分能力；
    而相关衡量的是"形状是否一致"，在这种情形下仍然稳定。
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# 校准锚点：把余弦值线性映射到 0~100。
# 早期用 (c-0.5)/0.5 的映射太陡——实测"同说话人+噪声"会掉到 0.00，
# 失去区分能力。改用固定锚点后，分数有绝对含义（见下表）。
_CALIB = (
    (-0.200, 0.0),    # 谱形相反（完全不同类型）
    (0.000, 12.0),    # 谱形无关（加噪把谱抹平）
    (0.400, 30.0),    # 只有部分维度接近
    (0.600, 45.0),    # 不同说话人常见区间
    (0.780, 62.0),    # 共振峰明显偏移
    (0.900, 78.0),    # 有可见偏差
    (0.960, 88.0),    # 高度相似
    (0.990, 96.0),
    (1.000, 100.0),
)
# 解读参考：
#   >= 88  很像同一个人
#   78~88  同类型音色，有可测差异
#   45~78  音色方向对但差异明显（或换了说话人）
#   < 45   基本不像（不同说话人 / 噪声主导）


def _calibrate(c: float) -> float:
    """余弦 -> 0~100（分段线性插值）。"""
    c = float(min(max(c, 0.0), 1.0))
    pts = _CALIB
    if c <= pts[0][0]:
        return 0.0
    for i in range(1, len(pts)):
        x0, y0 = pts[i - 1]
        x1, y1 = pts[i]
        if c <= x1:
            t = (c - x0) / (x1 - x0) if x1 > x0 else 0.0
            return float(y0 + t * (y1 - y0))
    return 100.0


def similarity(x_ref: np.ndarray, sr_ref: int,
               x_test: np.ndarray, sr_test: int,
               sr_common: int = SR_DEFAULT) -> float:
    """音色相似度 0~100（相对比较用的代理指标）。

    实现说明与两次关键修正：
      特征 = 逐帧跨维 z-score 的帧间均值（"谱形模式"）与标准差。
      1) 相似度函数从**皮尔逊相关**改为**余弦**。
         皮尔逊会做去均值，把 male/female 之间最主要的差异（谱倾斜方向）
         一起减掉——实测"男声 vs 女声"的皮尔逊高达 0.92，
         与"同一人不同相位"的 0.92 几乎无法区分，指标失效。
         余弦对方向敏感，能保留倾斜信息。
      2) 两端先重采样到同一采样率，避免采样率差异污染指标
         （实测：本机 16kHz 与数据集 48kHz 直接比较会引入偏差）。

    ⚠️ 这是**相对比较用**的代理指标，不是严格说话人验证。
    若要更可靠的判据，建议换用预训练说话人嵌入（CAM++ / ECAPA / WavLM-SV）。
    """
    x_ref = _resample(np.asarray(x_ref, dtype=np.float64), sr_ref, sr_common)
    x_test = _resample(np.asarray(x_test, dtype=np.float64), sr_test, sr_common)
    a = timbre_fingerprint(x_ref, sr_common)
    b = timbre_fingerprint(x_test, sr_common)
    n = a.size // 3
    if n < 2:
        return 0.0
    c_pattern = _cosine(a[:n], b[:n])           # 谱形模式（主）
    c_std = _cosine(a[n:2 * n], b[n:2 * n])     # 谱形起伏（辅）
    c = 0.80 * c_pattern + 0.20 * c_std
    return _calibrate(c)


def similarity_detail(x_ref, sr_ref, x_test, sr_test,
                      sr_common: int = SR_DEFAULT) -> dict:
    """给出分项明细，便于判断差异来自哪类特征。"""
    x_ref = _resample(np.asarray(x_ref, dtype=np.float64), sr_ref, sr_common)
    x_test = _resample(np.asarray(x_test, dtype=np.float64), sr_test, sr_common)
    a = timbre_fingerprint(x_ref, sr_common)
    b = timbre_fingerprint(x_test, sr_common)
    n = a.size // 3
    return {
        "谱形模式(余弦)": round(_cosine(a[:n], b[:n]), 4),
        "谱形模式(皮尔逊)": round(_pearson(a[:n], b[:n]), 4),
        "谱形起伏(余弦)": round(_cosine(a[n:2 * n], b[n:2 * n]), 4),
        "总体": round(similarity(x_ref, sr_common, x_test, sr_common), 2),
    }


# ---------------------------------------------------------------- 自检
def _self_test() -> None:
    import io
    import sys

    sys.stdout = io.TextIOWrapper(open("_sim_selftest.txt", "wb"), encoding="utf-8")

    sr = 16000
    rng = np.random.default_rng(0)

    def synth(f0, formants, dur=2.0, seed=0, noise=0.0):
        n = int(dur * sr)
        t = np.arange(n) / sr
        r = np.random.default_rng(seed)
        y = np.zeros(n)
        k = 1
        while f0 * k < sr * 0.45 and k <= 60:
            y += (1.0 / k ** 2) * np.sin(2 * np.pi * f0 * k * t + r.uniform(0, 6.28))
            k += 1
        from timbreforge.engine import svf_bandpass
        out = np.zeros(n)
        for fc, bw in formants:
            out += svf_bandpass(y, fc, bw, sr)
        if noise > 0:
            out = out + noise * r.standard_normal(n)
        return out / (np.max(np.abs(out)) + 1e-9)

    A = synth(120, [(700, 90), (1200, 110), (2600, 140)])          # 说话人 A
    A_same = synth(120, [(700, 90), (1200, 110), (2600, 140)], dur=1.5, seed=9)
    B = synth(210, [(400, 80), (2000, 120), (2900, 150)])          # 说话人 B（完全不同）
    A_shift = synth(120, [(560, 90), (960, 110), (2080, 140)])     # A 但共振峰下移 20%
    A_noisy = synth(120, [(700, 90), (1200, 110), (2600, 140)], noise=0.15)

    print("=== 客观音色相似度自检 ===")
    pairs = [
        ("同一说话人（不同长度/相位）", A, A_same, "应很高 >90"),
        ("不同说话人（音高+共振峰都不同）", A, B, "应明显低"),
        ("同说话人但共振峰下移20%", A, A_shift, "应居中"),
        ("同说话人但加噪声", A, A_noisy, "应较高"),
    ]
    vals = {}
    for name, u, v, expect in pairs:
        s = similarity(u, sr, v, sr)
        vals[name] = s
        print(f"  {name:32s} 相似度={s:6.2f}   ({expect})")

    print("\n=== 明细示例（不同说话人）===")
    for k, v in similarity_detail(A, sr, B, sr).items():
        print(f"  {k:16s} {v}")

    print("\n=== 判据 ===")
    ok = True

    def ck(n, cond, d=""):
        nonlocal ok
        ok &= cond
        print(f"  {'通过' if cond else '失败'}  {n}   {d}")

    ck("同一说话人 > 90", vals["同一说话人（不同长度/相位）"] > 90,
       f"{vals['同一说话人（不同长度/相位）']:.1f}")
    ck("不同说话人 < 同一说话人", vals["不同说话人（音高+共振峰都不同）"]
       < vals["同一说话人（不同长度/相位）"] - 10,
       f"{vals['不同说话人（音高+共振峰都不同）']:.1f}")
    ck("共振峰偏移介于两者之间",
       vals["不同说话人（音高+共振峰都不同）"]
       < vals["同说话人但共振峰下移20%"]
       < vals["同一说话人（不同长度/相位）"],
       f"{vals['同说话人但共振峰下移20%']:.1f}")
    print(f"\n整体: {'指标可用' if ok else '仍需调整'}")
    sys.stdout.flush()
    return ok


if __name__ == "__main__":
    _self_test()
