"""TimbreForge —— 感知特征层（P0-8）

把音频映射到**可解释的听感维度**，用途：
  1. 验证"宏真的改变了听感"（而不是随机出声）
  2. 给优化式反解提供目标函数（P2）
  3. 输出人类可读的音色画像，方便人工核对

注意：这些是**工程近似特征**，不等于真实主观听感；只作粗筛，
      最终判断仍以人工 A/B 试听为准（见风险 R6）。
"""
from __future__ import annotations

import numpy as np

FEATURE_NAMES = (
    "brightness",   # 频谱质心（归一化 kHz）
    "rolloff",      # 85% 能量滚降点（kHz）
    "noisiness",    # 谱平坦度（噪声性/粗糙度）
    "inharmonic",   # 非谐波性（金属感/钟感）
    "attack",       # 起音速度（瞬态锐度）
    "sustain",      # 持续段占比（能否拉长音）
    "motion",       # 调制活动量（颤抖/流动）
)

EPS = 1e-12


def _spectrum(x: np.ndarray, sr: float):
    w = np.hanning(x.size)
    mag = np.abs(np.fft.rfft(x * w)) + EPS
    freq = np.fft.rfftfreq(x.size, 1.0 / sr)
    return mag, freq


def _envelope(x: np.ndarray, sr: float, hop: int = 256) -> np.ndarray:
    frames = max(x.size // hop, 1)
    trimmed = x[:frames * hop].reshape(frames, hop)
    return np.sqrt(np.mean(trimmed ** 2, axis=1) + EPS)


def _estimate_f0_cepstrum(x: np.ndarray, sr: float, fmin: float = 60.0,
                          fmax: float = 1000.0) -> float:
    """倒谱法估基频。

    为什么不用自相关：人声元音有 3 个高 Q 共振峰，自相关会被共振"振铃"
    带偏（实测 226Hz 被估成 330Hz、120Hz 被估成 430Hz）。
    倒谱把"声道包络"和"声门周期"分离，对共振峰不敏感。
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    if x.size < 512 or np.max(np.abs(x)) < 1e-9:
        return 0.0
    n = x.size
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(x * win, 2 * n)) + 1e-12
    logspec = np.log(spec)
    cep = np.fft.irfft(logspec)
    # 在合理的基频周期范围里找倒谱峰
    lo = max(int(sr / fmax), 1)
    hi = min(int(sr / fmin), cep.size - 1)
    if hi <= lo + 1:
        return 0.0
    seg = cep[lo:hi]
    lag = int(np.argmax(seg)) + lo
    if 0 < lag < cep.size - 1:
        a, b, c = cep[lag - 1], cep[lag], cep[lag + 1]
        den = (a - 2 * b + c)
        if abs(den) > 1e-12:
            lag = lag + 0.5 * (a - c) / den
    f0 = sr / lag if lag > 0 else 0.0
    if not (fmin <= f0 <= fmax):
        return 0.0
    # 峰显著性：倒谱峰的相对高度（过低说明无周期性）
    med = float(np.median(seg))
    peak_val = float(cep[int(round(lag))])
    if peak_val <= 0 or (peak_val - med) < 0.02 * (abs(peak_val) + 1e-9):
        # 允许较弱的情形，但排除完全无峰
        if peak_val <= 1e-6:
            return 0.0
    return float(f0)


def _acf_f0(x: np.ndarray, sr: float, fmin: float, fmax: float) -> float:
    """自相关 + **候选基频评分**估基频（人声稳健版）。

    为什么不用"取最强峰 + 倍频纠错"：自相关的最强峰几乎总是出现在
    1 个真实周期处，向长周期纠错必然把 110Hz 纠成 55Hz（实测踩过）。

    改用有明确判据的候选评分：
      对每个候选 f0，把它各次谐波对应的 lag 处的自相关值累加起来评分，
      再对"谐波太少的低音候选"做轻微惩罚，取分最高者。
    真基频的谐波序列在同一信号里必然都有较强自相关，而
    "倍频候选"（2*f0）只有一半谐波能对上，因此会被区分开。
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    if not np.isfinite(x).all() or np.max(np.abs(x)) < 1e-9:
        return 0.0
    dec = max(int(sr / 12000), 1)
    y = x[::dec]
    sr2 = sr / dec
    y = y * np.hanning(y.size)
    n = y.size
    if n < 256:
        return 0.0
    spec = np.fft.rfft(y, 2 * n)
    ac = np.fft.irfft(spec * np.conj(spec))[:n]
    if ac[0] <= 0:
        return 0.0
    ac = ac / ac[0]

    def ac_at(freq: float) -> float:
        lag = sr2 / freq
        i = int(round(lag))
        if i < 1 or i >= n - 1:
            return 0.0
        # 取 lag 邻域最大值（容忍一点周期误差）
        a = max(0, i - 2)
        b = min(n, i + 3)
        return float(np.max(ac[a:b]))

    best_f, best_score = 0.0, -1e9
    f = fmin
    while f <= fmax:
        s = 0.0
        k = 1
        while f * k <= sr2 * 0.4 and k <= 25:
            s += ac_at(f * k) / (k ** 0.5)
            k += 1
        # 低音候选容易靠"更长的周期"占便宜，做轻微惩罚
        score = s - 0.15 * np.log2(max(fmax / f, 1.0))
        if score > best_score:
            best_score, best_f = score, f
        f *= 1.0025          # 0.25% 步进扫描
    if best_f <= 0 or best_score <= 0.2:
        return 0.0
    return float(best_f)


def _estimate_f0(x: np.ndarray, sr: float, fmin: float = 40.0,
                 fmax: float = 2000.0) -> float:
    """基频估计：自相关+倍频纠错为主，倒谱兜底。"""
    f = _acf_f0(x, sr, max(fmin, 50.0), min(fmax, 1000.0))
    if f > 0:
        return f
    return _estimate_f0_cepstrum(x, sr, fmin=max(fmin, 60.0), fmax=min(fmax, 1000.0))


def _inharmonicity(x: np.ndarray, sr: float, mag: np.ndarray,
                   freq: np.ndarray) -> float:
    """非谐波性 0~1（0=谐波乐音，1=非谐波/噪声/金属感）。

    实现要点（踩过两个坑后定稿）：
      * 不能只看"能量是否落在某 f0 的整数倍上"——低通锯齿会因为没有高频谐波被判成 1.0
      * 不能依赖单次自相关估基频——纯正弦会被估成低八度，导致判据反转
    做法：取谱峰，然后**枚举候选基频**（各谱峰本身 + 它们的间隔），
          选让"峰与整数倍偏差"最小的那个；再结合谱平坦度处理噪声情形。
    """
    # ---- 谱平坦度：噪声的主导判据 ----
    flat = float(np.exp(np.mean(np.log(mag + EPS))) / (np.mean(mag) + EPS))

    # ---- 找谱峰 ----
    df = float(freq[1] - freq[0]) if freq.size > 1 else 1.0
    m = mag.copy()
    m[:max(int(50 / df), 1)] = 0.0
    if m.size < 16:
        return 1.0
    idx = np.arange(2, m.size - 2)
    local = (m[idx] > m[idx - 1]) & (m[idx] > m[idx + 1]) \
        & (m[idx] > m[idx - 2]) & (m[idx] > m[idx + 2])
    peak_idx = idx[local]
    if peak_idx.size < 3:
        # 谱太平 -> 噪声类
        return float(np.clip(flat * 3.0, 0.0, 1.0))
    order = np.argsort(m[peak_idx])[::-1][:24]
    pf = np.sort(freq[peak_idx[order]])

    # ---- 枚举候选基频 ----
    cands = set()
    for f in pf:
        for div in (1, 2, 3, 4):
            cands.add(f / div)
    d = np.diff(pf)
    for g in d[d > 20]:
        cands.add(float(g))
    ac = _estimate_f0(x, sr)
    if ac > 0:
        cands.add(ac)

    best_dev = 1.0
    best_frac = 0.0
    for f0 in cands:
        if f0 < 30 or f0 > sr * 0.45:
            continue
        ratios = pf / f0
        dev = np.abs(ratios - np.round(ratios))
        # 只统计合理范围内的谐波次号，避免超大次号的数值噪声
        valid = ratios <= 40
        if valid.sum() < 2:
            continue
        dev_v = dev[valid]
        mean_dev = float(np.mean(dev_v))
        frac_aligned = float(np.mean(dev_v < 0.04))
        if mean_dev < best_dev:
            best_dev = mean_dev
            best_frac = frac_aligned

    # ---- 合成非谐波性 ----
    # 峰对齐得越整齐 -> 越谐波；平坦度越高 -> 越像噪声
    tonal = float(np.clip(1.0 - best_dev * 6.0, 0.0, 1.0))
    noisy = float(np.clip(flat * 3.0, 0.0, 1.0))
    inharm = 1.0 - (0.75 * tonal + 0.25 * (1.0 - noisy))
    return float(np.clip(inharm, 0.0, 1.0))


def extract(x: np.ndarray, sr: float = 48000.0) -> np.ndarray:
    """提取感知特征向量（按 FEATURE_NAMES 顺序）。"""
    x = np.asarray(x, dtype=np.float64)
    if x.size < 512:
        x = np.pad(x, (0, 512 - x.size))
    mag, freq = _spectrum(x, sr)
    total = mag.sum()

    centroid = float((freq * mag).sum() / total)
    cumsum = np.cumsum(mag)
    roll_idx = int(np.searchsorted(cumsum, 0.85 * cumsum[-1]))
    roll_idx = min(roll_idx, freq.size - 1)
    rolloff = float(freq[roll_idx])

    flatness = float(np.exp(np.mean(np.log(mag))) / np.mean(mag))

    # 非谐波性：先用自相关估基频，再看谱峰与它的整数倍对齐程度。
    # 早期用"第一个谱峰当基频"，遇到被低通削掉基频的锯齿波会误判成金属感，
    # 实测"温暖低音"曾错误得到 0.59，故改为自相关估基频。
    inharmonic = _inharmonicity(x, sr, mag, freq)

    env = _envelope(x, sr)
    if env.size >= 4:
        peak_i = int(np.argmax(env))
        attack = float(1.0 / (peak_i + 2))
        head = env[:max(peak_i + 2, 3)]
        # 持续段：后半段平均值 / 峰值
        tail = env[env.size // 2:]
        sustain = float(np.clip(np.mean(tail) / (env.max() + EPS), 0.0, 1.0))
        # 调制活动：包络的一阶差分能量（相对）
        d = np.diff(env)
        motion = float(np.clip(np.std(d) / (np.mean(env) + EPS), 0.0, 3.0) / 3.0)
        _ = head
    else:
        attack, sustain, motion = 0.5, 0.5, 0.0

    return np.array([
        centroid / 1000.0,
        rolloff / 1000.0,
        flatness,
        inharmonic,
        attack,
        sustain,
        motion,
    ], dtype=np.float64)


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """特征空间欧氏距离（已按各自量级缩放，便于比较）。"""
    scale = np.array([2.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    return float(np.sqrt(np.sum(((a - b) * scale) ** 2)))


def profile_text(feats: np.ndarray) -> str:
    """生成人类可读的音色画像。"""
    d = dict(zip(FEATURE_NAMES, feats))

    def band(v, lo, mid):
        return "低" if v < lo else ("中" if v < mid else "高")

    return (
        f"亮度={band(d['brightness'], 0.8, 2.5)}({d['brightness']:.2f}kHz)  "
        f"高频延展={band(d['rolloff'], 2, 6)}({d['rolloff']:.1f}kHz)  "
        f"噪感={band(d['noisiness'], 0.05, 0.2)}({d['noisiness']:.3f})  "
        f"金属感={band(d['inharmonic'], 0.25, 0.50)}({d['inharmonic']:.2f})  "
        f"起音={'快' if d['attack'] > 0.25 else '慢'}({d['attack']:.2f})  "
        f"持续={band(d['sustain'], 0.15, 0.45)}({d['sustain']:.2f})  "
        f"流动={band(d['motion'], 0.15, 0.35)}({d['motion']:.2f})"
    )


VOICE_FEATURE_NAMES = (
    "f0",        # 基频（Hz）—— 决定音高与性别
    "f1",        # 第一共振峰（Hz）—— 决定开口度
    "f2",        # 第二共振峰（Hz）—— 决定前后/元音身份
    "hfr",       # 高频占比 —— 气声/明亮度
    "vib",       # 颤音深度（相对）—— 流动感
    "hnr",       # 谐噪比近似 —— 沙哑/气声程度
)


def _spectral_envelope(x: np.ndarray, sr: float):
    """平滑谱包络，用于稳定地找共振峰。"""
    mag, freq = _spectrum(x, sr)
    # 对数域 + 移动平均平滑
    logmag = np.log(mag)
    win = max(int(120.0 / (freq[1] - freq[0])), 3)
    kern = np.ones(win) / win
    smooth = np.convolve(logmag, kern, mode="same")
    return smooth, freq


def extract_voice(x: np.ndarray, sr: float = 48000.0) -> np.ndarray:
    """人声专用特征：基频 + 共振峰 + 气声 + 颤音。

    共振峰提取的坑：谱包络上"相邻两个点此起彼伏"会被当成两个峰，
    实测得到 F1=673 / F2=675 这种明显错误的结果。所以这里用
    **峰值突出度(prominence)筛峰 + 最小间隔约束**，保证 F1/F2 真正分开。
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size < 1024:
        x = np.pad(x, (0, 1024 - x.size))

    f0 = _estimate_f0(x, sr, fmin=60.0, fmax=600.0)

    f1, f2 = _find_formants(x, sr)

    mag, fr = _spectrum(x, sr)
    hfr = float(mag[fr > 4000].sum() / (mag.sum() + EPS))

    # 颤音：用基频轨迹相对起伏估计（只取有声段）
    hop = int(0.05 * sr)
    f0s = []
    for start in range(0, max(x.size - hop, 1), hop):
        segx = x[start:start + hop]
        if segx.size >= 512:
            seg_rms = float(np.sqrt(np.mean(segx ** 2)))
            if seg_rms < 1e-4:
                continue
            f = _estimate_f0(segx, sr, fmin=60.0, fmax=600.0)
            if f > 0:
                f0s.append(f)
    if len(f0s) >= 3 and np.mean(f0s) > 0:
        # 归一化到 0~1：5% 的起伏就算很明显
        vib = float(np.clip(np.std(f0s) / np.mean(f0s) / 0.05, 0.0, 1.0))
    else:
        vib = 0.0

    # 谐噪比近似：谱峰能量 / 总能量（越高越"实"，越低越"气"）
    peak_energy = float(np.sort(mag)[-40:].sum()) if mag.size > 40 else float(mag.sum())
    hnr = float(peak_energy / (mag.sum() + EPS))

    return np.array([f0, f1, f2, hfr, vib, hnr], dtype=np.float64)


def _lpc_coeffs(x: np.ndarray, order: int = 14) -> np.ndarray:
    """Levinson-Durbin 解 LPC 系数（纯 numpy，不依赖 scipy）。

    LPC 是人声共振峰估计的标准做法：它把声道建模成全极点滤波器，
    比"直接在频谱上挑峰"稳健得多（谐波干扰不会让它跑偏）。
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    if np.max(np.abs(x)) < 1e-9:
        return np.array([1.0])
    # 预加重，补偿声门谱的 -12dB/oct 倾斜
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])
    x = x * np.hanning(x.size)
    n = x.size
    r = np.correlate(x, x, mode="full")[n - 1:n + order]
    if r[0] <= 0:
        return np.array([1.0])
    r = r / r[0]
    a = np.zeros(order + 1)
    a[0] = 1.0
    err = 1.0
    for i in range(1, order + 1):
        acc = r[i] + np.dot(a[1:i], r[i - 1:0:-1])
        k = -acc / (err + 1e-12)
        k = float(np.clip(k, -1.0, 1.0))
        prev = a.copy()
        a[i] = k
        for j in range(1, i):
            a[j] = prev[j] + k * prev[i - j]
        err *= (1.0 - k * k)
        if err <= 1e-12:
            break
    return a


def _lpc_formants(x: np.ndarray, sr: float, order: int = 14,
                  f_hi: float = 4500.0) -> list[tuple[float, float]]:
    """用 LPC 谱找共振峰，返回 [(频率, 带宽)]，按频率升序。

    性能要点：1/|A(z)| 的谱直接用一次 FFT 求（A(z) 的系数就是 FFT 输入），
    早期用 Python 循环逐点算 4096 个复数点，慢到不可用。
    """
    a = _lpc_coeffs(x, order=order)
    if a.size < 3:
        return []
    nfft = 4096
    spec = np.fft.rfft(a, nfft)
    mag = 1.0 / (np.abs(spec) + 1e-12)
    freq = np.fft.rfftfreq(nfft, 1.0 / sr)

    peaks: list[tuple[float, float]] = []
    logmag = np.log(mag + 1e-12)
    for i in range(1, logmag.size - 1):
        if logmag[i] > logmag[i - 1] and logmag[i] >= logmag[i + 1] and freq[i] < f_hi:
            y0, y1, y2 = logmag[i - 1], logmag[i], logmag[i + 1]
            den = (y0 - 2 * y1 + y2)
            delta = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
            df = freq[1] - freq[0]
            f_peak = freq[i] + delta * df
            # 带宽：-3dB 半功率点间距的近似
            half = y1 - np.log(2.0)
            j = i
            while j > 0 and logmag[j] > half:
                j -= 1
            j2 = i
            while j2 < logmag.size - 1 and logmag[j2] > half:
                j2 += 1
            bw = float(freq[j2] - freq[j])
            peaks.append((float(f_peak), max(bw, 30.0)))
    return peaks


def _find_formants(x: np.ndarray, sr: float) -> tuple[float, float]:
    """从 LPC 谱里确定 F1 / F2。

    用"与元音原型表最匹配"的方式选峰：先取 LPC 全部候选共振峰，
    再与 a/e/i/o/u 的原型（按性别大致缩放）比对，取误差最小的一组。
    这比"挑最突出的两个峰"稳定得多——高元音 /i/ 的 F2 远强于 F1，
    按强度挑必然选错（实测踩过：得到 2148/2329）。
    """
    peaks = _lpc_formants(x, sr)
    if not peaks:
        return 0.0, 0.0
    freqs = [f for f, _ in peaks if f > 150.0]
    if len(freqs) < 2:
        return (freqs[0] if freqs else 0.0), 0.0

    # 元音原型（男声基准），按 F1 尺度粗略适配不同音区
    proto = {
        "a": (730, 1090), "e": (530, 1840), "i": (270, 2290),
        "o": (570, 840), "u": (300, 870), "v": (440, 1020),
    }
    scale = 1.0
    f1_med = float(np.median(freqs[:3]))
    if f1_med > 0:
        # 用最接近的中低共振峰判断音区尺度
        scale = float(np.clip(f1_med / 600.0, 0.7, 1.6))

    best = None
    best_err = 1e18
    for i in range(len(freqs)):
        for j in range(i + 1, len(freqs)):
            f1c, f2c = freqs[i], freqs[j]
            if f2c - f1c < 120.0:
                continue
            for _, (p1, p2) in proto.items():
                e = ((f1c - p1 * scale) / 400.0) ** 2 + \
                    ((f2c - p2 * scale) / 900.0) ** 2
                if e < best_err:
                    best_err = e
                    best = (f1c, f2c)
    if best is None:
        fs = sorted(freqs)[:2]
        return float(fs[0]), float(fs[1]) if len(fs) > 1 else 0.0
    return float(best[0]), float(best[1])


def profile_voice_text(feats: np.ndarray) -> str:
    d = dict(zip(VOICE_FEATURE_NAMES, feats))
    return (
        f"基频={d['f0']:.0f}Hz  "
        f"F1={d['f1']:.0f}Hz  F2={d['f2']:.0f}Hz  "
        f"高频占比={d['hfr']:.3f}  颤音={d['vib']:.2f}  谐噪比={d['hnr']:.3f}"
    )


def compare(a: np.ndarray, b: np.ndarray) -> str:
    """逐维对比两个音色。"""
    lines = []
    for i, name in enumerate(FEATURE_NAMES):
        delta = b[i] - a[i]
        arrow = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "=")
        lines.append(f"  {name:11s} {a[i]:8.4f} -> {b[i]:8.4f}  {arrow}")
    lines.append(f"  距离 = {distance(a, b):.4f}")
    return "\n".join(lines)
