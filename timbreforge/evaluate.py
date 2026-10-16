"""TimbreForge —— 论文级评估指标（S3）

按照 TTS 论文的常用评估协议组织，并与"AI 生图"的评判体系对应：

    相似度类（= 像不像参考）      : SECS（CAM++ 余弦）
    自然度类（= 电音/气音）        : UTMOS22、DNSMOS(OVRL/SIG/BAK/P808)
    韵律类（= 语气/节奏像不像）    : DTW 韵律相关（F0 与能量轮廓）
    同文本声学差异（不是念字正确率）: experimental MFCC distortion (非标准 MCD)
    诊断类（= 为什么不好）         : 抖动、HNR、谱平坦度

本文件只用 numpy 实现**不需要模型**的部分（韵律 DTW、MCD、诊断），
模型部分（UTMOS/DNSMOS/CAM++）分别在 tools/ 下用 ComfyUI venv 调用。
"""
from __future__ import annotations

import numpy as np

SR_DEFAULT = 16000
N_FFT = 1024
HOP = 256


# ---------------------------------------------------------------- 工具
def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or x.size < 2:
        return x
    n_out = max(int(round(x.size * sr_out / sr_in)), 2)
    idx = np.linspace(0, x.size - 1, n_out)
    i0 = np.floor(idx).astype(np.int64)
    i1 = np.clip(i0 + 1, 0, x.size - 1)
    f = idx - i0
    return x[i0] * (1 - f) + x[i1] * f


def _mono(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x


# ---------------------------------------------------------------- F0 轨迹
def f0_track(x: np.ndarray, sr: int, fmin: float = 60.0,
             fmax: float = 500.0, hop_ms: float = 10.0) -> np.ndarray:
    """逐帧基频（自相关 + "取第一个高峰"判据），无声帧为 nan。

    复用 voicefx 里已验证的判据：真周期是最小的高自相关峰。
    """
    x = _mono(x)
    hop = max(int(sr * hop_ms / 1000), 1)
    win = max(int(sr * 0.04), 64)          # 40ms 窗
    n = max((x.size - win) // hop + 1, 1)
    out = np.full(n, np.nan)
    for i in range(n):
        seg = x[i * hop:i * hop + win]
        if seg.size < win:
            break
        seg = seg - seg.mean()
        if np.max(np.abs(seg)) < 1e-6:
            continue
        # 降采样提速
        dec = max(int(sr / 12000), 1)
        y = seg[::dec]
        sr2 = sr / dec
        y = y * np.hanning(y.size)
        m = y.size
        spec = np.fft.rfft(y, 2 * m)
        ac = np.fft.irfft(spec * np.conj(spec))[:m]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        lo = max(int(sr2 / fmax), 2)
        hi = min(int(sr2 / fmin), m - 1)
        if hi <= lo + 1:
            continue
        reg = ac[lo:hi]
        pk = float(np.max(reg))
        if pk < 0.3:
            continue
        thr = 0.9 * pk
        lag = None
        for j in range(1, reg.size - 1):
            if reg[j] >= thr and reg[j] >= reg[j - 1] and reg[j] >= reg[j + 1]:
                lag = lo + j
                break
        if lag is None:
            lag = int(np.argmax(reg)) + lo
        out[i] = sr2 / lag
    return out


def energy_track(x: np.ndarray, sr: int, hop_ms: float = 10.0) -> np.ndarray:
    """逐帧对数能量。"""
    x = _mono(x)
    hop = max(int(sr * hop_ms / 1000), 1)
    n = max(x.size // hop, 1)
    seg = x[:n * hop].reshape(n, hop)
    e = np.sqrt(np.mean(seg ** 2, axis=1) + 1e-12)
    return 20.0 * np.log10(e + 1e-9)


# ---------------------------------------------------------------- DTW
def dtw(a: np.ndarray, b: np.ndarray, band: int | None = None):
    """动态时间规整（纯 numpy）。

    返回 (归一化距离, 路径)。使用 Sakoe-Chiba 带以限制搜索范围。
    为控制内存，用滚动两行 DP + 回溯指针。
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    n, m = a.size, b.size
    if n == 0 or m == 0:
        return float("inf"), []
    if band is None:
        band = max(int(0.3 * max(n, m)), 8)
    INF = 1e18
    prev = np.full(m + 1, INF)
    prev[0] = 0.0
    back = np.zeros((n + 1, m + 1), dtype=np.int8)   # 0=对角 1=上 2=左
    for i in range(1, n + 1):
        cur = np.full(m + 1, INF)
        j_lo = max(1, i - band)
        j_hi = min(m, i + band)
        for j in range(j_lo, j_hi + 1):
            cost = abs(a[i - 1] - b[j - 1])
            d = prev[j - 1]
            u = prev[j]
            l = cur[j - 1]
            if d <= u and d <= l:
                cur[j] = cost + d
                back[i, j] = 0
            elif u <= l:
                cur[j] = cost + u
                back[i, j] = 1
            else:
                cur[j] = cost + l
                back[i, j] = 2
        prev = cur
    dist = float(prev[m])
    # 回溯
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        d = back[i, j]
        if d == 0:
            i -= 1
            j -= 1
        elif d == 1:
            i -= 1
        else:
            j -= 1
    path.reverse()
    return dist / max(n, m), path


def _interp_nan(v: np.ndarray) -> np.ndarray:
    """线性插值填补 nan；全是 nan 则返回等长零。"""
    v = np.asarray(v, dtype=np.float64)
    ok = np.isfinite(v)
    if ok.sum() < 2:
        return np.zeros_like(v)
    idx = np.arange(v.size)
    return np.interp(idx, idx[ok], v[ok])


def prosody_correlation(ref: np.ndarray, test: np.ndarray,
                        sr_ref: int, sr_test: int) -> dict:
    """DTW 对齐后的 F0 与能量轮廓相关系数（论文里常报的 prosody 指标）。

    ⚠️ 说明：论文多用"person correlation of DTW-aligned F0"。
    这里对齐后计算皮尔逊相关（0~1 越高越像），并给出归一化 DTW 距离。
    """
    a = _resample(_mono(ref), sr_ref, SR_DEFAULT)
    b = _resample(_mono(test), sr_test, SR_DEFAULT)

    out: dict = {}
    for name, track_fn in (("f0", f0_track), ("energy", energy_track)):
        ta = track_fn(a, SR_DEFAULT)
        tb = track_fn(b, SR_DEFAULT)
        ta_i = _interp_nan(ta)
        tb_i = _interp_nan(tb)
        dist, path = dtw(ta_i, tb_i)
        if path:
            pa = np.array([ta_i[i] for i, _ in path])
            pb = np.array([tb_i[j] for _, j in path])
            if pa.std() > 1e-9 and pb.std() > 1e-9:
                r = float(np.corrcoef(pa, pb)[0, 1])
            else:
                r = 0.0
        else:
            r = 0.0
        out[f"{name}_corr"] = round(r, 4)
        out[f"{name}_dtw"] = round(dist, 4)
        # 诊断：静音/无声帧比例（nan 比例）
        out[f"{name}_unvoiced_ratio"] = round(
            float(np.mean(~np.isfinite(ta))) if ta.size else 0.0, 4)
    return out


# ---------------------------------------------------------------- MCD
def mfcc_seq(x: np.ndarray, sr: int, n_mfcc: int = 13,
             n_fft: int = N_FFT, hop: int = HOP, n_mels: int = 40) -> np.ndarray:
    """MFCC 序列（用于 MCD）。"""
    x = _mono(x)
    if x.size < n_fft * 2:
        x = np.pad(x, (0, n_fft * 2 - x.size))
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])
    n = 1 + (x.size - n_fft) // hop
    win = np.hamming(n_fft)
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    frames = x[idx] * win
    spec = np.abs(np.fft.rfft(frames, n_fft)) ** 2
    # 梅尔滤波器组
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    mels = np.linspace(hz2mel(40.0), hz2mel(sr / 2), n_mels + 2)
    bins = np.clip(np.floor((n_fft + 1) * mel2hz(mels) / sr).astype(int),
                   0, n_fft // 2)
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
    mel = np.log(spec @ fb.T + 1e-10)
    k = np.arange(n_mfcc)[:, None]
    nn = np.arange(n_mels)[None, :]
    dct = np.cos(np.pi * k * (2 * nn + 1) / (2 * n_mels)) * np.sqrt(2.0 / n_mels)
    return mel @ dct.T


def mcd(ref: np.ndarray, test: np.ndarray, sr_ref: int, sr_test: int) -> float:
    """Experimental MFCC distortion (legacy MCD-shaped number), lower is better.

    This is not a content-recognition metric and must not be interpreted as CER/WER.
    It only describes acoustic difference when text and alignment are comparable.
    """
    a = mfcc_seq(_resample(_mono(ref), sr_ref, SR_DEFAULT), SR_DEFAULT)[:, 1:13]
    b = mfcc_seq(_resample(_mono(test), sr_test, SR_DEFAULT), SR_DEFAULT)[:, 1:13]
    if a.size == 0 or b.size == 0:
        return float("nan")
    # 用 DTW 对齐（对齐后比较才有意义）
    dist, path = dtw(a[:, 0], b[:, 0])
    if not path:
        L = min(a.shape[0], b.shape[0])
        diff = a[:L] - b[:L]
    else:
        pa = np.array([a[i] for i, _ in path])
        pb = np.array([b[j] for _, j in path])
        diff = pa - pb
    val = (10.0 / np.log(10.0)) * np.sqrt(2.0 * np.mean(diff ** 2))
    return float(val)


# ---------------------------------------------------------------- 诊断
def diagnostics(x: np.ndarray, sr: int) -> dict:
    """自然度辅助诊断：音高抖动、能量抖动、HNR、谱平坦度、高频占比。

    ⚠️ 关于"颤抖"：最初只测短时**能量**抖动，实测识别不出颤抖
    （人工注入 4.5Hz 音高颤抖后，能量抖动的变化方向都不对）。
    原因是听感上的"颤抖"主要是**音高抖动**（jitter/vibrato），
    因此这里同时给出 f0_jitter（逐帧音高的相对起伏）与 tremor（能量抖动）。

    这些不是论文标准指标，但能解释"为什么听起来电音/发虚/发抖"。
    """
    x = _mono(x)
    if x.size < 512:
        return {}
    # 短时能量抖动
    hop = max(int(sr * 0.01), 1)
    n = max(x.size // hop, 1)
    e = np.sqrt(np.mean(x[:n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    v = e > 0.05 * np.max(e)
    ev = e[v] if v.sum() >= 8 else e
    tremor = float(np.std(np.diff(ev)) / (np.mean(ev) + 1e-9))

    # 音高抖动：逐帧 f0 的相对起伏（有声帧）
    ft = f0_track(x, sr)
    fv = ft[np.isfinite(ft)]
    if fv.size >= 5 and np.mean(fv) > 0:
        f0_jitter = float(np.std(fv) / np.mean(fv))
        # 一阶差分抖动（更接近"发抖"而非"颤音"）
        f0_flutter = float(np.mean(np.abs(np.diff(fv))) / np.mean(fv))
    else:
        f0_jitter = f0_flutter = 0.0

    # HNR 近似：自相关最大值
    y = x - x.mean()
    m = min(y.size, int(sr * 1.0))
    y = y[:m] * np.hanning(m)
    spec = np.fft.rfft(y, 2 * m)
    ac = np.fft.irfft(spec * np.conj(spec))[:m]
    ac = ac / (ac[0] + 1e-12)
    lo = max(int(sr / 500), 1)
    hi = min(int(sr / 60), m - 1)
    r_max = float(np.max(ac[lo:hi])) if hi > lo else 0.0
    r_max = float(np.clip(r_max, 1e-4, 0.999))
    hnr_db = 10.0 * np.log10(r_max / (1.0 - r_max))

    # 谱平坦度与高频占比
    mag = np.abs(np.fft.rfft(x * np.hanning(x.size))) + 1e-12
    freq = np.fft.rfftfreq(x.size, 1.0 / sr)
    flat = float(np.exp(np.mean(np.log(mag))) / np.mean(mag))
    hf = float(mag[freq > 4000].sum() / mag.sum())
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x)))
    # 近似"气声"：高频占比高 + HNR 低
    breathy = float(np.clip(hf * 3.0, 0, 1) * np.clip(1.0 - (hnr_db + 5) / 25.0, 0, 1))
    return {
        "f0_jitter": round(f0_jitter, 4),
        "f0_flutter": round(f0_flutter, 4),
        "tremor": round(tremor, 4),
        "hnr_db": round(hnr_db, 2),
        "flatness": round(flat, 5),
        "hf_ratio": round(hf, 4),
        "rms": round(rms, 4),
        "peak": round(peak, 4),
        "breathy_score": round(breathy, 3),
    }


# ---------------------------------------------------------------- 汇总
def evaluate(ref_path: str, test_path: str, use_models: bool = False,
             device: str = "cpu") -> dict:
    """对一对音频做完整评估（模型部分可选）。"""
    import soundfile as sf

    xr, srr = sf.read(ref_path)
    xt, srt = sf.read(test_path)
    xr, xt = _mono(xr), _mono(xt)

    res: dict = {"ref": ref_path, "test": test_path}
    res.update(prosody_correlation(xr, xt, srr, srt))
    res["mcd"] = round(mcd(xr, xt, srr, srt), 3)
    res.update(diagnostics(xt, srt))

    if use_models:
        try:
            from timbreforge.mosmodels import score_naturalness, score_quality
            res.update(score_naturalness(test_path, device=device))
            res.update(score_quality(test_path, device=device))
        except Exception as e:
            res["model_error"] = f"{type(e).__name__}: {str(e)[:120]}"
    return res


def _self_test() -> None:
    import io
    import sys
    import tempfile

    import soundfile as sf

    sys.stdout = io.TextIOWrapper(open("_eval_selftest.txt", "wb"), encoding="utf-8")

    sr = 16000
    rng = np.random.default_rng(0)

    def voice(f0, dur=2.0, jitter=0.0, seed=0, tilt=0.0):
        n = int(dur * sr)
        t = np.arange(n) / sr
        r = np.random.default_rng(seed)
        # 带轻微音高起伏的谐波源
        f = f0 * (1.0 + jitter * np.sin(2 * np.pi * 4.5 * t))
        ph = np.cumsum(f) / sr
        y = np.zeros(n)
        for k in range(1, 40):
            if f0 * k > sr * 0.45:
                break
            y += (1.0 / k ** 2) * np.sin(2 * np.pi * k * ph)
        env = np.minimum(t / 0.05, 1.0) * np.minimum((dur - t) / 0.2, 1.0)
        y = y * np.clip(env, 0, 1)
        if tilt:
            y = y + tilt * r.standard_normal(n)
        return y / (np.max(np.abs(y)) + 1e-9)

    td = tempfile.mkdtemp()
    import os
    A = voice(150, 2.0, jitter=0.02, seed=1)
    A2 = voice(150, 2.0, jitter=0.02, seed=2)          # 同人不同段
    B = voice(220, 2.0, jitter=0.02, seed=3)           # 不同人
    A_trem = voice(150, 2.0, jitter=0.25, seed=1)      # 强烈颤抖
    A_noise = voice(150, 2.0, jitter=0.02, seed=1, tilt=0.3)

    paths = {}
    for name, sig in (("A", A), ("A2", A2), ("B", B),
                      ("A_trem", A_trem), ("A_noise", A_noise)):
        p = os.path.join(td, f"{name}.wav")
        sf.write(p, sig.astype(np.float32), sr)
        paths[name] = p

    print("=== 指标自检 ===")
    print(f"{'对照':16s} {'f0相关':>8s} {'f0_DTW':>8s} {'能量相关':>9s} "
          f"{'MCD':>8s} {'f0抖动':>8s} {'f0颤动':>8s} {'HNR':>7s}")
    print("-" * 86)
    rows = {}
    for name, p in (("同人不同段", paths["A2"]), ("不同音高", paths["B"]),
                    ("强烈颤抖", paths["A_trem"]), ("加噪声", paths["A_noise"])):
        r = evaluate(paths["A"], p)
        rows[name] = r
        print(f"{name:16s} {r['f0_corr']:8.3f} {r['f0_dtw']:8.3f} "
              f"{r['energy_corr']:9.3f} {r['mcd']:8.2f} {r['f0_jitter']:8.4f} "
              f"{r['f0_flutter']:8.4f} {r['hnr_db']:7.2f}")

    # 自己 vs 自己（上界）
    r_self = evaluate(paths["A"], paths["A"])
    print(f"{'自身（上界）':16s} {r_self['f0_corr']:8.3f} {r_self['f0_dtw']:8.3f} "
          f"{r_self['energy_corr']:9.3f} {r_self['mcd']:8.2f} "
          f"{r_self['f0_jitter']:8.4f} {r_self['f0_flutter']:8.4f} "
          f"{r_self['hnr_db']:7.2f}")

    print("\n=== 判据 ===")
    ok = True

    def ck(n, c, d=""):
        nonlocal ok
        ok &= c
        print(f"  {'通过' if c else '失败'}  {n}   {d}")

    ck("自身 MCD 最低", r_self["mcd"] <= min(
        rows[k]["mcd"] for k in rows) + 1e-6, f"{r_self['mcd']:.2f}")
    ck("同人 f0 相关 > 不同音高",
       rows["同人不同段"]["f0_corr"] >= rows["不同音高"]["f0_corr"],
       f"{rows['同人不同段']['f0_corr']:.3f} vs {rows['不同音高']['f0_corr']:.3f}")
    ck("颤抖样本的 f0 抖动最高",
       rows["强烈颤抖"]["f0_jitter"] >= max(
           rows[k]["f0_jitter"] for k in rows),
       f"{rows['强烈颤抖']['f0_jitter']:.4f} vs 其余 "
       f"{max(rows[k]['f0_jitter'] for k in rows if k != '强烈颤抖'):.4f}")
    ck("抖动指标能区分颤抖与正常",
       rows["强烈颤抖"]["f0_jitter"] > rows["同人不同段"]["f0_jitter"] * 1.5,
       f"{rows['强烈颤抖']['f0_jitter']:.4f} vs {rows['同人不同段']['f0_jitter']:.4f}")
    ck("噪声样本的 HNR 更低",
       rows["加噪声"]["hnr_db"] < rows["同人不同段"]["hnr_db"],
       f"{rows['加噪声']['hnr_db']:.2f} vs {rows['同人不同段']['hnr_db']:.2f}")
    print(f"\n整体: {'指标可用' if ok else '仍需调整'}")
    sys.stdout.flush()


if __name__ == "__main__":
    _self_test()
