"""TimbreForge —— SD 式批量抽卡与自动挑选（S2）

对标 AI 生图的「一次出 N 张 → 自动挑最好的」：

    一句声线描述 + 种子 + 参数
        → 批量合成 N 条候选（换种子 / 换参考片段 / 换参数变体）
        → 用 CAM++ 衡量参考说话人相似度（不是文字-声音匹配度）
        → 按分排序，自动选出最佳，并输出网格清单

这样**不需要逐条听评**：只看排序结果，或只试听前三名。
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timbreforge.backend_indextts import SynthParams

import numpy as np
import soundfile as sf


@dataclass
class Variant:
    """一个候选配置。"""
    label: str
    seed: int = 42
    ref_seconds: float = 6.0
    num_beams: int = 1
    do_sample: bool = True
    temperature: float = 0.8
    duration_factor: float = 1.0
    emotion_weight: float = 0.65
    extra: dict = field(default_factory=dict)


def default_variants(base_seed: int = 42, n: int = 8,
                     vary: tuple[str, ...] = ("seed", "ref_len")) -> list[Variant]:
    """Generate useful, unique variants rather than blind random samples."""
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    vary = tuple(dict.fromkeys(vary))
    valid = {"seed", "ref_len", "beams", "duration"}
    if not vary:
        raise ValueError("vary must not be empty")
    unknown = set(vary) - valid
    if unknown:
        raise ValueError(f"unknown vary dimension(s): {sorted(unknown)}")

    finite_values = {"ref_len": (4.0, 5.0, 7.0, 8.0),
                     "beams": (2, 3), "duration": (0.95, 1.05)}
    if "seed" not in vary:
        capacity = 1 + sum(len(finite_values[name]) for name in vary if name != "seed")
        if n > capacity:
            raise ValueError(f"requested n={n} exceeds {capacity} unique configurations "
                             "available without varying seed")

    variants = [Variant("基准", seed=base_seed)]
    seen: set[tuple] = set()

    def config_key(candidate: Variant) -> tuple:
        return (candidate.seed, candidate.ref_seconds, candidate.num_beams,
                candidate.do_sample, candidate.temperature,
                candidate.duration_factor, candidate.emotion_weight,
                tuple(sorted(candidate.extra.items())))

    seen.add(config_key(variants[0]))
    i = 0
    exhausted: set[str] = set()
    while len(variants) < n:
        which = vary[i % len(vary)]
        k = i // len(vary)
        i += 1
        if which in exhausted:
            if "seed" not in vary and len(exhausted) == len(vary):
                raise ValueError("not enough unique configurations for requested n")
            continue
        if which == "seed":
            candidate = Variant(f"seed{base_seed + k + 1}", seed=base_seed + k + 1)
        elif which == "ref_len":
            values = finite_values[which]
            if k >= len(values):
                exhausted.add(which)
                continue
            secs = values[k]
            candidate = Variant(f"参考{secs:.0f}s", seed=base_seed, ref_seconds=secs)
        elif which == "beams":
            values = finite_values[which]
            if k >= len(values):
                exhausted.add(which)
                continue
            beams = values[k]
            candidate = Variant(f"beams{beams}", seed=base_seed, num_beams=beams)
        else:  # duration
            values = finite_values[which]
            if k >= len(values):
                exhausted.add(which)
                continue
            duration = values[k]
            candidate = Variant(f"时长{duration:.2f}", seed=base_seed,
                                duration_factor=duration)
        key = config_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        variants.append(candidate)
    return variants


def rms_peak(path) -> tuple[float, float]:
    x, sr = sf.read(str(path))
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = np.asarray(x, dtype=np.float64)
    return (float(np.sqrt(np.mean(x ** 2))), float(np.max(np.abs(x))))


def tremor(path, hop_ms: float = 10.0) -> float:
    """短时能量抖动（越小越平稳）。"""
    x, sr = sf.read(str(path))
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = np.asarray(x, dtype=np.float64)
    hop = max(int(sr * hop_ms / 1000), 1)
    n = x.size // hop
    if n < 8:
        return 0.0
    r = np.sqrt(np.mean(x[:n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    v = r > 0.05 * np.max(r)
    if v.sum() < 8:
        return 0.0
    rr = r[v]
    return float(np.std(np.diff(rr)) / (np.mean(rr) + 1e-9))


def run_grid(prompt: str, text: str, variants: list[Variant],
             outdir: str | Path = "outputs_grid",
             base_url: str = "http://127.0.0.1:8188",
             voice_cid: str | None = None,
             seed: int = 42, verbose: bool = True,
             use_mos: bool = True, mos_device: str = "cpu",
             ref_wav_override: str | None = None,
             base_params: SynthParams | None = None, top_n: int = 3) -> dict:
    """批量合成并按参考相似度/自然度诊断排序；组合权重仅为启发式。

    ``ref_wav_override`` 是候选的 conditioning source。外部音频每种
    ref_seconds 使用单独准备的连续窗口；所有结果仍对同一完整评估参考评分。
    """
    from timbreforge.backend_indextts import SynthParams, ping, synthesize
    from timbreforge.score import cosine_to_100, grade, score_pairs
    from timbreforge.reference import prepare_reference
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n <= 0:
        raise ValueError("top_n must be a positive integer")
    from timbreforge.voicebank import find_voice, get_bank

    ok, msg = ping(base_url)
    if not ok:
        raise RuntimeError(msg)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    bank = get_bank()

    # Prepare the evaluation reference once. External crop variants condition on
    # distinct cached crops, while every candidate is scored against this fixed ref.
    evaluation_ref = None
    prepared_refs: dict[float, Path] = {}
    if ref_wav_override:
        entry = None
        evaluation_ref = prepare_reference(ref_wav_override, max_seconds=None)
        if verbose:
            print(f"使用外部参考音频: {ref_wav_override}")
    elif voice_cid:
        entry = bank.get(voice_cid)
        if entry is None:
            raise ValueError(f"找不到声线 {voice_cid}")
    else:
        entry, _ = find_voice(prompt, seed=seed, top_k=6)
    if verbose and entry is not None:
        print(f"声线: {entry.name} ({entry.cid})  {entry.gender} {entry.age}")
    if evaluation_ref is None:
        evaluation_ref = bank.to_wav(entry, force=False, max_seconds=None)

    rows = []
    for i, v in enumerate(variants):
        # 按变体要求生成参考（不同截取长度 -> 不同 WAV）
        if ref_wav_override:
            if v.ref_seconds not in prepared_refs:
                prepared_refs[v.ref_seconds] = prepare_reference(
                    ref_wav_override, max_seconds=v.ref_seconds)
            ref = prepared_refs[v.ref_seconds]
        else:
            ref = bank.to_wav(entry, force=False, max_seconds=v.ref_seconds)
            if evaluation_ref is None:
                evaluation_ref = ref
        base = base_params or SynthParams()
        params = replace(base, text=text, seed=v.seed, num_beams=v.num_beams,
                         do_sample=v.do_sample, temperature=v.temperature,
                         duration_factor=v.duration_factor,
                         emotion_weight=v.emotion_weight)
        t0 = time.time()
        try:
            res = synthesize(str(ref), params, base=base_url, verbose=False)
        except Exception as e:
            if verbose:
                print(f"  [{i+1}/{len(variants)}] {v.label} 失败: "
                      f"{type(e).__name__}")
            continue
        dst = outdir / f"{i+1:02d}_{v.label}.wav"
        shutil.copy2(res["file"], dst)
        # Keep the generated waveform unmodified for fair evaluation.
        r, _ = rms_peak(dst)
        rows.append({
            "idx": i + 1, "label": v.label, "file": str(dst),
            "ref": str(evaluation_ref), "conditioning_ref": str(ref),
            "seed": v.seed, "ref_seconds": v.ref_seconds,
            "num_beams": v.num_beams, "duration_factor": v.duration_factor,
            "params": params.to_dict(),
            "elapsed": round(time.time() - t0, 1), "rms": round(r, 4),
            "tremor": round(tremor(dst), 4),
        })

    if not rows:
        raise RuntimeError("所有变体都合成失败")

    # 自动打分（CAM++ 相似度，一次进程批量算）
    if verbose:
        print(f"\n自动打分（CAM++ 相似度）…")
    pairs = [{"label": r["label"], "ref": str(evaluation_ref), "test": r["file"]}
             for r in rows]
    scored = score_pairs(pairs)
    smap = {s["label"]: s["cosine"] for s in scored}
    if len(smap) != len(rows):
        raise RuntimeError("CAM++ results missing for one or more generated rows")
    for r in rows:
        c = smap.get(r["label"])
        if c is None or not np.isfinite(float(c)):
            raise RuntimeError(f"CAM++ result missing/nonfinite for {r['label']}")
        r["cosine"] = round(float(c), 4)
        r["score"] = round(cosine_to_100(c), 1)
        r["grade"] = grade(c)

    # 自动打分（自然度 + 质量，UTMOS22 / DNSMOS）
    if use_mos:
        from timbreforge.mosmodels import combined_score, grade_combined, score_files
        if verbose:
            print("自动打分（UTMOS 自然度 + DNSMOS 质量）…")
        mrecs = score_files([r["file"] for r in rows], device=mos_device)
        mmap = {m["file"]: m for m in mrecs}
        if len(mmap) != len(rows):
            raise RuntimeError("MOS results missing for one or more generated rows")
        for r in rows:
            m = mmap.get(r["file"], {})
            u, o = m.get("utmos"), m.get("dnsmos_ovrl")
            if u is None or o is None or not np.isfinite(float(u)) or not np.isfinite(float(o)):
                raise RuntimeError(f"MOS results missing/nonfinite for {r['file']}")
            r["utmos"] = float(u)
            r["dnsmos_ovrl"] = float(o)
            r["dnsmos_sig"] = m.get("dnsmos_sig")
            r["dnsmos_bak"] = m.get("dnsmos_bak")
            c = combined_score(r["score"], r["utmos"], r["dnsmos_ovrl"])
            r["combined"] = c["combined"]
            r["naturalness"] = c["naturalness100"]
            r["verdict"] = grade_combined(c["combined"])
    else:
        for r in rows:
            r["combined"] = r["score"]
            r["verdict"] = r["grade"]

    # 综合排序：有 MOS 时用综合分，否则用相似度
    rows.sort(key=lambda r: -(r.get("combined") or r["score"]))
    best = rows[0]

    # 注意：使用外部参考（ref_wav_override）时 entry 为 None，
    # 此处必须做空值保护——曾因直接访问 entry.cid 导致整个试点失败。
    summary = {
        "prompt": prompt, "text": text,
        "voice": (entry.cid if entry is not None else None),
        "voice_name": (entry.name if entry is not None else
                       Path(ref_wav_override).name),
        "gender": (entry.gender if entry is not None else None),
        "age": (entry.age if entry is not None else None),
        "ref_override": ref_wav_override,
        "n_variants": len(rows), "best": best, "rows": rows,
        "used_mos": bool(use_mos),
    }
    (outdir / "grid.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    if verbose:
        print(f"\n{'='*92}")
        print(f"自动排序结果（启发式综合分：相似度 45% + 自然度 40% + 质量 15%）"
              if use_mos else "自动排序结果（仅相似度）")
        print(f"{'='*92}")
        if use_mos:
            print(f"{'#':>3s} {'变体':12s} {'种子':>5s} {'参考':>5s} {'相似':>6s} "
                  f"{'UTMOS':>6s} {'OVRL':>6s} {'综合':>6s} {'档位':6s} {'RMS':>7s}")
            for r in rows[:top_n]:
                print(f"{r['idx']:3d} {r['label']:12s} {r['seed']:5d} "
                      f"{r['ref_seconds']:4.0f}s {r['score']:6.1f} "
                      f"{(r.get('utmos') or 0):6.2f} "
                      f"{(r.get('dnsmos_ovrl') or 0):6.2f} "
                      f"{(r.get('combined') or 0):6.1f} {r.get('verdict',''):6s} "
                      f"{r['rms']:7.4f}")
        else:
            print(f"{'#':>3s} {'变体':12s} {'种子':>5s} {'参考':>5s} {'beams':>5s} "
                  f"{'余弦':>8s} {'分数':>6s} {'档位':6s} {'RMS':>7s}")
            for r in rows[:top_n]:
                print(f"{r['idx']:3d} {r['label']:12s} {r['seed']:5d} "
                      f"{r['ref_seconds']:4.0f}s {r['num_beams']:5d} "
                      f"{r['cosine']:+8.4f} {r['score']:6.1f} {r['grade']:6s} "
                      f"{r['rms']:7.4f}")
        print(f"\n最佳: {best['label']}  综合 {best.get('combined', best['score'])} "
              f"({best.get('verdict', best['grade'])})  -> {best['file']}")
        print(f"清单: {outdir/'grid.json'}")
    return summary


def _self_test() -> None:
    import io
    print("=== 变体生成自检 ===")
    vs = default_variants(base_seed=42, n=8)
    for v in vs:
        print(f"  {v.label:14s} seed={v.seed:<4d} 参考={v.ref_seconds:.0f}s "
              f"beams={v.num_beams} 时长={v.duration_factor:.2f}")


if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(open("_grid_selftest.txt", "wb"), encoding="utf-8")
    _self_test()
    sys.stdout.flush()
