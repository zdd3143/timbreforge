"""TimbreForge —— 命令行入口（P0-9）

示例：
    python -m timbreforge.cli "温暖厚实的复古模拟低音" -o warm_bass.wav
    python -m timbreforge.cli "冰冷尖锐的金属拨弦" --seed 7 --explain
    python -m timbreforge.cli "空灵飘渺的女声铺底" --batch 5 --outdir out/
    python -m timbreforge.cli --list-families
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from timbreforge import (  # noqa: E402
    FAMILIES, MACRO_NAMES, build_patch, compare, explain, extract,
    extract_voice, profile_text, profile_voice_text, render_patch, to_dict,
)
from timbreforge.engine import SR  # noqa: E402


def _write_wav(path: str, audio: np.ndarray, sr: float) -> None:
    try:
        import soundfile as sf
    except ImportError:
        raise SystemExit("需要 soundfile 才能写出 wav：pip install soundfile")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sf.write(path, audio, int(sr))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="timbreforge",
        description="用自然语言（中文）生成合成器音色。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("description", nargs="?", default=None,
                    help='音色描述，例如 "温暖厚实的复古模拟低音"')
    ap.add_argument("-o", "--output", default=None, help="输出 wav 路径")
    ap.add_argument("--outdir", default="outputs_timbre", help="批量输出目录")
    ap.add_argument("--seed", type=int, default=0, help="随机种子（同 seed 完全复现）")
    ap.add_argument("--duration", type=float, default=None, help="时长（秒），默认按乐器族")
    ap.add_argument("--sr", type=float, default=SR, help=f"采样率，默认 {int(SR)}")
    ap.add_argument("--jitter", type=float, default=0.12, help="随机扰动幅度 0~0.3")
    ap.add_argument("--batch", type=int, default=0, help="批量生成 N 个变体（seed 递增）")
    ap.add_argument("--voice", action="store_true",
                    help="强制人声（元音/共振峰）模式")
    ap.add_argument("--vowel", default=None, choices=["a", "e", "i", "o", "u", "v", "n"],
                    help="指定元音：a/e/i/o/u/v(中性)/n(鼻音)")
    ap.add_argument("--gender", default=None, choices=["female", "male", "child"],
                    help="人声性别/音区")
    ap.add_argument("--narration", default=None,
                    help="把旁白合成到指定 wav 路径（需本机有可用 TTS）")
    ap.add_argument("--helm", default=None,
                    help="同时导出 Helm preset JSON 到指定路径")
    ap.add_argument("--helm-state", default=None,
                    help="同时导出 Helm 二进制状态文件到指定路径")
    ap.add_argument("--explain", action="store_true", help="显示这句话是怎么被理解的")
    ap.add_argument("--profile", action="store_true", help="显示音色感知画像")
    ap.add_argument("--json", default=None, help="把 patch 参数另存为 json")
    ap.add_argument("--compare", default=None, help="与另一个 wav 对比感知特征")
    ap.add_argument("--list-families", action="store_true", help="列出乐器族")
    ap.add_argument("--list-macros", action="store_true", help="列出宏定义")

    args = ap.parse_args(argv)

    if args.list_families:
        print("可用乐器族（可作为描述的一部分）：")
        for k, v in FAMILIES.items():
            print(f"  {k:8s} 波形={v['waveform']:8s} 默认时长={v['duration']}s")
        print("\n提示：描述里写「贝斯/铺底/拨弦/主音/钢琴/铃」即可选中对应族。")
        return 0

    if args.list_macros:
        print("6 个宏（映射到合成参数的中间层）：")
        for m in MACRO_NAMES:
            print(f"  {m}")
        return 0

    if not args.description:
        ap.print_help()
        return 2

    # ---------------------------------------------------------- 解释
    if args.explain:
        print(explain(args.description))
        print()

    # ---------------------------------------------------------- 批量
    if args.batch and args.batch > 0:
        os.makedirs(args.outdir, exist_ok=True)
        files = []
        print(f"批量生成 {args.batch} 个变体 → {args.outdir}/")
        for i in range(args.batch):
            seed = args.seed + i
            p = build_patch(args.description, seed=seed, jitter=args.jitter,
                            force_voice=args.voice, vowel=args.vowel,
                            gender=args.gender)
            audio = render_patch(p, sr=args.sr, duration=args.duration)
            name = f"tf_{seed:04d}_{p.family}.wav"
            path = os.path.join(args.outdir, name)
            _write_wav(path, audio, args.sr)
            if getattr(p, "mode", "synth") == "voice":
                feat = extract_voice(audio, args.sr)
                prof = profile_voice_text(feat)
            else:
                feat = extract(audio, args.sr)
                prof = profile_text(feat)
            files.append((name, p, feat))
            print(f"  [{i+1}/{args.batch}] {name}  {prof}")
        # 变体间距离：同族不应差异过大
        if len(files) >= 2:
            ds = [np.linalg.norm(files[i][2] - files[0][2]) for i in range(1, len(files))]
            print(f"\n  变体间平均特征距离 = {np.mean(ds):.4f}（越小说明同族一致性越好）")
        return 0

    # ---------------------------------------------------------- 单个
    t0 = time.time()
    p = build_patch(args.description, seed=args.seed, jitter=args.jitter,
                    force_voice=args.voice, vowel=args.vowel, gender=args.gender)
    t_build = time.time() - t0

    t0 = time.time()
    audio = render_patch(p, sr=args.sr, duration=args.duration)
    t_render = time.time() - t0

    out = args.output or f"{p.family}_seed{args.seed}.wav"
    _write_wav(out, audio, args.sr)

    is_voice = getattr(p, "mode", "synth") == "voice"
    feat = extract_voice(audio, args.sr) if is_voice else extract(audio, args.sr)
    profile = profile_voice_text(feat) if is_voice else profile_text(feat)
    dur_actual = audio.size / args.sr

    print(f"描述   : {args.description}")
    print(f"模式   : {'人声（元音/共振峰）' if is_voice else '合成音色'}"
          + (f"   元音={p.vowel}  性别={p.gender}" if is_voice else f"   乐器族={p.family}"))
    print(f"宏     : " + "  ".join(f"{k[:4]}={p.macros[k]:.2f}" for k in MACRO_NAMES))
    print(f"输出   : {out}  ({audio.size} 样本 @ {int(args.sr)}Hz, {dur_actual:.2f}s)")
    print(f"画像   : {profile}")
    print(f"耗时   : 构建 {t_build*1000:.1f}ms + 渲染 {t_render*1000:.1f}ms"
          f"  ({dur_actual/max(t_render,1e-9):.0f}x 实时)")
    print(f"峰值   : {float(np.max(np.abs(audio))):.3f}")

    if args.narration:
        from timbreforge.narrate import narrate, narration_script
        txt = narration_script(args.description, profile)
        got = narrate(txt, args.narration)
        print(f"旁白   : {got if got else '生成失败（本机无可用的 TTS 后端）'}")

    if args.profile:
        print("\n特征明细:")
        names = (__import__("timbreforge").VOICE_FEATURE_NAMES if is_voice
                 else __import__("timbreforge").FEATURE_NAMES)
        for n, v in zip(names, feat):
            print(f"  {n:12s} {v:.4f}")

    if args.compare:
        import soundfile as sf
        ref, sr_ref = sf.read(args.compare)
        if ref.ndim > 1:
            ref = ref.mean(axis=1)
        ref_feat = extract(ref.astype(np.float64), sr_ref)
        print(f"\n与 {args.compare} 对比:")
        print(compare(ref_feat, feat))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(to_dict(p), f, ensure_ascii=False, indent=1)
        print(f"参数已存: {args.json}")

    if args.helm or args.helm_state:
        try:
            from timbreforge.helm import export_preset, export_state_blob
            if args.helm:
                got = export_preset(p, args.helm, name=args.description)
                print(f"Helm preset: {got}")
            if args.helm_state:
                got = export_state_blob(p, args.helm_state, name=args.description)
                print(f"Helm state : {got}")
            print("  注意：本机未安装 Helm，导出文件尚未在真实 Helm 中验证加载。")
        except Exception as e:
            print(f"Helm 导出失败: {type(e).__name__}: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
