"""TimbreForge —— 角色声线生成 CLI（数据集检索版）

工作流（对标 AI 生图）：

    声线描述 ──检索──> 已准备的数据集角色声线──IndexTTS──> 语音
        种子 ──┘（同种子同声线，换种子抽同族不同声线）

可选数据来源：HuggingFace `TTS-AGI/moss-character-reference-voices`（CC-BY-4.0）。
发布包不附带数据集或索引，需另行准备；--ref-wav 不需要数据集。

用法示例
--------
    # 只看检索到什么声线（不合成，秒级）
    python -m timbreforge.voicegen "冰冷沙哑的成熟女声" --seed 42 --no-synth

    # 合成语音（需要 ComfyUI 在运行）
    python -m timbreforge.voicegen "睿智沉稳的老者" --seed 7 \
        --text "年轻人，我等你很久了。" --out elder.wav

    # 抽卡：同描述换种子
    python -m timbreforge.voicegen "空灵飘渺的女声" --batch 6 --text "听得到我吗？"

    # 复现指定声线
    python -m timbreforge.voicegen --cid k91_age5_bg0 --text "再次合成"

    # 看描述被怎么理解
    python -m timbreforge.voicegen "非常沙哑、有点冷的女声" --explain
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from timbreforge.voicebank import get_bank, find_voice
from timbreforge.voicelib import parse_voice_text
from timbreforge.config import get_config, set_config_path

EMOTION_KEYS = ("happy", "angry", "sad", "fearful",
                "disgusted", "melancholic", "surprised", "calm")


def entry_block(e, rank: int | None = None) -> str:
    tag = f"[{rank}] " if rank is not None else ""
    return (f"{tag}{e.name}\n"
            f"      cid={e.cid}  性别={e.gender}  年龄={e.age}  时长={e.duration:.1f}s\n"
            f"      {e.tagline}\n"
            f"      音区: {e.register[:110]}\n"
            f"      标签: {'、'.join(e.tags[:14])}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="voicegen",
        description="用自然语言从数据集检索角色声线并合成语音。",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--config", help="TOML configuration path")
    ap.add_argument("prompt", nargs="?", default=None, help="声线描述")
    ap.add_argument("--text", default="你好，这是用自然语言生成的角色声线测试。",
                    help="要合成的台词")
    ap.add_argument("--seed", type=int, default=0, help="种子（同种子同声线）")
    ap.add_argument("--top", type=int, default=6, help="显示前 N 个候选声线")
    ap.add_argument("--cid", default=None, help="直接指定数据集里的声线 id")
    ap.add_argument("--age", type=float, default=None, help="年龄感 0~1 覆盖")
    ap.add_argument("--gender", default=None,
                    choices=["female", "male", "neutral"], help="性别覆盖")
    ap.add_argument("--out", default=None, help="输出语音 wav 路径")
    ap.add_argument("--outdir", default=None, help="输出目录（默认使用配置的 outputs_dir/voicegen）")
    ap.add_argument("--batch", type=int, default=0, help="抽卡数量（seed 递增）")
    ap.add_argument("--grid", type=int, default=0,
                    help="SD 式批量抽卡 + 自动打分挑选（N 个变体，自动排序）")
    ap.add_argument("--grid-out", default=None, help="网格输出目录（默认使用配置的 outputs_dir/grid）")
    ap.add_argument("--grid-vary", default="seed,ref_len",
                    help="变体维度：seed,ref_len,beams,duration")
    ap.add_argument("--ref-wav", default=None,
                    help="直接用指定参考音频（如 AISHELL-3 真人参考）")
    ap.add_argument("--no-mos", action="store_true",
                    help="跳过 UTMOS/DNSMOS 自然度打分（只算相似度）")
    ap.add_argument("--explain", action="store_true", help="显示描述解析结果")
    ap.add_argument("--no-synth", action="store_true", help="只检索，不合成")
    norm_group = ap.add_mutually_exclusive_group()
    norm_group.add_argument("--normalize", dest="normalize", action="store_true",
                            help="仅为播放副本做峰值归一化（默认关闭，不影响 raw 输出）")
    norm_group.add_argument("--no-normalize", dest="normalize", action="store_false",
                            help="不做播放响度归一化（兼容旧参数）")
    ap.set_defaults(normalize=False)
    # 采样参数（对标生图）
    ap.add_argument("--backend", choices=["indextts", "cosyvoice3"], default="indextts")
    ap.add_argument("--ref-text", default="", help="CosyVoice 零样本模式：参考音频的准确转录")
    ap.add_argument("--instruction", default="", help="CosyVoice 语气/方言指令，不是新声线设计")
    ap.add_argument("--model", default="IndexTTS-2.5")
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=30)
    # Sampling is the upstream baseline, not a proven cause of tremor.
    # On 8GB VRAM use one beam first; compare greedy as an explicit ablation.
    ap.add_argument("--num-beams", type=int, default=1)
    ap.add_argument("--sample", action=argparse.BooleanOptionalAction, default=True,
                    help="随机采样（默认开启；--no-sample 用于确定性 GPT 对照）")
    ap.add_argument("--rep-penalty", type=float, default=3.0)
    ap.add_argument("--max-mel-tokens", type=int, default=1500)
    ap.add_argument("--segment-tokens", type=int, default=120)
    ap.add_argument("--duration-factor", type=float, default=1.0)
    ap.add_argument("--diffusion-steps", type=int, default=25,
                    help="S2Mel 扩散步数（质量/速度折中，1~100）")
    ap.add_argument("--inference-cfg-rate", type=float, default=0.7,
                    help="S2Mel CFG 强度（0~2；过高可能有伪影）")
    ap.add_argument("--silence-ms", type=int, default=100)
    ap.add_argument("--emotion-mode", default="same_as_speaker")
    ap.add_argument("--emotion-weight", type=float, default=0.65)
    ap.add_argument("--extra-emotion-text", default="")
    ap.add_argument("--base-url", default=None, help="覆盖配置中的 ComfyUI URL")
    for k in EMOTION_KEYS:
        ap.add_argument(f"--{k}", type=float, default=0.0)
    args = ap.parse_args(argv)
    if args.config:
        set_config_path(args.config)
    cfg = get_config()
    args.base_url = args.base_url or cfg.comfy_url
    args.outdir = args.outdir or str(cfg.outputs_dir / "voicegen")
    args.grid_out = args.grid_out or str(cfg.outputs_dir / "grid")

    from timbreforge.backend_indextts import SynthParams
    emotion = parse_voice_text(args.prompt)["emotion"].copy() if args.prompt else {}
    emotion.update({k: getattr(args, k) for k in EMOTION_KEYS if getattr(args, k) > 0})
    params = SynthParams(
        text=args.text, model=args.model, precision=args.precision,
        seed=args.seed, temperature=args.temperature, top_p=args.top_p,
        top_k=args.top_k, num_beams=args.num_beams, do_sample=bool(args.sample),
        repetition_penalty=args.rep_penalty, max_mel_tokens=args.max_mel_tokens,
        max_text_tokens_per_segment=args.segment_tokens,
        interval_silence_ms=args.silence_ms, duration_factor=args.duration_factor,
        diffusion_steps=args.diffusion_steps, inference_cfg_rate=args.inference_cfg_rate,
        emotion_mode=("extra_emotion_text" if args.extra_emotion_text and args.emotion_mode == "same_as_speaker"
                      else "emotion_vector" if emotion and args.emotion_mode == "same_as_speaker"
                      else args.emotion_mode),
        emotion_weight=args.emotion_weight,
        extra_emotion_text=args.extra_emotion_text, emotion=emotion)
    bank = None
    if not args.ref_wav:
        try:
            bank = get_bank()
        except Exception as exc:
            print(f"数据集检索不可用：{type(exc).__name__}: {exc}。请准备数据集与索引，或使用 --ref-wav。")
            return 2
        print(f"声线库: {len(bank)} 条（TTS-AGI/moss-character-reference-voices, CC-BY-4.0）")

    # ---------------------------------------------------------- SD 式网格抽卡
    if args.grid and args.grid > 0:
        if args.backend != "indextts":
            ap.error("CosyVoice3 当前仅支持单条对照；--grid 暂限 IndexTTS，避免误用其采样参数。")
        if not args.prompt and not args.cid and not args.ref_wav:
            ap.print_help()
            return 2
        from timbreforge.grid import default_variants, run_grid
        vary = tuple(x.strip() for x in args.grid_vary.split(",") if x.strip())
        variants = default_variants(base_seed=args.seed, n=args.grid, vary=vary)
        # Preserve explicit CLI settings for fields not varied by this experiment.
        from dataclasses import replace
        variants = [replace(v, do_sample=params.do_sample,
                            temperature=params.temperature,
                            emotion_weight=params.emotion_weight,
                            num_beams=(v.num_beams if "beams" in vary else params.num_beams),
                            duration_factor=(v.duration_factor if "duration" in vary else params.duration_factor))
                    for v in variants]
        run_grid(args.prompt or "", args.text, variants,
                 outdir=args.grid_out, base_url=args.base_url,
                 voice_cid=args.cid, seed=args.seed,
                 use_mos=not args.no_mos,
                 ref_wav_override=args.ref_wav, base_params=params)
        return 0

    # ---------------------------------------------------------- 解释
    if args.explain and args.prompt:
        p = parse_voice_text(args.prompt)
        print(f'\n描述解析: "{args.prompt}"')
        print(f"  命中词: {'、'.join(p['hits']) or '（无）'}")
        if p["negated"]:
            print(f"  否定词: {'、'.join(p['negated'])}")
        print(f"  声线参数: tilt={p['tilt']:+.3f} formant={p['formant']:+.3f} "
              f"breath={p['breath']:.3f} rate={p['rate']:.3f} age={p['age']:+.2f}")
        print(f"  情感向量: {p['emotion'] or '（无）'}")

    # ---------------------------------------------------------- 选声线
    # An explicit reference is a valid source even without a bank prompt/cid.
    # Previously --ref-wav worked only in --grid and was silently ignored here.
    if args.ref_wav:
        e, ranked = None, []
        print(f"\n指定外部参考: {args.ref_wav}")
    elif args.cid:
        e = bank.get(args.cid)
        if e is None:
            print(f"找不到声线 {args.cid}")
            return 2
        ranked = [(e, 1.0)]
        print(f"\n指定声线: {args.cid}")
    else:
        if not args.prompt:
            ap.print_help()
            return 2
        e, ranked = find_voice(args.prompt, seed=args.seed, top_k=args.top,
                               age=args.age, gender=args.gender)

    if e is not None:
        print(f"\n{'='*78}\n检索结果\n{'='*78}")
        for i, (x, s) in enumerate(ranked):
            mark = " <== 选中" if x.cid == e.cid else ""
            print(f"\n  {entry_block(x, i+1)}   匹配度={s:.4f}{mark}")

    # 情感向量
    emotion = {}
    if args.prompt:
        parsed = parse_voice_text(args.prompt)
        emotion.update(parsed["emotion"])
    for k in EMOTION_KEYS:
        v = getattr(args, k)
        if v > 0:
            emotion[k] = v
    if emotion:
        print("\n情感向量: " + "  ".join(f"{k}={v:.2f}" for k, v in emotion.items()))

    # ---------------------------------------------------------- 参考音频
    t0 = time.time()
    if args.backend == "cosyvoice3":
        # CosyVoice transcripts describe the ORIGINAL waveform. Preparation or
        # trimming here would misreport conditioning and could break alignment.
        wav = Path(args.ref_wav).resolve() if args.ref_wav else bank.fetch_audio(e)
    elif args.ref_wav:
        from timbreforge.reference import prepare_reference
        wav = prepare_reference(args.ref_wav, max_seconds=6.0)
    else:
        wav = bank.to_wav(e)
    voice_id = e.cid if e is not None else wav.stem
    print(f"\n参考音频已就绪: {wav}  ({wav.stat().st_size/1024:.0f} KB, "
          f"{time.time()-t0:.1f}s)")

    info = {
        "prompt": args.prompt, "seed": args.seed, "voice": voice_id,
        "name": e.name if e is not None else Path(args.ref_wav).name,
        "gender": e.gender if e is not None else None,
        "age": e.age if e is not None else None,
        "tags": e.tags if e is not None else [],
        "emotion": emotion, "ref_wav": str(wav),
    }
    meta_dir = Path(args.outdir)
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / f"{voice_id}_seed{args.seed}.meta.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------------------------------------------------------- 抽卡
    if args.batch and args.batch > 0:
        print(f"\n抽卡 {args.batch} 个（seed={args.seed}…）：")
        picks = []
        for i in range(args.batch):
            x, _ = find_voice(args.prompt, seed=args.seed + i, top_k=1,
                              age=args.age, gender=args.gender)
            picks.append((args.seed + i, x))
            print(f"  seed={args.seed+i:<4d} {x.cid:22s} {x.name[:34]:34s} "
                  f"{x.gender:11s} {x.age}")
        uniq = len({x.cid for _, x in picks})
        print(f"  唯一声线 {uniq}/{args.batch}")
        (meta_dir / f"batch_seed{args.seed}.json").write_text(
            json.dumps([{"seed": s, "cid": x.cid, "name": x.name,
                         "gender": x.gender, "age": x.age}
                        for s, x in picks], ensure_ascii=False, indent=1),
            encoding="utf-8")
        if args.no_synth:
            return 0

    if args.no_synth:
        print("\n（--no-synth：跳过语音合成）")
        return 0

    # ---------------------------------------------------------- 合成
    from timbreforge.backend_indextts import ping
    if args.backend == "cosyvoice3":
        from timbreforge.backend_cosyvoice import synthesize
        ok, msg = True, "CosyVoice3 本地隔离后端"
    else:
        from timbreforge.backend_indextts import synthesize
        ok, msg = ping(args.base_url)
    print(f"\n后端: {msg}")
    if not ok:
        print("无法合成语音。请启动配置地址对应的 ComfyUI 并检查 IndexTTS 节点。")
        print("（参考音频已就绪，后端起来后可用 --cid 直接复现）")
        return 1

    # backend_indextts.build_graph also guards these modes, so the same
    # SynthParams is used by normal synthesis and grid experiments.
    if emotion and params.emotion_mode == "same_as_speaker":
        params.emotion_mode = "emotion_vector"
        print("（检测到情感向量，已自动切换 emotion_mode -> emotion_vector）")
    elif params.extra_emotion_text and params.emotion_mode == "same_as_speaker":
        params.emotion_mode = "extra_emotion_text"
        print("（检测到情感文字，已自动切换 emotion_mode -> extra_emotion_text）")
    print(f"台词: {args.text}")
    print(f"采样: seed={params.seed} temp={params.temperature} "
          f"top_p={params.top_p} top_k={params.top_k} beams={params.num_beams} "
          f"rep={params.repetition_penalty}")

    t0 = time.time()
    if args.backend == "cosyvoice3":
        res = synthesize(str(wav), params, ref_text=args.ref_text,
                         instruction=args.instruction)
    else:
        res = synthesize(str(wav), params, base=args.base_url)
    print(f"合成完成: {res['file']}  ({res['elapsed']:.1f}s)")

    out = args.out or str(meta_dir / f"{voice_id}_seed{args.seed}.wav")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(res["file"], out)

    # Playback-only level adjustment; this cannot restore natural prosody.
    # Keep the native render alongside the listening copy for objective scoring.
    info["raw_output"] = res["file"]
    if args.normalize:
        import soundfile as _sf
        _x, _sr = _sf.read(out)
        _mono = _x if _x.ndim == 1 else _x.mean(axis=1)
        _pk = float(np.max(np.abs(_x)))
        if 1e-6 < _pk < 0.95:
            gain = 0.95 / _pk
            _x = np.clip(_x * gain, -1.0, 1.0)
            _sf.write(out, _x.astype(np.float32), _sr)
            print(f"响度归一化: 峰值 {_pk:.3f} -> 0.950 (增益 {gain:.2f}x)")
    _x2, _sr2 = (lambda a: (a[0], a[1]))(
        __import__("soundfile").read(out))
    _m = _x2 if _x2.ndim == 1 else _x2.mean(axis=1)
    print(f"输出: {out}  ({os.path.getsize(out)/1024:.0f} KB, "
          f"{len(_m)/_sr2:.2f}s, 峰值 {float(np.max(np.abs(_m))):.3f}, "
          f"RMS {float(np.sqrt(np.mean(_m**2))):.3f})")

    info["output"] = out
    info["elapsed"] = res["elapsed"]
    info["params"] = params.to_dict()
    info["backend"] = args.backend
    if args.backend == "cosyvoice3":
        info["backend_metadata"] = res
    (meta_dir / f"{voice_id}_seed{args.seed}.meta.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
