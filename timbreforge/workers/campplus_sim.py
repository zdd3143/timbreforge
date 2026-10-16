"""用 IndexTTS 自带的 CAM++ 说话人嵌入计算音色相似度（高可靠版）。

严格复现 IndexTTS 的用法（infer_v2_5.py:649-651）：
    audio 16k -> torchaudio kaldi.fbank(num_mel_bins=80, dither=0)
              -> 减帧均值 -> CAMPPlus -> 192 维说话人嵌入
相似度 = 两个嵌入的余弦。

这就是"语音界的 CLIP score"：参考音频与合成音频在**说话人身份空间**里的贴近度。

必须用 ComfyUI 的 venv 运行（需要 torch / torchaudio / indextts）。

用法：
    python campplus_sim.py --pairs pairs.json --out result.json
    pairs.json = [{"label": "...", "ref": "a.wav", "test": "b.wav"}, ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

def resolve_model_paths(indextts_repo=None, checkpoint=None):
    """CLI overrides environment/config; do not assume a ComfyUI installation."""
    from pathlib import Path

    repo = indextts_repo or os.environ.get("TIMBREFORGE_INDEXTTS_REPO")
    ckpt = checkpoint or os.environ.get("TIMBREFORGE_CAMPPLUS_CHECKPOINT")
    if not repo or not ckpt:
        # Direct file execution in a separate model runtime need not have the
        # TimbreForge wheel installed. The sibling config remains available.
        package_root = str(Path(__file__).resolve().parents[2])
        if package_root not in sys.path:
            sys.path.insert(0, package_root)
        from timbreforge.config import get_config
        cfg = get_config()
        repo = repo or getattr(cfg, "indextts_repo", None)
        ckpt = ckpt or getattr(cfg, "campplus_checkpoint", None)
    if not repo or not ckpt:
        raise ValueError("Set --indextts-repo / --checkpoint or "
                         "TIMBREFORGE_INDEXTTS_REPO / TIMBREFORGE_CAMPPLUS_CHECKPOINT")
    return Path(repo).expanduser().resolve(), Path(ckpt).expanduser().resolve()


def load_audio_16k(path: str):
    """读音频并重采样到 16k 单声道（返回 torch tensor）。"""
    import soundfile as sf
    import torch
    import torchaudio

    x, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(x).transpose(0, 1)          # [C, T]
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    return wav  # [1, T]


def build_model(device: str = "cpu", indextts_repo=None, checkpoint=None):
    """加载 CAM++（复用 IndexTTS 的模型定义）。"""
    import torch
    repo, ckpt = resolve_model_paths(indextts_repo, checkpoint)
    if not repo.is_dir():
        raise FileNotFoundError(f"IndexTTS repo not found: {repo}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus

    if not ckpt.is_file():
        raise FileNotFoundError(f"CAM++ checkpoint not found: {ckpt}")
    model = CAMPPlus(feat_dim=80, embedding_size=192)
    state = torch.load(ckpt, map_location="cpu")
    # 兼容可能的封装
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    model.load_state_dict(state)
    model = model.to(device).eval()
    return model, ckpt


def embed(model, wav, device: str = "cpu") -> np.ndarray:
    """按 IndexTTS 的方式提取 192 维说话人嵌入。"""
    import torch
    import torchaudio

    feat = torchaudio.compliance.kaldi.fbank(
        wav.to(device), num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    with torch.no_grad():
        e = model(feat.unsqueeze(0))
    v = e.squeeze(0).cpu().numpy().astype(np.float64)
    return v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="JSON: [{label, ref, test}, ...]")
    ap.add_argument("--out", required=True, help="输出 JSON")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--indextts-repo", help="IndexTTS source repo (or config/env)")
    ap.add_argument("--checkpoint", help="Local campplus_cn_common.bin (or config/env)")
    ap.add_argument("--self-test", action="store_true",
                    help="额外报告自相似分数作为上界")
    args = ap.parse_args()

    pairs = json.loads(open(args.pairs, encoding="utf-8").read())
    model, ckpt = build_model(args.device, args.indextts_repo, args.checkpoint)
    print(f"CAM++ 权重: {ckpt}", file=sys.stderr)
    print(f"设备: {args.device}   待测 {len(pairs)} 对", file=sys.stderr)

    results = []
    cache: dict[str, np.ndarray] = {}
    for i, p in enumerate(pairs):
        ref, test = p["ref"], p["test"]
        if ref not in cache:
            cache[ref] = embed(model, load_audio_16k(ref), args.device)
        if test not in cache:
            cache[test] = embed(model, load_audio_16k(test), args.device)
        c = cosine(cache[ref], cache[test])
        rec = {"label": p.get("label", f"pair{i}"), "ref": ref, "test": test,
               "cosine": round(c, 6)}
        if args.self_test:
            rec["ref_self"] = round(cosine(cache[ref], cache[ref]), 6)
        results.append(rec)
        print(f"  {rec['label']:34s} cosine={c:+.4f}", file=sys.stderr)

    # 自我一致性基线：同一文件对自身应为 1.0，用于确认管线正确
    if cache:
        k0 = next(iter(cache))
        print(f"自检（同一文件 vs 自身）: {cosine(cache[k0], cache[k0]):.6f}",
              file=sys.stderr)

    open(args.out, "w", encoding="utf-8").write(
        json.dumps(results, ensure_ascii=False, indent=1))
    print(f"已写入 {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
