"""MOS 指标推理（UTMOS22 自然度 + DNSMOS 质量）。

必须用 ComfyUI venv 运行：
    python -m timbreforge.workers.mos_infer --files files.json --out out.json --model-dir models_mos

两个模型都是社区维护的原生 PyTorch 移植版，数值分别与
官方 `tarepan/SpeechMOS` 与 Microsoft DNSMOS ONNX 一致（见各自 README）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def load_models(model_dir: str, device: str):
    import torch
    from transformers import AutoModel

    ut_dir = os.path.join(model_dir, "utmos22")
    dn_dir = os.path.join(model_dir, "dnsmos")
    ut = dn = None
    if os.path.isdir(ut_dir):
        ut = AutoModel.from_pretrained(ut_dir, trust_remote_code=True, local_files_only=True).eval().to(device)
    if os.path.isdir(dn_dir):
        dn = AutoModel.from_pretrained(dn_dir, trust_remote_code=True, local_files_only=True).eval().to(device)
    return ut, dn


def read_wav(path: str):
    import soundfile as sf
    import torch

    x, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(x).transpose(0, 1)      # [C, T]
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav.squeeze(0), sr                       # [T], sr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", required=True, help="JSON 数组，音频路径列表")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-dir", help="Local MOS model directory (or config/env)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if args.model_dir is None:
        from pathlib import Path
        package_root = str(Path(__file__).resolve().parents[2])
        if package_root not in sys.path:
            sys.path.insert(0, package_root)
        from timbreforge.config import get_config
        args.model_dir = str(get_config().mos_models_dir)

    import torch

    paths = json.loads(open(args.files, encoding="utf-8").read())
    print(f"设备={args.device}  待评 {len(paths)} 个文件", file=sys.stderr)
    ut, dn = load_models(args.model_dir, args.device)
    print(f"UTMOS={'已加载' if ut is not None else '缺失'}  "
          f"DNSMOS={'已加载' if dn is not None else '缺失'}", file=sys.stderr)

    results = []
    for i, p in enumerate(paths):
        rec = {"file": p}
        try:
            wav, sr = read_wav(p)
            dev_wav = wav.to(args.device)
            if ut is not None:
                with torch.inference_mode():
                    s = ut(dev_wav, sampling_rate=int(sr)).scores[0]
                rec["utmos"] = round(float(s), 4)
            if dn is not None:
                with torch.inference_mode():
                    sc = dn(dev_wav, sampling_rate=int(sr))
                for key, attr in (("dnsmos_sig", "sig"), ("dnsmos_bak", "bak"),
                                  ("dnsmos_ovrl", "ovrl"), ("dnsmos_p808", "p808")):
                    try:
                        rec[key] = round(float(getattr(sc, attr)[0]), 4)
                    except Exception:
                        pass
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        results.append(rec)
        print(f"  [{i+1}/{len(paths)}] {os.path.basename(p):34s} "
              f"UTMOS={rec.get('utmos')} OVRL={rec.get('dnsmos_ovrl')} "
              f"{rec.get('error','')}", file=sys.stderr)

    open(args.out, "w", encoding="utf-8").write(
        json.dumps(results, ensure_ascii=False, indent=1))
    print(f"已写入 {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
