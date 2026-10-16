"""File-JSON protocol worker for the bundled local CosyVoice3 environment."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

HELPFUL = "You are a helpful assistant."
DELIMITER = "<|endofprompt|>"


def run(request: dict) -> dict:
    # Must be set before importing the model libraries. Missing local assets fail;
    # neither the parent nor this worker invokes snapshot_download explicitly.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    repo = Path(request["repo"]).resolve()
    model_dir = Path(request["model_dir"]).resolve()
    reference = Path(request["ref_audio"]).resolve()
    output = Path(request["output"]).resolve()
    if not model_dir.is_dir() or not reference.is_file():
        raise FileNotFoundError("Local model directory or reference audio missing")
    sys.path.insert(0, str(repo / "third_party" / "Matcha-TTS"))
    sys.path.insert(0, str(repo))
    import numpy as np
    import soundfile as sf
    import torch

    params = request["params"]
    text = params["text"].strip()
    if not text:
        raise ValueError("Synthesis text must not be empty")
    transcript = request.get("ref_text", "").strip()
    instruction = request.get("instruction", "").strip()
    if DELIMITER in text or DELIMITER in transcript or DELIMITER in instruction:
        raise ValueError("Supply plain text without CosyVoice prompt delimiters")
    ref, ref_sr = sf.read(reference, dtype="float32", always_2d=True)
    if ref.size == 0 or ref_sr <= 0 or not np.isfinite(ref).all() or not np.any(ref):
        raise ValueError("Reference audio is empty, silent, or non-finite")
    ref_duration = len(ref) / ref_sr
    if ref_duration > 30:
        raise ValueError("Reference exceeds 30 seconds; provide a matching shorter audio/transcript pair (no automatic cropping)")
    seed = int(params.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CosyVoice3 requires the bundled CUDA-enabled GPU runtime")
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()
    from cosyvoice.cli.cosyvoice import AutoModel

    started = time.perf_counter()
    model = AutoModel(model_dir=str(model_dir), fp16=True,
                      load_trt=False, load_vllm=False)
    # Some model constructors initialize RNG state; restore the requested seed.
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    loaded = time.perf_counter()
    frontend = bool(params.get("text_normalization", True))
    common = {"stream": False, "speed": 1.0, "text_frontend": frontend}
    if instruction:
        mode = "instruct2"
        prompt = f"{HELPFUL} {instruction}{DELIMITER}"
        generated = model.inference_instruct2(text, prompt, str(reference), **common)
    elif transcript:
        mode = "zero_shot"
        prompt = f"{HELPFUL}{DELIMITER}{transcript}"
        generated = model.inference_zero_shot(text, prompt, str(reference), **common)
    else:
        mode = "cross_lingual"
        generated = model.inference_cross_lingual(
            f"{HELPFUL}{DELIMITER}{text}", str(reference), **common)
    chunks = []
    with torch.inference_mode():
        for chunk in generated:
            audio = chunk["tts_speech"].detach().float().cpu().numpy()
            if audio.ndim == 2 and audio.shape[0] == 1:
                audio = audio[0]
            if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
                raise ValueError("Model returned an invalid audio chunk")
            chunks.append(audio)
    if not chunks:
        raise ValueError("Model returned no audio chunks")
    audio = np.concatenate(chunks)
    if not np.any(audio):
        raise ValueError("Model returned silent audio")
    sample_rate = int(model.sample_rate)
    if sample_rate <= 0:
        raise ValueError("Model returned invalid sample rate")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Preserve amplitudes in floating-point WAV: no clipping or normalization.
    sf.write(output, audio, sample_rate, subtype="FLOAT")
    torch.cuda.synchronize()
    finished = time.perf_counter()
    supported = {"text", "seed", "text_normalization"}
    ignored = {key: value for key, value in params.items() if key not in supported}
    return {
        "file": str(output), "elapsed": finished - started,
        "sample_rate": sample_rate, "duration": len(audio) / sample_rate,
        "load_seconds": loaded - started, "inference_seconds": finished - loaded,
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "backend": "cosyvoice3", "model_dir": str(model_dir), "mode": mode,
        "seed": seed, "reference": str(reference),
        "reference_duration": ref_duration, "reference_cropped": False,
        "ref_text_used": bool(transcript and not instruction),
        "instruction": instruction, "output_subtype": "FLOAT",
        "capabilities": {"seed": True, "reference_audio": True,
                         "reference_transcript": True, "instruction": True,
                         "text_normalization": True, "speed": "fixed_1.0",
                         "indextts_sampling_controls": False,
                         "indextts_emotion_vector": False,
                         "duration_factor": False},
        "unsupported_params": ignored,
        "warnings": (["ref_text is not consumed in instruct2 mode"]
                     if transcript and instruction else []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, help="UTF-8 request JSON path")
    parser.add_argument("--result", required=True, help="UTF-8 result JSON path")
    args = parser.parse_args()
    try:
        request = json.loads(Path(args.request).read_text(encoding="utf-8"))
        result = run(request)
        code = 0
    except Exception as exc:
        traceback.print_exc()
        result = {"error": f"{type(exc).__name__}: {exc}"}
        code = 1
    result_path = Path(args.result)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
