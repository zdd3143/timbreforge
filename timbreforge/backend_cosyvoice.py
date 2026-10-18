"""Isolated, local-only CosyVoice3 adapter; never imports torch in the caller.

The worker runs in the bundled CosyVoice Python environment. It does not share
IndexTTS sampling/emotion controls: unsupported settings are reported, not mapped.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time
import uuid

from timbreforge.backend_indextts import SynthParams
from timbreforge.config import get_config
from timbreforge.resources import model_slot

WORKER = Path(__file__).resolve().parent / "workers" / "cosyvoice_worker.py"

# These remain available as module attributes for existing callers and tests,
# but are resolved only when accessed so CLI-selected config is respected.
def __getattr__(name: str):
    cfg = get_config()
    if name == "COSYVOICE_REPO":
        return cfg.cosyvoice_repo
    if name == "PYTHON":
        return cfg.cosyvoice_python
    if name == "MODEL_DIR":
        return cfg.cosyvoice_model
    if name == "OUTPUT_DIR":
        return cfg.outputs_dir / "cosyvoice3"
    raise AttributeError(name)


def synthesize(ref_audio: str | Path, params: SynthParams, ref_text: str = "",
               instruction: str = "", timeout: float = 1800,
               verbose: bool = True) -> dict:
    """Generate float WAV with the unmodified reference; return file/elapsed metadata.

    References over 30 seconds are rejected rather than cropped, so a supplied
    transcript remains aligned. Instructions select instruct2 over zero-shot.
    No model unloading, downloads, package installation, or loudness processing.
    """
    # Explicit module overrides (including mock patches) win over live config.
    repo, python, model_dir, output_dir = (
        Path(globals()[name] if name in globals() else __getattr__(name))
        for name in ("COSYVOICE_REPO", "PYTHON", "MODEL_DIR", "OUTPUT_DIR")
    )
    reference = Path(ref_audio).expanduser().resolve()
    if not reference.is_file():
        raise FileNotFoundError(f"Reference audio not found: {reference}")
    if not params.text.strip():
        raise ValueError("Synthesis text must not be empty")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    for path in (python, WORKER, model_dir / "cosyvoice3.yaml",
                 model_dir / "llm.pt", model_dir / "flow.pt", model_dir / "hift.pt",
                 model_dir / "campplus.onnx", model_dir / "speech_tokenizer_v3.onnx",
                 model_dir / "CosyVoice-BlankEN" / "config.json",
                 model_dir / "CosyVoice-BlankEN" / "model.safetensors"):
        if not path.is_file():
            raise FileNotFoundError(f"Required local CosyVoice asset missing: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    output = (output_dir / f"{run_id}.wav").resolve()
    request_path = output_dir / f"{run_id}.request.json"
    result_path = output_dir / f"{run_id}.result.json"
    log_path = output_dir / f"{run_id}.log"
    request = {
        "repo": str(repo), "model_dir": str(model_dir),
        "ref_audio": str(reference), "ref_text": ref_text,
        "instruction": instruction, "params": params.to_dict(),
        "output": str(output),
    }
    request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
    env = dict(os.environ, PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_HUB_DISABLE_TELEMETRY="1", GRADIO_ANALYTICS_ENABLED="False")
    command = [str(python), "-B", str(WORKER), "--request", str(request_path),
               "--result", str(result_path)]
    if verbose:
        print(f"CosyVoice3 local worker: {run_id}; log={log_path}")
    start = time.perf_counter()
    try:
        # File-backed stdout avoids Windows sandbox named-pipe restrictions and
        # keeps potentially large model logs out of the parent process memory.
        with model_slot(output_dir.parent), log_path.open("w", encoding="utf-8") as log:
            process = subprocess.run(command, cwd=str(repo), env=env,
                                     stdout=log, stderr=subprocess.STDOUT,
                                     timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"CosyVoice3 exceeded {timeout}s; see {log_path}") from exc
    result = None
    if result_path.is_file():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise RuntimeError(f"Invalid CosyVoice3 result JSON; see {log_path}") from exc
    if process.returncode != 0 or not isinstance(result, dict) or result.get("error"):
        detail = result.get("error", "No valid worker result") if isinstance(result, dict) else "No valid worker result"
        if 'out of memory' in str(detail).lower():
            raise RuntimeError('CosyVoice3 CUDA out of memory（GPU 显存不足）；已保存的候选和人工选择不会丢失。'
                               '请手动释放其他 GPU 工作负载后只重试失败句；不要关闭 weights_only 安全加载。'
                               f' See {log_path}')
        raise RuntimeError(f"CosyVoice3 worker failed (exit {process.returncode}): {detail}; see {log_path}")
    if result.get("file") != str(output) or not output.is_file():
        raise RuntimeError(f"CosyVoice3 worker returned missing/unexpected audio; see {log_path}")
    # Independently reject truncated, empty, silent, or non-finite worker output.
    import numpy as np
    import soundfile as sf
    try:
        audio, sample_rate = sf.read(output, dtype="float32", always_2d=True)
        if (audio.size == 0 or sample_rate <= 0 or not np.isfinite(audio).all()
                or not np.any(audio)):
            raise ValueError("empty, silent, or non-finite audio")
        if sample_rate != result.get("sample_rate"):
            raise ValueError("sample-rate metadata does not match audio")
    except Exception as exc:
        raise RuntimeError(f"Invalid CosyVoice3 audio output: {exc}; see {log_path}") from exc
    return {**result, "elapsed": time.perf_counter() - start,
            "prompt_id": run_id, "filename": output.name,
            "log": str(log_path), "metadata_file": str(result_path)}
