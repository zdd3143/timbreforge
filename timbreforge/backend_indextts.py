"""TimbreForge —— IndexTTS 后端（经 ComfyUI HTTP API）

调用链：LoadAudio(参考音频) -> IndexTTS25ModelLoader -> IndexTTS25Synthesize -> SaveAudio

采样参数与 IndexTTS 节点一一对应（这就是"AI 生图式参数"在语音侧的实现）：
    seed / temperature / top_p / top_k / num_beams / repetition_penalty
    + 8 维情感向量 + duration_factor（时长）

前置条件：ComfyUI 正在运行（默认 http://127.0.0.1:8188）。
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from timbreforge.config import get_config

def __getattr__(name: str):
    fields = {'DEFAULT_BASE': 'comfy_url', 'COMFY_INPUT': 'comfy_input_dir',
              'COMFY_OUTPUT': 'comfy_output_dir'}
    if name not in fields:
        raise AttributeError(name)
    return getattr(get_config(), fields[name])


def _setting(name: str):
    # Explicit overrides remain supported for integrations and unit tests.
    return globals()[name] if name in globals() else __getattr__(name)

EMOTION_KEYS = ("happy", "angry", "sad", "fearful",
                "disgusted", "melancholic", "surprised", "calm")


@dataclass
class SynthParams:
    """一次语音合成的全部可控参数（对标生图的采样参数）。"""
    text: str = "欢迎使用 TimbreForge。"
    language: str = "ZH"
    model: str = "IndexTTS-2.5"
    precision: str = "bf16"
    seed: int = 0
    temperature: float = 0.8
    top_p: float = 0.8
    top_k: int = 30
    num_beams: int = 1
    repetition_penalty: float = 3.0
    max_mel_tokens: int = 1500
    max_text_tokens_per_segment: int = 120
    interval_silence_ms: int = 100
    duration_factor: float = 1.0
    diffusion_steps: int = 25
    inference_cfg_rate: float = 0.7
    emotion_mode: str = "same_as_speaker"
    emotion_weight: float = 0.65
    extra_emotion_text: str = ""
    emotion: dict = field(default_factory=dict)
    text_normalization: bool = True
    do_sample: bool = True

    def to_dict(self) -> dict:
        return {**self.__dict__}


class ComfyUnavailable(RuntimeError):
    pass


def ping(base: str | None = None, timeout: float = 4.0) -> tuple[bool, str]:
    """检查 ComfyUI 是否在运行。"""
    base = base or _setting('DEFAULT_BASE')
    try:
        with urllib.request.urlopen(base + "/system_stats", timeout=timeout) as r:
            data = json.load(r)
        dev = (data.get("devices") or [{}])[0]
        return True, f"ComfyUI 在线  设备={dev.get('name', '?')}  " \
                     f"显存={round(dev.get('vram_total', 0)/1024**3, 1)}GB"
    except Exception as e:
        return False, f"ComfyUI 未响应（{type(e).__name__}）: {base}"


def _post(base: str, path: str, payload: dict, timeout: float = 120):
    req = urllib.request.Request(
        base + path, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _get(base: str, path: str, timeout: float = 60):
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return json.load(r)


def copy_to_comfy_input(src: str | Path, name: str | None = None) -> str:
    """把参考音频复制到 ComfyUI 的 input 目录，返回其文件名。"""
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(f"参考音频不存在: {src}")
    input_dir = Path(_setting('COMFY_INPUT'))
    if not input_dir.is_dir():
        raise FileNotFoundError('ComfyUI input directory does not exist. Configure [comfy].input_dir to the actual shared ComfyUI input folder: ' + str(input_dir))
    # ComfyUI caches LoadAudio by name; content identity prevents stale conditions
    # when two sources share a basename or an existing reference changes.
    digest = hashlib.sha256(src.read_bytes()).hexdigest()[:24]
    dst_name = name or f"tf_ref_{digest}{src.suffix.lower()}"
    if Path(dst_name).name != dst_name or '/' in dst_name or '\\' in dst_name:
        raise ValueError('Reference destination must be a plain filename')
    dst = input_dir / dst_name
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return dst_name


def build_graph(ref_name: str, params: SynthParams) -> dict:
    """构造 ComfyUI API 格式的计算图。"""
    inputs = {
        "model": ["2", 0],
        "speaker_audio": ["1", 0],
        "text": params.text,
        "language": params.language,
        "emotion_mode": (
            "extra_emotion_text" if params.extra_emotion_text.strip()
            and params.emotion_mode == "same_as_speaker" else
            "emotion_vector" if any(float(v) > 0 for v in params.emotion.values())
            and params.emotion_mode == "same_as_speaker" else params.emotion_mode
        ),
        "emotion_weight": params.emotion_weight,
        "extra_emotion_text": params.extra_emotion_text,
        "emotion_random": False,
        "duration_factor": params.duration_factor,
        "diffusion_steps": params.diffusion_steps,
        "inference_cfg_rate": params.inference_cfg_rate,
        "interval_silence_ms": params.interval_silence_ms,
        "text_normalization": params.text_normalization,
        "max_text_tokens_per_segment": params.max_text_tokens_per_segment,
        "do_sample": params.do_sample,
        "temperature": params.temperature,
        "top_p": params.top_p,
        "top_k": params.top_k,
        "num_beams": params.num_beams,
        "repetition_penalty": params.repetition_penalty,
        "length_penalty": 0.0,
        "max_mel_tokens": params.max_mel_tokens,
        "seed": params.seed,
    }
    for k in EMOTION_KEYS:
        inputs[k] = float(params.emotion.get(k, 0.0))
    return {
        "1": {"class_type": "LoadAudio", "inputs": {"audio": ref_name}},
        "2": {"class_type": "IndexTTS25ModelLoader", "inputs": {
            "model_name": params.model,
            "precision": params.precision,
            "device": "auto",
            "cuda_kernel": False,
            "deepspeed": False,
            "gpt_accel": False,
            "torch_compile": False,
            "need_emo_text": bool(params.extra_emotion_text),
        }},
        "3": {"class_type": "IndexTTS25Synthesize", "inputs": inputs},
        "4": {"class_type": "SaveAudio", "inputs": {
            "audio": ["3", 0],
            "filename_prefix": "audio/timbreforge",
        }},
    }


def synthesize(ref_audio: str | Path, params: SynthParams,
               base: str | None = None, timeout: float = 1800,
               verbose: bool = True) -> dict:
    from timbreforge.resources import model_slot, release_idle_models
    with model_slot(release_comfy=False):
        try:
            return _synthesize(ref_audio, params, base, timeout, verbose)
        finally:
            # Standard Comfy free alone cannot clear IndexTTS's module cache.
            # Never let a cleanup error hide the actual synthesis exception.
            try:
                release_idle_models(base or _setting('DEFAULT_BASE'))
            except (OSError, ValueError, RuntimeError) as exc:
                if verbose:
                    print(f'闲置模型释放未完成：{exc}')


def _synthesize(ref_audio: str | Path, params: SynthParams,
               base: str | None = None, timeout: float = 1800,
               verbose: bool = True) -> dict:
    """执行一次合成，返回 {"file", "elapsed", "prompt_id", "filename"}。"""
    base = base or _setting('DEFAULT_BASE')
    ok, msg = ping(base)
    if not ok:
        raise ComfyUnavailable(msg)

    # Disk patches do not reload a running ComfyUI Python process. Fail clearly
    # rather than claim the new decoder controls are active on a stale node.
    if params.model == "IndexTTS-2.5":
        schema = _get(base, "/object_info/IndexTTS25Synthesize")
        declared = schema.get("IndexTTS25Synthesize", {}).get("input", {})
        optional = declared.get("optional", {})
        if "diffusion_steps" not in optional or "inference_cfg_rate" not in optional:
            raise RuntimeError("ComfyUI 的 IndexTTS 节点尚未加载生成侧修复；请重启 ComfyUI 后重试。")
    ref_name = copy_to_comfy_input(ref_audio)
    graph = build_graph(ref_name, params)
    cid = str(uuid.uuid4())
    t0 = time.time()
    try:
        res = _post(base, "/prompt", {"prompt": graph, "client_id": cid})
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:1500]
        raise RuntimeError(f"提交失败 HTTP {e.code}: {detail}") from e
    pid = res.get("prompt_id")
    if not pid:
        raise RuntimeError(f"提交失败: {json.dumps(res, ensure_ascii=False)[:800]}")

    if verbose:
        print(f"  prompt_id={pid}  已提交，等待合成…")
    while True:
        time.sleep(3)
        elapsed = time.time() - t0
        hist = _get(base, f"/history/{pid}")
        if pid in hist:
            entry = hist[pid]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                errors = [m[1] for m in status.get('messages', [])
                          if isinstance(m, (list, tuple)) and len(m) > 1 and m[0] == 'execution_error' and isinstance(m[1], dict)]
                detail = '; '.join(f"{e.get('node_type', '')}: {e.get('exception_message', '')}" for e in errors)
                raise RuntimeError('ComfyUI 执行报错：' + (detail[:2000] or '请查看 ComfyUI 日志。'))
            for _nid, out in (entry.get("outputs") or {}).items():
                for item in out.get("audio", []) or []:
                    sub = item.get("subfolder") or ""
                    output_root = Path(_setting('COMFY_OUTPUT')).resolve()
                    p = (output_root / sub / item['filename']).resolve()
                    if not p.is_relative_to(output_root):
                        raise RuntimeError('ComfyUI returned an unsafe output path')
                    if p.is_file():
                        return {"file": str(p), "elapsed": elapsed,
                                "prompt_id": pid, "filename": item["filename"]}
            raise RuntimeError("执行完成但没有音频输出")
        if elapsed > timeout:
            raise TimeoutError(f"等待超过 {timeout}s")
        if verbose and int(elapsed) % 30 < 3:
            print(f"    …已等待 {elapsed:.0f}s")


def load_audio(path: str | Path):
    """读回合成结果。"""
    import soundfile as sf
    x, sr = sf.read(str(path))
    return x, sr
