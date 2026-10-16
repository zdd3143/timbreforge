"""Qwen3-TTS 0.1.1 isolated local file-JSON adapter (no torch imports)."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import uuid

from timbreforge.config import get_config
from timbreforge.resources import is_memory_error, model_slot, release_idle_models

WORKER = Path(__file__).resolve().parent / 'workers' / 'qwen_worker.py'


@dataclass
class QwenParams:
    seed: int = 42
    language: str = 'Chinese'
    do_sample: bool = True
    temperature: float = .9
    top_p: float = 1.
    top_k: int = 50
    repetition_penalty: float = 1.05
    max_new_tokens: int = 1024
    subtalker_dosample: bool = True
    subtalker_temperature: float = .9
    subtalker_top_p: float = 1.
    subtalker_top_k: int = 50

    def validate(self):
        for key in ('seed', 'top_k', 'subtalker_top_k', 'max_new_tokens'):
            if type(getattr(self, key)) is not int:
                raise ValueError(f'{key} must be an integer')
        if not 0 <= self.seed <= 0x7fffffffffffffff:
            raise ValueError('seed out of range')
        if not 0 <= self.top_k <= 1000 or not 0 <= self.subtalker_top_k <= 1000:
            raise ValueError('top_k out of range')
        if not 1 <= self.max_new_tokens <= 8192:
            raise ValueError('max_new_tokens must be 1–8192')
        for key in ('do_sample', 'subtalker_dosample'):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f'{key} must be boolean')
        for key in ('temperature', 'subtalker_temperature', 'top_p', 'subtalker_top_p', 'repetition_penalty'):
            value = getattr(self, key)
            if type(value) not in (float, int) or not math.isfinite(value):
                raise ValueError(f'{key} must be finite')
        if not 0 < self.temperature <= 5 or not 0 < self.subtalker_temperature <= 5:
            raise ValueError('temperature must be >0 and <=5')
        if not 0 < self.top_p <= 1 or not 0 < self.subtalker_top_p <= 1:
            raise ValueError('top_p must be >0 and <=1')
        if not 0 < self.repetition_penalty <= 20:
            raise ValueError('repetition_penalty out of range')
        if not isinstance(self.language, str) or self.language not in {
            'Auto', 'Chinese', 'English', 'Japanese', 'Korean', 'German',
            'French', 'Russian', 'Portuguese', 'Spanish', 'Italian'}:
            raise ValueError('Unsupported Qwen language')
        return self

    def to_dict(self):
        self.validate()
        return asdict(self)


def _text(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 5000:
        raise ValueError(f'{name} must be nonempty text of at most 5000 characters')
    return value.strip()


def _options(device, dtype, attention):
    if device != 'cpu' and (not isinstance(device, str) or not device.startswith('cuda:') or not device[5:].isdigit()):
        raise ValueError('device must be cpu or cuda:N')
    if dtype not in ('float16', 'bfloat16', 'float32'):
        raise ValueError('dtype must be float16/bfloat16/float32')
    if device == 'cpu' and dtype != 'float32':
        raise ValueError('CPU requires explicit dtype=float32; there is no automatic fallback')
    if attention not in ('sdpa', 'eager', 'flash_attention_2'):
        raise ValueError('attention must be sdpa/eager/flash_attention_2')


def _audio(path, *, reference=False):
    import numpy as np
    import soundfile as sf
    p = Path(path).expanduser().resolve()
    info = sf.info(p)
    if info.frames <= 0 or info.samplerate <= 0 or info.channels > 32:
        raise ValueError('Invalid audio dimensions')
    if reference and info.duration > 30:
        raise ValueError('Reference exceeds 30 seconds; no automatic trimming')
    if info.frames * info.channels > 24000 * 3600:
        raise ValueError('Audio exceeds validation sample cap')
    x, sr = sf.read(p, dtype='float32', always_2d=True)
    if not x.size or not np.isfinite(x).all() or not np.any(x):
        raise ValueError('Audio must be finite, nonempty and nonsilent')
    if not reference and (info.format != 'WAV' or info.subtype != 'FLOAT'):
        raise ValueError('Qwen output must be FLOAT WAV')
    return p, sr, len(x) / sr


def _run(mode, text, params, *, instruction='', ref_audio=None, ref_text='',
         device='cuda:0', dtype='float16', attention='sdpa', timeout=1800):
    if not isinstance(params, QwenParams):
        raise ValueError('params must be QwenParams')
    values = params.to_dict()
    text = _text(text, 'text')
    _options(device, dtype, attention)
    if type(timeout) not in (float, int) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('timeout must be positive and finite')
    cfg = get_config()
    python = Path(cfg.qwen_python).expanduser().resolve()
    model = Path(cfg.qwen_design_model if mode == 'voice_design' else cfg.qwen_base_model).expanduser().resolve()
    for required in (python, WORKER, model / 'config.json'):
        if not required.is_file():
            raise FileNotFoundError(f'Required local Qwen file missing: {required}')
    # Sharded and single safetensor releases are both supported; loader checks all shards.
    if not any(model.glob('*.safetensors')):
        raise FileNotFoundError(f'Local Qwen safetensors missing: {model}')
    reference_hash = None
    reference = None
    if mode == 'voice_clone':
        ref_text = _text(ref_text, 'exact reference transcript')
        reference, _, _ = _audio(ref_audio, reference=True)
        reference_hash = hashlib.sha256(reference.read_bytes()).hexdigest()
    else:
        instruction = _text(instruction, 'instruction')
    config_hash = hashlib.sha256((model / 'config.json').read_bytes()).hexdigest()
    outdir = Path(cfg.outputs_dir).resolve() / 'qwen'
    outdir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    output = outdir / f'{run_id}.wav'
    request_file = outdir / f'{run_id}.request.json'
    result_file = outdir / f'{run_id}.result.json'
    log_file = outdir / f'{run_id}.log'
    request = dict(mode=mode, text=text, instruction=instruction, ref_audio=str(reference) if reference else None,
                   ref_text=ref_text, reference_sha256=reference_hash, params=values, model_dir=str(model),
                   model_config_sha256=config_hash, output=str(output), device=device, dtype=dtype, attention=attention)
    request_file.write_text(json.dumps(request, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    env = dict(os.environ, PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
    start = time.perf_counter()
    attempts = 0
    with model_slot(cfg.outputs_dir):
        for attempt in range(2):
            attempts = attempt + 1
            if attempt:
                # The failed process has exited; only this request's partial files
                # are removed. Its cache and model cannot survive the retry.
                result_file.unlink(missing_ok=True)
                output.unlink(missing_ok=True)
                log_file = outdir / f'{run_id}.retry.log'
                release_idle_models()
            try:
                with log_file.open('w', encoding='utf-8') as log:
                    process = subprocess.run([str(python), '-B', str(WORKER), '--request', str(request_file),
                        '--result', str(result_file)], cwd=str(model), env=env, stdout=log,
                        stderr=subprocess.STDOUT, timeout=timeout, check=False,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(f'Qwen exceeded {timeout}s; worker stopped; see {log_file}') from exc
            try:
                result = json.loads(result_file.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                result = None
            if not process.returncode and isinstance(result, dict) and not result.get('error'):
                break
            detail = result.get('error', 'invalid result') if isinstance(result, dict) else 'no valid worker result'
            with log_file.open('rb') as stream:
                stream.seek(max(0, log_file.stat().st_size - 8192))
                tail = stream.read().decode('utf-8', errors='replace')
            memory_failure = is_memory_error(f'{detail} {tail} {process.returncode}')
            if memory_failure and attempt == 0:
                continue
            if memory_failure:
                detail = ('内存或 GPU 显存不足：已退出失败模型并释放闲置功能，自动重试一次仍失败。'
                          '请缩短单次台词或释放其它应用占用；no automatic CPU fallback。')
            raise RuntimeError(f'Qwen worker failed (exit {process.returncode}): {detail}; see {log_file}')
    expected = dict(file=str(output), mode=mode, params=values, seed=params.seed,
                    model_dir=str(model), device=device, dtype=dtype, attention=attention,
                    model_config_sha256=config_hash, qwen_package_version='0.1.1',
                    reference_sha256=reference_hash)
    if any(result.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f'Qwen result path/provenance mismatch; see {log_file}')
    _, sr, duration = _audio(output)
    if type(result.get('sample_rate')) is not int or result['sample_rate'] != sr or type(result.get('duration')) not in (int, float) or not math.isfinite(result['duration']) or abs(result['duration']-duration)>1/sr:
        raise RuntimeError('Qwen result audio metadata mismatch')
    for key in ('load_seconds', 'inference_seconds'):
        if type(result.get(key)) not in (int, float) or not math.isfinite(result[key]) or result[key] < 0:
            raise RuntimeError('Qwen result timing metadata invalid')
    if hashlib.sha256((model/'config.json').read_bytes()).hexdigest() != config_hash:
        raise RuntimeError('Qwen model config changed during generation')
    if reference and hashlib.sha256(reference.read_bytes()).hexdigest() != reference_hash:
        raise RuntimeError('Reference changed during generation')
    return {**result, 'attempts': attempts, 'resource_policy':'one model process; release idle caches; exit after request',
            'elapsed': time.perf_counter()-start, 'log': str(log_file),
            'metadata_file': str(result_file), 'prompt_id': run_id, 'filename': output.name}


def design(text, instruction, params, *, device='cuda:0', dtype='float16', attention='sdpa', timeout=1800):
    return _run('voice_design', text, params, instruction=instruction,
                device=device, dtype=dtype, attention=attention, timeout=timeout)


def clone(ref_audio, ref_text, text, params, *, device='cuda:0', dtype='float16', attention='sdpa', timeout=1800):
    return _run('voice_clone', text, params, ref_audio=ref_audio, ref_text=ref_text,
                device=device, dtype=dtype, attention=attention, timeout=timeout)
