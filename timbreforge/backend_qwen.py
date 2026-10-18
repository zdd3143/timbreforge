"""Qwen3-TTS 0.1.1 isolated local file-JSON adapter (no torch imports).

Three request modes share one worker protocol: ``voice_design`` (prompt-driven
design), ``voice_clone`` (reference audio, or a cached clone prompt) and
``clone_prompt`` (build and persist the reusable conditioning tensors).
"""
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
QWEN_PACKAGE_VERSION = '0.1.1'
CLONE_MODES = ('icl', 'xvector')
_HASH_RE_LENGTH = 64


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


def _hash(value, name):
    if not isinstance(value, str) or len(value) != _HASH_RE_LENGTH or any(
            character not in '0123456789abcdef' for character in value):
        raise ValueError(f'{name} must be a lowercase 64-character SHA-256 digest')
    return value


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


def _base_model():
    cfg = get_config()
    python = Path(cfg.qwen_python).expanduser().resolve()
    model = Path(cfg.qwen_base_model).expanduser().resolve()
    for required in (python, WORKER, model / 'config.json'):
        if not required.is_file():
            raise FileNotFoundError(f'Required local Qwen file missing: {required}')
    if not any(model.glob('*.safetensors')):
        raise FileNotFoundError(f'Local Qwen safetensors missing: {model}')
    config_hash = hashlib.sha256((model / 'config.json').read_bytes()).hexdigest()
    return cfg, python, model, config_hash


def _write_request(path, request):
    path.write_text(json.dumps(request, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding='utf-8')


def _worker_env():
    return dict(os.environ, PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1',
                TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')


def _kill_process(process):
    """Terminate a worker and any child it may have started."""
    try:
        if os.name == 'nt':
            subprocess.run(['taskkill', '/F', '/T', '/PID', str(process.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           creationflags=subprocess.CREATE_NO_WINDOW, timeout=15)
        else:
            process.kill()
    except (OSError, subprocess.SubprocessError):
        try:
            process.kill()
        except OSError:
            pass


def _spawn(command, *, cwd, env, log_file, timeout, cancel):
    """Run the worker; ``subprocess.run`` when no cancel token is supplied.

    Project tests and simple CLI paths patch ``subprocess.run`` directly, so the
    default path keeps that contract. A polling Popen path is used only when a
    caller needs prompt cancellation (UI/resource manager).
    """
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    if cancel is None:
        return subprocess.run(command, cwd=str(cwd), env=env, stdout=log_file,
                              stderr=subprocess.STDOUT, timeout=timeout, check=False,
                              creationflags=flags)
    process = subprocess.Popen(command, cwd=str(cwd), env=env, stdout=log_file,
                               stderr=subprocess.STDOUT, creationflags=flags)
    deadline = time.monotonic() + timeout
    try:
        while process.poll() is None:
            if cancel.is_set():
                _kill_process(process)
                raise RuntimeError('任务已取消；已停止模型进程并清理临时文件。')
            if time.monotonic() >= deadline:
                _kill_process(process)
                raise TimeoutError(f'Qwen exceeded {timeout}s; worker stopped')
            time.sleep(.1)
    finally:
        if process.poll() is None:
            _kill_process(process)
    return process


def _execute(request, run_id, *, timeout, cancel=None):
    """Run one worker request under the cross-process model slot."""
    cfg = get_config()
    output = Path(request['output'])
    outdir = output.parent
    outdir.mkdir(parents=True, exist_ok=True)
    request_file = outdir / f'{run_id}.request.json'
    result_file = outdir / f'{run_id}.result.json'
    log_file = outdir / f'{run_id}.log'
    _write_request(request_file, request)
    env = _worker_env()
    command = [str(Path(cfg.qwen_python).expanduser().resolve()), '-B', str(WORKER),
               '--request', str(request_file), '--result', str(result_file)]
    model_dir = Path(request['model_dir'])
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
                    process = _spawn(command, cwd=model_dir, env=env, log_file=log,
                                     timeout=timeout, cancel=cancel)
            except subprocess.TimeoutExpired as exc:
                output.unlink(missing_ok=True)
                raise TimeoutError(f'Qwen exceeded {timeout}s; worker stopped; see {log_file}') from exc
            except RuntimeError:
                # Cancellation (or any early stop) must not publish a partial file.
                output.unlink(missing_ok=True)
                result_file.unlink(missing_ok=True)
                raise
            try:
                result = json.loads(result_file.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                result = None
            if not process.returncode and isinstance(result, dict) and not result.get('error'):
                return {**result, 'attempts': attempts, 'log': str(log_file),
                        'metadata_file': str(result_file), 'request_file': str(request_file),
                        'elapsed': time.perf_counter() - start}
            detail = result.get('error', 'invalid result') if isinstance(result, dict) else 'no valid worker result'
            with log_file.open('rb') as stream:
                stream.seek(max(0, log_file.stat().st_size - 8192))
                tail = stream.read().decode('utf-8', errors='replace')
            if result is None and tail.strip():
                detail = f'no valid worker result; worker log tail: {tail.strip()[-4000:]}'
            memory_failure = is_memory_error(f'{detail} {tail} {process.returncode}')
            if memory_failure and attempt == 0:
                continue
            if memory_failure:
                pagefile_failure = any(token in f'{detail} {tail}'.lower()
                                       for token in ('1455', 'pagefile', '页面文件'))
                if pagefile_failure:
                    detail = ('Windows 页面文件不足：已退出失败模型并释放闲置功能，自动重试一次仍失败。'
                              '请将系统页面文件上限提高到至少 32 GB 后重启电脑，再重新生成。')
                else:
                    detail = ('内存或 GPU 显存不足：已退出失败模型并释放闲置功能，自动重试一次仍失败。'
                              '请缩短单次台词或释放其它应用占用；no automatic CPU fallback。')
            raise RuntimeError(f'Qwen worker failed (exit {process.returncode}): {detail}; see {log_file}')
    raise RuntimeError('Qwen worker failed without a result')  # pragma: no cover


def _verify_timings(result):
    for key in ('load_seconds', 'inference_seconds'):
        if type(result.get(key)) not in (int, float) or not math.isfinite(result[key]) or result[key] < 0:
            raise RuntimeError('Qwen result timing metadata invalid')


def _verify_audio_result(result, request, expected):
    """Validate a design/clone result against the request that produced it."""
    if any(result.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Qwen result path/provenance mismatch; see {result.get('log', '')}")
    path, sr, duration = _audio(result['file'])
    if (type(result.get('sample_rate')) is not int or result['sample_rate'] != sr
            or type(result.get('duration')) not in (int, float)
            or not math.isfinite(result['duration'])
            or abs(result['duration'] - duration) > 1 / sr):
        raise RuntimeError('Qwen result audio metadata mismatch')
    _verify_timings(result)
    model = Path(request['model_dir'])
    if hashlib.sha256((model / 'config.json').read_bytes()).hexdigest() != request['model_config_sha256']:
        raise RuntimeError('Qwen model config changed during generation')
    return path


def _run(mode, text, params, *, instruction='', ref_audio=None, ref_text='',
         device='cuda:0', dtype='float16', attention='sdpa', timeout=1800,
         clone_prompt=None, clone_mode='icl', reference_sha256=None, cancel=None,
         adapter_path=None, adapter_scale=1.0, adapter_sha256=None):
    if mode not in ('voice_design', 'voice_clone'):
        raise ValueError('Unknown Qwen mode')
    if not isinstance(params, QwenParams):
        raise ValueError('params must be QwenParams')
    values = params.to_dict()
    text = _text(text, 'text')
    _options(device, dtype, attention)
    if adapter_path is not None:
        if not isinstance(adapter_path, (str, Path)) or not str(adapter_path).strip():
            raise ValueError('adapter_path 必须是本地适配器目录。')
        if not isinstance(adapter_scale, (int, float)) or not 0 <= float(adapter_scale) <= 2:
            raise ValueError('adapter_scale 必须在 0–2 之间。')
        if not isinstance(adapter_sha256, str) or len(adapter_sha256) != 64:
            raise ValueError('adapter_sha256 必须用于校验社区适配器。')
    if type(timeout) not in (float, int) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('timeout must be positive and finite')
    if clone_mode not in CLONE_MODES:
        raise ValueError('clone_mode must be icl or xvector')
    if mode == 'voice_design':
        cfg = get_config()
        python = Path(cfg.qwen_python).expanduser().resolve()
        model = Path(cfg.qwen_design_model).expanduser().resolve()
        for required in (python, WORKER, model / 'config.json'):
            if not required.is_file():
                raise FileNotFoundError(f'Required local Qwen file missing: {required}')
        if not any(model.glob('*.safetensors')):
            raise FileNotFoundError(f'Local Qwen safetensors missing: {model}')
        config_hash = hashlib.sha256((model / 'config.json').read_bytes()).hexdigest()
        instruction = _text(instruction, 'instruction')
        prompt_path = None
        prompt_hash = None
        if ref_audio is not None or reference_sha256 is not None:
            raise ValueError('VoiceDesign does not accept a reference recording or prompt')
    else:
        cfg, python, model, config_hash = _base_model()
        ref_text = _text(ref_text, 'exact reference transcript')
        if clone_prompt is not None:
            if ref_audio is not None:
                raise ValueError('Pass either ref_audio or clone_prompt, not both')
            prompt_path = Path(clone_prompt).expanduser().resolve()
            if not prompt_path.is_file():
                raise FileNotFoundError(f'Clone prompt not found: {prompt_path}')
            prompt_hash = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
            reference_hash = _hash(reference_sha256, 'reference_sha256')
            reference_value = None
        else:
            prompt_path = None
            prompt_hash = None
            reference, _, _ = _audio(ref_audio, reference=True)
            reference_hash = hashlib.sha256(reference.read_bytes()).hexdigest()
            reference_value = str(reference)

    outdir = Path(cfg.outputs_dir).resolve() / 'qwen'
    outdir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    output = outdir / f'{run_id}.wav'
    request = dict(mode=mode, text=text, instruction=instruction,
                   ref_audio=reference_value if mode == 'voice_clone' else None,
                   ref_text=ref_text if mode == 'voice_clone' else '',
                   reference_sha256=reference_hash if mode == 'voice_clone' else None,
                   clone_prompt=str(prompt_path) if prompt_path else None,
                   clone_prompt_sha256=prompt_hash,
                   adapter_path=str(adapter_path) if adapter_path else None,
                   adapter_scale=float(adapter_scale), adapter_sha256=adapter_sha256,
                   clone_mode=clone_mode if mode == 'voice_clone' else 'none',
                   params=values, model_dir=str(model), model_config_sha256=config_hash,
                   output=str(output), device=device, dtype=dtype, attention=attention)
    if mode == 'voice_design':
        # Design has no reference; keep provenance explicit instead of absent keys.
        request.update(ref_audio=None, ref_text='', reference_sha256=None,
                       clone_prompt=None, clone_prompt_sha256=None, clone_mode='none')
    result = _execute(request, run_id, timeout=timeout, cancel=cancel)
    expected = dict(file=str(output), mode=mode, params=values, seed=params.seed,
                    model_dir=str(model), device=device, dtype=dtype, attention=attention,
                    model_config_sha256=config_hash, qwen_package_version=QWEN_PACKAGE_VERSION,
                    reference_sha256=request['reference_sha256'], clone_mode=request['clone_mode'])
    _verify_audio_result(result, request, expected)
    if mode == 'voice_clone':
        if prompt_path is not None:
            if result.get('clone_prompt_sha256') != prompt_hash:
                raise RuntimeError('Qwen clone prompt provenance mismatch')
        elif result.get('clone_prompt_sha256') not in (None, ''):
            raise RuntimeError('Qwen reported a clone prompt that was not requested')
    if request['reference_sha256'] is not None:
        reference = Path(ref_audio).expanduser().resolve() if ref_audio else None
        if reference is not None and hashlib.sha256(reference.read_bytes()).hexdigest() != reference_hash:
            raise RuntimeError('Reference changed during generation')
    return {**result, 'resource_policy': 'one model process; release idle caches; exit after request',
            'filename': output.name, 'prompt_id': run_id}


def design(text, instruction, params, *, device='cuda:0', dtype='float16', attention='sdpa',
           timeout=1800, cancel=None):
    return _run('voice_design', text, params, instruction=instruction,
                device=device, dtype=dtype, attention=attention, timeout=timeout,
                cancel=cancel)


def clone(ref_audio, ref_text, text, params, *, device='cuda:0', dtype='float16',
          attention='sdpa', timeout=1800, clone_prompt=None, clone_mode='icl',
          reference_sha256=None, cancel=None, adapter_path=None,
          adapter_scale=1.0, adapter_sha256=None):
    """Clone with raw reference audio or a cached clone prompt.

    A cached prompt skips reference re-encoding; its bytes and the Card audio
    hash are verified so a stale cache can never silently impersonate a voice.
    """
    return _run('voice_clone', text, params, ref_audio=ref_audio, ref_text=ref_text,
                device=device, dtype=dtype, attention=attention, timeout=timeout,
                clone_prompt=clone_prompt, clone_mode=clone_mode,
                reference_sha256=reference_sha256, cancel=cancel,
                adapter_path=adapter_path, adapter_scale=adapter_scale,
                adapter_sha256=adapter_sha256)


def create_clone_prompt(ref_audio, ref_text, *, clone_mode='icl', device='cuda:0',
                        dtype='float16', attention='sdpa', timeout=1800,
                        output_dir=None, cancel=None):
    """Build the reusable Qwen conditioning tensors and persist them as one file."""
    if clone_mode not in CLONE_MODES:
        raise ValueError('clone_mode must be icl or xvector')
    ref_text = _text(ref_text, 'exact reference transcript')
    _options(device, dtype, attention)
    if type(timeout) not in (float, int) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('timeout must be positive and finite')
    cfg, python, model, config_hash = _base_model()
    reference, _, _ = _audio(ref_audio, reference=True)
    reference_hash = hashlib.sha256(reference.read_bytes()).hexdigest()
    outdir = Path(output_dir or (Path(cfg.outputs_dir) / 'qwen' / 'clone-prompts')).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    output = outdir / f'{run_id}.pt'
    request = dict(mode='clone_prompt', text='', instruction='', ref_audio=str(reference),
                   ref_text=ref_text, reference_sha256=reference_hash,
                   clone_prompt=None, clone_prompt_sha256=None, clone_mode=clone_mode,
                   params=None, model_dir=str(model), model_config_sha256=config_hash,
                   output=str(output), device=device, dtype=dtype, attention=attention)
    result = _execute(request, run_id, timeout=timeout, cancel=cancel)
    expected = dict(file=str(output), mode='clone_prompt', model_dir=str(model),
                    device=device, dtype=dtype, attention=attention,
                    model_config_sha256=config_hash, qwen_package_version=QWEN_PACKAGE_VERSION,
                    reference_sha256=reference_hash, clone_mode=clone_mode)
    if any(result.get(key) != value for key, value in expected.items()):
        raise RuntimeError('Qwen clone prompt provenance mismatch')
    _verify_timings(result)
    if not output.is_file() or output.stat().st_size <= 0:
        raise RuntimeError('Qwen clone prompt file was not created')
    prompt_hash = hashlib.sha256(output.read_bytes()).hexdigest()
    if result.get('prompt_sha256') != prompt_hash:
        raise RuntimeError('Qwen clone prompt hash mismatch')
    tensor_info = result.get('tensor_info')
    if not isinstance(tensor_info, dict) or not tensor_info:
        raise RuntimeError('Qwen clone prompt tensor metadata missing')
    if hashlib.sha256(reference.read_bytes()).hexdigest() != reference_hash:
        raise RuntimeError('Reference changed while creating the clone prompt')
    if hashlib.sha256((model / 'config.json').read_bytes()).hexdigest() != config_hash:
        raise RuntimeError('Qwen model config changed while creating the clone prompt')
    return {**result, 'prompt_file': str(output), 'prompt_sha256': prompt_hash,
            'prompt_bytes': output.stat().st_size,
            'resource_policy': 'one model process; release idle caches; exit after request',
            'filename': output.name, 'prompt_id': run_id}

