"""Standalone Qwen-TTS 0.1.1 worker. File protocol; one model/process.

Modes:
  * ``voice_design``: prompt-driven design from an instruction.
  * ``voice_clone``: generate from raw reference audio or a cached clone prompt.
  * ``clone_prompt``: build and persist the reusable conditioning tensors.

A clone prompt file is a ``torch.save`` payload of plain tensors. It is loaded
with ``weights_only=True`` and its transcript/mode must match the request, so a
stale cache can never silently impersonate another take.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import tempfile
import time
import traceback

GENERATION_KEYS = frozenset({'do_sample','temperature','top_p','top_k','repetition_penalty',
    'max_new_tokens','subtalker_dosample','subtalker_temperature','subtalker_top_p','subtalker_top_k'})
CLONE_MODES = ('icl', 'xvector')
CLONE_PROMPT_FORMAT = 'timbreforge-qwen-clone-prompt'
CLONE_PROMPT_VERSION = 1


def _ensure_sox_windows():
    """Use SoX-ng when the legacy ``sox.exe`` is unavailable or incompatible."""
    if os.name != 'nt' or shutil.which('sox'):
        return
    candidate = shutil.which('sox_ng')
    if not candidate:
        roots_list = []
        for base in (os.environ.get('LOCALAPPDATA', ''),
                     str(Path(os.environ.get('USERPROFILE', '')) / 'AppData' / 'Local')):
            if base:
                roots_list.append(Path(base) / 'Microsoft' / 'WinGet' / 'Packages')
        matches = []
        for roots in roots_list:
            if roots.is_dir():
                matches = list(roots.glob('sox_ng.sox_ng_*/sox_ng.exe'))
                if not matches:
                    matches = list(roots.glob('sox_ng.sox_ng_*/*/sox_ng.exe'))
                if matches:
                    break
        candidate = str(matches[0]) if matches else None
    if not candidate:
        return
    shim_dir = Path(tempfile.gettempdir()) / 'timbreforge-sox-shim'
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim = shim_dir / 'sox.exe'
    if not shim.is_file():
        shutil.copy2(candidate, shim)
    os.environ['PATH'] = str(shim_dir) + os.pathsep + os.environ.get('PATH', '')


def clone_segments(text, limit=100):
    """Bound clone KV-cache and decoder peaks; preserve all source characters."""
    segments, current = [], ''
    for sentence in re.findall(r'[^。！？!?\n]*[。！？!?\n]|[^。！？!?\n]+$', text):
        while sentence:
            room = limit - len(current)
            current += sentence[:room]
            sentence = sentence[room:]
            if len(current) == limit:
                segments.append(current)
                current = ''
        if len(current) >= limit // 2:
            segments.append(current)
            current = ''
    if current:
        segments.append(current)
    return segments


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _cpu_tensor(torch, tensor, label):
    if tensor is None:
        return None
    if not torch.is_tensor(tensor):
        raise ValueError(f'{label} must be a torch tensor')
    if tensor.numel() <= 0:
        raise ValueError(f'{label} must not be empty')
    value = tensor.detach().to('cpu').contiguous()
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise ValueError(f'{label} contains non-finite values')
    return value


def _tensor_info(tensor):
    if tensor is None:
        return None
    return {'shape': [int(size) for size in tensor.shape],
            'dtype': str(tensor.dtype), 'numel': int(tensor.numel())}


def _save_prompt(torch, output, *, clone_mode, ref_text, item):
    ref_code = _cpu_tensor(torch, getattr(item, 'ref_code', None), 'ref_code')
    embedding = _cpu_tensor(torch, getattr(item, 'ref_spk_embedding', None), 'ref_spk_embedding')
    if embedding is None:
        raise ValueError('clone prompt requires a speaker embedding')
    if clone_mode == 'icl' and ref_code is None:
        raise ValueError('ICL clone prompt requires reference speech codes')
    payload = {
        'format': CLONE_PROMPT_FORMAT,
        'version': CLONE_PROMPT_VERSION,
        'clone_mode': clone_mode,
        'ref_text': ref_text,
        'x_vector_only': clone_mode == 'xvector',
        'icl_mode': clone_mode == 'icl',
        'ref_code': ref_code,
        'ref_spk_embedding': embedding,
    }
    temporary = Path(output).with_name('.' + Path(output).name + '.tmp')
    try:
        torch.save(payload, str(temporary))
        if Path(output).exists():
            raise FileExistsError(f'clone prompt output already exists: {output}')
        os.rename(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {'ref_code': _tensor_info(ref_code), 'ref_spk_embedding': _tensor_info(embedding)}


def _load_prompt(torch, request, *, device):
    from qwen_tts import VoiceClonePromptItem
    path = Path(request['clone_prompt']).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'clone prompt file missing: {path}')
    if _sha256(path) != request['clone_prompt_sha256']:
        raise ValueError('Clone prompt file hash mismatch; rebuild the cache')
    payload = torch.load(str(path), map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or payload.get('format') != CLONE_PROMPT_FORMAT:
        raise ValueError('Not a TimbreForge clone prompt file')
    if type(payload.get('version')) is not int or payload['version'] != CLONE_PROMPT_VERSION:
        raise ValueError('Unsupported clone prompt version')
    if payload.get('clone_mode') != request['clone_mode']:
        raise ValueError('Clone prompt mode does not match the request')
    if payload.get('ref_text') != request['ref_text']:
        raise ValueError('Clone prompt transcript does not match the reference transcript')
    target = torch.device(device)
    ref_code = payload.get('ref_code')
    embedding = payload.get('ref_spk_embedding')
    if ref_code is not None:
        if not torch.is_tensor(ref_code) or ref_code.numel() <= 0:
            raise ValueError('Clone prompt ref_code is invalid')
        ref_code = ref_code.to(target)
    if not torch.is_tensor(embedding) or embedding.numel() <= 0:
        raise ValueError('Clone prompt speaker embedding is invalid')
    embedding = embedding.to(target)
    return [VoiceClonePromptItem(
        ref_code=ref_code,
        ref_spk_embedding=embedding,
        x_vector_only_mode=bool(payload.get('x_vector_only')),
        icl_mode=bool(payload.get('icl_mode')),
        ref_text=payload.get('ref_text'),
    )]


def _adapter_artifact_hash(directory):
    digest = hashlib.sha256()
    files = []
    root = Path(directory).resolve()
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('Adapter directory must not contain symlinks')
        if path.is_file():
            files.append((path.relative_to(root).as_posix(), path))
    for relative, path in sorted(files):
        digest.update(relative.encode('utf-8'))
        digest.update(b'\\0')
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b''):
                digest.update(chunk)
    return digest.hexdigest()


def _load_lora_adapter(model, request):
    """Attach a PEFT LoRA to Qwen's inner talker model.

    Community Qwen3-TTS adapters target the inner ``talker.model``.  We keep
    the attachment in the worker process so the base model and adapter never
    become mutable global state in the web process.
    """
    adapter_path = request.get('adapter_path')
    if not adapter_path:
        return None
    if request.get('mode') != 'voice_clone':
        raise ValueError('Community adapters are supported only for voice_clone')
    directory = Path(adapter_path).resolve()
    config_path = directory / 'adapter_config.json'
    weights_path = directory / 'adapter_model.safetensors'
    if not directory.is_dir() or directory.is_symlink() or not config_path.is_file():
        raise FileNotFoundError(f'Adapter directory is missing: {directory}')
    if not weights_path.is_file():
        raise ValueError('Adapter must contain adapter_model.safetensors')
    expected = request.get('adapter_sha256')
    if not isinstance(expected, str) or _adapter_artifact_hash(directory) != expected:
        raise ValueError('Adapter artifact hash mismatch; re-import it in TimbreForge')
    scale = request.get('adapter_scale', 1.0)
    if not isinstance(scale, (int, float)) or not 0 <= float(scale) <= 2:
        raise ValueError('Invalid adapter scale')
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError('加载社区 LoRA 需要在 Qwen 环境安装 peft。') from exc
    root = getattr(model, 'model', None)
    if root is None:
        raise RuntimeError('Qwen3-TTS wrapper has no underlying model for LoRA')
    parent = getattr(root, 'talker', None)
    if parent is None:
        raise RuntimeError('Qwen3-TTS model has no talker module for LoRA')
    inner = getattr(parent, 'model', None)
    if inner is None:
        raise RuntimeError('Qwen3-TTS talker has no inner model for LoRA')
    wrapped = PeftModel.from_pretrained(inner, str(directory), is_trainable=False)
    try:
        wrapped.set_adapter('default')
    except (AttributeError, ValueError):
        pass
    # PEFT stores per-adapter multipliers on every injected LoRA layer.
    for module in wrapped.modules():
        scaling = getattr(module, 'scaling', None)
        if isinstance(scaling, dict) and 'default' in scaling:
            scaling['default'] = float(scale)
    parent.model = wrapped
    return {'path': str(directory), 'sha256': expected, 'scale': float(scale),
            'base_target': 'talker.model'}


def run(request):
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
    _ensure_sox_windows()
    import numpy as np
    import soundfile as sf
    import torch
    # Avoid pathological thread oversubscription in CPU-side reference features
    # and checkpoint staging on Windows laptops. Does not change sampling.
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    from qwen_tts import Qwen3TTSModel
    version = importlib.metadata.version('qwen-tts')
    if version != '0.1.1':
        raise RuntimeError(f'qwen-tts==0.1.1 required, installed {version}')
    mode = request['mode']
    if mode not in ('voice_design', 'voice_clone', 'clone_prompt'):
        raise ValueError('Unknown Qwen mode')
    clone_mode = request.get('clone_mode') or ('none' if mode == 'voice_design' else 'icl')
    if mode == 'voice_design':
        clone_mode = 'none'
    if clone_mode not in CLONE_MODES + ('none',):
        raise ValueError('Invalid clone mode')
    params = request.get('params')
    seed = None
    if mode != 'clone_prompt':
        if not isinstance(params, dict):
            raise ValueError('Generation modes require parameters')
        if set(params) != GENERATION_KEYS | {'seed', 'language'}:
            raise ValueError('Unexpected or missing generation parameters')
        seed = params['seed']
        if type(seed) is not int or not 0 <= seed <= 0x7fffffffffffffff:
            raise ValueError('Invalid seed')
    device, dtype, attention = (request[k] for k in ('device','dtype','attention'))
    if device != 'cpu' and (not isinstance(device,str) or not device.startswith('cuda:') or not device[5:].isdigit()):
        raise ValueError('Invalid device')
    if dtype not in ('float16','bfloat16','float32') or device == 'cpu' and dtype != 'float32':
        raise ValueError('CPU requires explicit float32')
    if attention not in ('sdpa','eager','flash_attention_2'):
        raise ValueError('Invalid attention')
    cuda = device != 'cpu'
    if cuda and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA device is not available; no automatic fallback')
    model_dir = Path(request['model_dir']).resolve()
    config_hash = hashlib.sha256((model_dir/'config.json').read_bytes()).hexdigest()
    if config_hash != request['model_config_sha256']:
        raise ValueError('Model config hash mismatch')
    reference = None
    reference_hash = None
    prompt_items = None
    prompt_hash = request.get('clone_prompt_sha256')
    if mode == 'voice_design':
        if not request['instruction'].strip():
            raise ValueError('Design requires an instruction')
    elif mode == 'clone_prompt':
        reference = Path(request['ref_audio']).resolve()
        reference_hash = _sha256(reference)
        if reference_hash != request['reference_sha256']:
            raise ValueError('Reference audio hash mismatch')
        if not request['ref_text'].strip():
            raise ValueError('Clone prompt requires the exact reference transcript')
        x, sr = sf.read(reference, dtype='float32', always_2d=True)
        if not x.size or not np.isfinite(x).all() or not np.any(x) or len(x)/sr > 30:
            raise ValueError('Reference invalid; never auto-trim')
        del x
    else:  # voice_clone
        if not request['ref_text'].strip():
            raise ValueError('Clone requires the exact reference transcript')
        if request.get('clone_prompt'):
            reference_hash = request['reference_sha256']
            if not isinstance(reference_hash, str) or len(reference_hash) != 64:
                raise ValueError('Cached clone requires the reference audio hash')
        else:
            reference = Path(request['ref_audio']).resolve()
            reference_hash = _sha256(reference)
            if reference_hash != request['reference_sha256']:
                raise ValueError('Reference transcript/hash mismatch')
            x, sr = sf.read(reference, dtype='float32', always_2d=True)
            if not x.size or not np.isfinite(x).all() or not np.any(x) or len(x)/sr > 30:
                raise ValueError('Reference invalid; never auto-trim')
            del x  # Validation must not retain a duplicate reference during inference.
    print(f'Qwen: loading {mode} on {device}, dtype={dtype}, attention={attention}', flush=True)
    started = time.perf_counter()
    # Windows torch 2.6 direct CUDA safetensors get_slice can access-violate.
    # Stage safe tensor reads on CPU, then Transformers explicitly transfers each
    # parameter to the requested device. This is not CPU inference or offloading.
    # Process-local only: no installed package or checkpoint modifications.
    cpu_staging = sys.platform == 'win32' and cuda
    if cpu_staging:
        import transformers.modeling_utils as modeling_utils
        original_safe_open = modeling_utils.safe_open
        def staged_safe_open(filename, *args, **kwargs):
            kwargs['device'] = 'cpu'
            return original_safe_open(filename, *args, **kwargs)
        modeling_utils.safe_open = staged_safe_open
    model_kwargs = dict(device_map=device, dtype=getattr(torch,dtype),
                        attn_implementation=attention, local_files_only=True,
                        low_cpu_mem_usage=True)
    try:
        try:
            model = Qwen3TTSModel.from_pretrained(str(model_dir), **model_kwargs)
        except OSError as exc:
            # Windows error 1455 means the temporary CPU staging map exceeds
            # the page-file commit limit. A direct CUDA load uses less system
            # commit and is safe to try after the staged attempt is released.
            message = str(exc).lower()
            if not (cpu_staging and ('1455' in message or 'pagefile' in message or '页面文件' in message)):
                raise
            print('Qwen: CPU staging exceeded the Windows page-file limit; retrying direct CUDA load.', flush=True)
            import gc
            gc.collect()
            if cuda:
                torch.cuda.empty_cache()
            modeling_utils.safe_open = original_safe_open
            model = Qwen3TTSModel.from_pretrained(str(model_dir), **model_kwargs)
    finally:
        if cpu_staging:
            modeling_utils.safe_open = original_safe_open
    adapter_info = _load_lora_adapter(model, request)
    loaded = time.perf_counter()
    print(f'Qwen: model loaded in {loaded-started:.2f}s; starting {mode}', flush=True)
    output = Path(request['output']).resolve()

    if mode == 'clone_prompt':
        # Constructors may alter RNG, but prompt creation performs no sampling.
        result_prompt = model.create_voice_clone_prompt(
            ref_audio=str(reference), ref_text=request['ref_text'],
            x_vector_only_mode=clone_mode == 'xvector')
        if not isinstance(result_prompt, (list, tuple)) or len(result_prompt) != 1:
            raise ValueError('Expected exactly one clone prompt item')
        tensor_info = _save_prompt(torch, output, clone_mode=clone_mode,
                                   ref_text=request['ref_text'], item=result_prompt[0])
        return dict(file=str(output), mode=mode, model_dir=str(model_dir),
                    model_config_sha256=config_hash, qwen_package_version=version,
                    device=device, dtype=dtype, attention=attention,
                    reference_sha256=reference_hash, clone_mode=clone_mode,
                    clone_prompt_sha256=None, prompt_sha256=_sha256(output),
                    adapter=adapter_info,
                    tensor_info=tensor_info, prompt_bytes=output.stat().st_size,
                    safetensors_load_staging='cpu_to_requested_device' if cpu_staging else 'standard',
                    load_seconds=loaded-started,
                    inference_seconds=time.perf_counter()-loaded,
                    cuda_peak_memory_bytes=int(torch.cuda.max_memory_allocated(device)) if cuda else None)

    # Constructors may alter RNG; set all generators immediately before generation.
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    generation = {key:params[key] for key in GENERATION_KEYS}
    common = dict(text=request['text'], language=params['language'], non_streaming_mode=True, **generation)
    segments = [request['text']]
    if mode == 'voice_design':
        waves, sample_rate = model.generate_voice_design(instruct=request['instruction'], **common)
    else:
        segments = [part for part in clone_segments(request['text']) if part.strip()]
        pieces, sample_rate = [], None
        # Build reference conditioning once; reusing it avoids repeated audio
        # encodes for long text. Each chunk is decoded and moved to CPU separately.
        if request.get('clone_prompt'):
            prompt = _load_prompt(torch, request, device=device)
        else:
            prompt = model.create_voice_clone_prompt(ref_audio=str(reference),
                ref_text=request['ref_text'], x_vector_only_mode=clone_mode == 'xvector')
        for index, segment in enumerate(segments):
            print(f'Qwen: clone segment {index + 1}/{len(segments)}', flush=True)
            chunk, rate = model.generate_voice_clone(voice_clone_prompt=prompt,
                **dict(common, text=segment))
            if len(chunk) != 1 or (sample_rate is not None and rate != sample_rate):
                raise ValueError('Invalid clone chunk or sample rate')
            piece = np.asarray(chunk[0], dtype=np.float32)
            if piece.ndim != 1 or not piece.size or not np.isfinite(piece).all():
                raise ValueError('Invalid clone chunk waveform')
            pieces.append(piece)
            sample_rate = rate
        waves = [np.concatenate(pieces)]
    generated = time.perf_counter()
    if not isinstance(waves, (list, tuple)) or len(waves) != 1:
        raise ValueError('Expected exactly one generated waveform')
    wave = np.asarray(waves[0], dtype=np.float32)
    if wave.ndim != 1 or not wave.size or not np.isfinite(wave).all() or not np.any(wave):
        raise ValueError('Generated audio must be finite, nonsilent mono')
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError('Invalid output sample rate')
    # Exclusive create prevents accidental overwrites; no normalization.
    with output.open('xb') as stream:
        sf.write(stream, wave, sample_rate, format='WAV', subtype='FLOAT')
    return dict(file=str(output), mode=mode, params=params, seed=seed, sample_rate=sample_rate,
        duration=len(wave)/sample_rate, output_subtype='FLOAT', model_dir=str(model_dir),
        model_config_sha256=config_hash, qwen_package_version=version, device=device,
        dtype=dtype, attention=attention, reference_sha256=reference_hash,
        clone_mode=clone_mode, clone_prompt_sha256=prompt_hash if mode == 'voice_clone' and request.get('clone_prompt') else None,
        adapter=adapter_info,
        reference_cropped=False, ref_text_used=mode=='voice_clone',
        text_segments=len(segments) if mode=='voice_clone' else 1,
        safetensors_load_staging='cpu_to_requested_device' if cpu_staging else 'standard',
        load_seconds=loaded-started,inference_seconds=generated-loaded,
        cuda_peak_memory_bytes=int(torch.cuda.max_memory_allocated(device)) if cuda else None)


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--request',required=True)
    parser.add_argument('--result',required=True)
    args=parser.parse_args(argv)
    try:
        request=json.loads(Path(args.request).read_text(encoding='utf-8'))
        result=run(request)
        code=0
    except Exception as exc:
        traceback.print_exc()
        result={'error':f'{type(exc).__name__}: {exc}'}
        code=1
    Path(args.result).write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return code


if __name__=='__main__':
    raise SystemExit(main())

