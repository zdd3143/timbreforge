"""Standalone Qwen-TTS 0.1.1 worker. File protocol; one model/process."""
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
import sys
import time
import traceback

GENERATION_KEYS = frozenset({'do_sample','temperature','top_p','top_k','repetition_penalty',
    'max_new_tokens','subtalker_dosample','subtalker_temperature','subtalker_top_p','subtalker_top_k'})


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


def run(request):
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
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
    if mode not in ('voice_design', 'voice_clone'):
        raise ValueError('Unknown Qwen mode')
    params = request['params']
    if set(params) != GENERATION_KEYS | {'seed','language'}:
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
    reference_hash = None
    if mode == 'voice_clone':
        reference = Path(request['ref_audio']).resolve()
        reference_hash = hashlib.sha256(reference.read_bytes()).hexdigest()
        if reference_hash != request['reference_sha256'] or not request['ref_text'].strip():
            raise ValueError('Reference transcript/hash mismatch')
        x, sr = sf.read(reference, dtype='float32', always_2d=True)
        if not x.size or not np.isfinite(x).all() or not np.any(x) or len(x)/sr>30:
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
    try:
        model = Qwen3TTSModel.from_pretrained(str(model_dir), device_map=device,
            dtype=getattr(torch,dtype), attn_implementation=attention, local_files_only=True,
            low_cpu_mem_usage=True)
    finally:
        if cpu_staging:
            modeling_utils.safe_open = original_safe_open
    loaded = time.perf_counter()
    print(f'Qwen: model loaded in {loaded-started:.2f}s; starting {mode}', flush=True)
    # Constructors may alter RNG; set all generators immediately before generation.
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    generation = {key:params[key] for key in GENERATION_KEYS}
    common = dict(text=request['text'], language=params['language'], non_streaming_mode=True, **generation)
    if mode == 'voice_design':
        if not request['instruction'].strip():
            raise ValueError('Design requires an instruction')
        waves, sample_rate = model.generate_voice_design(instruct=request['instruction'], **common)
    else:
        segments = [part for part in clone_segments(request['text']) if part.strip()]
        pieces, sample_rate = [], None
        # Build reference conditioning once; reusing it avoids repeated audio
        # encodes for long text. Each chunk is decoded and moved to CPU separately.
        prompt = model.create_voice_clone_prompt(ref_audio=str(reference),
            ref_text=request['ref_text'], x_vector_only_mode=False)
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
    output = Path(request['output']).resolve()
    # Exclusive create prevents accidental overwrites; no normalization.
    with output.open('xb') as stream:
        sf.write(stream, wave, sample_rate, format='WAV', subtype='FLOAT')
    return dict(file=str(output), mode=mode, params=params, seed=seed, sample_rate=sample_rate,
        duration=len(wave)/sample_rate, output_subtype='FLOAT', model_dir=str(model_dir),
        model_config_sha256=config_hash, qwen_package_version=version, device=device,
        dtype=dtype, attention=attention, reference_sha256=reference_hash,
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
