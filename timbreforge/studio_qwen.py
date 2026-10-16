"""Qwen Base reference identity reuse, separate from IndexTTS parameters."""
from dataclasses import replace
import json
import hashlib
from pathlib import Path
import uuid
import numpy as np
import soundfile as sf
from timbreforge.config import get_config


def generate_character(request, reference, transcript, character, character_store, *, output_dir=None):
    from timbreforge.backend_qwen import QwenParams, clone
    from timbreforge.studio import _reference, _GENERATION_LOCK
    values = request.qwen_params
    if values is None and character:
        values = character['params'].get('qwen', {})
    params = QwenParams(**(values or {}))
    if request.seed_override is not None:
        params = replace(params, seed=request.seed_override)
    params.validate()
    if request.device not in ('cuda:0', 'cpu'):
        raise ValueError('Qwen device must be cuda:0 or cpu')
    if request.qwen_dtype not in ('float16', 'bfloat16', 'float32'):
        raise ValueError('Qwen dtype must be float16, bfloat16 or float32')
    ref, info = _reference(reference)
    with _GENERATION_LOCK:
        result = clone(ref, transcript, request.text, params, device=request.device,
                       dtype='float32' if request.device == 'cpu' else request.qwen_dtype)
        if hashlib.sha256(ref.read_bytes()).hexdigest() != info['sha256']:
            raise RuntimeError('Reference changed during Qwen generation; no result published')
        # Adapter independently validates FLOAT output and reference consistency.
        x, sr = sf.read(result['file'], dtype='float32', always_2d=True)
        if not x.size or not np.isfinite(x).all() or not np.any(x):
            raise RuntimeError('Invalid Qwen clone output')
    run_id = uuid.uuid4().hex
    directory = Path(output_dir or get_config().outputs_dir / 'studio').resolve()
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / (run_id + '.wav')
    with destination.open('xb') as stream:
        sf.write(stream, x, sr, format='WAV', subtype='FLOAT')
    metadata = dict(schema_version=1, run_id=run_id, task=request.task, mode='voice_clone',
                    backend='qwen3', text=request.text.strip(), reference_audio=str(ref),
                    reference_text=transcript, reference_info=info, instruction='',
                    params={'qwen':params.to_dict()}, warnings=[
                        '声音身份来自固定设计参考与转录，不是重新运行提示词。',
                        'Qwen Base 不支持独立表演指令；未映射 IndexTTS 参数。'],
                    elapsed=result.get('elapsed'), sample_rate=sr, duration=len(x)/sr,
                    file=str(destination), raw_output=result['file'], backend_result=result,
                    character_id=character['id'] if character else None)
    if result.get('text_segments', 1) > 1:
        metadata['warnings'].append(f"长台词已分 {result['text_segments']} 段顺序生成以降低显存峰值，段落间语气衔接请试听确认。")
    if request.save_character_name.strip():
        from timbreforge.characters import CharacterStore
        store = character_store or CharacterStore()
        card = store.create(request.save_character_name, ref, reference_text=transcript,
                            description=request.description, backend='qwen3', params={'qwen':params.to_dict()},
                            provenance={'run_id':run_id,'reference_info':info})
        metadata['character_id'] = card['id']
    path = directory / (run_id+'.json')
    metadata['metadata_file'] = str(path)
    path.write_text(json.dumps(metadata,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return metadata
