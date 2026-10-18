"""The single supported path from a character card to new speech.

Every generation records the character ID and role version, model identity,
parameters, seed, reference/transcript hashes, clone-prompt decision and output
hash next to the audio. Failures never publish a partial take, and stale clone
prompts fall back to an ordinary clone with a user-visible reason.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import numpy as np
import soundfile as sf

from timbreforge import backend_qwen, clone_prompts
from timbreforge.characters import CharacterStore, audio_path
from timbreforge.config import get_config

SCHEMA_VERSION = 1
MAX_TEXT = 5000


class GenerationError(RuntimeError):
    """A generation failed after its temporary files were cleaned up."""


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_text(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('台词不能为空。')
    if len(text) > MAX_TEXT:
        raise ValueError(f'单次台词不得超过 {MAX_TEXT} 字；长文本请使用长文本生成。')
    return text.strip()


def _params(card, params, seed):
    from timbreforge.backend_qwen import QwenParams
    if params is None:
        values = (card.get('params') or {}).get('qwen') or {}
        params = QwenParams(**(values if isinstance(values, dict) else {}))
    elif isinstance(params, dict):
        params = QwenParams(**params)
    elif not isinstance(params, QwenParams):
        raise ValueError('params 必须是 QwenParams 或字典。')
    params.validate()
    if seed is not None:
        if type(seed) is not int or not 0 <= seed <= 0x7FFFFFFFFFFFFFFF:
            raise ValueError('seed 必须是 0 到 2^63-1 的整数。')
        params = replace(params, seed=seed)
    elif params.seed is None:
        params = replace(params, seed=42)
    return params


def _options(device, dtype):
    if device != 'cpu' and (not isinstance(device, str) or not device.startswith('cuda:')
                            or not device[5:].isdigit()):
        raise ValueError('device 必须是 cpu 或 cuda:N。')
    if dtype not in ('float16', 'bfloat16', 'float32'):
        raise ValueError('dtype 必须是 float16/bfloat16/float32。')
    if device == 'cpu' and dtype != 'float32':
        raise ValueError('CPU 生成必须显式使用 float32。')


def generate_for_character(role_id, text, *, params=None, seed=None, device='cuda:0',
                           dtype='float16', use_clone_prompt=True, create_prompt=False,
                           store=None, output_dir=None, cancel=None, progress=None,
                           backend=None) -> dict:
    """Generate one take from a character card and persist full provenance."""
    store = store or CharacterStore()
    text = _validate_text(text)
    _options(device, dtype)
    card = store.load(role_id)
    if card.get('backend') != 'qwen3':
        raise GenerationError('统一生成入口目前只支持 Qwen3 角色；'
                              'IndexTTS/CosyVoice 后端仍可在进阶页面单独使用。')
    transcript = (card.get('reference_text') or '').strip()
    if not transcript:
        raise GenerationError('这张角色卡没有参考音频逐字稿。请先在「音频修整」或「创建角色」'
                              '中填写准确逐字稿，再生成。')
    values = _params(card, params, seed)
    reference = Path(audio_path(card)).resolve()
    if not reference.is_file():
        raise GenerationError(f'角色参考音频不存在：{reference}。请检查角色卡文件。')
    audio_hash = card.get('canonical_sha256') or card.get('reference_sha256')
    warnings = ['声音身份来自角色卡的参考音频与逐字稿；不是重新运行提示词。',
                'Qwen Base 不支持独立表演指令；未映射 IndexTTS 表演参数。']
    adapter_info = None
    try:
        from timbreforge.adapters import AdapterError, AdapterRegistry
        adapter_info = AdapterRegistry().resolve_binding(card)
    except AdapterError as exc:
        raise GenerationError(f'角色绑定的社区适配器不可用：{exc}') from exc
    if adapter_info:
        warnings.append(f'已加载社区适配器：{adapter_info["entry"]["name"]}，scale={adapter_info["scale"]:g}。')
    prompt_info = {'used': False, 'path': None, 'sha256': None, 'status': 'disabled',
                   'reason': ''}
    prompt = None
    if use_clone_prompt:
        report = clone_prompts.validate_clone_prompt(role_id, store=store)
        status = report['status']
        if status == 'valid':
            prompt = clone_prompts.load_clone_prompt(role_id, store=store)
            prompt_info = {'used': True, 'path': prompt['path'],
                           'sha256': prompt['metadata']['prompt_sha256'],
                           'status': 'valid', 'reason': ''}
            warnings.append('已复用本机 Clone Prompt 缓存，跳过了参考音频重新编码。')
        elif create_prompt and status in ('missing', 'stale', 'corrupt'):
            try:
                created = clone_prompts.create_clone_prompt(
                    role_id, store=store, device=device, dtype=dtype, cancel=cancel)
                prompt = {'path': created['path'],
                          'metadata': created['metadata'],
                          'mode': created['metadata']['clone_mode'],
                          'sha256': created['metadata']['prompt_sha256']}
                prompt_info = {'used': True, 'path': created['path'],
                               'sha256': created['metadata']['prompt_sha256'],
                               'status': 'created', 'reason': ''}
                warnings.append('本次已创建并复用 Clone Prompt 缓存。')
            except (clone_prompts.ClonePromptError, OSError) as exc:
                prompt_info = {'used': False, 'path': None, 'sha256': None,
                               'status': 'fallback', 'reason': str(exc)}
                warnings.append(f'Clone Prompt 创建失败，已回退到普通克隆：{exc}')
        else:
            prompt_info = {'used': False, 'path': None, 'sha256': None,
                           'status': 'fallback', 'reason': clone_prompts.status_text(report)}
            warnings.append('Clone Prompt 未使用：' + clone_prompts.status_text(report))
            if status in ('stale', 'corrupt'):
                # Drop tensors that provably no longer match the card so they can
                # never be reused by another page.
                try:
                    clone_prompts.invalidate_clone_prompt(
                        role_id, '生成前检测到缓存失效：' + '；'.join(report['reasons']),
                        store=store)
                except (OSError, ValueError):
                    pass
    if progress is not None:
        progress(0.2, desc='准备 Qwen 克隆')
    directory = Path(output_dir or (Path(get_config().outputs_dir) / 'generation' / role_id)).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    destination = directory / f'{run_id}.wav'
    metadata_file = directory / f'{run_id}.json'
    backend_output = None
    try:
        if prompt is not None:
            result = _synthesize(backend, 'clone', text, values, device=device, dtype=dtype,
                                 clone_prompt=prompt['path'],
                                 clone_mode=prompt['mode'], ref_text=transcript,
                                 reference_sha256=audio_hash, cancel=cancel,
                                 adapter_path=adapter_info['path'] if adapter_info else None,
                                 adapter_scale=adapter_info['scale'] if adapter_info else 1.0,
                                 adapter_sha256=adapter_info['artifact_sha256'] if adapter_info else None)
        else:
            result = _synthesize(backend, 'clone', text, values, device=device, dtype=dtype,
                                 ref_audio=str(reference), ref_text=transcript,
                                 reference_sha256=audio_hash, cancel=cancel,
                                 adapter_path=adapter_info['path'] if adapter_info else None,
                                 adapter_scale=adapter_info['scale'] if adapter_info else 1.0,
                                 adapter_sha256=adapter_info['artifact_sha256'] if adapter_info else None)
        backend_output = Path(result['file'])
        _validate_generated_audio(backend_output)
        audio_hash_out = _sha(backend_output)
        with destination.open('xb') as stream, backend_output.open('rb') as source:
            shutil.copyfileobj(source, stream)
        if _sha(destination) != audio_hash_out:
            raise GenerationError('生成音频复制后哈希不一致，已放弃发布。')
        samples, sample_rate = sf.read(str(destination), dtype='float32', always_2d=True)
        duration = len(samples) / sample_rate
        metadata = {
            'schema_version': SCHEMA_VERSION,
            'run_id': run_id,
            'kind': 'character_generation',
            'created_at': _now(),
            'role_id': card['id'],
            'role_name': card['name'],
            'role_version': card.get('version', 1),
            'backend': 'qwen3',
            'mode': 'voice_clone',
            'model_name': card.get('model_name', ''),
            'model_revision': card.get('model_revision', ''),
            'model_dir': result.get('model_dir'),
            'model_config_sha256': result.get('model_config_sha256'),
            'qwen_package_version': result.get('qwen_package_version'),
            'params': values.to_dict(),
            'seed': values.seed,
            'text': text,
            'reference': {'audio': str(reference), 'sha256': audio_hash,
                          'transcript': transcript,
                          'transcript_sha256': hashlib.sha256(transcript.encode('utf-8')).hexdigest(),
                          'clone_mode': (prompt['mode'] if prompt else card.get('clone_mode', 'icl')),
                          'edit_chain_sha256': hashlib.sha256(json.dumps(
                              card.get('audio_edit_chain', []), ensure_ascii=False,
                              sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()},
            'clone_prompt': prompt_info,
            'adapter': ({'id': adapter_info['entry']['id'],
                         'name': adapter_info['entry']['name'],
                         'scale': adapter_info['scale'],
                         'artifact_sha256': adapter_info['artifact_sha256']}
                        if adapter_info else None),
            'audio': {'file': destination.name, 'sha256': audio_hash_out,
                      'duration': duration, 'sample_rate': sample_rate},
            'warnings': warnings,
            'backend_result': _trim_result(result),
        }
        if result.get('text_segments', 1) > 1:
            metadata['warnings'].append(
                f"长台词已分 {result['text_segments']} 段顺序生成以降低显存峰值，段落间语气衔接请试听确认。")
        _write_metadata(metadata_file, metadata)
        return {**metadata, 'file': str(destination), 'metadata_file': str(metadata_file),
                'duration': duration, 'sample_rate': sample_rate,
                'result': result}
    except BaseException:
        destination.unlink(missing_ok=True)
        metadata_file.unlink(missing_ok=True)
        raise
    finally:
        if backend_output is not None and backend_output.exists():
            # The raw Qwen output is an intermediate; the published copy and the
            # metadata are the durable artifacts.
            backend_output.unlink(missing_ok=True)


def available_backend(name='qwen3'):
    """Return a registered adapter only when its local files are present."""
    try:
        from timbreforge.backends import BackendUnavailable, default_registry
    except ImportError:
        return None
    try:
        backend = default_registry().get(name)
        ok, _reason = backend.available()
        return backend if ok else None
    except (BackendUnavailable, OSError, ValueError):
        return None


def _synthesize(backend, mode, text, params, **kwargs):
    """Route through a unified adapter when supplied, else the Qwen adapter."""
    if backend is None:
        return backend_qwen.clone(kwargs.pop('ref_audio', None), kwargs.pop('ref_text'),
                                  text, params, **kwargs)
    if not hasattr(backend, 'synthesize'):
        raise TypeError('backend must implement synthesize(request) or be None')
    from timbreforge.backends import BackendRequest
    request = BackendRequest(mode=mode, text=text, params=params,
                             instruction=kwargs.pop('instruction', ''),
                             ref_audio=kwargs.pop('ref_audio', None),
                             ref_text=kwargs.pop('ref_text', ''),
                             clone_prompt=kwargs.pop('clone_prompt', None),
                             clone_mode=kwargs.pop('clone_mode', 'icl'),
                             reference_sha256=kwargs.pop('reference_sha256', None),
                             device=kwargs.pop('device', 'cuda:0'),
                             dtype=kwargs.pop('dtype', 'float16'),
                             attention=kwargs.pop('attention', 'sdpa'),
                             timeout=kwargs.pop('timeout', 1800.0),
                             cancel=kwargs.pop('cancel', None),
                             adapter_path=kwargs.pop('adapter_path', None),
                             adapter_scale=kwargs.pop('adapter_scale', 1.0),
                             adapter_sha256=kwargs.pop('adapter_sha256', None))
    if kwargs:
        raise TypeError(f'unexpected generation options: {sorted(kwargs)}')
    return backend.synthesize(request).raw


def _validate_generated_audio(path):
    info = sf.info(str(path))
    if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
        raise GenerationError('模型返回的音频为空。')
    samples, sample_rate = sf.read(str(path), dtype='float32', always_2d=True)
    if not samples.size or not np.isfinite(samples).all() or not np.any(samples):
        raise GenerationError('模型返回的音频包含无效采样。')
    if not 8000 <= sample_rate <= 192000:
        raise GenerationError('模型返回的音频采样率异常。')
    if len(samples) / sample_rate > 600:
        raise GenerationError('单次生成音频超过 10 分钟上限。')


def _trim_result(result):
    """Keep provenance but drop bulky or soon-deleted duplicate fields."""
    return {key: value for key, value in result.items()
            if key not in {'params', 'ref_text_used', 'reference_cropped', 'file'}}


def _write_metadata(path, metadata):
    temporary = Path(str(path) + '.tmp')
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def list_generations(role_id=None, *, output_dir=None, limit=50):
    """Newest-first generation records; corrupt files are skipped."""
    root = Path(output_dir) if output_dir is not None else Path(
        get_config().outputs_dir) / 'generation'
    if role_id is not None and not isinstance(role_id, str):
        raise ValueError('role_id must be text')
    if role_id is None:
        directories = sorted(root.glob('*'))
    else:
        # Support both the default role-subdirectory layout and an explicit flat
        # output directory chosen by the caller/UI.
        candidate = root / role_id
        directories = [candidate] if candidate.is_dir() else [root]
    records = []
    for directory in directories:
        if not directory.is_dir():
            continue
        for path in directory.glob('*.json'):
            try:
                data = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                continue
            if not isinstance(data, dict) or data.get('schema_version') != SCHEMA_VERSION:
                continue
            if role_id is not None and data.get('role_id') != role_id:
                continue
            audio = data.get('audio')
            name = audio.get('file', '') if isinstance(audio, dict) else ''
            if not isinstance(name, str) or not name:
                continue
            candidate_path = directory / Path(name).name
            if not candidate_path.is_file():
                continue
            data['file'] = str(candidate_path)
            data['metadata_file'] = str(path)
            records.append(data)
    records.sort(key=lambda item: item.get('created_at', ''), reverse=True)
    return records[:max(1, int(limit))]


def load_generation(metadata_file):
    path = Path(metadata_file).expanduser().resolve()
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict) or data.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('生成记录版本不兼容。')
    audio = path.parent / data['audio']['file']
    if not audio.is_file():
        raise FileNotFoundError(f'生成音频不存在：{audio}')
    data['file'] = str(audio)
    data['metadata_file'] = str(path)
    return data


def delete_generation(metadata_file):
    """Delete one published take and its metadata; never touches the role card."""
    record = load_generation(metadata_file)
    metadata = Path(record['metadata_file'])
    audio = Path(record['file'])
    if audio.parent != metadata.parent:
        raise ValueError('生成音频不在记录目录内，拒绝删除。')
    audio.unlink(missing_ok=True)
    metadata.unlink(missing_ok=True)
    return record


def cleanup_generation_staging(role_id=None, *, older_than_hours=24, output_dir=None):
    """Remove leftover Qwen staging files from failed runs (best effort)."""
    root = Path(output_dir) if output_dir is not None else Path(
        get_config().outputs_dir) / 'qwen'
    if not root.is_dir():
        return 0
    import time
    cutoff = time.time() - older_than_hours * 3600
    removed = 0
    for pattern in ('*.request.json', '*.result.json', '*.retry.log'):
        for path in root.glob(pattern):
            try:
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            except OSError:
                continue
    return removed

