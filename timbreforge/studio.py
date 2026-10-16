"""Task-oriented voice studio: reference cloning, performance and character reuse.

Inspired by the separation of VoiceDesign / Base / CustomVoice, not an adapter
claiming those Qwen models are installed. No model import or network at startup.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import hashlib
import json
import math
from pathlib import Path
import threading
import uuid

import numpy as np
import soundfile as sf

from timbreforge.backend_indextts import SynthParams
from timbreforge.config import get_config
from timbreforge.studio_qwen import generate_character as _generate_qwen_character

_GENERATION_LOCK = threading.Lock()


@dataclass
class GenerationRequest:
    task: str = 'clone'
    backend: str = 'indextts'
    text: str = ''
    reference_audio: str | None = None
    reference_text: str = ''
    instruction: str = ''
    character_id: str | None = None
    params: SynthParams | None = None
    seed_override: int | None = None
    base_url: str = ''
    save_character_name: str = ''
    description: str = ''
    qwen_params: dict | None = None
    device: str = 'cuda:0'
    qwen_dtype: str = 'bfloat16'


def capabilities() -> list[dict]:
    """Local asset checks only. available is NOT proof of CUDA/runtime readiness."""
    from timbreforge import backend_cosyvoice as cv
    required = [cv.PYTHON, cv.WORKER, cv.MODEL_DIR / 'cosyvoice3.yaml',
                cv.MODEL_DIR / 'llm.pt', cv.MODEL_DIR / 'flow.pt',
                cv.MODEL_DIR / 'hift.pt', cv.MODEL_DIR / 'campplus.onnx',
                cv.MODEL_DIR / 'speech_tokenizer_v3.onnx',
                cv.MODEL_DIR / 'CosyVoice-BlankEN' / 'config.json',
                cv.MODEL_DIR / 'CosyVoice-BlankEN' / 'model.safetensors']
    return [
        dict(backend='indextts', label='IndexTTS 2.5 / ComfyUI',
             tasks=['clone', 'instruct', 'character'], available=True,
             note='已接入；需要 ComfyUI 在线并加载修复节点。文字指令仅控制情感，不保证方言。'),
        dict(backend='cosyvoice3', label='CosyVoice3 本地隔离环境',
             tasks=['clone', 'instruct', 'character'], available=all(p.is_file() for p in required),
             note='本地文件就绪不代表推理已验收。转录用于零样本克隆；指令模式不消费转录。'),
        dict(backend='qwen3_design', label='Qwen3-TTS VoiceDesign 1.7B',
             tasks=['design'], available=all(p.is_file() for p in (get_config().qwen_python,
                 get_config().qwen_design_model / 'config.json', get_config().qwen_design_model / 'model.safetensors',
                 get_config().qwen_design_model / 'speech_tokenizer' / 'model.safetensors')),
             note='真实文字声线设计；本地文件检查不代表GPU推理已就绪。首页使用独立Qwen采样参数。'),
        dict(backend='qwen3', label='Qwen3-TTS Base 0.6B 角色复用',
             tasks=['clone','character'], available=all(p.is_file() for p in (get_config().qwen_python,
                 get_config().qwen_base_model / 'config.json', get_config().qwen_base_model / 'model.safetensors',
                 get_config().qwen_base_model / 'speech_tokenizer' / 'model.safetensors')),
             note='用已选设计参考和准确转录固定声音条件；Base不支持独立表演指令。'),
    ]


def _reference(path: str | Path) -> tuple[Path, dict]:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f'参考音频不存在：{p}')
    info = sf.info(p)
    if info.duration > 30 or info.channels > 32 or info.frames * info.channels > 24000 * 3600:
        raise ValueError('参考音频请控制在 30 秒内，声道或采样数量过大；不会自动裁切。')
    x, sr = sf.read(p, dtype='float32', always_2d=True)
    if x.size == 0 or sr <= 0 or not np.isfinite(x).all() or not np.any(x):
        raise ValueError('参考音频为空、静音或包含无效采样')
    duration = len(x) / sr
    if duration > 30:
        raise ValueError('参考音频请控制在 30 秒内，并提供与音频对应的准确转录；不会自动裁切。')
    return p, dict(sample_rate=sr, duration=duration,
                   sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def _params(value: SynthParams) -> None:
    numeric = ('temperature', 'top_p', 'repetition_penalty', 'duration_factor',
               'emotion_weight', 'inference_cfg_rate')
    for key in numeric:
        v = getattr(value, key)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f'{key} 必须是有限数值')
    for key in ('seed', 'top_k', 'num_beams', 'max_mel_tokens',
                'max_text_tokens_per_segment', 'interval_silence_ms', 'diffusion_steps'):
        v = getattr(value, key)
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f'{key} 必须是整数')
    if not 0 <= value.seed <= 0x7FFFFFFFFFFFFFFF:
        raise ValueError('seed 超出有效范围')
    if not 1 <= value.diffusion_steps <= 100 or not 0 <= value.inference_cfg_rate <= 2:
        raise ValueError('扩散步数应在 1–100，CFG 应在 0–2')
    if not 0 < value.temperature <= 2 or not 0 <= value.top_p <= 1 or not 0 <= value.top_k <= 100:
        raise ValueError('temperature/top_p/top_k 超出有效范围')
    if not 1 <= value.num_beams <= 10 or not 0 < value.repetition_penalty <= 20:
        raise ValueError('beams/repetition_penalty 超出有效范围')
    if not .5 <= value.duration_factor <= 2 or not 0 <= value.emotion_weight <= 1:
        raise ValueError('duration_factor/emotion_weight 超出有效范围')
    if not 0 <= value.interval_silence_ms <= 5000:
        raise ValueError('静音间隔应在 0–5000ms')
    if not 20 <= value.max_text_tokens_per_segment <= 400 or not 50 <= value.max_mel_tokens <= 4096:
        raise ValueError('文本或语音 token 上限超出有效范围')
    if not isinstance(value.emotion, dict) or any(
        k not in {'happy', 'angry', 'sad', 'fearful', 'disgusted', 'melancholic', 'surprised', 'calm'}
        or isinstance(v, bool) or not isinstance(v, (int, float))
        or not math.isfinite(v) or not 0 <= v <= 1 for k, v in value.emotion.items()
    ):
        raise ValueError('情感向量必须是已知维度的 0–1 有限数值')
    if not isinstance(value.do_sample, bool) or not isinstance(value.text_normalization, bool):
        raise ValueError('采样及文本规范化开关必须是布尔值')


def generate(request: GenerationRequest, *, character_store=None,
             output_dir: str | Path | None = None) -> dict:
    """Render once, persist unnormalized audio + request/result provenance.

    A process-local lock serializes studio tasks; backend resource slots also
    serialize CLI workers and release idle ComfyUI models. Character references
    are portable audio assets, not backend-incompatible pickled embeddings.
    """
    if request.task not in {'clone', 'instruct', 'character', 'design'}:
        raise ValueError('未知任务类型')
    if request.task == 'design':
        raise ValueError('VoiceDesign 请使用首页设计会话，或 DesignStore.create；角色复用不重新运行设计提示词。')
    if not isinstance(request.text, str) or not request.text.strip():
        raise ValueError('请填写要生成的台词')
    if len(request.text) > 5000:
        raise ValueError('单次台词超过 5000 字，请拆成场景或段落')
    for key in ('reference_text', 'instruction', 'save_character_name', 'description'):
        if not isinstance(getattr(request, key), str):
            raise ValueError(f'{key} 必须是文字')
    if request.save_character_name and (not request.save_character_name.strip() or len(request.save_character_name) > 120):
        raise ValueError('角色名称必须为 1–120 字')
    if request.params is not None and not isinstance(request.params, SynthParams):
        raise ValueError('params 必须是 SynthParams')
    backend = request.backend
    reference = request.reference_audio
    transcript = request.reference_text.strip()
    instruction = request.instruction.strip()
    character = None
    params = request.params or SynthParams()
    if request.task == 'character':
        if not request.character_id:
            raise ValueError('请先选择角色卡')
        if request.reference_audio or transcript:
            raise ValueError('角色复用使用已保存的参考；不要同时指定新参考或转录')
        if character_store is None:
            from timbreforge.characters import CharacterStore
            character_store = CharacterStore(get_config().characters_dir)
        character = character_store.load(request.character_id)
        backend = character['backend']
        reference = character['reference_audio']
        transcript = character['reference_text']
        instruction = instruction or character['instruction']
        if request.params is None:
            allowed = {f.name for f in fields(SynthParams)}
            params = SynthParams(**{k: v for k, v in character['params'].items() if k in allowed})
    if backend == 'qwen3':
        if request.task == 'instruct':
            raise ValueError('Qwen Base 不支持 instruct 任务；请选择 clone 或 character')
        if instruction:
            raise ValueError('Qwen Base 参考克隆不消费表演指令；请清空该句指令。设计描述仅在 VoiceDesign 页面使用。')
        if request.params is not None:
            raise ValueError('Qwen 使用独立 qwen_params，不映射 IndexTTS 采样参数')
        if not reference or not transcript:
            raise ValueError('Qwen Base 需要参考音频和对应的准确完整转录')
        return _generate_qwen_character(request, reference, transcript, character, character_store, output_dir=output_dir)
    if backend not in {'indextts', 'cosyvoice3'}:
        raise ValueError('后端必须是 indextts、cosyvoice3 或 qwen3')
    if not reference:
        raise ValueError('请上传或指定参考音频')
    if request.task == 'instruct' and not instruction:
        raise ValueError('指令模式需要填写表演指令')
    if request.task == 'clone' and instruction:
        raise ValueError('克隆模式不消费表演指令；请选择 instruct 模式')
    ref, ref_info = _reference(reference)
    if request.seed_override is not None:
        params = replace(params, seed=request.seed_override)
    _params(params)
    if not isinstance(params.extra_emotion_text, str) or not isinstance(params.emotion_mode, str):
        raise ValueError('情感模式和文字必须是字符串')
    params = replace(params, text=request.text.strip(), emotion=dict(params.emotion))
    warnings = []
    mode = 'instruct' if instruction else 'clone'
    if backend == 'indextts':
        if mode == 'instruct':
            if params.emotion:
                raise ValueError('文字情感指令与情感向量不可混用，请清空其中一个')
            params = replace(params, emotion_mode='extra_emotion_text', extra_emotion_text=instruction)
            warnings.append('IndexTTS 文字指令仅作为情感文本，不保证语速、方言或音色设计；会额外加载情感模型。')
        elif params.extra_emotion_text.strip():
            raise ValueError('请通过 instruct 任务填写情感文字，而非在 clone 中隐藏指令')
        elif params.emotion:
            params = replace(params, emotion_mode='emotion_vector')
        elif params.emotion_mode != 'same_as_speaker':
            raise ValueError('无情感输入时请选择 same_as_speaker 模式')
        if ref_info['duration'] > 15:
            warnings.append('IndexTTS 上游最多读取约 15 秒参考；本服务不重写原音频。')
        if transcript:
            warnings.append('IndexTTS 不消费参考转录；转录仅随角色卡保存。')
    else:
        if params.emotion or params.extra_emotion_text.strip() or params.emotion_mode != 'same_as_speaker':
            raise ValueError('CosyVoice3 不支持 IndexTTS 情感向量；请使用 instruct 任务和文字指令')
        if mode == 'clone' and not transcript:
            warnings.append('未提供参考转录，将使用不含 LLM 参考转录的 cross_lingual 模式，而非转录条件克隆。')
        if mode == 'instruct' and transcript:
            warnings.append('CosyVoice3 instruct2 不使用参考转录；角色卡仍保留原转录。')
        warnings.append('CosyVoice3 使用模型自身采样设置，IndexTTS 的 beams/扩散/时长参数不适用。')
    cfg = get_config()
    if not request.base_url:
        request = replace(request, base_url=cfg.comfy_url)
    outdir = Path(output_dir if output_dir is not None else cfg.outputs_dir / 'studio').resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    with _GENERATION_LOCK:
        if backend == 'indextts':
            from timbreforge.backend_indextts import synthesize
            result = synthesize(str(ref), params, base=request.base_url, verbose=False)
        else:
            from timbreforge.backend_cosyvoice import synthesize
            result = synthesize(str(ref), params, ref_text=transcript,
                                instruction=instruction, verbose=False)
        raw = Path(result['file']).resolve()
        audio, sr = sf.read(raw, dtype='float32', always_2d=True)
        if not audio.size or not np.isfinite(audio).all() or not np.any(audio):
            raise RuntimeError('后端生成了空、静音或无效音频')
        if hashlib.sha256(ref.read_bytes()).hexdigest() != ref_info['sha256']:
            raise RuntimeError('生成期间参考音频被修改，请重试')
        output = outdir / f'{run_id}.wav'
        # IndexTTS SaveAudio may produce FLAC; decode to FLOAT WAV, never rename
        # FLAC bytes to .wav. This preserves amplitudes without peak processing.
        sf.write(output, audio, sr, subtype='FLOAT')
        metadata = dict(schema_version=1, run_id=run_id, task=request.task, mode=mode,
                        backend=backend, text=params.text, reference_audio=str(ref),
                        reference_text=transcript, reference_info=ref_info,
                        instruction=instruction, params=params.to_dict(), warnings=warnings,
                        elapsed=result.get('elapsed'), sample_rate=sr, duration=len(audio) / sr,
                        file=str(output), raw_output=str(raw), backend_result=result,
                        character_id=character['id'] if character else None)
        if request.save_character_name.strip():
            if character_store is None:
                from timbreforge.characters import CharacterStore
                character_store = CharacterStore()
            card = character_store.create(
                request.save_character_name, str(ref), reference_text=transcript,
                description=request.description, backend=backend, instruction=instruction,
                params=params.to_dict(), provenance=dict(run_id=run_id, reference_info=ref_info))
            metadata['character_id'] = card['id']
        meta_path = outdir / f'{run_id}.json'
        metadata['metadata_file'] = str(meta_path)
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return metadata
