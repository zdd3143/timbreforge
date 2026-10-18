"""Guided voice tuning: one-dimension candidates, A/B audition and promotion.

Candidates are generated with VoiceDesign from the role's structured recipe, so
each take changes exactly one delivery/text dimension while timbre and identity
stay fixed. A promoted take becomes the role's new canonical reference audio and
its spoken text becomes the new transcript, which invalidates any clone prompt
and requires the user's explicit confirmation.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

import numpy as np
import soundfile as sf

from timbreforge import backend_qwen
from timbreforge.audio_edit_schema import make_event
from timbreforge.candidates import compare_audio
from timbreforge.characters import CharacterStore, audio_path
from timbreforge.config import get_config
from timbreforge.voice_recipe import (RecipeError, design_prompt, normalize,
                                      variants as recipe_variants)

SCHEMA_VERSION = 1
MAX_TEXT = 500
MAX_CANDIDATES = 8
_ID = re.compile(r'[0-9a-f]{32}\Z')


class CandidateError(ValueError):
    """Raised when a tuning session or candidate is invalid."""


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _text(value):
    if not isinstance(value, str) or not value.strip():
        raise CandidateError('试听台词不能为空。')
    if len(value) > MAX_TEXT:
        raise CandidateError(f'声线微调试听台词不得超过 {MAX_TEXT} 字，请使用短句。')
    return value.strip()


def _params(card, seed):
    from timbreforge.backend_qwen import QwenParams
    values = (card.get('params') or {}).get('qwen') or {}
    base = QwenParams(**(values if isinstance(values, dict) else {}))
    base.validate()
    if type(seed) is not int or not 0 <= seed <= 0x7FFFFFFFFFFFFFFF:
        raise CandidateError('seed 必须是 0 到 2^63-1 的整数。')
    return replace(base, seed=seed)


class CandidateStore:
    """Sessions live under ``outputs/tuning``; the character card stays authoritative."""

    def __init__(self, root=None):
        self.root = Path(root if root is not None else Path(get_config().outputs_dir) / 'tuning').expanduser().resolve()

    def _directory(self, session_id):
        if not isinstance(session_id, str) or not _ID.fullmatch(session_id):
            raise CandidateError('声线微调会话 ID 无效。')
        directory = self.root / session_id
        if directory.is_symlink():
            raise CandidateError('声线微调会话目录不安全。')
        return directory

    def _save(self, session):
        directory = self._directory(session['id'])
        temporary = directory / f'.session-{uuid.uuid4().hex}.tmp'
        try:
            with temporary.open('x', encoding='utf-8') as stream:
                json.dump(session, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, directory / 'session.json')
        finally:
            temporary.unlink(missing_ok=True)

    def load(self, session_id):
        directory = self._directory(session_id)
        path = directory / 'session.json'
        if path.is_symlink() or not path.is_file():
            raise CandidateError('声线微调会话不存在。')
        try:
            session = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            raise CandidateError(f'声线微调会话损坏：{exc}') from exc
        _validate_session(session, session_id, directory)
        return session

    def list(self, role_id=None):
        if not self.root.is_dir():
            return []
        sessions = []
        for directory in sorted(self.root.iterdir()):
            if not _ID.fullmatch(directory.name):
                continue
            try:
                session = self.load(directory.name)
            except (OSError, ValueError, CandidateError):
                continue
            if role_id is not None and session['role_id'] != role_id:
                continue
            sessions.append(session)
        return sorted(sessions, key=lambda item: item['created_at'], reverse=True)


def _validate_session(session, session_id, directory):
    if not isinstance(session, dict) or session.get('schema_version') != SCHEMA_VERSION:
        raise CandidateError('声线微调会话版本不兼容。')
    if session.get('id') != session_id:
        raise CandidateError('声线微调会话 ID 不匹配。')
    if not isinstance(session.get('role_id'), str) or not _ID.fullmatch(session['role_id']):
        raise CandidateError('声线微调会话角色 ID 无效。')
    if not isinstance(session.get('text'), str) or not session['text'].strip():
        raise CandidateError('声线微调会话缺少试听台词。')
    try:
        normalize(session.get('base_recipe'))
    except RecipeError as exc:
        raise CandidateError(f'声线微调配方无效：{exc}') from exc
    if not isinstance(session.get('candidates'), list):
        raise CandidateError('声线微调候选列表无效。')
    seen = set()
    for candidate in session['candidates']:
        if not isinstance(candidate, dict):
            raise CandidateError('声线微调候选记录无效。')
        if not isinstance(candidate.get('id'), str) or not _ID.fullmatch(candidate['id']):
            raise CandidateError('声线微调候选 ID 无效。')
        if candidate['id'] in seen:
            raise CandidateError('声线微调候选 ID 重复。')
        seen.add(candidate['id'])
        if not isinstance(candidate.get('prompt'), str) or not candidate['prompt']:
            raise CandidateError('声线微调候选缺少提示词。')
        if type(candidate.get('seed')) is not int:
            raise CandidateError('声线微调候选随机种子无效。')
        if not isinstance(candidate.get('params'), dict):
            raise CandidateError('声线微调候选参数无效。')
        if not isinstance(candidate.get('role_id'), str) or candidate['role_id'] != session['role_id']:
            raise CandidateError('声线微调候选角色 ID 不匹配。')
        try:
            normalize(candidate.get('recipe'))
        except RecipeError as exc:
            raise CandidateError(f'声线微调候选配方无效：{exc}') from exc
        if not isinstance(candidate.get('changed'), dict):
            raise CandidateError('声线微调候选缺少变化维度。')
        if not isinstance(candidate.get('profile'), dict):
            raise CandidateError('声线微调候选缺少模型信息。')
        name = candidate.get('file')
        if not isinstance(name, str) or Path(name).name != name:
            raise CandidateError('声线微调候选文件名不安全。')
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise CandidateError('声线微调候选音频缺失。')
        if _sha(path) != candidate.get('sha256'):
            raise CandidateError('声线微调候选音频哈希不匹配。')
    if session.get('selected') is not None and session['selected'] not in seen:
        raise CandidateError('声线微调选中的候选不存在。')


def create_session(role_id, text, *, store=None, output_dir=None, count=4,
                   dimensions=None, recipe=None, seed=42, device='cuda:0',
                   dtype='float16', design_callable=None) -> dict:
    """Create a session and generate ``count`` single-dimension candidates."""
    store = store or CharacterStore()
    card = store.load(role_id)
    text = _text(text)
    if type(count) is not int or not 1 <= count <= MAX_CANDIDATES:
        raise CandidateError(f'候选数量必须在 1–{MAX_CANDIDATES} 之间。')
    if device not in ('cuda:0', 'cpu'):
        raise CandidateError('device 必须是 cuda:0 或 cpu。')
    if dtype not in ('float16', 'bfloat16', 'float32'):
        raise CandidateError('dtype 必须是 float16/bfloat16/float32。')
    try:
        base_recipe = normalize(recipe if recipe is not None else card.get('voice_recipe') or {})
    except RecipeError as exc:
        raise CandidateError(f'声线配方无效：{exc}') from exc
    choices = recipe_variants(base_recipe, dimensions=dimensions, limit=count)
    params = _params(card, seed)
    tuning = CandidateStore(output_dir)
    session_id = uuid.uuid4().hex
    directory = tuning.root / session_id
    directory.mkdir(parents=True, exist_ok=False)
    session = {
        'schema_version': SCHEMA_VERSION,
        'id': session_id,
        'role_id': card['id'],
        'role_name': card['name'],
        'role_version': card.get('version', 1),
        'text': text,
        'base_recipe': base_recipe,
        'params': params.to_dict(),
        'device': device,
        'dtype': dtype,
        'candidates': [],
        'selected': None,
        'promoted': None,
        'created_at': _now(),
    }
    tuning._save(session)
    try:
        session = generate_candidates(session_id, store=store, output_dir=output_dir,
                                      count=count, seed=seed, device=device, dtype=dtype,
                                      design_callable=design_callable)
    except BaseException:
        import shutil
        shutil.rmtree(directory, ignore_errors=True)
        raise
    return session


def generate_candidates(session_id, *, store=None, output_dir=None, count=None, seed=None,
                        device=None, dtype=None, design_callable=None) -> dict:
    """Append more candidates; every take records prompt, seed, params and role."""
    store = store or CharacterStore()
    tuning = CandidateStore(output_dir)
    session = tuning.load(session_id)
    card = store.load(session['role_id'])
    total = len(session['candidates']) + (count if count is not None else 1)
    if total > MAX_CANDIDATES:
        raise CandidateError(f'一个会话最多 {MAX_CANDIDATES} 个候选。')
    if count is not None and (type(count) is not int or count < 1):
        raise CandidateError('count 必须是正整数。')
    batch = count if count is not None else 1
    if not session['candidates']:
        choices = recipe_variants(session['base_recipe'], limit=MAX_CANDIDATES)
    else:
        choices = recipe_variants(session['base_recipe'], limit=MAX_CANDIDATES)
    start = len(session['candidates'])
    if start + batch > len(choices):
        batch = max(0, len(choices) - start)
    if batch == 0:
        raise CandidateError('这个配方已经没有可用的单维候选了。')
    base_params = _params(card, seed if seed is not None else session['params']['seed'])
    device = device or session['device']
    dtype = dtype or session['dtype']
    designer = design_callable or backend_qwen.design
    directory = tuning._directory(session_id)
    for choice in choices[start:start + batch]:
        params = replace(base_params, seed=base_params.seed + start)
        prompt = design_prompt(choice['recipe'])
        result = designer(session['text'], prompt, params, device=device,
                          dtype='float32' if device == 'cpu' else dtype)
        source = Path(result['file'])
        samples, sample_rate = sf.read(str(source), dtype='float32', always_2d=True)
        if not samples.size or not np.isfinite(samples).all() or not np.any(samples):
            raise CandidateError('模型返回的候选音频无效。')
        candidate_id = uuid.uuid4().hex
        destination = directory / f'{candidate_id}.wav'
        with destination.open('xb') as stream:
            sf.write(stream, samples, sample_rate, format='WAV', subtype='FLOAT')
        session['candidates'].append({
            'id': candidate_id,
            'file': destination.name,
            'sha256': _sha(destination),
            'duration': len(samples) / sample_rate,
            'sample_rate': sample_rate,
            'prompt': prompt,
            'seed': params.seed,
            'params': params.to_dict(),
            'recipe': choice['recipe'],
            'changed': {'dimension': choice['dimension'], 'from': choice['from'],
                        'to': choice['to'], 'label': choice['label']},
            'role_id': session['role_id'],
            'role_version': session['role_version'],
            'profile': {'model_dir': result.get('model_dir'),
                        'model_config_sha256': result.get('model_config_sha256'),
                        'qwen_package_version': result.get('qwen_package_version'),
                        'device': result.get('device', device),
                        'dtype': result.get('dtype', dtype)},
            'selected': False,
            'created_at': _now(),
        })
        start += 1
        tuning._save(session)
        try:
            source.unlink(missing_ok=True)
        except OSError:
            pass
    return session


def audition(session_id, *, store=None, output_dir=None):
    """Per-candidate playback entries plus the selected take, if any."""
    tuning = CandidateStore(output_dir)
    session = tuning.load(session_id)
    directory = tuning._directory(session_id)
    return {
        'session': session,
        'audio': [{'id': candidate['id'], 'file': str(directory / candidate['file']),
                   'sha256': candidate['sha256'], 'label': candidate['changed']['label'],
                   'seed': candidate['seed'], 'selected': candidate['id'] == session['selected']}
                  for candidate in session['candidates']],
    }


def compare(session_id, a_id, b_id, *, match_volume=False, store=None, output_dir=None):
    """Create A/B listening copies; raw candidate files stay untouched."""
    tuning = CandidateStore(output_dir)
    session = tuning.load(session_id)
    directory = tuning._directory(session_id)
    lookup = {candidate['id']: candidate for candidate in session['candidates']}
    for candidate_id in (a_id, b_id):
        if candidate_id not in lookup:
            raise CandidateError('A/B 候选不在该会话中。')
    result = compare_audio(directory / lookup[a_id]['file'],
                           directory / lookup[b_id]['file'],
                           match_volume=match_volume, output_dir=directory / 'ab')
    result['a_meta'] = lookup[a_id]['changed']
    result['b_meta'] = lookup[b_id]['changed']
    return result


def select_candidate(session_id, candidate_id, *, store=None, output_dir=None):
    tuning = CandidateStore(output_dir)
    session = tuning.load(session_id)
    if not any(candidate['id'] == candidate_id for candidate in session['candidates']):
        raise CandidateError('候选不在该会话中。')
    for candidate in session['candidates']:
        candidate['selected'] = candidate['id'] == candidate_id
    session['selected'] = candidate_id
    tuning._save(session)
    return session


def promote_candidate(session_id, candidate_id, *, confirmed=False, create_prompt=True,
                      store=None, output_dir=None, prompt_device='cuda:0',
                      prompt_dtype='float16'):
    """Promote the selected take into a new version of the character card."""
    if confirmed is not True:
        raise CandidateError('必须确认候选音频与试听台词逐字一致后再保存为新版本。')
    store = store or CharacterStore()
    tuning = CandidateStore(output_dir)
    session = tuning.load(session_id)
    if session['selected'] != candidate_id:
        raise CandidateError('请先试听并选中一个候选，再保存为新版本。')
    candidate = next(item for item in session['candidates'] if item['id'] == candidate_id)
    card = store.load(session['role_id'])
    directory = Path(card['reference_audio']).resolve().parent
    audio_directory = directory / 'audio'
    audio_directory.mkdir(exist_ok=True)
    source = tuning._directory(session_id) / candidate['file']
    digest = _sha(source)
    destination_name = f'promoted-{digest[:16]}.wav'
    destination = audio_directory / destination_name
    if not destination.exists():
        temporary = audio_directory / f'.promote-{uuid.uuid4().hex}.tmp'
        try:
            samples, sample_rate = sf.read(str(source), dtype='float32', always_2d=True)
            with sf.SoundFile(str(temporary), mode='x', samplerate=sample_rate,
                              channels=samples.shape[1], format='WAV',
                              subtype='FLOAT') as stream:
                stream.write(samples)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    if _sha(destination) != digest:
        raise CandidateError('已存在的候选派生音频哈希不匹配，请检查角色目录后重试。')
    previous = Path(audio_path(card)).resolve()
    event = make_event(
        event='promote_candidate',
        source_audio=previous.relative_to(directory).as_posix(),
        source_hash=card.get('canonical_sha256') or card.get('reference_sha256'),
        output_audio=f'audio/{destination_name}', output_hash=digest,
        note=f"采用声线微调候选：{candidate['changed']['label']}；"
             f"提示词、seed {candidate['seed']} 与模型信息保存在调音会话 {session_id}。")
    chain = list(card.get('audio_edit_chain', [])) + [event]
    card = store.update(
        session['role_id'],
        canonical_audio=f'audio/{destination_name}', canonical_sha256=digest,
        audio_edit_chain=chain,
        reference_text=session['text'],
        voice_recipe=candidate['recipe'],
        source='voice_tuning/candidate',
        seed=candidate['seed'],
        provenance_patch={'tuning': {
            'session_id': session_id, 'candidate_id': candidate_id,
            'changed': candidate['changed'], 'prompt': candidate['prompt'],
            'params': candidate['params'],
            'profile': candidate['profile'],
        }},
        invalidate_reason='采用声线微调候选，参考音频与逐字稿已更新')
    session['promoted'] = {'candidate_id': candidate_id, 'role_id': card['id'],
                           'role_version': card.get('version'),
                           'promoted_at': _now()}
    tuning._save(session)
    prompt_result = {'status': 'skipped', 'reason': '未请求创建 Clone Prompt'}
    if create_prompt:
        from timbreforge import clone_prompts
        try:
            created = clone_prompts.create_clone_prompt(
                card['id'], store=store, device=prompt_device, dtype=prompt_dtype)
            card = created['card']
            prompt_result = {'status': 'created' if created['created'] else 'reused',
                             'path': created['path'],
                             'sha256': created['metadata']['prompt_sha256']}
        except (clone_prompts.ClonePromptError, OSError, RuntimeError) as exc:
            prompt_result = {'status': 'unavailable', 'reason': str(exc)}
    return {'ok': True, 'card': card, 'session': session, 'candidate': candidate,
            'clone_prompt': prompt_result,
            'warnings': ([] if prompt_result['status'] in ('created', 'reused')
                         else ['新版本已保存，但 Clone Prompt 暂不可用；下次生成会回退到普通克隆。'
                               + prompt_result.get('reason', '')])}


def discard_session(session_id, *, output_dir=None):
    """Delete one tuning session and its disposable audio; never touches the card."""
    tuning = CandidateStore(output_dir)
    directory = tuning._directory(session_id)
    if not directory.exists():
        raise CandidateError('声线微调会话不存在。')
    session = tuning.load(session_id)
    import shutil
    shutil.rmtree(directory)
    return session


def status_text(session):
    """Short Chinese status for the tuning page."""
    count = len(session.get('candidates', []))
    if session.get('promoted'):
        promoted = session['promoted']
        index = next((position + 1 for position, candidate in enumerate(session['candidates'])
                      if candidate['id'] == promoted.get('candidate_id')), 0)
        return (f"已采用第 {index} 个候选为新版本（角色版本 {promoted.get('role_version')}）。")
    if session.get('selected'):
        return f'已选中一个候选；共 {count} 个候选，可保存为新版本。'
    return f'共 {count} 个候选；请 A/B 试听后选中一个。'

