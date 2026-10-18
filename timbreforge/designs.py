"""Prompt-driven VoiceDesign sessions, explicit selection and anchored identities."""
from __future__ import annotations
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import uuid
import numpy as np
import soundfile as sf
from timbreforge.config import get_config
from timbreforge.characters import CharacterStore

_LOCK = threading.RLock()
_ID = re.compile(r'[0-9a-f]{32}\Z')


def _id(value):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError('Invalid design/candidate ID')
    return value


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DesignStore:
    def __init__(self, root=None):
        self.root = Path(root or get_config().outputs_dir / 'designs').expanduser().resolve()

    def _dir(self, session_id):
        directory = self.root / _id(session_id)
        if directory.is_symlink() or directory.resolve().parent != self.root:
            raise ValueError('Unsafe design directory')
        return directory

    def _save(self, session):
        directory = self._dir(session['id'])
        temporary = directory / ('.session-' + uuid.uuid4().hex + '.tmp')
        try:
            with temporary.open('x', encoding='utf-8') as stream:
                json.dump(session, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, directory / 'session.json')
        finally:
            temporary.unlink(missing_ok=True)

    def load(self, session_id):
        directory = self._dir(session_id)
        manifest = directory / 'session.json'
        if manifest.is_symlink():
            raise ValueError('Unsafe session manifest')
        session = json.loads(manifest.read_text(encoding='utf-8'))
        if not isinstance(session, dict) or session.get('schema_version') != 1 or session.get('id') != session_id:
            raise ValueError('Invalid design session')
        if not isinstance(session.get('prompt'), str) or not isinstance(session.get('text'), str):
            raise ValueError('Invalid design text')
        if not isinstance(session.get('candidates'), list):
            raise ValueError('Invalid candidates')
        from timbreforge.backend_qwen import QwenParams
        from timbreforge.characters import _validate_json_value
        _validate_json_value(session, 'design session')
        QwenParams(**session['params']).validate()
        if session.get('device') not in ('cpu', 'cuda:0'):
            raise ValueError('Invalid design device')
        ids = set()
        for candidate in session['candidates']:
            if not isinstance(candidate, dict) or not isinstance(candidate.get('result'), dict):
                raise ValueError('Invalid candidate result')
            if type(candidate.get('seed')) is not int or not 0 <= candidate['seed'] <= 0x7fffffffffffffff:
                raise ValueError('Invalid candidate seed')
            if type(candidate.get('favorite')) is not bool or not isinstance(candidate.get('note'), str):
                raise ValueError('Invalid candidate review')
            if not isinstance(candidate.get('sha256'), str) or not re.fullmatch(r'[0-9a-f]{64}', candidate['sha256']):
                raise ValueError('Invalid candidate hash')
            cid = _id(candidate['id'])
            if cid in ids or candidate['file'] != cid + '.wav':
                raise ValueError('Invalid design candidate path or duplicate ID')
            ids.add(cid)
            path = directory / candidate['file']
            if path.is_symlink() or not path.is_file() or _sha(path) != candidate['sha256']:
                raise ValueError('Design candidate is missing or changed')
        if session.get('selected_candidate_id') is not None and session['selected_candidate_id'] not in ids:
            raise ValueError('Selected design candidate is missing')
        return session

    def list(self):
        if not self.root.is_dir():
            return []
        result = []
        for directory in sorted(self.root.iterdir()):
            if _ID.fullmatch(directory.name):
                try:
                    result.append(self.load(directory.name))
                except (OSError, ValueError, KeyError, TypeError):
                    pass
        return result

    def create(self, prompt, text, params, *, count=1, device='cuda:0', confirmed=False):
        from timbreforge.backend_qwen import QwenParams
        if confirmed is not True:
            raise ValueError('Confirm authorized synthetic voice design before generation')
        if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 2000:
            raise ValueError('Voice description must contain 1–2000 characters')
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= 500:
            raise ValueError('Design preview must contain 1–500 characters; prefer a short complete sentence')
        if not isinstance(params, QwenParams):
            raise ValueError('params must be QwenParams')
        params.validate()
        self._check_count(count, params.seed)
        if device not in ('cuda:0', 'cpu'):
            raise ValueError('Select cuda:0 or cpu explicitly')
        with _LOCK:
            session = dict(schema_version=1, id=uuid.uuid4().hex, prompt=prompt.strip(), text=text.strip(),
                           created_at=datetime.now(timezone.utc).isoformat(), params=params.to_dict(),
                           device=device, candidates=[], selected_candidate_id=None)
            self._dir(session['id']).mkdir(parents=True, exist_ok=False)
            self._save(session)
            return self.append(session['id'], params=params, count=count, device=device, confirmed=True)

    @staticmethod
    def _check_count(count, seed):
        if type(count) is not int or not 1 <= count <= 4:
            raise ValueError('Generate 1–4 candidates at a time')
        if type(seed) is not int or not 0 <= seed <= 0x7FFFFFFFFFFFFFFF - count + 1:
            raise ValueError('Invalid seed range')

    def append(self, session_id, *, params=None, count=1, device=None, confirmed=False):
        from timbreforge.backend_qwen import QwenParams, design
        if confirmed is not True:
            raise ValueError('Explicit design confirmation required')
        with _LOCK:
            session = self.load(session_id)
            value = params if params is not None else QwenParams(**session['params'])
            if not isinstance(value, QwenParams):
                raise ValueError('params must be QwenParams')
            value.validate()
            self._check_count(count, value.seed)
            device = device or session['device']
            if device not in ('cuda:0', 'cpu'):
                raise ValueError('Select cuda:0 or cpu explicitly')
            from timbreforge.studio import _GENERATION_LOCK
            for index in range(count):
                actual = replace(value, seed=value.seed + index)
                with _GENERATION_LOCK:
                    result = design(session['text'], session['prompt'], actual, device=device,
                                    dtype='float32' if device == 'cpu' else 'float16')
                source = Path(result['file']).resolve()
                x, sr = sf.read(source, dtype='float32', always_2d=True)
                if not x.size or sr <= 0 or not np.isfinite(x).all() or not np.any(x):
                    raise ValueError('Invalid design audio')
                cid = uuid.uuid4().hex
                destination = self._dir(session_id) / (cid + '.wav')
                with destination.open('xb') as stream:
                    sf.write(stream, x, sr, format='WAV', subtype='FLOAT')
                session['candidates'].append(dict(id=cid, file=destination.name, sha256=_sha(destination),
                                                  seed=actual.seed, result=result, favorite=False, note=''))
                self._save(session)
            return session

    def audio_path(self, session_id, candidate_id):
        session = self.load(session_id)
        candidate = self._candidate(session, candidate_id)
        return str(self._dir(session_id) / candidate['file'])

    @staticmethod
    def _candidate(session, candidate_id):
        _id(candidate_id)
        for candidate in session['candidates']:
            if candidate['id'] == candidate_id:
                return candidate
        raise ValueError('Candidate not in design session')

    def select(self, session_id, candidate_id, *, favorite=None, note=None):
        with _LOCK:
            session = self.load(session_id)
            candidate = self._candidate(session, candidate_id)
            if favorite is not None:
                if type(favorite) is not bool:
                    raise ValueError('favorite must be boolean')
                candidate['favorite'] = favorite
            if note is not None:
                if not isinstance(note, str):
                    raise ValueError('note must be text')
                candidate['note'] = note
            session['selected_candidate_id'] = candidate_id
            candidate.setdefault('reviews', []).append(dict(at=datetime.now(timezone.utc).isoformat(),
                                                            favorite=candidate['favorite'], note=candidate['note']))
            self._save(session)
            return session

    def save_character(self, session_id, candidate_id, name, *, description=None, confirmed=False):
        if confirmed is not True:
            raise ValueError('Confirm reference transcript matches rendered audio before saving identity')
        with _LOCK:
            session = self.load(session_id)
            candidate = self._candidate(session, candidate_id)
            if session.get('selected_candidate_id') != candidate_id:
                raise ValueError('Manually select the candidate before saving a character')
            path = self.audio_path(session_id, candidate_id)
            params = candidate['result'].get('params', session['params'])
            if description is None:
                description = session['prompt']
            if not isinstance(description, str) or len(description) > 2000:
                raise ValueError('角色描述必须是不超过 2000 字的文本')
            return CharacterStore().create(name, path, reference_text=session['text'],
                                           description=description.strip(), backend='qwen3', instruction='',
                                           params={'qwen': params}, provenance={
                                               'kind': 'qwen_voice_design', 'session_id': session_id,
                                               'candidate_id': candidate_id, 'design_prompt': session['prompt'],
                                               'design_result': candidate['result'],
                                               'reference_transcript_user_confirmed': True,
                                               'identity_strategy': 'fixed designed reference + transcript, Base clone; not prompt regeneration'})

