"""CPU-only, explicit-choice dialogue projects; no inference or scoring at import."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import threading
import uuid
import zipfile

import numpy as np
import soundfile as sf

from timbreforge.config import get_config
from timbreforge import studio
from timbreforge.characters import CharacterStore, _validate_json_value

_LOCK = threading.RLock()  # shared by every store instance in this process
_ID = re.compile(r'[0-9a-f]{32}\Z')
_HASH = re.compile(r'[0-9a-f]{64}\Z')
_AXES = {'speaker_similarity', 'content_accuracy', 'quality'}
RATE = 24000
MAX_EXPORT_SECONDS = 1800
MAX_EXPORT_SAMPLE_BUDGET = RATE * MAX_EXPORT_SECONDS * 4


def _now():
    return datetime.now(timezone.utc).isoformat()


def _id(value):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError('ID must be a lowercase UUID32')
    return value


def _hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _parse(script):
    if not isinstance(script, str) or len(script) > 2_600_000:
        raise ValueError('script must be a string of at most 2600000 characters')
    lines = []
    for number, raw in enumerate(script.splitlines(), 1):
        if not raw.strip():
            continue
        match = re.fullmatch(r'\s*([^：:〔〕\n]+?)(?:〔([^〔〕]*)〕)?\s*[：:]\s*(.+?)\s*', raw)
        if not match or not match[1].strip() or not match[3].strip():
            raise ValueError(f'malformed script at line {number}: expected speaker〔instruction〕：text')
        if len(lines) >= 500 or len(match[1].strip()) > 120 or len(match[2] or '') > 1000 or len(match[3].strip()) > 5000:
            raise ValueError(f'script limits exceeded at line {number}: max 500 lines, speaker120/instruction1000/text5000')
        lines.append(dict(id=uuid.uuid4().hex, line_number=number, speaker=match[1].strip(),
                          instruction=(match[2] or '').strip(), text=match[3].strip(),
                          candidates=[], selected_candidate_id=None))
    if not lines:
        raise ValueError('script must contain at least one dialogue line')
    return lines


class ProjectStore:
    """JSON schema1 projects. Hashes detect damage, not signed authenticity.

    All file fields on candidates are relative to the project directory. Exports
    alone return absolute paths for immediate download. The edit lock is local to
    this process, not a distributed/multi-process filesystem lock.
    """
    def __init__(self, root=None):
        self.root = Path(root if root is not None else get_config().outputs_dir / 'projects').expanduser().resolve()

    def _directory(self, project_id):
        path = self.root / _id(project_id)
        if path.is_symlink() or path.resolve().parent != self.root:
            raise ValueError('unsafe project directory')
        return path

    def _local(self, directory, name, *, exists=True):
        if not isinstance(name, str) or '\\' in name or ':' in name:
            raise ValueError('unsafe project-relative file')
        relative = PurePosixPath(name)
        if relative.is_absolute() or any(p in {'..', '.'} for p in relative.parts) or not relative.parts:
            raise ValueError('unsafe project-relative file')
        path = directory
        for part in relative.parts:
            path = path / part
            if path.is_symlink():
                raise ValueError('project files must not be symlinks')
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError('file escapes project')
        if exists and not path.is_file():
            raise ValueError('project file is missing')
        return path

    def _save(self, project):
        directory = self._directory(project['id'])
        project['updated_at'] = _now()
        payload = json.dumps(project, ensure_ascii=False, allow_nan=False, indent=2)
        descriptor, temporary = tempfile.mkstemp(prefix='.project-', suffix='.tmp', dir=directory)
        try:
            with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, directory / 'project.json')
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def create(self, name, script):
        if not isinstance(name, str) or not name.strip() or len(name) > 120:
            raise ValueError('name must contain 1–120 characters')
        lines = _parse(script)
        with _LOCK:
            project = dict(schema_version=1, id=uuid.uuid4().hex, name=name.strip(), script=script,
                           created_at=_now(), updated_at=_now(), bindings={}, lines=lines)
            directory = self._directory(project['id'])
            directory.mkdir(parents=True, exist_ok=False)
            (directory / 'candidates').mkdir()
            self._save(project)
            return project

    def load(self, project_id):
        with _LOCK:
            directory = self._directory(project_id)
            if not directory.is_dir():
                raise FileNotFoundError(f'project not found: {project_id}')
            path = self._local(directory, 'project.json')
            try:
                project = json.loads(path.read_text(encoding='utf-8'))
                self._validate(project, project_id, directory)
            except (KeyError, TypeError, UnicodeError, RecursionError, json.JSONDecodeError) as exc:
                raise ValueError(f'corrupt project manifest: {exc}') from exc
            return project

    def _validate(self, p, project_id, directory):
        fields = {'schema_version', 'id', 'name', 'script', 'created_at', 'updated_at', 'bindings', 'lines'}
        if not isinstance(p, dict) or set(p) != fields or type(p['schema_version']) is not int or p['schema_version'] != 1 or p['id'] != project_id:
            raise ValueError('invalid project schema or ID')
        _validate_json_value(p, 'project')
        if not isinstance(p['name'], str) or not p['name'].strip() or len(p['name']) > 120:
            raise ValueError('invalid project name')
        for key in ('created_at', 'updated_at'):
            if not isinstance(p[key], str) or datetime.fromisoformat(p[key]).tzinfo is None:
                raise ValueError('invalid project timestamp')
        parsed = _parse(p['script'])
        if not isinstance(p['lines'], list) or len(parsed) != len(p['lines']) or not isinstance(p['bindings'], dict):
            raise ValueError('invalid lines or bindings')
        speakers = {line['speaker'] for line in parsed}
        for speaker, binding in p['bindings'].items():
            if speaker not in speakers:
                raise ValueError('unknown bound speaker')
            _id(binding)
        seen = set()
        for line, original in zip(p['lines'], parsed):
            if set(line) != set(original):
                raise ValueError('invalid line schema')
            for key in ('line_number', 'speaker', 'text', 'instruction'):
                if line[key] != original[key] or type(line[key]) is not type(original[key]):
                    raise ValueError('line does not match script')
            _id(line['id'])
            if line['id'] in seen or not isinstance(line['candidates'], list):
                raise ValueError('duplicate line ID or invalid candidates')
            seen.add(line['id'])
            candidate_ids = set()
            for candidate in line['candidates']:
                expected = {'id', 'file', 'sha256', 'seed', 'provenance', 'favorite', 'issues', 'note', 'scores'}
                if not isinstance(candidate, dict) or set(candidate) != expected:
                    raise ValueError('invalid candidate schema')
                cid = _id(candidate['id'])
                if cid in seen or candidate['file'] != f'candidates/{cid}.wav':
                    raise ValueError('duplicate candidate ID or invalid file')
                seen.add(cid)
                candidate_ids.add(cid)
                audio = self._local(directory, candidate['file'])
                if not isinstance(candidate['sha256'], str) or not _HASH.fullmatch(candidate['sha256']) or _hash(audio) != candidate['sha256']:
                    raise ValueError('candidate audio hash mismatch')
                if type(candidate['seed']) is not int or not 0 <= candidate['seed'] <= 0x7FFFFFFFFFFFFFFF:
                    raise ValueError('invalid candidate seed')
                if not isinstance(candidate['provenance'], dict) or candidate['provenance'].get('rights_confirmed') is not True:
                    raise ValueError('invalid candidate provenance')
                if type(candidate['favorite']) is not bool or not isinstance(candidate['note'], str) or not isinstance(candidate['issues'], list) or any(not isinstance(v, str) for v in candidate['issues']):
                    raise ValueError('invalid candidate review')
                scores = candidate['scores']
                if not isinstance(scores, dict) or not _AXES.issubset(scores) or set(scores) - (_AXES | {'signal', 'warnings'}):
                    raise ValueError('invalid score axes')
                for axis in _AXES:
                    value = scores[axis]
                    if value is not None and (not isinstance(value, dict) or not isinstance(value.get('status'), str) or 'value' not in value or not isinstance(value.get('method'), str)):
                        raise ValueError('invalid independent score axis')
            if line['selected_candidate_id'] is not None and line['selected_candidate_id'] not in candidate_ids:
                raise ValueError('selected candidate does not belong to line')

    def list(self):
        with _LOCK:
            result = []
            if self.root.is_dir():
                for entry in sorted(self.root.iterdir()):
                    if _ID.fullmatch(entry.name):
                        try:
                            result.append(self.load(entry.name))
                        except (ValueError, OSError):
                            continue
            return result

    def bind(self, project_id, speaker, character_id):
        with _LOCK:
            project = self.load(project_id)
            if speaker not in {line['speaker'] for line in project['lines']}:
                raise ValueError('speaker not present in project')
            CharacterStore().load(_id(character_id))
            project['bindings'][speaker] = character_id
            self._save(project)
            return project

    @staticmethod
    def _line(project, line_id):
        _id(line_id)
        for line in project['lines']:
            if line['id'] == line_id:
                return line
        raise ValueError('line not found in project')

    def generate_line(self, project_id, line_id, *, count=1, seed=42, confirmed=False, character_store=None, device='cuda:0'):
        if confirmed is not True:
            raise ValueError('explicit rights confirmation is required')
        if device not in ('cuda:0', 'cpu'):
            raise ValueError('device must be cuda:0 or cpu; no automatic fallback')
        if type(count) is not int or not 1 <= count <= 4:
            raise ValueError('count must be 1–4')
        if type(seed) is not int or not 0 <= seed <= 0x7FFFFFFFFFFFFFFF - count + 1:
            raise ValueError('invalid seed range')
        with _LOCK:
            project = self.load(project_id)
            line = self._line(project, line_id)
            binding = project['bindings'].get(line['speaker'])
            if not binding:
                raise ValueError('bind a character before generating')
            store = character_store if character_store is not None else CharacterStore()
            card = store.load(binding)
            reference_hash = _hash(Path(card['reference_audio']))
            directory = self._directory(project_id)
            self._local(directory, 'candidates/.placeholder', exists=False)  # validate directory containment
            for index in range(count):
                cid = uuid.uuid4().hex
                request = studio.GenerationRequest(task='character', character_id=binding, text=line['text'],
                                                   instruction=line['instruction'], seed_override=seed + index, device=device)
                metadata = studio.generate(request, character_store=store, output_dir=directory / 'candidates')
                if metadata.get('character_id') != binding or metadata.get('reference_info', {}).get('sha256') != reference_hash:
                    raise ValueError('generation provenance does not match bound character/reference')
                if _hash(Path(card['reference_audio'])) != reference_hash:
                    raise ValueError('bound reference changed during generation')
                source = Path(metadata['file']).resolve()
                target = self._local(directory, f'candidates/{cid}.wav', exists=False)
                if target.exists():
                    raise FileExistsError(target)
                # Validate source and committed copy; backend metadata alone is not audio evidence.
                def validate_generated(path):
                    with sf.SoundFile(path) as recording:
                        if recording.format != 'WAV' or recording.subtype != 'FLOAT' or recording.frames <= 0 or recording.samplerate <= 0 or recording.channels > 32:
                            raise ValueError('generated candidate must be nonempty FLOAT WAV')
                        if recording.frames * recording.channels > MAX_EXPORT_SAMPLE_BUDGET:
                            raise ValueError('generated candidate exceeds sample cap')
                        peak = 0.
                        for block in recording.blocks(blocksize=65536, dtype='float64', always_2d=True):
                            if not np.isfinite(block).all():
                                raise ValueError('generated candidate contains nonfinite samples')
                            peak = max(peak, float(np.max(np.abs(block))))
                        if peak <= 1e-8:
                            raise ValueError('generated candidate is silent')
                validate_generated(source)
                with source.open('rb') as src, target.open('xb') as dst:
                    shutil.copyfileobj(src, dst)
                validate_generated(target)
                provenance = dict(metadata)
                provenance['rights_confirmed'] = True
                candidate = dict(id=cid, file=f'candidates/{cid}.wav', sha256=_hash(target), seed=seed + index,
                                 provenance=provenance, favorite=False, issues=[], note='',
                                 scores={axis: None for axis in sorted(_AXES)})
                _validate_json_value(candidate, 'candidate')
                line['candidates'].append(candidate)
                self._save(project)  # commit each success before attempting the next render
            return project

    def review(self, project_id, line_id, candidate_id, *, selected=None, favorite=None, issues=None, note=None):
        with _LOCK:
            project = self.load(project_id)
            line = self._line(project, line_id)
            _id(candidate_id)
            candidate = next((c for c in line['candidates'] if c['id'] == candidate_id), None)
            if candidate is None:
                raise ValueError('candidate not found in line')
            for key, value in (('selected', selected), ('favorite', favorite)):
                if value is not None and type(value) is not bool:
                    raise ValueError(f'{key} must be a bool')
            if issues is not None and (not isinstance(issues, list) or any(not isinstance(v, str) for v in issues)):
                raise ValueError('issues must be a list of strings')
            if note is not None and not isinstance(note, str):
                raise ValueError('note must be a string')
            if selected is True:
                line['selected_candidate_id'] = candidate_id
            elif selected is False and line['selected_candidate_id'] == candidate_id:
                line['selected_candidate_id'] = None
            for key, value in (('favorite', favorite), ('issues', issues), ('note', note)):
                if value is not None:
                    candidate[key] = value
            candidate['provenance'].setdefault('review_history', []).append(
                dict(timestamp=_now(), selected=selected, favorite=favorite,
                     issues=issues[:] if issues is not None else None, note=note))
            self._save(project)
            return project

    def assess(self, project_id, line_id, candidate_id, *, observed_text=None, run_speaker=False, run_quality=False, character_store=None):
        """Persist auxiliary evidence; never change human selection or favorites.

        Speaker comparison uses the original generation character and reference
        digest, never a later speaker re-binding or an arbitrary provenance path.
        """
        from timbreforge.candidates import measure_candidate
        if type(run_speaker) is not bool or type(run_quality) is not bool:
            raise ValueError('measurement switches must be bools')
        with _LOCK:
            project = self.load(project_id)
            line = self._line(project, line_id)
            _id(candidate_id)
            candidate = next((c for c in line['candidates'] if c['id'] == candidate_id), None)
            if candidate is None:
                raise ValueError('candidate not found in line')
            reference = None
            if run_speaker:
                provenance = candidate['provenance']
                original_id = provenance.get('character_id')
                store = character_store if character_store is not None else CharacterStore()
                card = store.load(_id(original_id))
                reference = card['reference_audio']
                expected_hash = provenance.get('reference_info', {}).get('sha256')
                if not expected_hash or _hash(Path(reference)) != expected_hash:
                    raise ValueError('original generation reference is missing or changed; cannot compare speaker')
            path = self._local(self._directory(project_id), candidate['file'])
            measured = measure_candidate(path, reference=reference, expected_text=line['text'],
                                         observed_text=observed_text, run_speaker=run_speaker, run_quality=run_quality)
            # Preserve previously measured independent axes when not requested.
            previous = candidate['scores']
            for axis, requested in (('speaker_similarity', run_speaker), ('quality', run_quality),
                                    ('content_accuracy', observed_text is not None)):
                if not requested and previous.get(axis) is not None:
                    measured[axis] = previous[axis]
            _validate_json_value(measured, 'scores')
            candidate['scores'] = measured
            candidate['provenance'].setdefault('assessment_history', []).append(
                dict(timestamp=_now(), observed_text=observed_text, run_speaker=run_speaker,
                     run_quality=run_quality, scores=measured))
            self._save(project)
            return project

    def export(self, project_id, *, gap_seconds=.25):
        if type(gap_seconds) not in (int, float) or not math.isfinite(gap_seconds) or not 0 <= gap_seconds <= 60:
            raise ValueError('gap_seconds must be finite and within 0–60')
        with _LOCK:
            project = self.load(project_id)
            directory = self._directory(project_id)
            selected = []
            total = 0
            gap = round(gap_seconds * RATE)
            speakers = list(dict.fromkeys(line['speaker'] for line in project['lines']))
            for index, line in enumerate(project['lines']):
                cid = line['selected_candidate_id']
                if cid is None:
                    raise ValueError(f'every line must be manually selected; missing line {line["line_number"]}')
                candidate = next(c for c in line['candidates'] if c['id'] == cid)
                path = self._local(directory, candidate['file'])
                info = sf.info(path)
                if info.frames <= 0 or info.samplerate <= 0 or info.channels > 32 or info.frames * info.channels > MAX_EXPORT_SAMPLE_BUDGET:
                    raise ValueError('invalid selected audio')
                frames = max(1, round(info.frames * RATE / info.samplerate))
                total += frames + (gap if index else 0)
                if total > MAX_EXPORT_SECONDS * RATE or total * (len(speakers) + 1) > MAX_EXPORT_SAMPLE_BUDGET:
                    raise ValueError('export duration/memory cap exceeded; split the project')
                selected.append((line, candidate, path, info, frames))
            eid = uuid.uuid4().hex
            output = self._local(directory, f'exports/{eid}/mixed.wav', exists=False).parent
            output.mkdir(parents=True, exist_ok=False)
            (output / 'lines').mkdir()
            (output / 'stems').mkdir()
            timeline = []
            # One stem at a time bounds memory regardless of speaker count. float64
            # accumulation, FLOAT output, no limiter/normalization/clipping applied.
            mixed = np.zeros(total, dtype=np.float64)
            stem_paths = []
            conversions = []
            cursor = 0
            for index, (line, candidate, path, info, frames) in enumerate(selected):
                if index:
                    cursor += gap
                raw_name = f'lines/line_{index + 1:04d}.wav'
                shutil.copyfile(path, output / raw_name)
                if _hash(output / raw_name) != candidate['sha256'] or _hash(path) != candidate['sha256']:
                    raise ValueError('selected audio changed during export copy')
                timeline.append(dict(line_id=line['id'], candidate_id=candidate['id'], speaker=line['speaker'],
                                     text=line['text'], instruction=line['instruction'], file=raw_name,
                                     seed=candidate['seed'], sha256=candidate['sha256'], provenance=candidate['provenance'],
                                     favorite=candidate['favorite'], issues=candidate['issues'], note=candidate['note'], scores=candidate['scores'],
                                     start_frame=cursor, end_frame=cursor + frames,
                                     start_seconds=cursor / RATE, end_seconds=(cursor + frames) / RATE))
                conversions.append(dict(line_id=line['id'], original_sample_rate=info.samplerate,
                                        original_channels=info.channels, original_frames=info.frames,
                                        output_frames=frames, resampled=info.samplerate != RATE,
                                        downmixed=info.channels != 1))
                cursor += frames
            for ordinal, speaker in enumerate(speakers, 1):
                stem = np.zeros(total, dtype=np.float64)
                for item, (line, candidate, path, info, frames) in zip(timeline, selected):
                    if line['speaker'] != speaker:
                        continue
                    # Check input size before decode as well as canonical duration.
                    if info.frames * info.channels > MAX_EXPORT_SAMPLE_BUDGET:
                        raise ValueError('source audio memory cap exceeded')
                    # Decode our verified raw export copy, then verify both files
                    # again so concurrent source edits cannot silently alter stems.
                    copy_path = output / item['file']
                    audio, sr = sf.read(copy_path, dtype='float64', always_2d=True)
                    if _hash(copy_path) != candidate['sha256'] or _hash(path) != candidate['sha256']:
                        raise ValueError('selected audio changed during export decode')
                    if not np.isfinite(audio).all():
                        raise ValueError('selected audio contains nonfinite samples')
                    mono = audio.mean(axis=1, dtype=np.float64)
                    if sr != RATE:
                        mono = np.interp(np.arange(frames, dtype=np.float64) * sr / RATE,
                                         np.arange(len(mono), dtype=np.float64), mono)
                    if len(mono) != frames:
                        raise ValueError('selected audio changed during export')
                    stem[item['start_frame']:item['end_frame']] += mono
                name = f'stems/speaker_{ordinal:03d}.wav'
                sf.write(output / name, stem, RATE, subtype='FLOAT')
                mixed += stem
                stem_paths.append(dict(speaker=speaker, file=name))
            sf.write(output / 'mixed.wav', mixed, RATE, subtype='FLOAT')
            with (output / 'dialogue.csv').open('w', newline='', encoding='utf-8-sig') as stream:
                writer = csv.DictWriter(stream, fieldnames=['line_id', 'candidate_id', 'speaker', 'text', 'instruction', 'file', 'start_seconds', 'end_seconds', 'start_frame', 'end_frame'])
                writer.writeheader()
                writer.writerows({key: row[key] for key in writer.fieldnames} for row in timeline)
            manifest = dict(schema_version=1, id=eid, project_id=project_id, created_at=_now(),
                            sample_rate=RATE, channels=1, subtype='FLOAT', mixing_dtype='float64',
                            normalization=False, gap_seconds=gap / RATE, requested_gap_seconds=gap_seconds,
                            duration_seconds=total / RATE, frames=total,
                            resampling='numpy.interp linear; frame count rounded to nearest canonical sample',
                            channel_conversion='arithmetic mean to mono', conversions=conversions,
                            mixed_audio='mixed.wav', stems=stem_paths, timeline=timeline)
            for name, data in (('dialogue.json', timeline), ('manifest.json', manifest)):
                (output / name).write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
            archive = output / 'project.zip'
            with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as bundle:
                for path in sorted(output.rglob('*')):
                    if path.is_file() and path != archive:
                        bundle.write(path, path.relative_to(output).as_posix())
            return dict(id=eid, directory=str(output), zip=str(archive), file=str(output / 'mixed.wav'),
                        mixed_audio=str(output / 'mixed.wav'), manifest=str(output / 'manifest.json'),
                        csv=str(output / 'dialogue.csv'), dialogue=str(output / 'dialogue.json'),
                        stems=[dict(speaker=s['speaker'], file=str(output / s['file'])) for s in stem_paths],
                        line_audio=[str(output / row['file']) for row in timeline], timeline=timeline)
