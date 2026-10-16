"""Project integration tests: real local WAVs, mocked inference, CPU only."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import uuid
import zipfile

import numpy as np
import soundfile as sf

from timbreforge.projects import ProjectStore, RATE


class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = ProjectStore(self.root / 'projects')
        self.character_id = uuid.uuid4().hex
        self.reference = self.root / 'reference.wav'
        sf.write(self.reference, np.full(240, .2), RATE, subtype='FLOAT')
        self.reference_hash = hashlib.sha256(self.reference.read_bytes()).hexdigest()
        self.card = dict(id=self.character_id, reference_audio=str(self.reference))
        self.load_patch = patch('timbreforge.projects.CharacterStore.load', return_value=self.card)
        self.load_patch.start()
        self.addCleanup(self.load_patch.stop)
        self.calls = []
        self.generate_patch = patch('timbreforge.projects.studio.generate', side_effect=self.render)
        self.generate_patch.start()
        self.addCleanup(self.generate_patch.stop)

    def render(self, request, *, character_store, output_dir):
        self.calls.append(request)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / (uuid.uuid4().hex + '.wav')
        # Alternate rates and channel counts; amplitude > 1 detects normalization.
        if request.text == 'two':
            sf.write(path, np.full((120, 2), 1.5), 12000, subtype='FLOAT')
        else:
            sf.write(path, np.full(240, .5), RATE, subtype='FLOAT')
        return dict(file=str(path), character_id=request.character_id, instruction=request.instruction,
                    params={'seed': request.seed_override}, reference_info={'sha256': self.reference_hash},
                    reference_audio=str(self.reference), backend='mock')

    def project(self, script='A〔quiet〕:one\nB：two'):
        project = self.store.create('scene', script)
        for speaker in dict.fromkeys(line['speaker'] for line in project['lines']):
            project = self.store.bind(project['id'], speaker, self.character_id)
        return project

    def generate_all(self, project):
        for line in project['lines']:
            project = self.store.generate_line(project['id'], line['id'], confirmed=True)
            candidate = next(l for l in project['lines'] if l['id'] == line['id'])['candidates'][-1]
            project = self.store.review(project['id'], line['id'], candidate['id'], selected=True)
        return project

    def test_parse_and_persist(self):
        p = self.store.create('夜', '旁白：夜已经很深了。\n\n角色A〔压低声音〕:你听见了吗？\n角色B〔迟疑〕：可能只是风。')
        self.assertEqual([l['line_number'] for l in p['lines']], [1, 3, 4])
        self.assertEqual(p['lines'][1]['instruction'], '压低声音')
        self.assertEqual(ProjectStore(self.store.root).load(p['id']), p)
        self.assertEqual(self.store.list(), [p])
        with self.assertRaisesRegex(ValueError, 'line 2'):
            self.store.create('bad', 'A:ok\nnot a dialogue')
        with self.assertRaises(ValueError):
            self.store.create('empty', '\n  ')

    def test_generation_rights_count_requests_append_and_manual_choice(self):
        p = self.project('A〔quiet〕:one')
        pid, lid = p['id'], p['lines'][0]['id']
        for options in ({}, {'confirmed': True, 'count': 0}, {'confirmed': True, 'count': 5}, {'confirmed': True, 'seed': True}):
            with self.assertRaises(ValueError):
                self.store.generate_line(pid, lid, **options)
        p = self.store.generate_line(pid, lid, confirmed=True, count=2, seed=51)
        line = p['lines'][0]
        self.assertIsNone(line['selected_candidate_id'])
        self.assertEqual([r.seed_override for r in self.calls], [51, 52])
        self.assertEqual(self.calls[0].task, 'character')
        self.assertEqual(self.calls[0].instruction, 'quiet')
        candidate = line['candidates'][0]
        self.assertEqual(candidate['scores'], dict(speaker_similarity=None, content_accuracy=None, quality=None))
        self.assertTrue(candidate['provenance']['rights_confirmed'])
        p = self.store.review(pid, lid, candidate['id'], selected=True, favorite=True, issues=['pace'], note='keep')
        p = self.store.generate_line(pid, lid, confirmed=True)
        self.assertEqual(len(p['lines'][0]['candidates']), 3)
        self.assertEqual(p['lines'][0]['selected_candidate_id'], candidate['id'])
        self.assertEqual(p['lines'][0]['candidates'][0]['note'], 'keep')
        p = self.store.review(pid, lid, p['lines'][0]['candidates'][1]['id'], selected=False)
        self.assertEqual(p['lines'][0]['selected_candidate_id'], candidate['id'])
        self.assertEqual(self.store.load(pid), p)

    def test_explicit_cpu_device_reaches_studio_without_fallback(self):
        p = self.project('A:one')
        self.store.generate_line(p['id'], p['lines'][0]['id'], confirmed=True, device='cpu')
        self.assertEqual(self.calls[-1].device, 'cpu')
        self.store.generate_line(p['id'], p['lines'][0]['id'], confirmed=True)
        self.assertEqual(self.calls[-1].device, 'cuda:0')
        before = len(self.calls)
        with self.assertRaisesRegex(ValueError, 'device'):
            self.store.generate_line(p['id'], p['lines'][0]['id'], confirmed=True, device='auto')
        self.assertEqual(len(self.calls), before)

    def test_failure_keeps_previous_success_and_retry(self):
        p = self.project('A:one')
        pid, lid = p['id'], p['lines'][0]['id']
        original = self.render
        def failing(request, **kwargs):
            if request.seed_override == 43:
                raise RuntimeError('backend down')
            return original(request, **kwargs)
        with patch('timbreforge.projects.studio.generate', side_effect=failing):
            with self.assertRaisesRegex(RuntimeError, 'backend down'):
                self.store.generate_line(pid, lid, count=3, confirmed=True)
        self.assertEqual(len(self.store.load(pid)['lines'][0]['candidates']), 1)
        p = self.store.generate_line(pid, lid, confirmed=True)
        self.assertEqual(len(p['lines'][0]['candidates']), 2)

    def test_export_alignment_raw_provenance_and_unique(self):
        p = self.project()
        with self.assertRaisesRegex(ValueError, 'every line'):
            self.store.export(p['id'])
        p = self.generate_all(p)
        result = self.store.export(p['id'], gap_seconds=.005)
        mixed, rate = sf.read(result['file'], dtype='float64')
        self.assertEqual(rate, RATE)
        self.assertEqual(len(mixed), 600)
        np.testing.assert_allclose(mixed[:240], .5)
        np.testing.assert_allclose(mixed[240:360], 0)
        np.testing.assert_allclose(mixed[360:], 1.5)
        first, _ = sf.read(result['stems'][0]['file'])
        second, _ = sf.read(result['stems'][1]['file'])
        np.testing.assert_allclose(first + second, mixed)
        np.testing.assert_allclose(first[240:], 0)
        np.testing.assert_allclose(second[:360], 0)
        self.assertEqual(sf.info(result['file']).subtype, 'FLOAT')
        manifest = json.loads(Path(result['manifest']).read_text(encoding='utf-8'))
        self.assertTrue(manifest['conversions'][1]['resampled'])
        self.assertTrue(manifest['conversions'][1]['downmixed'])
        self.assertFalse(manifest['normalization'])
        self.assertEqual(result['timeline'][1]['start_frame'], 360)
        for line, raw in zip(p['lines'], result['line_audio']):
            candidate = line['candidates'][0]
            self.assertEqual(Path(raw).read_bytes(), (self.store.root / p['id'] / candidate['file']).read_bytes())
        with zipfile.ZipFile(result['zip']) as archive:
            self.assertIn('manifest.json', archive.namelist())
            self.assertIn('dialogue.csv', archive.namelist())
            self.assertIn('stems/speaker_001.wav', archive.namelist())
        self.assertNotEqual(self.store.export(p['id'])['directory'], result['directory'])

    def test_safe_speaker_filename_and_caps(self):
        p = self.generate_all(self.project('../../outside:one'))
        result = self.store.export(p['id'])
        self.assertEqual(Path(result['stems'][0]['file']).name, 'speaker_001.wav')
        with patch('timbreforge.projects.MAX_EXPORT_SECONDS', .001):
            with self.assertRaisesRegex(ValueError, 'cap'):
                self.store.export(p['id'])
        for gap in (-1, float('nan'), 61, True):
            with self.assertRaises(ValueError):
                self.store.export(p['id'], gap_seconds=gap)

    def test_assess_persisted_without_automatic_choice(self):
        p = self.generate_all(self.project('A:one'))
        line = p['lines'][0]
        candidate = line['candidates'][0]
        p = self.store.assess(p['id'], line['id'], candidate['id'], observed_text='one')
        scores = p['lines'][0]['candidates'][0]['scores']
        self.assertEqual(scores['content_accuracy']['value'], 0)
        self.assertEqual(scores['speaker_similarity']['status'], 'unverified')
        self.assertEqual(p['lines'][0]['selected_candidate_id'], candidate['id'])
        self.assertEqual(self.store.load(p['id']), p)
        with patch('timbreforge.projects._hash', return_value='0' * 64):
            with self.assertRaises(ValueError):
                self.store.assess(p['id'], line['id'], candidate['id'], run_speaker=True)

    def test_corrupt_paths_ids_schema_hash_and_selection(self):
        p = self.generate_all(self.project('A:one'))
        path = self.store.root / p['id'] / 'project.json'
        original = path.read_text(encoding='utf-8')
        mutations = [lambda d: d.update(schema_version=2), lambda d: d.update(id=uuid.uuid4().hex),
                     lambda d: d['lines'][0]['candidates'][0].update(file='../escape.wav'),
                     lambda d: d['lines'][0].update(selected_candidate_id=uuid.uuid4().hex),
                     lambda d: d['lines'][0]['candidates'][0].update(sha256='0' * 64)]
        for mutate in mutations:
            value = json.loads(original)
            mutate(value)
            path.write_text(json.dumps(value), encoding='utf-8')
            with self.assertRaises(ValueError):
                self.store.load(p['id'])
            self.assertEqual(self.store.list(), [])
        path.write_text(original, encoding='utf-8')
        for pid in ('../bad', 'A' * 32, 'x', '/etc/passwd'):
            with self.assertRaises(ValueError):
                self.store.load(pid)
        candidate_path = self.store.root / p['id'] / p['lines'][0]['candidates'][0]['file']
        candidate_path.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'hash'):
            self.store.load(p['id'])

    def test_invalid_generated_audio_and_script_limits(self):
        p = self.project('A:one')
        source = self.root / 'invalid.wav'
        def invalid(request, **kwargs):
            return dict(file=str(source), character_id=self.character_id, reference_info={'sha256': self.reference_hash})
        for samples, subtype in ((np.zeros(20), 'FLOAT'), (np.ones(20), 'PCM_16'),
                                  (np.full(20, float('nan')), 'FLOAT')):
            sf.write(source, samples, RATE, subtype=subtype)
            with patch('timbreforge.projects.studio.generate', side_effect=invalid):
                with self.assertRaises(ValueError):
                    self.store.generate_line(p['id'], p['lines'][0]['id'], confirmed=True)
        self.assertEqual(self.store.load(p['id'])['lines'][0]['candidates'], [])
        for script in ('A:' + 'x' * 5001, 'A:x\n' * 501, 'A' * 121 + ':x'):
            with self.assertRaises(ValueError):
                self.store.create('oversized', script)

    def test_review_audit_and_export_review_metadata(self):
        p = self.generate_all(self.project('A:one'))
        line = p['lines'][0]
        cid = line['candidates'][0]['id']
        p = self.store.review(p['id'], line['id'], cid, favorite=True, note='human preference')
        candidate = p['lines'][0]['candidates'][0]
        self.assertEqual(len(candidate['provenance']['review_history']), 2)
        self.assertTrue(candidate['provenance']['review_history'][-1]['favorite'])
        result = self.store.export(p['id'])
        self.assertEqual(result['timeline'][0]['note'], 'human preference')
        self.assertTrue(result['timeline'][0]['favorite'])
        self.assertEqual(result['timeline'][0]['scores'], candidate['scores'])

    def test_unknown_bind_and_candidate_rejected(self):
        p = self.project('A:one')
        with self.assertRaises(ValueError):
            self.store.bind(p['id'], 'missing', self.character_id)
        with self.assertRaises(ValueError):
            self.store.bind(p['id'], 'A', '../invalid')
        with self.assertRaises(ValueError):
            self.store.review(p['id'], p['lines'][0]['id'], uuid.uuid4().hex, selected=True)


if __name__ == '__main__':
    unittest.main()
