"""Single-dimension tuning candidates, A/B audition and promotion to the card."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import types

import numpy as np
import soundfile as sf

from timbreforge import clone_prompts, voice_candidates
from timbreforge.characters import CharacterStore
from timbreforge.voice_recipe import RECIPE_EXAMPLES, normalize


def tone(path, *, sample_rate=24000, seconds=0.5, frequency=200.0):
    time_axis = np.arange(int(sample_rate * seconds)) / sample_rate
    samples = 0.2 * np.sin(2 * np.pi * frequency * time_axis)
    sf.write(str(path), samples, sample_rate, format='WAV', subtype='FLOAT')
    return Path(path)


def differences(first, second):
    changed = []
    for group in first:
        for field, value in first[group].items():
            if second[group][field] != value:
                changed.append((group, field))
    return changed


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.store = CharacterStore(self.base / 'characters')
        self.source = tone(self.base / 'source.wav')
        self.recipe = RECIPE_EXAMPLES['cold_anime_girl']['recipe']
        self.card = self.store.create('微调角色', self.source, reference_text='原始台词。',
                                      backend='qwen3', voice_recipe=self.recipe)
        self.tuning_root = self.base / 'tuning'
        self.model = self.base / 'model'
        self.model.mkdir()
        (self.model / 'config.json').write_text('{}')
        self.config = patch.object(clone_prompts, 'get_config',
                                   return_value=types.SimpleNamespace(qwen_base_model=self.model))
        self.config.start()
        self.addCleanup(self.config.stop)
        self.designs = []

    def designer(self, text, prompt, params, *, device='cuda:0', dtype='float16'):
        self.designs.append({'text': text, 'prompt': prompt, 'seed': params.seed,
                             'device': device, 'dtype': dtype})
        target = self.base / f'design-{len(self.designs)}.wav'
        tone(target, frequency=180 + 20 * len(self.designs))
        return {'file': str(target), 'model_dir': str(self.model),
                'model_config_sha256': hashlib.sha256(b'{}').hexdigest(),
                'qwen_package_version': '0.1.1', 'device': device, 'dtype': dtype}

    def create(self, count=4, **kwargs):
        return voice_candidates.create_session(
            self.card['id'], '试听台词，请保持冷淡。', store=self.store,
            output_dir=self.tuning_root, count=count, design_callable=self.designer, **kwargs)

    def test_candidates_change_exactly_one_dimension_and_record_provenance(self):
        session = self.create(count=4)
        self.assertEqual(len(session['candidates']), 4)
        base = normalize(self.recipe)
        for index, candidate in enumerate(session['candidates']):
            self.assertEqual(candidate['role_id'], self.card['id'])
            self.assertEqual(candidate['role_version'], 1)
            self.assertEqual(candidate['seed'], 42 + index)
            self.assertEqual(candidate['params']['seed'], 42 + index)
            self.assertTrue(candidate['prompt'])
            self.assertEqual(candidate['profile']['model_dir'], str(self.model))
            self.assertEqual(len(differences(base, candidate['recipe'])), 1)
            self.assertTrue(candidate['changed']['label'])
            self.assertNotEqual(candidate['changed']['from'], candidate['changed']['to'])
            path = self.tuning_root / session['id'] / candidate['file']
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), candidate['sha256'])
        dimensions = [candidate['changed']['dimension'] for candidate in session['candidates']]
        self.assertEqual(len(set(dimensions)), 4)
        self.assertEqual(len(self.designs), 4)
        self.assertTrue(all(len(design['prompt']) > 20 for design in self.designs))
        reloaded = voice_candidates.CandidateStore(self.tuning_root).load(session['id'])
        self.assertEqual(reloaded['candidates'], session['candidates'])
        self.assertIn('4 个候选', voice_candidates.status_text(session))
        self.assertIn('试听', voice_candidates.status_text(session))

    def test_audition_and_ab_compare_do_not_modify_candidates(self):
        session = self.create(count=2)
        before = {candidate['id']: (self.tuning_root / session['id'] / candidate['file']).read_bytes()
                  for candidate in session['candidates']}
        listening = voice_candidates.audition(session['id'], output_dir=self.tuning_root)
        self.assertEqual(len(listening['audio']), 2)
        first, second = [candidate['id'] for candidate in session['candidates']]
        result = voice_candidates.compare(session['id'], first, second, match_volume=True,
                                          output_dir=self.tuning_root)
        self.assertTrue(Path(result['a']).is_file())
        self.assertTrue(Path(result['b']).is_file())
        self.assertEqual(result['a_meta']['dimension'], session['candidates'][0]['changed']['dimension'])
        after = {candidate['id']: (self.tuning_root / session['id'] / candidate['file']).read_bytes()
                 for candidate in session['candidates']}
        self.assertEqual(before, after)

    def test_promotion_requires_selection_and_confirmation(self):
        session = self.create(count=2)
        candidate = session['candidates'][0]
        with self.assertRaises(voice_candidates.CandidateError):
            voice_candidates.promote_candidate(session['id'], candidate['id'], store=self.store,
                                               output_dir=self.tuning_root, create_prompt=False)
        voice_candidates.select_candidate(session['id'], candidate['id'],
                                          output_dir=self.tuning_root)
        with self.assertRaises(voice_candidates.CandidateError):
            voice_candidates.promote_candidate(session['id'], candidate['id'], store=self.store,
                                               output_dir=self.tuning_root, create_prompt=False)

    def test_promotion_updates_card_without_touching_original(self):
        session = self.create(count=2)
        candidate = session['candidates'][0]
        voice_candidates.select_candidate(session['id'], candidate['id'],
                                          output_dir=self.tuning_root)
        original = Path(self.card['reference_audio']).read_bytes()
        result = voice_candidates.promote_candidate(
            session['id'], candidate['id'], confirmed=True, create_prompt=False,
            store=self.store, output_dir=self.tuning_root)
        card = result['card']
        self.assertEqual(card['version'], 2)
        self.assertEqual(Path(card['reference_audio']).read_bytes(), original)
        self.assertEqual(Path(card['canonical_audio']).read_bytes(),
                         (self.tuning_root / session['id'] / candidate['file']).read_bytes())
        self.assertEqual(card['reference_text'], '试听台词，请保持冷淡。')
        self.assertEqual(card['voice_recipe'], normalize(candidate['recipe']))
        self.assertEqual(card['seed'], candidate['seed'])
        self.assertEqual(card['source'], 'voice_tuning/candidate')
        self.assertEqual(card['audio_edit_chain'][-1]['event'], 'promote_candidate')
        self.assertEqual(card['provenance']['tuning']['candidate_id'], candidate['id'])
        self.assertEqual(card['provenance']['tuning']['prompt'], candidate['prompt'])
        self.assertEqual(result['clone_prompt']['status'], 'skipped')
        self.assertTrue(result['warnings'])
        promoted = voice_candidates.CandidateStore(self.tuning_root).load(session['id'])['promoted']
        self.assertEqual(promoted['candidate_id'], candidate['id'])
        self.assertEqual(promoted['role_version'], 2)

    def test_promotion_can_create_a_new_clone_prompt(self):
        session = self.create(count=1)
        candidate = session['candidates'][0]
        voice_candidates.select_candidate(session['id'], candidate['id'],
                                          output_dir=self.tuning_root)

        def fake_prompt(ref_audio, ref_text, *, clone_mode='icl', device='cuda:0',
                        dtype='float16', output_dir=None, cancel=None):
            path = Path(output_dir) / 'raw.pt'
            data = b'prompt'
            path.write_bytes(data)
            return {'prompt_file': str(path), 'prompt_sha256': hashlib.sha256(data).hexdigest(),
                    'clone_mode': clone_mode, 'model_config_sha256': hashlib.sha256(b'{}').hexdigest(),
                    'qwen_package_version': '0.1.1', 'device': device, 'dtype': dtype,
                    'tensor_info': {'ref_spk_embedding': {'shape': [8]}}}
        with patch.object(clone_prompts.backend_qwen, 'create_clone_prompt',
                          side_effect=fake_prompt):
            result = voice_candidates.promote_candidate(
                session['id'], candidate['id'], confirmed=True, create_prompt=True,
                store=self.store, output_dir=self.tuning_root)
        self.assertEqual(result['clone_prompt']['status'], 'created')
        self.assertIsNotNone(result['card']['clone_prompt'])
        self.assertEqual(result['warnings'], [])

    def test_failed_prompt_creation_keeps_the_promoted_version(self):
        session = self.create(count=1)
        candidate = session['candidates'][0]
        voice_candidates.select_candidate(session['id'], candidate['id'],
                                          output_dir=self.tuning_root)
        with patch.object(clone_prompts, 'create_clone_prompt',
                          side_effect=FileNotFoundError('model missing')):
            result = voice_candidates.promote_candidate(
                session['id'], candidate['id'], confirmed=True, create_prompt=True,
                store=self.store, output_dir=self.tuning_root)
        self.assertEqual(result['card']['version'], 2)
        self.assertIsNotNone(result['card']['canonical_audio'])
        self.assertEqual(result['clone_prompt']['status'], 'unavailable')
        self.assertIn('普通克隆', result['warnings'][0])

    def test_discard_session_removes_only_session_files(self):
        session = self.create(count=1)
        directory = self.tuning_root / session['id']
        self.assertTrue(directory.is_dir())
        discarded = voice_candidates.discard_session(session['id'], output_dir=self.tuning_root)
        self.assertEqual(discarded['id'], session['id'])
        self.assertFalse(directory.exists())
        self.assertTrue(self.store.load(self.card['id']))

    def test_candidate_count_and_recipe_are_validated(self):
        with self.assertRaises(voice_candidates.CandidateError):
            self.create(count=9)
        with self.assertRaises(voice_candidates.CandidateError):
            self.create(count=0)
        with self.assertRaises(voice_candidates.CandidateError):
            self.create(recipe={'timbre': {'pitch': 'ultra'}})
        session = self.create(count=8)
        self.assertEqual(len(session['candidates']), 8)
        with self.assertRaises(voice_candidates.CandidateError):
            voice_candidates.generate_candidates(session['id'], store=self.store,
                                                 output_dir=self.tuning_root, count=1,
                                                 design_callable=self.designer)


if __name__ == '__main__':
    unittest.main()

