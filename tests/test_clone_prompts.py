"""Clone Prompt cache: creation, validation, invalidation and fallback reasons."""
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

from timbreforge import clone_prompts
from timbreforge.characters import CharacterStore


class ClonePromptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.model = self.base / 'model'
        self.model.mkdir()
        (self.model / 'config.json').write_text('{}')
        self.config = patch.object(clone_prompts, 'get_config',
                                   return_value=types.SimpleNamespace(qwen_base_model=self.model))
        self.config.start()
        self.addCleanup(self.config.stop)
        self.store = CharacterStore(self.base / 'characters')
        self.source = self.base / 'voice.wav'
        sf.write(self.source, np.full(24000, 0.08, dtype=np.float32), 24000, subtype='FLOAT')
        self.card = self.store.create('缓存角色', self.source, reference_text='参考台词。',
                                      backend='qwen3')
        self.calls = []

    def fake_backend(self, ref_audio, ref_text, *, clone_mode='icl', device='cuda:0',
                     dtype='float16', output_dir=None, cancel=None):
        self.calls.append({'ref_audio': str(ref_audio), 'ref_text': ref_text,
                           'clone_mode': clone_mode, 'device': device,
                           'output_dir': str(output_dir)})
        target = Path(output_dir) / 'raw-prompt.pt'
        data = f'tensors-{clone_mode}-{len(self.calls)}'.encode()
        target.write_bytes(data)
        config_hash = hashlib.sha256((self.model / 'config.json').read_bytes()).hexdigest()
        return {
            'prompt_file': str(target),
            'prompt_sha256': hashlib.sha256(data).hexdigest(),
            'prompt_bytes': len(data),
            'clone_mode': clone_mode,
            'model_config_sha256': config_hash,
            'qwen_package_version': '0.1.1',
            'device': device,
            'dtype': dtype,
            'tensor_info': {'ref_code': {'shape': [4, 2]},
                            'ref_spk_embedding': {'shape': [16]}},
        }

    def create_prompt(self, **kwargs):
        with patch.object(clone_prompts.backend_qwen, 'create_clone_prompt',
                          side_effect=self.fake_backend):
            return clone_prompts.create_clone_prompt(self.card['id'], store=self.store, **kwargs)

    def test_create_records_fingerprint_and_pointer(self):
        result = self.create_prompt()
        self.assertTrue(result['created'])
        card = result['card']
        self.assertEqual(Path(card['clone_prompt']).parent.name, 'prompts')
        self.assertEqual(Path(card['clone_prompt']).name, result['metadata']['prompt_file'])
        self.assertRegex(Path(card['clone_prompt']).name,
                         r'^clone-icl-[0-9a-f]{12}\.pt$')
        directory = Path(card['reference_audio']).parent / 'prompts'
        self.assertTrue((directory / 'clone_prompt.json').is_file())
        self.assertEqual(list(directory.glob('raw-prompt.pt')), [])
        report = clone_prompts.validate_clone_prompt(card['id'], store=self.store)
        self.assertTrue(report['ok'])
        self.assertEqual(report['status'], 'valid')
        self.assertEqual(report['metadata']['role_id'], card['id'])
        self.assertEqual(report['metadata']['tensor_info']['ref_code']['shape'], [4, 2])
        loaded = clone_prompts.load_clone_prompt(card['id'], store=self.store)
        self.assertEqual(loaded['mode'], 'icl')
        self.assertIn('已缓存', clone_prompts.status_text(report))
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0]['ref_text'], '参考台词。')

    def test_valid_cache_is_reused_unless_forced(self):
        first = self.create_prompt()
        second = self.create_prompt()
        self.assertFalse(second['created'])
        self.assertEqual(first['metadata']['prompt_sha256'], second['metadata']['prompt_sha256'])
        self.assertEqual(len(self.calls), 1)
        forced = self.create_prompt(force=True)
        self.assertTrue(forced['created'])
        self.assertEqual(len(self.calls), 2)
        directory = Path(first['card']['reference_audio']).parent / 'prompts'
        prompts = [item for item in directory.glob('*.pt')]
        self.assertEqual(len(prompts), 1)
        self.assertTrue(clone_prompts.validate_clone_prompt(self.card['id'], store=self.store)['ok'])

    def test_transcript_change_makes_cache_stale_and_rebuilds(self):
        self.create_prompt()
        manifest = Path(self.card['reference_audio']).parent / 'character.json'
        data = json.loads(manifest.read_text(encoding='utf-8'))
        data['reference_text'] = '被外部改过的台词。'
        manifest.write_text(json.dumps(data), encoding='utf-8')
        report = clone_prompts.validate_clone_prompt(self.card['id'], store=self.store)
        self.assertFalse(report['ok'])
        self.assertEqual(report['status'], 'stale')
        self.assertIn('逐字稿已变化', '；'.join(report['reasons']))
        with self.assertRaises(clone_prompts.ClonePromptStale) as caught:
            clone_prompts.load_clone_prompt(self.card['id'], store=self.store)
        self.assertIn('逐字稿', str(caught.exception))
        rebuilt = self.create_prompt()
        self.assertTrue(rebuilt['created'])
        self.assertEqual(self.calls[-1]['ref_text'], '被外部改过的台词。')
        self.assertTrue(clone_prompts.validate_clone_prompt(self.card['id'], store=self.store)['ok'])

    def test_audio_edit_invalidates_cache_through_the_card(self):
        from timbreforge import character_audio
        self.create_prompt()
        character_audio.render_preset(self.card['id'], 'clean_voice', store=self.store)
        card = self.store.load(self.card['id'])
        self.assertIsNone(card['clone_prompt'])
        report = clone_prompts.validate_clone_prompt(self.card['id'], store=self.store)
        self.assertEqual(report['status'], 'stale')
        self.assertIn('参考音频已变化', '；'.join(report['reasons']))
        self.assertIn('回退', clone_prompts.status_text(report))
        reasons = card['provenance'].get('clone_prompt_invalidations', [])
        self.assertTrue(reasons)
        self.assertIn('参考音频', reasons[-1]['reason'])

    def test_clone_mode_switch_invalidates_and_rebuilds_in_new_mode(self):
        self.create_prompt()
        self.store.update(self.card['id'], clone_mode='xvector')
        stale = clone_prompts.validate_clone_prompt(self.card['id'], store=self.store)
        self.assertEqual(stale['status'], 'stale')
        self.assertIn('克隆模式', '；'.join(stale['reasons']))
        rebuilt = self.create_prompt()
        self.assertEqual(rebuilt['metadata']['clone_mode'], 'xvector')
        self.assertRegex(rebuilt['metadata']['prompt_file'], r'^clone-xvector-')

    def test_corrupt_prompt_file_is_reported(self):
        result = self.create_prompt()
        Path(result['path']).write_bytes(b'corrupted')
        report = clone_prompts.validate_clone_prompt(self.card['id'], store=self.store)
        self.assertEqual(report['status'], 'corrupt')
        self.assertIn('哈希', '；'.join(report['reasons']))
        self.assertIn('损坏', clone_prompts.status_text(report))
        with self.assertRaises(clone_prompts.ClonePromptStale):
            clone_prompts.load_clone_prompt(self.card['id'], store=self.store)

    def test_invalidate_removes_files_and_is_idempotent(self):
        result = self.create_prompt()
        directory = Path(result['card']['reference_audio']).parent / 'prompts'
        invalidated = clone_prompts.invalidate_clone_prompt(
            self.card['id'], '用户手动清理', store=self.store)
        self.assertTrue(invalidated['removed'])
        self.assertIsNone(invalidated['card']['clone_prompt'])
        self.assertEqual(list(directory.glob('*.pt')), [])
        self.assertFalse((directory / 'clone_prompt.json').exists())
        again = clone_prompts.invalidate_clone_prompt(self.card['id'], '再次清理', store=self.store)
        self.assertEqual(again['removed'], [])

    def test_unsupported_backend_is_reported_without_cache_write(self):
        card = self.store.create('其他后端', self.source, backend='indextts')
        report = clone_prompts.validate_clone_prompt(card['id'], store=self.store)
        self.assertEqual(report['status'], 'unsupported')
        with patch.object(clone_prompts.backend_qwen, 'create_clone_prompt') as backend:
            with self.assertRaises(clone_prompts.ClonePromptError):
                clone_prompts.create_clone_prompt(card['id'], store=self.store)
        backend.assert_not_called()

    def test_missing_model_config_is_explained(self):
        self.create_prompt()
        with patch.object(clone_prompts, 'get_config',
                          return_value=types.SimpleNamespace(qwen_base_model=self.base / 'gone')):
            report = clone_prompts.validate_clone_prompt(self.card['id'], store=self.store)
            self.assertEqual(report['status'], 'stale')
            self.assertIn('模型缺失', '；'.join(report['reasons']))
            with self.assertRaises(FileNotFoundError):
                self.create_prompt()

    def test_unsafe_prompt_filename_in_metadata_is_rejected(self):
        result = self.create_prompt()
        directory = Path(result['card']['reference_audio']).parent / 'prompts'
        metadata_path = directory / 'clone_prompt.json'
        data = json.loads(metadata_path.read_text(encoding='utf-8'))
        data['prompt_file'] = '../../outside.pt'
        metadata_path.write_text(json.dumps(data), encoding='utf-8')
        report = clone_prompts.validate_clone_prompt(self.card['id'], store=self.store)
        self.assertEqual(report['status'], 'missing')
        self.assertIn('文件名字段不安全', '；'.join(report['reasons']))


if __name__ == '__main__':
    unittest.main()

