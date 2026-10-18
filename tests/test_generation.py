"""Unified character generation: provenance, clone-prompt decisions, cleanup."""
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

from timbreforge import clone_prompts, generation
from timbreforge.characters import CharacterStore


def write_audio(path, samples=None, sample_rate=24000):
    if samples is None:
        time_axis = np.arange(int(sample_rate * 0.4)) / sample_rate
        samples = 0.2 * np.sin(2 * np.pi * 180 * time_axis)
    sf.write(str(path), samples, sample_rate, format='WAV', subtype='FLOAT')
    return Path(path)


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.store = CharacterStore(self.base / 'characters')
        self.source = write_audio(self.base / 'source.wav')
        self.card = self.store.create('主角色', self.source, reference_text='参考台词。',
                                      backend='qwen3', params={'qwen': {'seed': 17}})
        self.outputs = self.base / 'generation'
        self.model = self.base / 'model'
        self.model.mkdir()
        (self.model / 'config.json').write_text('{}')
        self.config = patch.object(clone_prompts, 'get_config',
                                   return_value=types.SimpleNamespace(qwen_base_model=self.model))
        self.config.start()
        self.addCleanup(self.config.stop)
        self.calls = []

    def fake_clone(self, ref_audio, ref_text, text, params, **kwargs):
        self.calls.append({'ref_audio': ref_audio, 'ref_text': ref_text, 'text': text,
                           'params': params, 'kwargs': kwargs})
        raw = self.base / f'raw-{len(self.calls)}.wav'
        write_audio(raw)
        return {'file': str(raw), 'mode': 'voice_clone', 'params': params.to_dict(),
                'seed': params.seed, 'model_dir': str(self.model),
                'model_config_sha256': hashlib.sha256(b'{}').hexdigest(),
                'qwen_package_version': '0.1.1', 'device': kwargs.get('device'),
                'dtype': kwargs.get('dtype'), 'text_segments': 1,
                'sample_rate': 24000, 'duration': 0.4, 'log': 'log.txt'}

    def generate(self, **kwargs):
        with patch.object(generation.backend_qwen, 'clone', side_effect=self.fake_clone):
            return generation.generate_for_character(
                self.card['id'], kwargs.pop('text', '新的台词。'),
                store=self.store, output_dir=self.outputs, **kwargs)

    def test_generation_records_full_provenance_and_cleans_raw_output(self):
        result = self.generate(seed=99)
        self.assertTrue(Path(result['file']).is_file())
        self.assertTrue(Path(result['metadata_file']).is_file())
        self.assertEqual(result['run_id'], Path(result['file']).stem)
        self.assertEqual(result['role_id'], self.card['id'])
        self.assertEqual(result['role_version'], 1)
        self.assertEqual(result['seed'], 99)
        self.assertEqual(result['params']['seed'], 99)
        self.assertEqual(result['text'], '新的台词。')
        self.assertEqual(result['reference']['transcript'], '参考台词。')
        self.assertEqual(result['reference']['sha256'], self.card['reference_sha256'])
        self.assertEqual(result['clone_prompt']['used'], False)
        self.assertEqual(result['clone_prompt']['status'], 'fallback')
        self.assertTrue(result['warnings'])
        self.assertEqual(Path(result['file']).read_bytes()[:4], b'RIFF')
        # The intermediate raw model file is not published, only the copied take.
        self.assertFalse(list(self.base.glob('raw-*.wav')))
        self.assertEqual(len(list(self.outputs.rglob('raw-*.wav'))), 0)
        stored = generation.load_generation(result['metadata_file'])
        self.assertEqual(stored['audio']['sha256'],
                         hashlib.sha256(Path(result['file']).read_bytes()).hexdigest())
        self.assertEqual(stored['run_id'], result['run_id'])

    def test_seed_defaults_to_character_and_can_be_overridden(self):
        result = self.generate()
        self.assertEqual(result['seed'], 17)
        second = self.generate(seed=5)
        self.assertEqual(second['seed'], 5)

    def test_character_without_transcript_is_rejected_before_generation(self):
        card = self.store.create('无逐字稿', self.source, backend='qwen3')
        with patch.object(generation.backend_qwen, 'clone') as clone:
            with self.assertRaises(generation.GenerationError):
                generation.generate_for_character(card['id'], '台词', store=self.store,
                                                  output_dir=self.outputs)
        clone.assert_not_called()

    def test_non_qwen_character_is_rejected(self):
        card = self.store.create('其他后端', self.source, backend='indextts')
        with self.assertRaises(generation.GenerationError):
            generation.generate_for_character(card['id'], '台词', store=self.store,
                                              output_dir=self.outputs)

    def test_valid_clone_prompt_is_reused_without_reference_audio(self):
        def fake_prompt(ref_audio, ref_text, *, clone_mode='icl', device='cuda:0',
                        dtype='float16', output_dir=None, cancel=None):
            target = Path(output_dir) / 'raw.pt'
            data = b'prompt-tensors'
            target.write_bytes(data)
            return {'prompt_file': str(target),
                    'prompt_sha256': hashlib.sha256(data).hexdigest(),
                    'clone_mode': clone_mode,
                    'model_config_sha256': hashlib.sha256(b'{}').hexdigest(),
                    'qwen_package_version': '0.1.1', 'device': device, 'dtype': dtype,
                    'tensor_info': {'ref_code': {'shape': [2, 2]}}}
        with patch.object(clone_prompts.backend_qwen, 'create_clone_prompt',
                          side_effect=fake_prompt):
            clone_prompts.create_clone_prompt(self.card['id'], store=self.store)
        result = self.generate()
        self.assertTrue(result['clone_prompt']['used'])
        self.assertEqual(result['clone_prompt']['status'], 'valid')
        self.assertIsNone(self.calls[0]['ref_audio'])
        self.assertIsNotNone(self.calls[0]['kwargs']['clone_prompt'])
        self.assertEqual(self.calls[0]['kwargs']['clone_mode'], 'icl')
        self.assertIn('复用', ' '.join(result['warnings']))

    def test_stale_clone_prompt_falls_back_and_is_removed(self):
        def fake_prompt(ref_audio, ref_text, *, clone_mode='icl', device='cuda:0',
                        dtype='float16', output_dir=None, cancel=None):
            target = Path(output_dir) / 'raw.pt'
            data = b'prompt-tensors'
            target.write_bytes(data)
            return {'prompt_file': str(target),
                    'prompt_sha256': hashlib.sha256(data).hexdigest(),
                    'clone_mode': clone_mode,
                    'model_config_sha256': hashlib.sha256(b'{}').hexdigest(),
                    'qwen_package_version': '0.1.1', 'device': device, 'dtype': dtype,
                    'tensor_info': {'ref_code': {'shape': [2, 2]}}}
        with patch.object(clone_prompts.backend_qwen, 'create_clone_prompt',
                          side_effect=fake_prompt):
            created = clone_prompts.create_clone_prompt(self.card['id'], store=self.store)
        manifest = Path(self.card['reference_audio']).parent / 'character.json'
        data = json.loads(manifest.read_text(encoding='utf-8'))
        data['reference_text'] = '被改过的逐字稿。'
        manifest.write_text(json.dumps(data), encoding='utf-8')
        result = self.generate()
        self.assertFalse(result['clone_prompt']['used'])
        self.assertEqual(result['clone_prompt']['status'], 'fallback')
        self.assertIn('逐字稿', result['clone_prompt']['reason'])
        self.assertEqual(Path(self.calls[0]['ref_audio']), Path(self.card['reference_audio']))
        self.assertEqual(list((Path(created['path']).parent).glob('*.pt')), [])

    def test_failure_never_publishes_partial_audio_or_metadata(self):
        def failing(ref_audio, ref_text, text, params, **kwargs):
            raw = self.base / 'partial.wav'
            write_audio(raw)
            raise RuntimeError('CUDA out of memory')
        with patch.object(generation.backend_qwen, 'clone', side_effect=failing):
            with self.assertRaisesRegex(RuntimeError, 'out of memory'):
                generation.generate_for_character(self.card['id'], '台词', store=self.store,
                                                  output_dir=self.outputs)
        self.assertEqual(list(self.outputs.rglob('*.wav')), [])
        self.assertEqual(list(self.outputs.rglob('*.json')), [])
        # The character card itself is untouched by a failed generation.
        self.assertEqual(self.store.load(self.card['id'])['version'], 1)

    def test_list_and_delete_generations(self):
        first = self.generate(text='第一句。')
        second = self.generate(text='第二句。')
        records = generation.list_generations(self.card['id'], output_dir=self.base / 'generation')
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]['run_id'], second['run_id'])
        deleted = generation.delete_generation(first['metadata_file'])
        self.assertEqual(deleted['run_id'], first['run_id'])
        self.assertFalse(Path(first['file']).exists())
        self.assertEqual(len(generation.list_generations(self.card['id'],
                                                         output_dir=self.base / 'generation')), 1)

    def test_unified_adapter_is_used_when_supplied(self):
        class Adapter:
            def __init__(self):
                self.request = None

            def synthesize(self, request):
                self.request = request
                raw = self.fake_result(request)
                return types.SimpleNamespace(raw=raw)

            def fake_result(self, request):
                raw_path = self.temp / 'adapter-raw.wav'
                write_audio(raw_path)
                return {'file': str(raw_path), 'mode': 'voice_clone',
                        'params': request.params.to_dict(), 'seed': request.params.seed,
                        'model_dir': str(self.model),
                        'model_config_sha256': hashlib.sha256(b'{}').hexdigest(),
                        'qwen_package_version': '0.1.1', 'device': request.device,
                        'dtype': request.dtype, 'text_segments': 1,
                        'sample_rate': 24000, 'duration': 0.4}

        adapter = Adapter()
        adapter.temp = self.base
        adapter.model = self.model
        result = generation.generate_for_character(
            self.card['id'], '适配器台词。', store=self.store, output_dir=self.outputs,
            backend=adapter)
        self.assertIsNotNone(adapter.request)
        self.assertEqual(adapter.request.mode, 'clone')
        self.assertEqual(adapter.request.ref_text, '参考台词。')
        self.assertEqual(Path(adapter.request.ref_audio), Path(self.card['reference_audio']))
        self.assertEqual(result['seed'], 17)
        self.assertTrue(Path(result['file']).is_file())

    def test_text_validation(self):
        for text in ('', '   ', 'x' * 5001):
            with self.assertRaises(ValueError):
                self.generate(text=text)
        with self.assertRaises(ValueError):
            self.generate(device='mps')
        with self.assertRaises(ValueError):
            self.generate(device='cpu', dtype='float16')


if __name__ == '__main__':
    unittest.main()

