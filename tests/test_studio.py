import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from timbreforge.backend_indextts import SynthParams
from timbreforge.studio import GenerationRequest, generate, capabilities


class StudioTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.ref = self.root / 'ref.wav'
        self.raw = self.root / 'output.flac'
        sf.write(self.ref, .1 * np.sin(np.arange(16000)), 16000, subtype='FLOAT')
        sf.write(self.raw, .2 * np.sin(np.arange(8000)), 24000)
        self.result = {'file': str(self.raw), 'elapsed': 2.5}

    def run_request(self, **kwargs):
        return generate(GenerationRequest(text='你好。', reference_audio=str(self.ref), **kwargs),
                        output_dir=self.root / 'out')

    def test_clone_preserves_reference_and_wav_encoding(self):
        with patch('timbreforge.backend_cosyvoice.synthesize', return_value=self.result) as synth:
            result = self.run_request(backend='cosyvoice3', reference_text='原始转录。')
        self.assertEqual(synth.call_args.args[0], str(self.ref))
        self.assertEqual(synth.call_args.kwargs['ref_text'], '原始转录。')
        self.assertEqual(sf.info(result['file']).format, 'WAV')
        self.assertEqual(sf.info(result['file']).subtype, 'FLOAT')
        np.testing.assert_array_equal(sf.read(result['file'])[0], sf.read(self.raw)[0])
        saved = json.loads(Path(result['metadata_file']).read_text(encoding='utf-8'))
        self.assertEqual(saved['reference_text'], '原始转录。')
        self.assertEqual(saved['sample_rate'], 24000)

    def test_instruction_routes_index_emotion_text(self):
        with patch('timbreforge.backend_indextts.synthesize', return_value=self.result) as synth:
            result = self.run_request(task='instruct', instruction='平静、克制。')
        p = synth.call_args.args[1]
        self.assertEqual(p.emotion_mode, 'extra_emotion_text')
        self.assertEqual(p.extra_emotion_text, '平静、克制。')
        self.assertTrue(result['warnings'])

    def test_rejects_ambiguous_tasks_before_inference(self):
        for kwargs in ({'task': 'design'}, {'task': 'instruct'},
                       {'task': 'clone', 'instruction': '快乐'},
                       {'backend': 'unknown'},
                       {'backend': 'cosyvoice3', 'params': SynthParams(emotion={'happy': .5})},
                       {'params': SynthParams(inference_cfg_rate=float('nan'))},
                       {'params': SynthParams(seed=-1)},
                       {'params': SynthParams(emotion=None)},
                       {'save_character_name': 'x' * 121}):
            with self.subTest(kwargs=kwargs), patch('timbreforge.backend_indextts.synthesize') as synth:
                with self.assertRaises(ValueError):
                    self.run_request(**kwargs)
                synth.assert_not_called()

    def test_character_reuses_reference_and_saved_settings(self):
        from timbreforge.characters import CharacterStore
        store = CharacterStore(self.root / 'cards')
        card = store.create('测试角色', self.ref, reference_text='原始句子。', backend='cosyvoice3',
                            params=SynthParams(seed=17).to_dict())
        with patch('timbreforge.backend_cosyvoice.synthesize', return_value=self.result) as synth:
            result = generate(GenerationRequest(task='character', character_id=card['id'], text='新句子。'),
                              character_store=store, output_dir=self.root / 'out')
        self.assertEqual(synth.call_args.args[1].seed, 17)
        self.assertEqual(synth.call_args.args[1].text, '新句子。')
        self.assertEqual(result['character_id'], card['id'])
        self.assertEqual(result['reference_text'], '原始句子。')

    def test_character_seed_override_keeps_sampling_settings(self):
        from timbreforge.characters import CharacterStore
        store = CharacterStore(self.root / 'cards')
        card = store.create('参数保留', self.ref, params=SynthParams(seed=17, top_p=.55, diffusion_steps=40).to_dict())
        with patch('timbreforge.backend_indextts.synthesize', return_value=self.result) as synth:
            generate(GenerationRequest(task='character', character_id=card['id'], text='新句子。', seed_override=99),
                     character_store=store, output_dir=self.root / 'out')
        self.assertEqual(synth.call_args.args[1].seed, 99)
        self.assertEqual(synth.call_args.args[1].top_p, .55)
        self.assertEqual(synth.call_args.args[1].diffusion_steps, 40)

    def test_save_character_after_success(self):
        from timbreforge.characters import CharacterStore
        store = CharacterStore(self.root / 'cards')
        with patch('timbreforge.backend_indextts.synthesize', return_value=self.result):
            result = generate(GenerationRequest(text='你好。', reference_audio=str(self.ref),
                                                save_character_name='新角色'),
                              character_store=store, output_dir=self.root / 'out')
        self.assertEqual(store.load(result['character_id'])['name'], '新角色')

    def test_invalid_backend_audio_does_not_save_character(self):
        sf.write(self.raw, np.zeros(1000), 16000)
        with patch('timbreforge.backend_indextts.synthesize', return_value=self.result):
            with self.assertRaises(RuntimeError):
                self.run_request()
        self.assertFalse(list((self.root / 'out').glob('*.json')))

    def test_capabilities_does_not_claim_design_available(self):
        design = [x for x in capabilities() if 'design' in x['tasks']]
        self.assertEqual(len(design), 1)
        self.assertFalse(design[0]['available'])


if __name__ == '__main__':
    unittest.main()
