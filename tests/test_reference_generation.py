import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from timbreforge.reference import prepare_reference
from timbreforge.backend_indextts import SynthParams, build_graph, copy_to_comfy_input
from timbreforge.voicegen import main


class ReferenceTests(unittest.TestCase):
    def test_duration_content_cache_and_native_rate(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / 'source.wav'
            sr = 24000
            x = .05 * np.sin(2 * np.pi * 170 * np.arange(sr * 9) / sr)
            sf.write(src, x, sr)
            a = prepare_reference(src, 4, root / 'cache')
            b = prepare_reference(src, 6, root / 'cache')
            self.assertNotEqual(a, b)
            self.assertEqual(sf.info(a).samplerate, sr)
            self.assertEqual(sf.info(a).frames, sr * 4)
            self.assertEqual(sf.info(b).frames, sr * 6)
            self.assertLess(np.max(np.abs(sf.read(a)[0])), .051)
            self.assertEqual(a, prepare_reference(src, 4, root / 'cache'))
            sf.write(src, x * .5, sr)
            self.assertNotEqual(a, prepare_reference(src, 4, root / 'cache'))

    def test_empty_silence_and_invalid_duration_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'silent.wav'
            sf.write(p, np.zeros(16000), 16000)
            for value in (0, -1, float('nan')):
                with self.assertRaises(ValueError):
                    prepare_reference(p, value, Path(td) / 'cache')
            with self.assertRaises(ValueError):
                prepare_reference(p, 1, Path(td) / 'cache')

    def test_comfy_names_cover_contents(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            a, b = root / 'a' / 'same.wav', root / 'b' / 'same.wav'
            a.parent.mkdir(); b.parent.mkdir()
            a.write_bytes(b'first'); b.write_bytes(b'second')
            (root / 'input').mkdir()
            with patch('timbreforge.backend_indextts.COMFY_INPUT', root / 'input'):
                self.assertNotEqual(copy_to_comfy_input(a), copy_to_comfy_input(b))

    def test_emotions_and_decoder_controls_reach_graph(self):
        p = SynthParams(emotion={'happy': .3}, do_sample=False, diffusion_steps=40,
                        inference_cfg_rate=.5)
        g = build_graph('ref.wav', p)['3']['inputs']
        self.assertEqual(g['emotion_mode'], 'emotion_vector')
        self.assertFalse(g['do_sample'])
        self.assertEqual(g['diffusion_steps'], 40)
        self.assertEqual(g['inference_cfg_rate'], .5)
        p = SynthParams(extra_emotion_text='沉稳')
        self.assertEqual(build_graph('ref.wav', p)['3']['inputs']['emotion_mode'],
                         'extra_emotion_text')

    def test_plain_cli_honors_external_reference_without_prompt(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ref = root / 'ref.wav'
            sf.write(ref, .05 * np.sin(np.arange(16000 * 2)), 16000)
            with patch('timbreforge.voicegen.get_bank', return_value=[]), \
                 patch('timbreforge.voicegen.find_voice', side_effect=AssertionError('must not retrieve')), \
                 contextlib.redirect_stdout(io.StringIO()):
                code = main(['--ref-wav', str(ref), '--no-synth', '--outdir', str(root / 'out')])
            self.assertEqual(code, 0)
            meta = json.loads(next((root / 'out').glob('*.meta.json')).read_text(encoding='utf-8'))
            self.assertTrue(Path(meta['ref_wav']).is_file())

    def test_cosy_cli_uses_original_reference_and_records_it(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ref, raw = root / 'ref.wav', root / 'raw.wav'
            sf.write(ref, .05 * np.sin(np.arange(16000 * 9)), 16000)
            sf.write(raw, .1 * np.sin(np.arange(1600)), 16000, subtype='FLOAT')
            with patch('timbreforge.voicegen.get_bank', return_value=[]), \
                 patch('timbreforge.voicegen.find_voice', side_effect=AssertionError('must not retrieve')), \
                 patch('timbreforge.reference.prepare_reference', side_effect=AssertionError('must not crop')), \
                 patch('timbreforge.backend_cosyvoice.synthesize', return_value={
                     'file': str(raw), 'elapsed': 1.0, 'reference': str(ref)}) as synth, \
                 contextlib.redirect_stdout(io.StringIO()):
                code = main(['开心的女声', '--backend', 'cosyvoice3', '--ref-wav', str(ref),
                             '--ref-text', '准确的完整转录。', '--outdir', str(root / 'out')])
            self.assertEqual(code, 0)
            self.assertEqual(synth.call_args.args[0], str(ref.resolve()))
            self.assertEqual(synth.call_args.kwargs['ref_text'], '准确的完整转录。')
            meta = json.loads(next((root / 'out').glob('*.meta.json')).read_text(encoding='utf-8'))
            self.assertEqual(meta['ref_wav'], str(ref.resolve()))
            self.assertEqual(meta['backend'], 'cosyvoice3')
            self.assertEqual(sf.info(meta['ref_wav']).duration, 9)
            np.testing.assert_array_equal(sf.read(meta['output'])[0], sf.read(raw)[0])

    def test_indextts_cli_keeps_custom_decoder_controls(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ref, raw = root / 'ref.wav', root / 'raw.wav'
            sf.write(ref, .05 * np.sin(np.arange(16000 * 2)), 16000)
            sf.write(raw, .1 * np.sin(np.arange(1600)), 16000, subtype='FLOAT')
            with patch('timbreforge.voicegen.get_bank', return_value=[]), \
                 patch('timbreforge.backend_indextts.ping', return_value=(True, 'ok')), \
                 patch('timbreforge.backend_indextts.synthesize', return_value={
                     'file': str(raw), 'elapsed': 1.0}) as synth, \
                 contextlib.redirect_stdout(io.StringIO()):
                code = main(['--ref-wav', str(ref), '--outdir', str(root / 'out'),
                             '--no-sample', '--diffusion-steps', '40',
                             '--inference-cfg-rate', '0.5'])
            self.assertEqual(code, 0)
            p = synth.call_args.args[1]
            self.assertFalse(p.do_sample)
            self.assertEqual(p.diffusion_steps, 40)
            self.assertEqual(p.inference_cfg_rate, .5)
            meta = json.loads(next((root / 'out').glob('*.meta.json')).read_text(encoding='utf-8'))
            self.assertEqual(meta['params']['diffusion_steps'], 40)

    def test_low_energy_reference_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / 'low.wav'
            sf.write(src, np.full(16000, 2e-6, dtype=np.float32), 16000, subtype='FLOAT')
            with self.assertRaisesRegex(ValueError, 'sustained signal'):
                prepare_reference(src, cache_dir=Path(td) / 'cache')

    def test_stale_comfy_schema_rejected_before_submission(self):
        from timbreforge.backend_indextts import synthesize
        with patch('timbreforge.backend_indextts.ping', return_value=(True, 'ok')), \
             patch('timbreforge.backend_indextts._get', return_value={}), \
             patch('timbreforge.backend_indextts._post') as submit:
            with self.assertRaisesRegex(RuntimeError, 'ComfyUI'):
                synthesize('unused.wav', SynthParams(), verbose=False)
        submit.assert_not_called()


if __name__ == '__main__':
    unittest.main()
