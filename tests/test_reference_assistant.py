import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf

from timbreforge.reference_assistant import inspect_reference, prepare_confirmed_reference


class ReferenceAssistantTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / '原始 reference.wav'

    def write(self, samples, sr=1000):
        sf.write(self.source, np.asarray(samples, dtype=np.float32), sr, subtype='FLOAT')
        return self.source

    def test_inspection_json_and_unverified(self):
        self.write(np.full((1000, 2), .25))
        before = self.source.read_bytes()
        r = inspect_reference(self.source, transcript='hello')
        self.assertEqual((r['duration_seconds'], r['channels'], r['samplerate']), (1, 2, 1000))
        self.assertEqual(r['peak'], .25)
        self.assertEqual(r['clipping_fraction'], 0)
        self.assertEqual(r['silence_frame_proportion'], 0)
        self.assertIn('unknown', r['unverified']['music'])
        self.assertIn('unknown', r['unverified']['multiple_speakers'])
        self.assertIn('unverified', r['unverified']['transcript_correspondence'])
        self.assertEqual(before, self.source.read_bytes())
        json.dumps(r, allow_nan=False)

    def test_duration_warning_strict_boundaries(self):
        for duration, backend, count in [(15, 'indextts', 0), (15.001, 'indextts', 1),
                                          (30, 'cosyvoice3', 0), (30.001, 'cosyvoice3', 1),
                                          (30.001, 'indextts', 2)]:
            self.write(np.full(round(duration * 1000), .25))
            r = inspect_reference(self.source, backend=backend)
            self.assertEqual(len(r['warnings']), count)

    def test_near_full_scale_not_clipping_certainty(self):
        self.write([.998, .9991, 1., -1., .2])
        r = inspect_reference(self.source)
        self.assertAlmostEqual(r['clipping_fraction'], .6)
        self.assertIn('not proof', ' '.join(r['warnings']))
        self.assertEqual(r['thresholds']['clipping_absolute_sample_gte'], .999)

    def test_silence_partial_frames_no_padding_and_antiphase(self):
        self.write(np.r_[np.zeros(20), np.full(20, 5e-6), [.1]])
        r = inspect_reference(self.source)
        self.assertAlmostEqual(r['silence_frame_proportion'], 2 / 3)
        self.write(np.column_stack([np.full(41, .25), np.full(41, -.25)]))
        self.assertEqual(inspect_reference(self.source)['silence_frame_proportion'], 0)
        r = prepare_confirmed_reference(self.source, confirmed=True, output_dir=self.root)
        self.assertEqual(r['channels'], 2)
        with self.assertRaisesRegex(ValueError, 'cancelled'):
            prepare_confirmed_reference(self.source, confirmed=True, mono=True, output_dir=self.root)

    def test_backend_modes_and_consumption(self):
        self.write([.1] * 100)
        for transcript, instruction, mode, text, ins in [('', '', 'cross_lingual', False, False),
                ('words', '', 'zero_shot', True, False), ('words', 'happy', 'instruct2', False, True)]:
            c = inspect_reference(self.source, backend='cosyvoice3', transcript=transcript,
                                  instruction=instruction)['backend_consumption']
            self.assertEqual((c['mode'], c['transcript_consumed'], c['instruction_consumed']), (mode, text, ins))
        self.assertFalse(inspect_reference(self.source, transcript='ignored')['backend_consumption']['transcript_consumed'])
        consumption = inspect_reference(self.source, instruction='平静')['backend_consumption']
        self.assertTrue(consumption['instruction_consumed'])
        self.assertIn('emotional text', consumption['explanation'])
        self.assertIsNone(inspect_reference(self.source, backend='other')['backend_consumption']['audio_consumed'])

    def test_consent_before_read_or_directory_creation(self):
        for consent in [False, None, 1, 'yes']:
            with self.assertRaisesRegex(ValueError, 'confirmation'):
                prepare_confirmed_reference(self.root / 'missing.wav', confirmed=consent,
                                            output_dir=self.root / 'not_created')
        self.assertFalse((self.root / 'not_created').exists())

    def test_invalid_ranges_empty_nonfinite_and_silent(self):
        self.write(np.full(1000, .2))
        for start, end in [(-1, None), (0, 0), (1, .5), (0, 1.001),
                           (float('nan'), None), (0, float('inf')), (.0001, .0002), (True, None)]:
            with self.assertRaises(ValueError):
                prepare_confirmed_reference(self.source, confirmed=True, start_seconds=start,
                                            end_seconds=end, output_dir=self.root / 'out')
        for values in [[], [float('nan')], [float('inf')]]:
            self.write(values)
            with self.assertRaises(ValueError):
                inspect_reference(self.source)
        self.write(np.zeros(100))
        self.assertEqual(inspect_reference(self.source)['silence_frame_proportion'], 1)
        with self.assertRaisesRegex(ValueError, 'silent'):
            prepare_confirmed_reference(self.source, confirmed=True, output_dir=self.root)
        self.write(np.r_[np.zeros(100), np.full(100, .2)])
        with self.assertRaisesRegex(ValueError, 'silent'):
            prepare_confirmed_reference(self.source, confirmed=True, end_seconds=.1, output_dir=self.root)

    def test_unique_float_output_exact_selection_and_original_hash(self):
        self.write(np.column_stack([np.arange(1000) / 2000, np.arange(1000) / 4000]))
        original = self.source.read_bytes()
        digest = hashlib.sha256(original).hexdigest()
        kwargs = dict(confirmed=True, start_seconds=.1, end_seconds=.4, mono=True, output_dir=self.root)
        r = prepare_confirmed_reference(self.source, **kwargs)
        s = prepare_confirmed_reference(self.source, **kwargs)
        self.assertNotEqual(r['output_path'], s['output_path'])
        self.assertNotEqual(Path(r['output_path']), self.source)
        self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual(r['original_sha256'], digest)
        info = sf.info(r['output_path'])
        self.assertEqual((info.subtype, info.frames, info.channels, info.samplerate), ('FLOAT', 300, 1, 1000))
        x = sf.read(self.source, always_2d=True)[0]
        np.testing.assert_array_equal(sf.read(r['output_path'])[0], x[100:400].mean(axis=1).astype(np.float32))
        self.assertFalse(r['processing_manifest']['enhancement'])
        self.assertEqual(r['processing_manifest']['start_frame'], 100)
        json.dumps(r, allow_nan=False)

    def test_transcript_invalidated_until_explicitly_confirmed_for_output(self):
        self.write(np.full(1000, .2))
        kw = dict(confirmed=True, start_seconds=.1, output_dir=self.root, transcript='new excerpt')
        r = prepare_confirmed_reference(self.source, **kw)
        self.assertEqual(r['transcript'], '')
        self.assertEqual(r['transcript_status'], 'invalidated_by_trim')
        r = prepare_confirmed_reference(self.source, transcript_confirmed=True, **kw)
        self.assertEqual(r['transcript'], 'new excerpt')
        self.assertEqual(r['transcript_status'], 'user_confirmed')
        self.assertIn('unverified', r['unverified']['transcript_correspondence'])
        r = prepare_confirmed_reference(self.source, confirmed=True, transcript='original', output_dir=self.root)
        self.assertEqual(r['transcript'], 'original')
        self.assertFalse(r['transcript_confirmed'])

    def test_default_output_directory_and_path_errors(self):
        self.write([.2] * 100)
        with patch('timbreforge.config.get_config', return_value=SimpleNamespace(outputs_dir=self.root / 'outputs')):
            r = prepare_confirmed_reference(str(self.source), confirmed=True)
        self.assertEqual(Path(r['output_path']).parent, self.root / 'outputs' / 'reference_assistant')
        with self.assertRaises(FileNotFoundError):
            inspect_reference(self.root / 'missing.wav')
        with self.assertRaises(OSError):
            prepare_confirmed_reference(self.source, confirmed=True, output_dir=self.source)


if __name__ == '__main__':
    unittest.main()
