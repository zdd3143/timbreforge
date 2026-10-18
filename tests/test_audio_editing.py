"""Non-destructive audio conditioning: chain validation, DSP sanity, hashing."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from timbreforge import audio_edit_pipeline as pipeline
from timbreforge import audio_presets
from timbreforge.audio_edit_schema import (ChainError, chain_digest, describe_operations,
                                           make_edit_document, summarize_chain,
                                           validate_document, validate_operations)
from timbreforge.audio_preview import build_preview, cleanup_previews, preview_path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def voice_like(seconds=2.5, sample_rate=24000, lead=0.4, tail=0.5, noise=0.0008, seed=7):
    """Deterministic voiced signal with leading/trailing digital silence."""
    generator = np.random.default_rng(seed)
    count = int(seconds * sample_rate)
    time_axis = np.arange(count) / sample_rate
    pitch = 140 + 8 * np.sin(2 * np.pi * 0.7 * time_axis)
    phase = 2 * np.pi * np.cumsum(pitch) / sample_rate
    signal = (0.28 * np.sin(phase) + 0.12 * np.sin(2 * phase)
              + 0.05 * np.sin(3 * phase) + 0.02 * np.sin(5 * phase))
    envelope = 0.55 + 0.45 * np.sin(2 * np.pi * 2.2 * time_axis + 0.4) ** 2
    signal = signal * envelope + generator.normal(0, noise, count)
    pad_lead = np.zeros(int(lead * sample_rate))
    pad_tail = np.zeros(int(tail * sample_rate))
    audio = np.concatenate([pad_lead, signal, pad_tail])[:, None]
    return audio, sample_rate


def write_audio(path, samples, sample_rate, subtype='FLOAT'):
    sf.write(str(path), samples, sample_rate, format='WAV', subtype=subtype)
    return Path(path)


class ChainSchemaTests(unittest.TestCase):
    def test_presets_validate_and_describe(self):
        self.assertEqual(audio_presets.preset_names(),
                         ('clean_voice', 'warm_voice', 'bright_voice', 'low_noise'))
        for name in audio_presets.preset_names():
            preset = audio_presets.get_preset(name)
            self.assertTrue(preset['operations'])
            self.assertEqual(validate_operations(preset['operations']), preset['operations'])
            self.assertIn('→', preset['summary'])

    def test_unknown_or_out_of_range_operations_are_rejected(self):
        with self.assertRaises(ChainError):
            validate_operations([{'name': 'reverb'}])
        with self.assertRaises(ChainError):
            validate_operations([{'name': 'normalize', 'target_lufs': -80}])
        with self.assertRaises(ChainError):
            validate_operations([{'name': 'trim_silence', 'threshold_db': -42, 'bogus': 1}])
        with self.assertRaises(ChainError):
            validate_operations([{'name': 'de_click', 'window': 4}])

    def test_single_instance_operations_may_not_repeat(self):
        with self.assertRaises(ChainError):
            validate_operations([{'name': 'normalize'}, {'name': 'normalize'}])
        # eq may appear twice, e.g. a correction before and after compression.
        operations = validate_operations([
            {'name': 'eq', 'bands': [{'type': 'highpass', 'freq': 80, 'gain_db': 0, 'q': 0.7}]},
            {'name': 'eq', 'bands': [{'type': 'highshelf', 'freq': 7000, 'gain_db': 1, 'q': 0.7}]},
        ])
        self.assertEqual(len(operations), 2)

    def test_digest_is_order_sensitive_and_ignores_default_notation(self):
        first = [{'name': 'trim_silence', 'threshold_db': -42, 'keep_leading_ms': 50,
                  'keep_trailing_ms': 80}]
        explicit = [{'name': 'trim_silence', 'threshold_db': -42.0, 'keep_leading_ms': 50.0,
                     'keep_trailing_ms': 80.0}]
        self.assertEqual(chain_digest(first), chain_digest(explicit))
        other = [{'name': 'trim_silence', 'threshold_db': -43, 'keep_leading_ms': 50,
                  'keep_trailing_ms': 80}]
        self.assertNotEqual(chain_digest(first), chain_digest(other))

    def test_document_requires_hashes_and_round_trips(self):
        operations = audio_presets.preset_operations('clean_voice')
        document = make_edit_document(
            source_audio='reference.wav', source_hash='a' * 64, operations=operations,
            output_audio='audio/canonical.wav', output_hash='b' * 64,
            preset='clean_voice', sample_rate=24000, channels=1,
            duration_in=2.0, duration_out=1.8)
        self.assertEqual(validate_document(document), document)
        damaged = dict(document, output_hash='nope')
        with self.assertRaises(ChainError):
            validate_document(damaged)
        summary = summarize_chain([document])
        self.assertEqual(summary['documents'], 1)
        self.assertEqual(summary['operations'], len(operations))

    def test_describe_operations_never_hides_a_step(self):
        labels = describe_operations(audio_presets.preset_operations('low_noise'))
        self.assertEqual(len(labels), 5)
        self.assertTrue(any('降噪' in label for label in labels))


class LoudnessTests(unittest.TestCase):
    def test_sine_loudness_matches_bs1770_convention(self):
        sample_rate = 48000
        time_axis = np.arange(sample_rate) / sample_rate
        tone = (0.1 * np.sin(2 * np.pi * 997 * time_axis))[:, None]
        measured = pipeline.measure_loudness(tone, sample_rate)
        # A 997 Hz sine at -20 dBFS reads about -23.7 LUFS once K-weighting and
        # the -0.691 offset are applied; allow the documented implementation
        # tolerance instead of pretending to be a certified meter.
        self.assertAlmostEqual(measured, -23.7, delta=0.8)

    def test_silence_reports_negative_infinity(self):
        self.assertEqual(pipeline.measure_loudness(np.zeros((4800, 1)), 24000), -math.inf)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.audio, self.rate = voice_like()
        self.source = write_audio(self.root / 'reference.wav', self.audio, self.rate)

    def tearDown(self):
        self.temp.cleanup()

    def test_trim_silence_removes_padding_but_keeps_speech(self):
        output, notes = pipeline.apply_operations(
            self.audio, self.rate, [{'name': 'trim_silence', 'threshold_db': -42,
                                     'keep_leading_ms': 50, 'keep_trailing_ms': 80}])
        self.assertLess(len(output), len(self.audio))
        self.assertGreater(len(output), int(1.5 * self.rate))
        self.assertEqual(output.shape[1], 1)
        self.assertTrue(any('去静音' in note for note in notes))

    def test_normalize_reaches_target_and_respects_ceiling(self):
        output, _notes = pipeline.apply_operations(
            self.audio, self.rate, [{'name': 'normalize', 'target_lufs': -16,
                                     'ceiling_dbfs': -1}])
        self.assertAlmostEqual(pipeline.measure_loudness(output, self.rate), -16, delta=1.0)
        self.assertLessEqual(pipeline.measure_peak_dbfs(output), -0.99)

    def test_normalize_refuses_absolute_silence(self):
        with self.assertRaises(ChainError):
            pipeline.apply_operations(np.zeros((2400, 1)), 24000,
                                      [{'name': 'normalize'}])

    def test_de_click_reduces_an_impulse(self):
        clicked = self.audio.copy()
        clicked[len(clicked) // 2] = 0.95
        output, _ = pipeline.apply_operations(
            clicked, self.rate, [{'name': 'de_click', 'threshold': 5.5, 'window': 5}])
        self.assertLess(float(np.max(np.abs(output))), 0.8)
        self.assertTrue(np.isfinite(output).all())

    def test_denoise_reduces_noise_floor_and_preserves_length(self):
        generator = np.random.default_rng(3)
        noise = generator.normal(0, 0.01, (self.rate * 2, 1))
        output, _ = pipeline.apply_operations(
            noise, self.rate, [{'name': 'denoise', 'strength': 0.8, 'window_ms': 32,
                                'noise_seconds': 0.5}])
        self.assertEqual(output.shape, noise.shape)
        self.assertLess(float(np.sqrt(np.mean(output ** 2))),
                        float(np.sqrt(np.mean(noise ** 2))))

    def test_eq_boost_is_applied_at_the_requested_band(self):
        sample_rate = 48000
        time_axis = np.arange(sample_rate // 2) / sample_rate
        tone = (0.05 * np.sin(2 * np.pi * 8000 * time_axis))[:, None]
        output, _ = pipeline.apply_operations(
            tone, sample_rate,
            [{'name': 'eq', 'bands': [
                # A shelf reaches roughly half of its gain at its centre
                # frequency, so the band is placed well below the probe tone.
                {'type': 'highshelf', 'freq': 2000, 'gain_db': 6.0, 'q': 0.707}]}])
        gain_db = 20 * math.log10(np.sqrt(np.mean(output ** 2)) /
                                  np.sqrt(np.mean(tone ** 2)))
        self.assertAlmostEqual(gain_db, 6.0, delta=1.2)

    def test_compression_reduces_dynamic_range(self):
        sample_rate = 24000
        loud = 0.6 * np.sin(2 * np.pi * 220 * np.arange(sample_rate) / sample_rate)
        quiet = 0.02 * np.sin(2 * np.pi * 220 * np.arange(sample_rate) / sample_rate)
        signal = np.concatenate([loud, quiet])[:, None]
        output, _ = pipeline.apply_operations(
            signal, sample_rate, [{'name': 'compress', 'threshold_db': -20, 'ratio': 4.0,
                                   'attack_ms': 5, 'release_ms': 60, 'makeup_db': 0}])
        before = 20 * math.log10(float(np.max(np.abs(signal[:sample_rate]))) /
                                float(np.max(np.abs(signal[sample_rate:]))))
        after = 20 * math.log10(float(np.max(np.abs(output[:sample_rate]))) /
                               float(np.max(np.abs(output[sample_rate:]))))
        self.assertLess(after, before)

    def test_render_chain_never_touches_the_source_and_records_hashes(self):
        original_hash = sha(self.source)
        original_mtime = self.source.stat().st_mtime_ns
        result = pipeline.render_chain(
            self.source, audio_presets.preset_operations('clean_voice'),
            dest_dir=self.root / 'audio', output_name='canonical.wav',
            source_name='reference.wav', preset='clean_voice')
        self.assertEqual(sha(self.source), original_hash)
        self.assertEqual(self.source.stat().st_mtime_ns, original_mtime)
        document = validate_document(result['document'])
        self.assertEqual(document['source_hash'], original_hash)
        self.assertEqual(document['output_hash'], sha(result['output_path']))
        self.assertEqual(document['output_audio'], 'canonical.wav')
        self.assertEqual(document['preset'], 'clean_voice')
        self.assertTrue(Path(result['output_path']).is_file())
        self.assertFalse(np.isnan(result['metrics']['lufs_out']))

    def test_render_chain_refuses_to_overwrite(self):
        destination = self.root / 'audio'
        destination.mkdir()
        (destination / 'canonical.wav').write_bytes(b'do not replace me')
        with self.assertRaises(FileExistsError):
            pipeline.render_chain(
                self.source, [{'name': 'normalize'}], dest_dir=destination,
                output_name='canonical.wav', source_name='reference.wav')
        self.assertEqual((destination / 'canonical.wav').read_bytes(), b'do not replace me')

    def test_trim_refuses_audio_that_is_entirely_below_threshold(self):
        quiet = np.zeros((self.rate, 1))
        quiet[:] = 1e-6
        with self.assertRaises(ChainError):
            pipeline.apply_operations(quiet, self.rate, [{'name': 'trim_silence'}])


class PreviewTests(unittest.TestCase):
    def test_preview_is_compressed_and_decodable(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            audio, rate = voice_like()
            source = write_audio(root / 'reference.wav', audio, rate)
            preview = build_preview(source, root / 'audio' / 'previews')
            self.assertIn(preview['format'], ('OGG', 'FLAC', 'WAV'))
            self.assertEqual(preview['source_sha256'], sha(source))
            info = sf.info(preview['path'])
            self.assertAlmostEqual(info.duration, preview['duration'], places=3)
            self.assertLess(preview['bytes'], source.stat().st_size * 2)
            self.assertIsNone(preview_path(root / 'nope'))
            self.assertEqual(preview_path(root), preview['path'])

    def test_cleanup_keeps_requested_number(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            for index in range(3):
                (folder / f'preview-{index}.ogg').write_bytes(b'x')
            self.assertEqual(cleanup_previews(folder, keep=1), 2)
            self.assertEqual(len(list(folder.iterdir())), 1)

    def test_preview_rejects_oversized_window(self):
        with tempfile.TemporaryDirectory() as td:
            audio, rate = voice_like(seconds=1)
            source = write_audio(Path(td) / 'a.wav', audio, rate)
            with self.assertRaises(ValueError):
                build_preview(source, Path(td) / 'p', max_seconds=600)


if __name__ == '__main__':
    unittest.main()

