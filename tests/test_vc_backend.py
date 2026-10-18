"""Experimental VC backend tests: real local subprocess, synthetic audio only."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

import numpy as np
import soundfile as sf

from timbreforge import vc_backend as vc
from timbreforge.characters import CharacterStore, audio_path


FAKE_SCRIPT = r'''
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf


def argument(name):
    return sys.argv[sys.argv.index(name) + 1]


request = json.loads(Path(argument('--request')).read_text(encoding='utf-8'))
result_path = Path(argument('--result'))
source = Path(request['source_audio'])
target = Path(request['target_audio'])
output = Path(request['output'])
mode = '__MODE__'

if mode == 'mutate_source':
    data, sample_rate = sf.read(source, dtype='float32')
    sf.write(source, data * 0.5, sample_rate, subtype='FLOAT')
elif mode == 'mutate_target':
    data, sample_rate = sf.read(target, dtype='float32')
    sf.write(target, data * 0.5, sample_rate, subtype='FLOAT')

if mode == 'pcm':
    data, sample_rate = sf.read(source, dtype='float32')
    sf.write(output, data, sample_rate, subtype='PCM_16')
elif mode == 'nonfinite':
    sf.write(output, np.full(1600, np.nan, dtype=np.float32), 16000, subtype='FLOAT')
else:
    shutil.copyfile(source, output)

if mode == 'partial_then_wait':
    (output.parent / 'converter.started').write_text('started', encoding='utf-8')
    time.sleep(30)

info = sf.info(output)
result = {
    'file': str(output),
    'sample_rate': info.samplerate,
    'duration': info.duration,
    'model_dir': request['model_dir'],
    'preserve_prosody': request['preserve_prosody'],
    'source_sha256': request['source_sha256'] if mode != 'wrong_hash' else '0' * 64,
    'target_sha256': request['target_sha256'],
}
result_path.write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
'''


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _has_chinese(text):
    return any('\u4e00' <= character <= '\u9fff' for character in str(text))


class VoiceConversionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

        self.store = CharacterStore(self.root / 'characters')
        self.target_source = self.root / 'target.wav'
        sf.write(self.target_source, self._tone(24000, 0.3, 220), 24000,
                 subtype='FLOAT')
        self.card = self.store.create(
            '目标角色', self.target_source, reference_text='目标角色参考音频。',
            backend='qwen3')
        self.target = Path(audio_path(self.card))

        self.source = self.root / 'source.wav'
        sf.write(self.source, self._tone(16000, 0.25, 330), 16000,
                 subtype='FLOAT')

        self.model_dir = self.root / 'vc-model'
        self.model_dir.mkdir()
        self.script = self.root / 'fake_vc.py'
        self._write_script('success')
        self.outdir = self.root / 'vc-output'
        self.backend = self._make_backend()

        vc.VC_REGISTRY.clear()
        self.addCleanup(vc.VC_REGISTRY.clear)

    @staticmethod
    def _tone(sample_rate, seconds, frequency):
        time_axis = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
        return (0.2 * np.sin(2 * np.pi * frequency * time_axis)).astype(np.float32)

    def _write_script(self, mode):
        self.script.write_text(FAKE_SCRIPT.replace('__MODE__', mode),
                               encoding='utf-8')

    def _make_backend(self, **overrides):
        values = dict(
            python=sys.executable,
            script=self.script,
            model_dir=self.model_dir,
            name='fake-vc',
            model_name='fake-vc-model',
            model_revision='test-revision',
            license_note='仅用于测试；使用前请自行确认模型许可证与声音授权。',
            estimated_vram_gb=2.0,
        )
        values.update(overrides)
        return vc.ExternalVoiceConversionBackend(**values)

    def _convert(self, **kwargs):
        return self.backend.convert(self.source, self.card['id'],
                                    store=self.store, output_dir=self.outdir,
                                    **kwargs)

    # -------------------------------------------------------------- success
    def test_success_returns_honest_metadata_and_keeps_inputs_untouched(self):
        source_before = _sha256(self.source)
        target_before = _sha256(self.target)
        metadata = self._convert(preserve_prosody=False)

        self.assertEqual(metadata['schema_version'], 1)
        self.assertEqual(metadata['backend'], 'fake-vc')
        self.assertIs(metadata['experimental'], True)
        self.assertEqual(metadata['target_role_id'], self.card['id'])
        self.assertEqual(metadata['source_audio'], str(self.source.resolve()))
        self.assertEqual(metadata['source_sha256'], source_before)
        self.assertEqual(metadata['target_audio'], str(self.target.resolve()))
        self.assertEqual(metadata['target_sha256'], target_before)
        self.assertIs(metadata['preserve_prosody'], False)
        self.assertEqual(metadata['model_dir'], str(self.model_dir))
        self.assertEqual(metadata['model_name'], 'fake-vc-model')
        self.assertEqual(metadata['license_note'], self.backend.spec.license_note)
        self.assertEqual(metadata['sample_rate'], 16000)
        self.assertAlmostEqual(metadata['duration'], 0.25, places=3)

        output = Path(metadata['output_audio'])
        self.assertTrue(output.is_file())
        self.assertEqual(metadata['output_sha256'], _sha256(output))
        self.assertEqual(sf.info(str(output)).subtype, 'FLOAT')
        audio, sample_rate = sf.read(str(output), dtype='float32', always_2d=True)
        self.assertEqual(sample_rate, 16000)
        self.assertTrue(np.isfinite(audio).all())
        self.assertTrue(np.any(audio))

        warnings = ' '.join(metadata['warnings'])
        for word in ('情绪', '语速', '咬字', '试听', '不是完美的音色替换'):
            self.assertIn(word, warnings)
        self.assertIn('preserve_prosody', warnings)

        self.assertEqual(_sha256(self.source), source_before)
        self.assertEqual(_sha256(self.target), target_before)
        self.assertEqual(list(self.outdir.glob('*.wav')), [output])

    def test_request_json_contains_exact_local_provenance(self):
        source_before = _sha256(self.source)
        target_before = _sha256(self.target)
        metadata = self._convert()
        request = json.loads(
            Path(metadata['request_file']).read_text(encoding='utf-8'))
        self.assertEqual(request['mode'], 'voice_conversion')
        self.assertEqual(request['source_audio'], str(self.source.resolve()))
        self.assertEqual(request['target_audio'], str(self.target.resolve()))
        self.assertEqual(request['target_role_id'], self.card['id'])
        self.assertIs(request['preserve_prosody'], True)
        self.assertEqual(request['output'], metadata['output_audio'])
        self.assertEqual(request['model_dir'], str(self.model_dir))
        self.assertEqual(request['source_sha256'], source_before)
        self.assertEqual(request['target_sha256'], target_before)

    def test_preserve_prosody_is_passed_through(self):
        self._write_script('success')
        metadata = self._convert(preserve_prosody=False)
        request = json.loads(
            Path(metadata['request_file']).read_text(encoding='utf-8'))
        self.assertIs(request['preserve_prosody'], False)
        self.assertIs(metadata['preserve_prosody'], False)
        self.assertIs(metadata['raw']['preserve_prosody'], False)
        self.assertIn('重点试听', ' '.join(metadata['warnings']))

    # ---------------------------------------------------------- availability
    def test_unavailable_backend_reports_reason_and_refuses_to_convert(self):
        cases = {
            'python': dict(python=self.root / 'missing-python.exe'),
            'script': dict(script=self.root / 'missing-script.py'),
            'model': dict(model_dir=self.root / 'missing-model'),
        }
        for label, overrides in cases.items():
            with self.subTest(component=label):
                backend = self._make_backend(**overrides)
                ok, reason = backend.available()
                self.assertFalse(ok)
                self.assertTrue(_has_chinese(reason))
                self.assertIn('不可用', reason)
                self.assertIn('不会自动下载', reason)
                with self.assertRaises(vc.VoiceConversionUnavailable) as caught:
                    backend.convert(self.source, self.card['id'],
                                    store=self.store, output_dir=self.outdir)
                self.assertEqual(str(caught.exception), reason)

    def test_missing_source_is_rejected_before_any_process(self):
        with self.assertRaises(vc.VoiceConversionError) as caught:
            self.backend.convert(self.root / 'missing.wav', self.card['id'],
                                 store=self.store, output_dir=self.outdir)
        self.assertIn('源音频不存在', str(caught.exception))
        self.assertTrue(_has_chinese(caught.exception))
        self.assertFalse(self.outdir.exists())

    def test_unknown_or_missing_target_card_is_rejected(self):
        for role_id in ('not-a-valid-role-id', '0' * 32):
            with self.subTest(role_id=role_id):
                with self.assertRaises(vc.VoiceConversionError) as caught:
                    self.backend.convert(self.source, role_id, store=self.store,
                                         output_dir=self.outdir)
                self.assertIn('目标角色卡', str(caught.exception))
                self.assertTrue(_has_chinese(caught.exception))

    def test_invalid_arguments_are_rejected(self):
        for kwargs in ({'preserve_prosody': 'yes'}, {'preserve_prosody': 1},
                       {'cancel': object()}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self._convert(**kwargs)
        event = threading.Event()
        event.set()
        with self.assertRaises(vc.VoiceConversionError) as caught:
            self._convert(cancel=event)
        self.assertIn('取消', str(caught.exception))
        self.assertFalse(self.outdir.exists())

    # ------------------------------------------------------------- rejection
    def test_tampered_results_are_rejected_and_partial_output_removed(self):
        for mode in ('wrong_hash', 'pcm', 'nonfinite'):
            with self.subTest(mode=mode):
                self._write_script(mode)
                with self.assertRaises(vc.VoiceConversionError) as caught:
                    self._convert()
                self.assertIn('日志', str(caught.exception))
                self.assertTrue(_has_chinese(caught.exception))
                self.assertEqual(list(self.outdir.glob('*.wav')), [])

    def test_source_mutation_during_conversion_is_rejected(self):
        self._write_script('mutate_source')
        source_before = _sha256(self.source)
        with self.assertRaises(vc.VoiceConversionError) as caught:
            self._convert()
        message = str(caught.exception)
        self.assertIn('源音频', message)
        self.assertIn('变化', message)
        self.assertIn('日志', message)
        self.assertNotEqual(_sha256(self.source), source_before)
        self.assertEqual(list(self.outdir.glob('*.wav')), [])

    def test_target_audio_mutation_during_conversion_is_rejected(self):
        self._write_script('mutate_target')
        target_before = _sha256(self.target)
        with self.assertRaises(vc.VoiceConversionError) as caught:
            self._convert()
        message = str(caught.exception)
        self.assertIn('目标角色音频', message)
        self.assertIn('变化', message)
        self.assertNotEqual(_sha256(self.target), target_before)
        self.assertEqual(list(self.outdir.glob('*.wav')), [])

    def test_silent_or_non_finite_source_is_rejected(self):
        silent = self.root / 'silent.wav'
        sf.write(silent, np.zeros(1600, dtype=np.float32), 16000, subtype='FLOAT')
        with self.assertRaises(vc.VoiceConversionError) as caught:
            self.backend.convert(silent, self.card['id'], store=self.store,
                                 output_dir=self.outdir)
        self.assertIn('静音', str(caught.exception))

        non_finite = self.root / 'non-finite.wav'
        sf.write(non_finite, np.full(1600, np.nan, dtype=np.float32), 16000,
                 subtype='FLOAT')
        with self.assertRaises(vc.VoiceConversionError) as caught:
            self.backend.convert(non_finite, self.card['id'], store=self.store,
                                 output_dir=self.outdir)
        self.assertIn('非有限', str(caught.exception))

    def test_cancel_kills_the_external_process_and_removes_partial_output(self):
        self._write_script('partial_then_wait')
        event = threading.Event()
        marker = self.outdir / 'converter.started'

        def wait_for_marker():
            deadline = time.monotonic() + 8.0
            while time.monotonic() < deadline:
                if marker.exists():
                    event.set()
                    return
                time.sleep(0.02)

        thread = threading.Thread(target=wait_for_marker, daemon=True)
        fallback = threading.Timer(10.0, event.set)
        fallback.daemon = True
        thread.start()
        fallback.start()
        try:
            with self.assertRaises(vc.VoiceConversionError) as caught:
                self._convert(cancel=event)
        finally:
            event.set()
            thread.join(timeout=5)
            fallback.cancel()
        self.assertIn('取消', str(caught.exception))
        self.assertIn('日志', str(caught.exception))
        self.assertEqual(list(self.outdir.glob('*.wav')), [])

    # ------------------------------------------------------------- status
    def test_status_markdown_is_honest_chinese_guidance(self):
        text = vc.vc_status_markdown()
        for word in ('实验', '许可证', '显存', '第一阶段', '试听', '情绪', '语速'):
            self.assertIn(word, text)
        self.assertIn('未登记', text)

    def test_registry_supports_one_unique_local_backend(self):
        vc.register_vc_backend(self.backend)
        self.assertEqual([backend.spec.name for backend in vc.default_vc_backends()],
                         ['fake-vc'])
        with self.assertRaises(vc.VoiceConversionError):
            vc.register_vc_backend(self.backend)
        text = vc.vc_status_markdown()
        self.assertIn('fake-vc', text)
        self.assertIn('可用', text)


if __name__ == '__main__':
    unittest.main()

