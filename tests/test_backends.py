"""Unified backend adapter tests: fake backends only, no model or GPU."""
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from timbreforge import backend_qwen
from timbreforge.backends import (
    BackendError,
    BackendRegistry,
    BackendRequest,
    BackendResult,
    BackendSpec,
    BackendUnavailable,
    CapabilityError,
    Qwen3Backend,
    TTSBackend,
    default_registry,
)


def _has_chinese(text):
    return any('\u4e00' <= character <= '\u9fff' for character in str(text))


def _spec(name='fake', kind='tts', capabilities=('design', 'clone'),
          model_name='fake-model', model_revision='1.0',
          description='测试用假后端；不加载任何模型。'):
    return BackendSpec(name=name, kind=kind, capabilities=capabilities,
                       model_name=model_name, model_revision=model_revision,
                       description=description)


class FakeBackend(TTSBackend):
    """A deterministic adapter used only for registry/capability tests."""

    def __init__(self, spec, *, available=True, reason='测试后端可用。'):
        self.spec = spec
        self._status = (available, reason)
        self.synthesize_calls = []
        self.prompt_calls = []

    def available(self):
        return self._status

    def synthesize(self, request):
        self.synthesize_calls.append(request)
        return BackendResult(
            audio_path='fake.wav', mode=request.mode, seed=0, params={},
            model_dir='fake-model-dir', model_config_sha256='a' * 64,
            clone_mode='none', clone_prompt_sha256=None, duration=1.0,
            sample_rate=24000, raw={})

    def create_clone_prompt(self, *args, **kwargs):
        self.prompt_calls.append((args, kwargs))
        return {'prompt_file': 'fake.pt'}


class BackendSpecTests(unittest.TestCase):
    def test_valid_spec_normalizes_capability_list(self):
        spec = _spec(capabilities=['design', 'clone', 'clone_prompt'])
        self.assertEqual(spec.capabilities, ('design', 'clone', 'clone_prompt'))
        self.assertTrue(_has_chinese(spec.description))

    def test_bad_specs_are_rejected_with_actionable_chinese_errors(self):
        bad = (
            {'name': 'Bad Name'},
            {'name': ''},
            {'kind': 'image'},
            {'capabilities': ('design', 'not-a-capability')},
            {'capabilities': ()},
            {'capabilities': ('design', 'design')},
            {'capabilities': ('clone_prompt',)},
            {'model_name': ''},
            {'model_revision': ' '},
            {'description': ''},
        )
        for overrides in bad:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError) as caught:
                    _spec(**overrides)
                self.assertTrue(_has_chinese(caught.exception))


class RegistryTests(unittest.TestCase):
    def test_register_get_list_and_names(self):
        registry = BackendRegistry()
        first = FakeBackend(_spec('fake-one', capabilities=('design',)))
        second = FakeBackend(_spec('fake-two', capabilities=('clone',)))
        registry.register(first)
        registry.register(second)
        self.assertIs(registry.get('fake-one'), first)
        self.assertEqual(registry.names(), ['fake-one', 'fake-two'])
        self.assertEqual([spec.name for spec in registry.list()], ['fake-one', 'fake-two'])
        self.assertEqual([spec.name for spec in registry.list(kind='tts')],
                         ['fake-one', 'fake-two'])
        self.assertEqual(registry.list(kind='vc'), [])

    def test_duplicate_name_is_rejected(self):
        registry = BackendRegistry()
        registry.register(FakeBackend(_spec('fake')))
        with self.assertRaises(BackendError) as caught:
            registry.register(FakeBackend(_spec('fake')))
        self.assertIn('fake', str(caught.exception))
        self.assertTrue(_has_chinese(caught.exception))

    def test_missing_name_raises_backend_unavailable(self):
        registry = BackendRegistry()
        registry.register(FakeBackend(_spec('fake')))
        with self.assertRaises(BackendUnavailable) as caught:
            registry.get('missing')
        self.assertIn('missing', str(caught.exception))
        self.assertIn('fake', str(caught.exception))
        self.assertTrue(_has_chinese(caught.exception))

    def test_register_rejects_non_adapter_and_bad_spec(self):
        registry = BackendRegistry()
        with self.assertRaises(BackendError):
            registry.register(object())
        broken = FakeBackend(_spec('broken'))
        broken.spec = types.SimpleNamespace(name='broken')
        with self.assertRaises(BackendError) as caught:
            registry.register(broken)
        self.assertIn('BackendSpec', str(caught.exception))

    def test_require_checks_availability_and_capability(self):
        registry = BackendRegistry()
        backend = FakeBackend(_spec('fake', capabilities=('design',)))
        registry.register(backend)
        self.assertIs(registry.require('fake'), backend)
        self.assertIs(registry.require('fake', 'design'), backend)
        with self.assertRaises(CapabilityError) as caught:
            registry.require('fake', 'clone')
        self.assertIn('clone', str(caught.exception))
        self.assertTrue(_has_chinese(caught.exception))
        with self.assertRaises(CapabilityError):
            registry.require('fake', 'bogus')

    def test_require_raises_when_unavailable(self):
        registry = BackendRegistry()
        registry.register(FakeBackend(
            _spec('offline', capabilities=('clone',)),
            available=False, reason='本地模型文件缺失；不会自动下载。'))
        with self.assertRaises(BackendUnavailable) as caught:
            registry.require('offline')
        self.assertIn('本地模型文件缺失', str(caught.exception))
        self.assertIn('不会自动下载', str(caught.exception))

    def test_bad_available_contract_is_rejected(self):
        class BadStatus(FakeBackend):
            def available(self):
                return 'yes'

        registry = BackendRegistry()
        registry.register(BadStatus(_spec('bad-status')))
        with self.assertRaises(BackendError) as caught:
            registry.require('bad-status')
        self.assertIn('available', str(caught.exception))


class QwenBackendTestBase(unittest.TestCase):
    """Temporary fake Qwen files; no model is imported or loaded."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.python = self.root / 'python.exe'
        self.python.write_bytes(b'fake-python')
        self.model = self.root / 'base-model'
        self.model.mkdir()
        (self.model / 'config.json').write_text('{}', encoding='utf-8')
        (self.model / 'model.safetensors').write_bytes(b'fake-weights')
        self.worker = self.root / 'qwen_worker.py'
        self.worker.write_text('# fake worker path\n', encoding='utf-8')
        self.outputs = self.root / 'outputs'
        self.config = types.SimpleNamespace(
            qwen_python=self.python, qwen_base_model=self.model,
            outputs_dir=self.outputs)
        self.patch_config = patch('timbreforge.backends.get_config',
                                  return_value=self.config)
        self.patch_config.start()
        self.addCleanup(self.patch_config.stop)
        self.patch_worker = patch('timbreforge.backends.QWEN_WORKER', self.worker)
        self.patch_worker.start()
        self.addCleanup(self.patch_worker.stop)

    def raw_result(self, mode='voice_design', *, seed=3, clone_mode='none',
                   prompt_hash=None, **overrides):
        raw = {
            'file': str(self.root / 'generated.wav'),
            'mode': mode,
            'seed': seed,
            'params': {'seed': seed, 'language': 'Chinese'},
            'model_dir': str(self.model),
            'model_config_sha256': 'a' * 64,
            'clone_mode': clone_mode,
            'clone_prompt_sha256': prompt_hash,
            'duration': 0.25,
            'sample_rate': 24000,
            'worker_note': 'kept-in-raw',
        }
        raw.update(overrides)
        return raw


class QwenAvailabilityTests(QwenBackendTestBase):
    def test_available_true_without_loading_a_model(self):
        ok, reason = Qwen3Backend().available()
        self.assertTrue(ok)
        self.assertTrue(_has_chinese(reason))

    def test_missing_python_returns_false_and_chinese_reason(self):
        self.python.unlink()
        ok, reason = Qwen3Backend().available()
        self.assertFalse(ok)
        self.assertIn('Python', reason)
        self.assertIn('不会自动下载', reason)
        self.assertTrue(_has_chinese(reason))

    def test_missing_config_returns_false_and_chinese_reason(self):
        (self.model / 'config.json').unlink()
        ok, reason = Qwen3Backend().available()
        self.assertFalse(ok)
        self.assertIn('config.json', reason)
        self.assertTrue(_has_chinese(reason))

    def test_missing_safetensors_returns_false_and_chinese_reason(self):
        (self.model / 'model.safetensors').unlink()
        ok, reason = Qwen3Backend().available()
        self.assertFalse(ok)
        self.assertIn('safetensors', reason)
        self.assertIn('不会自动下载', reason)
        self.assertTrue(_has_chinese(reason))

    def test_missing_worker_returns_false_and_chinese_reason(self):
        with patch('timbreforge.backends.QWEN_WORKER', self.root / 'missing-worker.py'):
            ok, reason = Qwen3Backend().available()
        self.assertFalse(ok)
        self.assertIn('worker', reason)
        self.assertTrue(_has_chinese(reason))

    def test_unreadable_config_returns_false_instead_of_raising(self):
        with patch('timbreforge.backends.get_config', side_effect=ValueError('坏配置')):
            ok, reason = Qwen3Backend().available()
        self.assertFalse(ok)
        self.assertIn('坏配置', reason)
        self.assertIn('不会自动下载', reason)


class QwenSynthesisTests(QwenBackendTestBase):
    def setUp(self):
        super().setUp()
        self.params = backend_qwen.QwenParams(seed=3)

    def test_design_delegates_and_maps_every_result_field(self):
        raw = self.raw_result()
        request = BackendRequest(
            mode='design', text='你好，世界。', params=self.params,
            instruction='温暖、克制的男声', device='cuda:0', dtype='float16',
            attention='sdpa', timeout=12.0)
        with patch.object(backend_qwen, 'design', return_value=raw) as design:
            result = Qwen3Backend().synthesize(request)
        design.assert_called_once_with(
            '你好，世界。', '温暖、克制的男声', self.params,
            device='cuda:0', dtype='float16', attention='sdpa',
            timeout=12.0, cancel=None)
        self.assertIsInstance(result, BackendResult)
        self.assertEqual(result.audio_path, raw['file'])
        self.assertEqual(result.mode, 'design')
        self.assertEqual(result.seed, 3)
        self.assertEqual(result.params, raw['params'])
        self.assertEqual(result.model_dir, str(self.model))
        self.assertEqual(result.model_config_sha256, 'a' * 64)
        self.assertEqual(result.clone_mode, 'none')
        self.assertIsNone(result.clone_prompt_sha256)
        self.assertEqual(result.duration, 0.25)
        self.assertEqual(result.sample_rate, 24000)
        self.assertEqual(result.raw['worker_note'], 'kept-in-raw')

    def test_design_rejects_clone_prompt_and_reference_audio(self):
        backend = Qwen3Backend()
        with patch.object(backend_qwen, 'design') as design:
            with self.assertRaises(CapabilityError) as caught:
                backend.synthesize(BackendRequest(
                    mode='design', text='台词', params=self.params,
                    clone_prompt='prompt.pt'))
            self.assertIn('Clone Prompt', str(caught.exception))
            with self.assertRaises(CapabilityError):
                backend.synthesize(BackendRequest(
                    mode='design', text='台词', params=self.params,
                    ref_audio='reference.wav'))
            design.assert_not_called()

    def test_unknown_mode_is_a_capability_error(self):
        with patch.object(backend_qwen, 'design') as design, \
                patch.object(backend_qwen, 'clone') as clone:
            with self.assertRaises(CapabilityError) as caught:
                Qwen3Backend().synthesize(BackendRequest(
                    mode='vc', text='台词', params=self.params))
        self.assertIn('vc', str(caught.exception))
        design.assert_not_called()
        clone.assert_not_called()

    def test_clone_with_reference_audio_maps_request(self):
        raw = self.raw_result('voice_clone', clone_mode='xvector')
        request = BackendRequest(
            mode='clone', text='新的台词。', params=self.params,
            ref_audio='reference.wav', ref_text='参考音频的完整转录。',
            clone_mode='xvector', timeout=9.0)
        with patch.object(backend_qwen, 'clone', return_value=raw) as clone:
            result = Qwen3Backend().synthesize(request)
        clone.assert_called_once_with(
            'reference.wav', '参考音频的完整转录。', '新的台词。', self.params,
            device='cuda:0', dtype='float16', attention='sdpa',
            timeout=9.0, clone_mode='xvector', cancel=None)
        self.assertNotIn('clone_prompt', clone.call_args.kwargs)
        self.assertNotIn('reference_sha256', clone.call_args.kwargs)
        self.assertEqual(result.mode, 'clone')
        self.assertEqual(result.clone_mode, 'xvector')
        self.assertIsNone(result.clone_prompt_sha256)

    def test_clone_with_prompt_passes_reference_hash_and_never_ref_audio(self):
        prompt = self.root / 'prompt.pt'
        prompt.write_bytes(b'fake-conditioning')
        raw = self.raw_result('voice_clone', clone_mode='icl', prompt_hash='b' * 64)
        request = BackendRequest(
            mode='clone', text='新的台词。', params=self.params,
            ref_text='参考音频的完整转录。', clone_prompt=str(prompt),
            clone_mode='icl', reference_sha256='c' * 64)
        with patch.object(backend_qwen, 'clone', return_value=raw) as clone:
            result = Qwen3Backend().synthesize(request)
        clone.assert_called_once_with(
            None, '参考音频的完整转录。', '新的台词。', self.params,
            device='cuda:0', dtype='float16', attention='sdpa',
            timeout=1800.0, clone_prompt=str(prompt), clone_mode='icl',
            reference_sha256='c' * 64, cancel=None)
        args, kwargs = clone.call_args
        self.assertIsNone(args[0])
        self.assertEqual(kwargs['reference_sha256'], 'c' * 64)
        self.assertEqual(result.clone_prompt_sha256, 'b' * 64)
        self.assertEqual(result.mode, 'clone')

    def test_clone_rejects_ambiguous_or_incomplete_sources(self):
        backend = Qwen3Backend()
        with patch.object(backend_qwen, 'clone') as clone:
            with self.assertRaises(CapabilityError) as caught:
                backend.synthesize(BackendRequest(
                    mode='clone', text='台词', params=self.params,
                    ref_audio='reference.wav', clone_prompt='prompt.pt',
                    reference_sha256='c' * 64))
            self.assertIn('不能同时', str(caught.exception))
            with self.assertRaises(ValueError):
                backend.synthesize(BackendRequest(
                    mode='clone', text='台词', params=self.params))
            with self.assertRaises(ValueError) as missing_hash:
                backend.synthesize(BackendRequest(
                    mode='clone', text='台词', params=self.params,
                    clone_prompt='prompt.pt'))
            self.assertIn('reference_sha256', str(missing_hash.exception))
        clone.assert_not_called()

    def test_non_qwen_params_are_rejected_before_subprocess(self):
        with patch.object(backend_qwen, 'design') as design:
            with self.assertRaises(ValueError) as caught:
                Qwen3Backend().synthesize(BackendRequest(
                    mode='design', text='台词', params={}, instruction='提示'))
        self.assertIn('QwenParams', str(caught.exception))
        design.assert_not_called()

    def test_missing_local_model_maps_to_backend_unavailable(self):
        with patch.object(backend_qwen, 'design',
                          side_effect=FileNotFoundError('Required local Qwen file missing')):
            with self.assertRaises(BackendUnavailable) as caught:
                Qwen3Backend().synthesize(BackendRequest(
                    mode='design', text='台词', params=self.params,
                    instruction='提示'))
        self.assertIn('不会自动下载', str(caught.exception))
        self.assertIn('Required local Qwen file missing', str(caught.exception))

    def test_malformed_adapter_result_is_rejected(self):
        raw = self.raw_result()
        del raw['sample_rate']
        with patch.object(backend_qwen, 'design', return_value=raw):
            with self.assertRaises(RuntimeError) as caught:
                Qwen3Backend().synthesize(BackendRequest(
                    mode='design', text='台词', params=self.params,
                    instruction='提示'))
        self.assertIn('sample_rate', str(caught.exception))
        self.assertTrue(_has_chinese(caught.exception))

    def test_capability_declaration_blocks_unsupported_mode(self):
        backend = Qwen3Backend()
        backend.spec = _spec('qwen3', capabilities=('clone',))
        with patch.object(backend_qwen, 'design') as design:
            with self.assertRaises(CapabilityError):
                backend.synthesize(BackendRequest(
                    mode='design', text='台词', params=self.params,
                    instruction='提示'))
        design.assert_not_called()


class QwenClonePromptTests(QwenBackendTestBase):
    def setUp(self):
        super().setUp()
        self.reference = self.root / 'reference.wav'
        self.reference.write_bytes(b'not-real-audio-but-unused')

    def test_create_clone_prompt_delegates_and_returns_adapter_dict(self):
        payload = {'prompt_file': 'prompt.pt', 'prompt_sha256': 'd' * 64}
        with patch.object(backend_qwen, 'create_clone_prompt',
                          return_value=payload) as create:
            result = Qwen3Backend().create_clone_prompt(
                self.reference, '参考音频的完整转录。', clone_mode='xvector',
                device='cuda:0', dtype='float16', attention='sdpa',
                timeout=33.0, output_dir=self.root / 'prompts')
        create.assert_called_once_with(
            self.reference, '参考音频的完整转录。', clone_mode='xvector',
            device='cuda:0', dtype='float16', attention='sdpa',
            timeout=33.0, output_dir=self.root / 'prompts', cancel=None)
        self.assertIs(result, payload)

    def test_create_clone_prompt_raises_when_unavailable(self):
        (self.model / 'model.safetensors').unlink()
        with patch.object(backend_qwen, 'create_clone_prompt') as create:
            with self.assertRaises(BackendUnavailable) as caught:
                Qwen3Backend().create_clone_prompt(
                    self.reference, '参考音频的完整转录。')
        self.assertIn('safetensors', str(caught.exception))
        self.assertIn('不会自动下载', str(caught.exception))
        create.assert_not_called()

    def test_create_clone_prompt_rejects_missing_capability(self):
        backend = Qwen3Backend()
        backend.spec = _spec('qwen3', capabilities=('clone',))
        with patch.object(backend_qwen, 'create_clone_prompt') as create:
            with self.assertRaises(CapabilityError):
                backend.create_clone_prompt(self.reference, '参考音频的完整转录。')
        create.assert_not_called()

    def test_create_clone_prompt_rejects_non_dict_result(self):
        with patch.object(backend_qwen, 'create_clone_prompt', return_value=[]):
            with self.assertRaises(RuntimeError) as caught:
                Qwen3Backend().create_clone_prompt(
                    self.reference, '参考音频的完整转录。')
        self.assertTrue(_has_chinese(caught.exception))


class DefaultRegistryTests(unittest.TestCase):
    def test_default_registry_does_not_touch_files_or_config(self):
        with patch('timbreforge.backends.get_config',
                   side_effect=AssertionError('default_registry 不得访问配置或文件')):
            registry = default_registry()
        self.assertEqual(registry.names(), ['qwen3'])
        spec = registry.list()[0]
        self.assertEqual(spec.name, 'qwen3')
        self.assertEqual(spec.kind, 'tts')
        self.assertEqual(spec.capabilities, ('design', 'clone', 'clone_prompt'))
        self.assertIn('0.1.1', spec.model_revision)
        self.assertIn('Qwen3-TTS 0.1.1', spec.description)
        self.assertIn('表演指令', spec.description)
        self.assertTrue(_has_chinese(spec.description))

    def test_default_registry_contains_a_working_qwen_backend(self):
        registry = default_registry()
        backend = registry.get('qwen3')
        self.assertIsInstance(backend, Qwen3Backend)


if __name__ == '__main__':
    unittest.main()

