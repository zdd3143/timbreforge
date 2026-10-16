import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from timbreforge import simple_ui as ui
from timbreforge.backend_qwen import QwenParams


class SimpleUITests(unittest.TestCase):
    def test_uses_model_recipe_with_bounded_preview_and_new_seed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / 'generation_config.json').write_text(json.dumps(dict(
                temperature=.65, top_k=32, subtalker_top_p=.8, max_new_tokens=8192,
                unrelated_model_setting='not forwarded')))
            with patch.object(ui, 'get_config', return_value=SimpleNamespace(qwen_design_model=root)), \
                 patch.object(ui.secrets, 'randbelow', return_value=1234):
                recipe = ui.automatic_recipe()
            self.assertEqual(recipe.temperature, .65)
            self.assertEqual(recipe.top_k, 32)
            self.assertEqual(recipe.subtalker_top_p, .8)
            self.assertEqual(recipe.max_new_tokens, 1024)
            self.assertEqual(recipe.seed, 1234)

    def test_empty_description_fails_before_runtime_or_storage(self):
        with patch.object(ui, 'automatic_device') as device, patch.object(ui, 'DesignStore') as store:
            for description in ('', '  ', None, 'a' * 2001):
                with self.assertRaises(ValueError):
                    ui.generate_simple(description)
            device.assert_not_called()
            store.assert_not_called()

    def test_description_only_reaches_real_service_contract(self):
        params = QwenParams(seed=712)
        with patch.object(ui, 'automatic_recipe', return_value=params), \
             patch.object(ui, 'automatic_device', return_value='cpu'), \
             patch.object(ui, 'DesignStore') as store:
            store.return_value.create.return_value = dict(id='session', prompt='温暖男声',
                text=ui.PREVIEW_TEXT, candidates=[dict(id='candidate')])
            store.return_value.audio_path.return_value = 'result.wav'
            audio, details = ui.generate_simple(' 温暖男声 ')
            store.return_value.create.assert_called_once_with('温暖男声', ui.PREVIEW_TEXT, params,
                count=1, device='cpu', confirmed=True)
            self.assertEqual(audio, 'result.wav')
            self.assertEqual(details['parameters']['seed'], 712)
            self.assertEqual(details['device'], 'cpu')

    def test_device_probe_failure_is_not_silently_cpu(self):
        with tempfile.TemporaryDirectory() as td:
            python = Path(td) / 'python.exe'
            python.write_bytes(b'test')
            with patch.object(ui, 'get_config', return_value=SimpleNamespace(qwen_python=python)), \
                 patch.object(ui.subprocess, 'run') as run:
                for device in ('cuda:0', 'cpu'):
                    run.return_value = SimpleNamespace(returncode=0, stdout=device+'\n', stderr='')
                    self.assertEqual(ui.automatic_device(), device)
                run.return_value = SimpleNamespace(returncode=1, stdout='', stderr='missing torch')
                with self.assertRaisesRegex(RuntimeError, 'missing torch'):
                    ui.automatic_device()

    def test_input_error_does_not_blame_models(self):
        self.assertEqual(ui.error_message(ValueError('请填写描述')), '请填写描述')
        self.assertIn('Qwen 路径', ui.error_message(FileNotFoundError('config.json')))

    @unittest.skipUnless(importlib.util.find_spec('gradio'), 'UI extra not installed')
    def test_simple_page_has_one_input_and_failure_clears_audio(self):
        import gradio as gr
        with patch.object(ui, 'automatic_recipe', side_effect=AssertionError('no IO during build')):
            with gr.Blocks(analytics_enabled=False) as demo:
                with gr.Tabs():
                    ui.build_simple_tab(gr, {'concurrency_id':'timbreforge-synthesis','concurrency_limit':1})
        components = demo.get_config_file()['components']
        self.assertEqual(sum(c['type'] == 'textbox' for c in components), 1)
        self.assertFalse(any(c['type'] in ('checkbox', 'slider', 'number') for c in components))
        fn = next(f.fn for f in demo.fns.values() if f.api_name == 'simple_generate')
        with patch.object(ui, 'generate_simple', side_effect=RuntimeError('GPU test failure')):
            outputs = list(fn('test', None))
        self.assertIsNone(outputs[0][0])
        self.assertIsNone(outputs[-1][0])
        self.assertIn('GPU test failure', outputs[-1][1])
        demo.close()


if __name__ == '__main__':
    unittest.main()
