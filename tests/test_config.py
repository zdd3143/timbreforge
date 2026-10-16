import os
import tempfile
import unittest
from pathlib import Path

from timbreforge.config import load_config


class ConfigTests(unittest.TestCase):
    def test_planner_paths_relative_and_env_override(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            path=root/'forge.toml'
            path.write_text('[planner]\npython="env/python.exe"\nmodel="models/planner"\n')
            with patch.dict(os.environ,{'TIMBREFORGE_PLANNER_MODEL':str(root/'override')}):
                cfg=load_config(path)
            self.assertEqual(cfg.planner_python,root/'env'/'python.exe')
            self.assertEqual(cfg.planner_model,root/'override')

    def test_toml_relative_paths_and_env_override(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = root / 'forge.toml'
            cfg.write_text('[comfy]\nurl = "http://example:8188"\ninput_dir = "in"\n[paths]\noutputs_dir = "out"\n', encoding='utf-8')
            old = os.environ.get('TIMBREFORGE_COMFY_URL')
            os.environ['TIMBREFORGE_COMFY_URL'] = 'http://env:1'
            try:
                value = load_config(cfg)
            finally:
                if old is None: os.environ.pop('TIMBREFORGE_COMFY_URL', None)
                else: os.environ['TIMBREFORGE_COMFY_URL'] = old
            self.assertEqual(value.comfy_url, 'http://env:1')
            self.assertEqual(value.comfy_input_dir, (root / 'in').resolve())
            self.assertEqual(value.outputs_dir, (root / 'out').resolve())

    def test_qwen_paths_relative_and_env_override(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = root / 'forge.toml'
            cfg.write_text('[qwen]\npython="runtime/python.exe"\ndesign_model="models/design"\nbase_model="models/base"\n', encoding='utf-8')
            with patch.dict(os.environ, {'TIMBREFORGE_QWEN_BASE_MODEL': str(root/'override')}):
                value = load_config(cfg)
            self.assertEqual(value.qwen_python, (root/'runtime/python.exe').resolve())
            self.assertEqual(value.qwen_design_model, (root/'models/design').resolve())
            self.assertEqual(value.qwen_base_model, (root/'override').resolve())

    def test_env_paths_are_cwd_relative(self):
        with tempfile.TemporaryDirectory() as td:
            old = os.environ.get('TIMBREFORGE_OUTPUTS_DIR')
            os.environ['TIMBREFORGE_OUTPUTS_DIR'] = 'relative-output'
            try:
                value = load_config()
            finally:
                if old is None: os.environ.pop('TIMBREFORGE_OUTPUTS_DIR', None)
                else: os.environ['TIMBREFORGE_OUTPUTS_DIR'] = old
            self.assertEqual(value.outputs_dir, (Path.cwd() / 'relative-output').resolve())


if __name__ == '__main__':
    unittest.main()
