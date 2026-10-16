import contextlib
import io
import os
import unittest
from unittest.mock import patch

from timbreforge.launcher import main


class LauncherTests(unittest.TestCase):
    def test_explicit_config_and_browser_options(self):
        with patch('timbreforge.config.set_config_path') as config, \
             patch('timbreforge.readiness.design_readiness', return_value={}), \
             patch('timbreforge.readiness.readiness_markdown', return_value='ready'), \
             patch('importlib.util.find_spec', return_value=object()), \
             patch('timbreforge.launcher.port_in_use', return_value=False), \
             patch('timbreforge.webui.main', return_value=0) as ui, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['--config', 'settings.toml', '--port', '7862']), 0)
        config.assert_called_once_with('settings.toml')
        ui.assert_called_once_with(['--port', '7862', '--config', 'settings.toml', '--open-browser'])

    def test_environment_config_and_no_browser(self):
        with patch.dict(os.environ, {'TIMBREFORGE_CONFIG': 'env.toml'}), \
             patch('timbreforge.config.set_config_path') as config, \
             patch('timbreforge.readiness.design_readiness', return_value={}), \
             patch('timbreforge.readiness.readiness_markdown', return_value='ready'), \
             patch('importlib.util.find_spec', return_value=object()), \
             patch('timbreforge.launcher.port_in_use', return_value=False), \
             patch('timbreforge.webui.main', return_value=0) as ui, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['--no-browser']), 0)
        config.assert_called_once_with('env.toml')
        self.assertNotIn('--open-browser', ui.call_args.args[0])

    def test_check_failure_does_not_launch(self):
        with patch('timbreforge.config.set_config_path'), \
             patch('timbreforge.readiness.design_readiness', return_value={'design_files_ready': False}), \
             patch('timbreforge.readiness.readiness_markdown', return_value='missing'), \
             patch('timbreforge.webui.main') as ui, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['--check']), 1)
        ui.assert_not_called()

    def test_occupied_port_is_explained_instead_of_crashing(self):
        output = io.StringIO()
        with patch('timbreforge.config.set_config_path'), \
             patch('timbreforge.readiness.design_readiness', return_value={'design_files_ready': True}), \
             patch('timbreforge.readiness.readiness_markdown', return_value='ready'), \
             patch('importlib.util.find_spec', return_value=object()), \
             patch('timbreforge.launcher.port_in_use', return_value=True), \
             patch('timbreforge.webui.main') as ui, \
             contextlib.redirect_stdout(output):
            self.assertEqual(main(['--no-browser']), 1)
        ui.assert_not_called()
        message = output.getvalue()
        self.assertIn('已被占用', message)
        self.assertIn('--port 7861', message)

    def test_free_port_launches_normally(self):
        with patch('timbreforge.config.set_config_path'), \
             patch('timbreforge.readiness.design_readiness', return_value={'design_files_ready': True}), \
             patch('timbreforge.readiness.readiness_markdown', return_value='ready'), \
             patch('importlib.util.find_spec', return_value=object()), \
             patch('timbreforge.launcher.port_in_use', return_value=False), \
             patch('timbreforge.webui.main', return_value=0) as ui, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['--config', 'settings.toml', '--no-browser']), 0)
        ui.assert_called_once_with(['--port', '7860', '--config', 'settings.toml'])


if __name__ == '__main__':
    unittest.main()
