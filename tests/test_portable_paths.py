"""No installed models: configuration changes must affect already-imported modules."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from timbreforge import backend_indextts as index
from timbreforge.characters import CharacterStore
from timbreforge.config import set_config_path
from timbreforge.studio_cli import main


class PortablePathTests(unittest.TestCase):
    def tearDown(self):
        set_config_path(None)

    def test_index_and_character_store_follow_new_config(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            config = root / 'settings.toml'
            config.write_text('[comfy]\nurl="http://localhost:9191"\ninput_dir="input"\noutput_dir="output"\n[paths]\ncharacters_dir="cards"\n', encoding='utf-8')
            set_config_path(config)
            self.assertEqual(index.DEFAULT_BASE, 'http://localhost:9191')
            self.assertEqual(index.COMFY_INPUT, root / 'input')
            self.assertEqual(CharacterStore().root, root / 'cards')
            # Missing ComfyUI input must not be silently fabricated in a wrong location.
            src = root / 'source.wav'
            src.write_bytes(b'reference')
            with self.assertRaisesRegex(FileNotFoundError, 'input directory'):
                index.copy_to_comfy_input(src)
            self.assertFalse((root / 'input').exists())
            (root / 'input').mkdir()
            with self.assertRaises(ValueError):
                index.copy_to_comfy_input(src, '../escape.wav')

    def test_cli_config_error_is_clean(self):
        with contextlib.redirect_stderr(io.StringIO()) as errors:
            with self.assertRaises(SystemExit) as caught:
                main(['--config', '/nonexistent-timbreforge-settings.toml', 'capabilities'])
        self.assertEqual(caught.exception.code, 1)
        self.assertIn('config file not found', errors.getvalue())

    def test_default_ping_uses_selected_url(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td).resolve() / 'settings.toml'
            p.write_text('[comfy]\nurl="http://localhost:9009"\n', encoding='utf-8')
            set_config_path(p)
            response = io.BytesIO(json.dumps({'devices': []}).encode())
            with patch.object(index.urllib.request, 'urlopen', return_value=response) as request:
                self.assertTrue(index.ping()[0])
            self.assertEqual(request.call_args.args[0], 'http://localhost:9009/system_stats')


if __name__ == '__main__':
    unittest.main()
