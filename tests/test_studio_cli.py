import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import soundfile as sf
from timbreforge.studio_cli import main


class StudioCliTests(unittest.TestCase):
    def test_capabilities_offline(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(main(['capabilities']), 0)
        self.assertFalse(json.loads(out.getvalue())[-1]['available'])

    def test_create_and_list_card_without_model(self):
        with tempfile.TemporaryDirectory() as td:
            ref = Path(td) / 'ref.wav'
            sf.write(ref, .1 * np.sin(np.arange(1600)), 16000)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                main(['save-character', '--name', '角色甲', '--reference', str(ref),
                      '--store', str(Path(td) / 'cards'), '--confirm-rights'])
            card = json.loads(out.getvalue())
            self.assertEqual(card['name'], '角色甲')
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                main(['characters', '--store', str(Path(td) / 'cards')])
            self.assertEqual(json.loads(out.getvalue())[0]['id'], card['id'])

    def test_requires_rights_confirmation(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as exc:
                main(['generate', '--text', '你好', '--reference', 'ref.wav'])
        self.assertEqual(exc.exception.code, 2)

    def test_character_seed_override_does_not_reset_saved_params(self):
        with patch('timbreforge.studio_cli.generate', return_value={'file': 'mock.wav'}) as render, \
             contextlib.redirect_stdout(io.StringIO()):
            main(['generate', '--task', 'character', '--character', 'a' * 32,
                  '--text', '新台词', '--seed', '27', '--confirm-rights'])
        request = render.call_args.args[0]
        self.assertIsNone(request.params)
        self.assertEqual(request.seed_override, 27)


if __name__ == '__main__':
    unittest.main()
