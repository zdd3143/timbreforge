"""Portable entrypoints need neither datasets nor private assets to start."""
from __future__ import annotations

import contextlib
import importlib
import io
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from timbreforge import config, voicegen, webui


class EntrypointTests(unittest.TestCase):
    def setUp(self):
        self.old_config = config._config_path
        self.addCleanup(config.set_config_path, self.old_config)

    def write_config(self, root):
        path = root / "portable.toml"
        path.write_text('[comfy]\nurl = "http://127.0.0.1:19222"\n'
                        '[paths]\noutputs_dir = "renders"\ncharacters_dir = "cards"\n',
                        encoding="utf-8")
        return path

    def test_canonical_creator_route_preserves_config(self):
        from timbreforge.studio_cli import main
        with patch('timbreforge.creation_cli.main', return_value=0) as creator:
            self.assertEqual(main(['--config', 'local.toml', 'create', 'list']), 0)
        creator.assert_called_once_with(['--config', 'local.toml', 'list'])

    def test_import_has_no_output_directory_side_effect(self):
        with patch.object(Path, "mkdir", side_effect=AssertionError("import mkdir")):
            importlib.reload(webui)
            importlib.reload(voicegen)

    def test_webui_config_selected_before_build_and_safe_launch_paths(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {k: v for k, v in os.environ.items() if not k.startswith("TIMBREFORGE_")}, clear=True):
            root = Path(td)
            path = self.write_config(root)
            demo = types.SimpleNamespace(launch=lambda **kwargs: launches.append(kwargs))
            launches = []
            def build(**kwargs):
                self.assertEqual(config.get_config().comfy_url, "http://127.0.0.1:19222")
                self.assertFalse(kwargs["include_legacy"])
                return demo
            with patch.object(webui, "build_ui", side_effect=build), \
                 patch.object(webui, "get_bank", side_effect=AssertionError("no dataset")):
                self.assertEqual(webui.main(["--config", str(path)]), 0)
            self.assertEqual(launches[0]["allowed_paths"],
                             [str((root / "renders").resolve()), str((root / "cards").resolve()),
                              str((Path(webui.__file__).resolve().parent.parent / "assets").resolve())])
            self.assertNotIn(str(root), launches[0]["allowed_paths"])

    def test_voicegen_config_controls_base_and_default_output_without_bank(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {k: v for k, v in os.environ.items() if not k.startswith("TIMBREFORGE_")}, clear=True):
            root = Path(td)
            path = self.write_config(root)
            reference, raw = root / "reference.wav", root / "raw.wav"
            samples = (.1 * np.sin(np.arange(3200))).astype(np.float32)
            sf.write(reference, samples, 16000, subtype="FLOAT")
            sf.write(raw, samples, 16000, subtype="FLOAT")
            with patch.object(voicegen, "get_bank", side_effect=AssertionError("no dataset")), \
                 patch("timbreforge.reference.prepare_reference", return_value=reference), \
                 patch("timbreforge.backend_indextts.ping", return_value=(True, "ok")) as ping, \
                 patch("timbreforge.backend_indextts.synthesize", return_value={
                     "file": str(raw), "elapsed": .1}) as synth, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(voicegen.main(["--config", str(path), "--ref-wav", str(reference)]), 0)
            ping.assert_called_once_with("http://127.0.0.1:19222")
            self.assertEqual(synth.call_args.kwargs["base"], "http://127.0.0.1:19222")
            self.assertTrue(list((root / "renders" / "voicegen").glob("*.wav")))


if __name__ == "__main__":
    unittest.main()

