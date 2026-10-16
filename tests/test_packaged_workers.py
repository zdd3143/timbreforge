"""Worker wiring tests: no torch, models, network, or external runtimes."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from timbreforge import backend_cosyvoice as cosyvoice
from timbreforge import mosmodels, score
from timbreforge.config import get_config, set_config_path
from timbreforge.workers import campplus_sim, mos_infer


class PackagedWorkerTests(unittest.TestCase):
    def setUp(self):
        cleanup=patch('timbreforge.resources.release_idle_models');cleanup.start();self.addCleanup(cleanup.stop)

    def test_worker_paths_are_inside_installed_package(self):
        package = Path(cosyvoice.__file__).resolve().parent
        for worker in (cosyvoice.WORKER, score.CAMPP_SCRIPT, mosmodels.MOS_SCRIPT):
            self.assertEqual(worker.parent, package / "workers")
            self.assertTrue(worker.is_file())
        self.assertTrue((package / "workers" / "__init__.py").is_file())

    def test_config_changes_after_adapter_import_are_visible(self):
        previous = get_config().config_path
        self.addCleanup(set_config_path, previous)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            for name in ("one", "two"):
                cfg = root / f"{name}.toml"
                cfg.write_text(f'[scorer]\npython = "{name}-python"\n'
                               f'mos_models = "{name}-mos"\n'
                               f'[cosyvoice]\nrepo = "{name}-repo"\n', encoding="utf-8")
                with patch.dict(os.environ, {"USERPROFILE": str(root), "LOCALAPPDATA": str(root)}, clear=True):
                    set_config_path(cfg)
                    self.assertEqual(Path(score.VENV_PY), root / f"{name}-python")
                    self.assertEqual(Path(mosmodels.VENV_PY), root / f"{name}-python")
                    self.assertEqual(mosmodels.MODEL_DIR, root / f"{name}-mos")
                    self.assertEqual(cosyvoice.COSYVOICE_REPO, root / f"{name}-repo")

    def test_cam_adapter_passes_configured_repo_checkpoint(self):
        cfg = SimpleNamespace(scorer_python=Path(sys.executable),
                              indextts_repo=Path("repo"), campplus_checkpoint=Path("checkpoint"))
        def fake_run(command, **kwargs):
            self.assertEqual(command[1], str(score.CAMPP_SCRIPT))
            self.assertEqual(command[command.index("--indextts-repo") + 1], "repo")
            self.assertEqual(command[command.index("--checkpoint") + 1], "checkpoint")
            Path(command[command.index("--out") + 1]).write_text('[{"cosine": 1.0}]', encoding="utf-8")
            return SimpleNamespace(returncode=0)
        with patch.object(score, "get_config", return_value=cfg), patch.object(score.subprocess, "run", side_effect=fake_run):
            self.assertEqual(score.score_pairs([]), [{"cosine": 1.0}])

    def test_mos_adapter_uses_current_model_path_and_python(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            for name in ("utmos22", "dnsmos"):
                (root / name).mkdir()
                (root / name / "model.safetensors").touch()
            cfg = SimpleNamespace(scorer_python=Path(sys.executable), mos_models_dir=root)
            def fake_run(command, **kwargs):
                self.assertEqual(command[0], sys.executable)
                self.assertEqual(command[1], str(mosmodels.MOS_SCRIPT))
                self.assertEqual(command[command.index("--model-dir") + 1], str(root))
                Path(command[command.index("--out") + 1]).write_text('[]', encoding="utf-8")
                return SimpleNamespace(returncode=0)
            with patch.object(mosmodels, "get_config", return_value=cfg), patch.object(mosmodels.subprocess, "run", side_effect=fake_run):
                self.assertEqual(mosmodels.score_files([]), [])

    def test_cam_worker_cli_overrides_environment_and_config(self):
        with patch.dict(os.environ, {"TIMBREFORGE_INDEXTTS_REPO": "env-repo", "TIMBREFORGE_CAMPPLUS_CHECKPOINT": "env-checkpoint"}):
            self.assertEqual(campplus_sim.resolve_model_paths("cli-repo", "cli-checkpoint"),
                             (Path("cli-repo").resolve(), Path("cli-checkpoint").resolve()))
            self.assertEqual(campplus_sim.resolve_model_paths(),
                             (Path("env-repo").resolve(), Path("env-checkpoint").resolve()))

    def test_cam_worker_can_use_config_without_loading_model(self):
        cfg = SimpleNamespace(indextts_repo=Path("config-repo"), campplus_checkpoint=Path("config-checkpoint"))
        with patch.dict(os.environ, {}, clear=True), patch("timbreforge.config.get_config", return_value=cfg):
            self.assertEqual(campplus_sim.resolve_model_paths(),
                             (Path("config-repo").resolve(), Path("config-checkpoint").resolve()))

    def test_worker_help_does_not_import_model_runtime(self):
        for module in (campplus_sim, mos_infer):
            with self.subTest(module=module.__name__), patch.object(sys, "argv", ["worker", "--help"]):
                with self.assertRaises(SystemExit) as result:
                    module.main()
                self.assertEqual(result.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
