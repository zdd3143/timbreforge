import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from timbreforge.backend_indextts import SynthParams
from timbreforge import backend_cosyvoice as adapter


class CosyVoiceAdapterTests(unittest.TestCase):
    def setUp(self):
        cleanup=patch('timbreforge.resources.release_idle_models');cleanup.start();self.addCleanup(cleanup.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "cosyvoice3"
        self.model = self.repo / "pretrained_models" / "Fun-CosyVoice3-0.5B"
        self.model.mkdir(parents=True)
        for relative in (
            "upy/python.exe", "cosyvoice3.yaml", "llm.pt", "flow.pt", "hift.pt",
            "campplus.onnx", "speech_tokenizer_v3.onnx",
            "CosyVoice-BlankEN/config.json", "CosyVoice-BlankEN/model.safetensors",
        ):
            p = (self.repo if relative.startswith("upy/") else self.model) / relative
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"local-placeholder")
        self.worker = self.root / "cosyvoice_worker.py"
        self.worker.write_text("# mocked worker path\n", encoding="utf-8")
        self.reference = self.root / "reference.wav"
        sf.write(self.reference, np.sin(np.arange(2400, dtype=np.float32)) * 0.1, 24000)
        self.outdir = self.root / "outputs"
        self.patches = [
            patch.object(adapter, "COSYVOICE_REPO", self.repo),
            patch.object(adapter, "PYTHON", self.repo / "upy" / "python.exe"),
            patch.object(adapter, "MODEL_DIR", self.model),
            patch.object(adapter, "WORKER", self.worker),
            patch.object(adapter, "OUTPUT_DIR", self.outdir),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    @staticmethod
    def fake_success(command, cwd, env, stdout, stderr, timeout, check):
        request_path = Path(command[command.index("--request") + 1])
        result_path = Path(command[command.index("--result") + 1])
        request = json.loads(request_path.read_text(encoding="utf-8"))
        output = Path(request["output"])
        sf.write(output, np.linspace(-0.2, 0.2, 2400, dtype=np.float32), 24000,
                 subtype="FLOAT")
        result_path.write_text(json.dumps({
            "file": str(output), "sample_rate": 24000, "duration": 0.1,
            "backend": "cosyvoice3", "mode": "zero_shot",
            "unsupported_params": {"temperature": 0.8},
            "capabilities": {"indextts_emotion_vector": False},
        }), encoding="utf-8")
        return type("Completed", (), {"returncode": 0})()

    def test_success_uses_isolated_worker_and_returns_valid_file(self):
        params = SynthParams(text="你好，世界。", seed=17)
        with patch.object(adapter.subprocess, "run", side_effect=self.fake_success) as run:
            result = adapter.synthesize(self.reference, params, ref_text="参考句子。",
                                        timeout=123, verbose=False)
        self.assertTrue(Path(result["file"]).is_file())
        self.assertGreater(result["elapsed"], 0)
        self.assertEqual(result["sample_rate"], 24000)
        self.assertEqual(result["backend"], "cosyvoice3")
        self.assertIn("unsupported_params", result)
        args, kwargs = run.call_args
        self.assertEqual(kwargs["cwd"], str(self.repo))
        self.assertEqual(kwargs["timeout"], 123)
        self.assertNotIn("capture_output", kwargs)
        self.assertTrue(kwargs["env"]["HF_HUB_OFFLINE"] == "1")
        request_path = Path(args[0][args[0].index("--request") + 1])
        request = json.loads(request_path.read_text(encoding="utf-8"))
        self.assertEqual(request["ref_audio"], str(self.reference.resolve()))
        self.assertEqual(request["ref_text"], "参考句子。")
        self.assertEqual(request["params"]["seed"], 17)

    def test_oom_does_not_advise_unsafe_pickle_loading(self):
        def failure(command, **kwargs):
            result_path = Path(command[command.index('--result') + 1])
            result_path.write_text(json.dumps({'error': 'UnpicklingError: weights_only=False might succeed; CUDA out of memory'}), encoding='utf-8')
            return type('Completed', (), {'returncode': 1})()
        with patch.object(adapter.subprocess, 'run', side_effect=failure):
            with self.assertRaisesRegex(RuntimeError, '显存不足') as caught:
                adapter.synthesize(self.reference, SynthParams(text='test'), verbose=False)
        self.assertNotIn('weights_only=False', str(caught.exception))

    def test_missing_reference_fails_before_subprocess(self):
        with patch.object(adapter.subprocess, "run") as run:
            with self.assertRaises(FileNotFoundError):
                adapter.synthesize(self.root / "missing.wav", SynthParams())
        run.assert_not_called()

    def test_missing_local_model_asset_fails_before_subprocess(self):
        (self.model / "flow.pt").unlink()
        with patch.object(adapter.subprocess, "run") as run:
            with self.assertRaisesRegex(FileNotFoundError, "flow.pt"):
                adapter.synthesize(self.reference, SynthParams())
        run.assert_not_called()

    def test_worker_failure_is_not_reported_as_success(self):
        def failure(command, **kwargs):
            result_path = Path(command[command.index("--result") + 1])
            result_path.write_text(json.dumps({"error": "CUDA out of memory"}), encoding="utf-8")
            return type("Completed", (), {"returncode": 1})()

        with patch.object(adapter.subprocess, "run", side_effect=failure):
            with self.assertRaisesRegex(RuntimeError, "CUDA out of memory"):
                adapter.synthesize(self.reference, SynthParams(), verbose=False)

    def test_worker_timeout_is_reported(self):
        with patch.object(adapter.subprocess, "run", side_effect=adapter.subprocess.TimeoutExpired("worker", 1)):
            with self.assertRaisesRegex(TimeoutError, "CosyVoice3 exceeded"):
                adapter.synthesize(self.reference, SynthParams(), timeout=1, verbose=False)

    def test_invalid_audio_output_is_rejected(self):
        def silent_success(command, cwd, env, stdout, stderr, timeout, check):
            request = json.loads(Path(command[command.index("--request") + 1]).read_text(encoding="utf-8"))
            result_path = Path(command[command.index("--result") + 1])
            sf.write(request["output"], np.zeros(2400, dtype=np.float32), 24000, subtype="FLOAT")
            result_path.write_text(json.dumps({"file": request["output"], "sample_rate": 24000}), encoding="utf-8")
            return type("Completed", (), {"returncode": 0})()

        with patch.object(adapter.subprocess, "run", side_effect=silent_success):
            with self.assertRaisesRegex(RuntimeError, "silent"):
                adapter.synthesize(self.reference, SynthParams(), verbose=False)


if __name__ == "__main__":
    unittest.main()
