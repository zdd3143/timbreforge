import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from timbreforge.readiness import _snapshot, _weights, design_readiness, readiness_markdown


class ReadinessTests(unittest.TestCase):
    def test_partial_model_is_not_ready(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / 'config.json').write_text('{"tts_model_type":"voice_design"}')
            (root / 'model.safetensors').write_bytes(b'fake test weights')
            result = _snapshot(root, 'voice_design')
            self.assertFalse(result['files_ready'])
            self.assertIn('vocab.json', result['missing'])
            self.assertIn('speech_tokenizer/config.json', result['missing'])

    def test_complete_snapshot_and_wrong_model_type(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / 'speech_tokenizer').mkdir()
            for file in ('generation_config.json', 'preprocessor_config.json', 'tokenizer_config.json',
                         'vocab.json', 'merges.txt', 'model.safetensors', 'speech_tokenizer/config.json',
                         'speech_tokenizer/preprocessor_config.json', 'speech_tokenizer/model.safetensors'):
                (root / file).write_bytes(b'test fixture')
            (root / 'config.json').write_text('{"tts_model_type":"voice_design"}')
            self.assertTrue(_snapshot(root, 'voice_design')['files_ready'])
            self.assertFalse(_snapshot(root, 'base')['files_ready'])

    def test_shards_must_be_complete_and_local(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            index = root / 'model.safetensors.index.json'
            index.write_text(json.dumps({'weight_map': {'a': 'one.safetensors', 'b': 'two.safetensors'}}))
            (root / 'one.safetensors').write_bytes(b'one')
            self.assertFalse(_weights(root))
            (root / 'two.safetensors').write_bytes(b'two')
            self.assertTrue(_weights(root))
            index.write_text(json.dumps({'weight_map': {'a': '../outside.safetensors'}}))
            self.assertFalse(_weights(root))
            index.write_text('[]')
            self.assertFalse(_weights(root))

    def test_missing_python_does_not_claim_ready(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = SimpleNamespace(qwen_python=root / 'missing.exe', qwen_design_model=root, qwen_base_model=root)
            with patch('timbreforge.readiness.get_config', return_value=cfg):
                report = design_readiness()
            self.assertFalse(report['design_files_ready'])
            self.assertFalse(report['base_files_ready'])
            self.assertIn('待配置', readiness_markdown(report))


if __name__ == '__main__':
    unittest.main()
