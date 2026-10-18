"""Community adapter registry and role binding tests."""
from pathlib import Path
import tempfile
import unittest
import json

import numpy as np
import soundfile as sf

from timbreforge.adapters import AdapterError, AdapterRegistry
from timbreforge.characters import CharacterStore


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.registry = AdapterRegistry(self.root / 'registry')
        self.adapter = self.root / 'adapter'
        self.adapter.mkdir()
        (self.adapter / 'adapter_config.json').write_text(json.dumps({
            'peft_type': 'LORA', 'r': 16, 'lora_alpha': 32,
            'target_modules': ['q_proj', 'v_proj'],
            'base_model_name_or_path': 'Qwen3-TTS-12Hz-0.6B-Base',
        }), encoding='utf-8')
        (self.adapter / 'adapter_model.safetensors').write_bytes(b'synthetic-safe-tensors')
        self.audio = self.root / 'voice.wav'
        t = np.arange(24000, dtype=np.float32) / 24000
        sf.write(self.audio, 0.1 * np.sin(2 * np.pi * 220 * t), 24000, subtype='FLOAT')

    def test_register_verify_and_remove(self):
        entry = self.registry.register(self.adapter, name='社区测试 LoRA', author='tester',
                                       license='Apache-2.0', default_scale=0.3)
        self.assertEqual(entry['adapter_type'], 'peft_lora')
        self.assertTrue(self.registry.verify(entry['id'])['ok'])
        (self.adapter / 'adapter_model.safetensors').write_bytes(b'changed')
        self.assertFalse(self.registry.verify(entry['id'])['ok'])
        self.registry.remove(entry['id'])
        self.assertEqual(self.registry.list(), [])

    def test_rejects_pickle_or_non_lora(self):
        (self.adapter / 'adapter_model.safetensors').unlink()
        (self.adapter / 'adapter_model.bin').write_bytes(b'pickle')
        with self.assertRaises(AdapterError):
            self.registry.register(self.adapter)

    def test_bind_and_unbind_role(self):
        store = CharacterStore(self.root / 'characters')
        card = store.create('社区角色', self.audio, reference_text='测试台词。', backend='qwen3')
        entry = self.registry.register(self.adapter, name='绑定测试')
        bound = self.registry.bind_role(card['id'], entry['id'], scale=0.35, store=store)
        self.assertEqual(bound['provenance']['adapter']['id'], entry['id'])
        resolved = self.registry.resolve_binding(bound)
        self.assertAlmostEqual(resolved['scale'], 0.35)
        unbound = self.registry.unbind_role(card['id'], store=store)
        self.assertIsNone((unbound.get('provenance') or {}).get('adapter'))

    def test_zip_import_rejects_path_traversal(self):
        import zipfile
        archive = self.root / 'bad.zip'
        with zipfile.ZipFile(archive, 'w') as bundle:
            bundle.writestr('../adapter_config.json', '{}')
        with self.assertRaises(AdapterError):
            self.registry.import_zip(archive)


if __name__ == '__main__':
    unittest.main()

