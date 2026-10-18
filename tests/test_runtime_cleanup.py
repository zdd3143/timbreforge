"""Regression tests for disposable runtime cleanup and preset isolation."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from timbreforge import character_audio
from timbreforge.characters import CharacterStore
from timbreforge.ui_tasks import clear_ephemeral_data


class RuntimeCleanupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.characters = self.root / 'characters'
        self.outputs = self.root / 'outputs'
        self.outputs.mkdir()
        self.audio = self.root / 'reference.wav'
        t = np.arange(24000, dtype=np.float32) / 24000
        sf.write(self.audio, 0.15 * np.sin(2 * np.pi * 220 * t), 24000,
                 subtype='FLOAT')

    def test_cleanup_removes_runtime_records_but_never_character_root(self):
        store = CharacterStore(self.characters)
        card = store.create('保留角色', self.audio, reference_text='测试台词。', backend='qwen3')
        (self.outputs / 'designs').mkdir()
        (self.outputs / 'designs' / 'session.json').write_text('{}', encoding='utf-8')
        (self.outputs / 'jobs.json').write_text('{}', encoding='utf-8')
        result = clear_ephemeral_data(outputs_dir=self.outputs, characters_dir=self.characters)
        self.assertEqual(result['failed'], [])
        self.assertEqual(result['removed'], 2)
        self.assertTrue(Path(card['reference_audio']).is_file())
        self.assertEqual(store.load(card['id'])['name'], '保留角色')

    def test_presets_always_start_from_original_reference(self):
        store = CharacterStore(self.characters)
        card = store.create('预设隔离', self.audio, reference_text='测试台词。', backend='qwen3')
        first = character_audio.render_preset(card['id'], 'clean_voice', store=store)
        second = character_audio.render_preset(card['id'], 'warm_voice', store=store)
        self.assertEqual(second['document']['source_audio'], 'reference.wav')
        self.assertEqual(second['document']['source_hash'], card['reference_sha256'])
        self.assertNotEqual(first['output_path'], second['output_path'])


if __name__ == '__main__':
    unittest.main()

