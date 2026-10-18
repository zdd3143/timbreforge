"""Role-card audio conditioning, duplication and portable packages."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import soundfile as sf

from timbreforge import character_audio
from timbreforge.character_package import duplicate_card, export_card, import_card
from timbreforge.characters import CharacterStore


def voice_like(seconds=2.0, sample_rate=24000, seed=11):
    generator = np.random.default_rng(seed)
    count = int(seconds * sample_rate)
    time_axis = np.arange(count) / sample_rate
    signal = 0.25 * np.sin(2 * np.pi * 150 * time_axis)
    signal += 0.05 * np.sin(2 * np.pi * 300 * time_axis)
    signal += generator.normal(0, 0.0005, count)
    pad = np.zeros(int(0.25 * sample_rate))
    return np.concatenate([pad, signal, pad])[:, None], sample_rate


class CharacterAudioTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'characters'
        self.store = CharacterStore(self.root)
        self.source = Path(self.temp.name) / 'voice.wav'
        audio, rate = voice_like()
        sf.write(self.source, audio, rate, subtype='FLOAT')
        self.card = self.store.create('音频修整', self.source, reference_text='测试台词。',
                                      backend='qwen3')

    def card_dir(self):
        return Path(self.card['reference_audio']).parent

    def test_render_preset_writes_derived_audio_and_preview(self):
        original = Path(self.card['reference_audio']).read_bytes()
        result = character_audio.render_preset(self.card['id'], 'clean_voice', store=self.store)
        card = result['card']
        self.assertTrue(Path(card['canonical_audio']).is_file())
        self.assertTrue(Path(card['canonical_audio']).resolve().is_relative_to(self.card_dir().resolve()))
        self.assertEqual(Path(self.card['reference_audio']).read_bytes(), original)
        self.assertEqual(card['version'], 2)
        self.assertEqual(len(card['audio_edit_chain']), 1)
        self.assertEqual(card['audio_edit_chain'][0]['preset'], 'clean_voice')
        self.assertEqual(result['preview']['format'], 'OGG')
        self.assertTrue(Path(result['preview']['path']).is_file())
        self.assertLessEqual(result['metrics']['peak_dbfs_out'], 0.01)
        self.assertTrue(result['notes'])

    def test_custom_chain_requires_at_least_one_operation(self):
        with self.assertRaises(ValueError):
            character_audio.render_custom(self.card['id'], [], store=self.store)

    def test_history_outputs_are_never_overwritten(self):
        first = character_audio.render_preset(self.card['id'], 'clean_voice', store=self.store)
        second = character_audio.render_preset(self.card['id'], 'warm_voice', store=self.store)
        self.assertNotEqual(first['output_path'], second['output_path'])
        self.assertEqual(len(second['card']['audio_edit_chain']), 2)
        self.assertTrue(Path(first['output_path']).is_file())
        reverted = character_audio.revert(self.card['id'], store=self.store)
        self.assertIsNone(reverted['card']['canonical_audio'])
        self.assertEqual(reverted['card']['audio_edit_chain'][-1]['event'], 'revert_to_original')
        self.assertTrue(Path(first['output_path']).is_file())

    def test_status_reports_current_audio_and_history(self):
        character_audio.render_preset(self.card['id'], 'low_noise', store=self.store)
        status = character_audio.status(self.card['id'], store=self.store)
        self.assertTrue(status['is_canonical'])
        self.assertGreater(status['duration'], 0)
        self.assertEqual(status['history']['documents'], 1)
        self.assertTrue(status['history']['labels'])
        self.assertIsNotNone(status['preview'])
        history = character_audio.history(self.card['id'], store=self.store)
        self.assertEqual(len(history), 1)
        self.assertIn('降噪', history[0]['summary'])


class CharacterPackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.store = CharacterStore(self.base / 'characters')
        self.source = self.base / 'voice.wav'
        audio, rate = voice_like()
        sf.write(self.source, audio, rate, subtype='FLOAT')
        self.card = self.store.create('可导出', self.source, reference_text='导出测试。',
                                      backend='qwen3', tags=['测试'],
                                      voice_design_prompt='低沉男声。')
        character_audio.render_preset(self.card['id'], 'clean_voice', store=self.store)
        self.card = self.store.load(self.card['id'])

    def test_export_and_reimport_round_trip(self):
        package = self.base / 'role.zip'
        exported = export_card(self.card['id'], package, store=self.store)
        self.assertTrue(package.is_file())
        self.assertGreater(exported['bytes'], 0)
        self.assertIn('reference.wav', exported['included_audio'])
        imported = import_card(package, store=self.store, name='重新导入')
        self.assertNotEqual(imported['id'], self.card['id'])
        self.assertEqual(imported['name'], '重新导入')
        self.assertIsNone(imported['clone_prompt'])
        self.assertEqual(imported['version'], 1)
        self.assertEqual(imported['provenance']['imported_from_package'], self.card['id'])
        self.assertTrue(Path(imported['canonical_audio']).is_file())
        self.assertEqual(Path(imported['canonical_audio']).read_bytes(),
                         Path(self.card['canonical_audio']).read_bytes())
        self.assertEqual(Path(imported['reference_audio']).read_bytes(),
                         Path(self.card['reference_audio']).read_bytes())
        self.assertEqual(imported['voice_recipe'], self.card['voice_recipe'])
        self.assertEqual(imported['tags'], ['测试'])
        self.assertEqual(len(self.store.list()), 2)

    def test_export_refuses_to_overwrite(self):
        package = self.base / 'role.zip'
        package.write_bytes(b'keep')
        with self.assertRaises(FileExistsError):
            export_card(self.card['id'], package, store=self.store)
        self.assertEqual(package.read_bytes(), b'keep')

    def test_import_rejects_unsafe_entries(self):
        package = self.base / 'bad.zip'
        with zipfile.ZipFile(package, 'w') as archive:
            archive.writestr('manifest.json', json.dumps({'schema_version': 2}))
            archive.writestr('../escape.wav', b'x')
        with self.assertRaisesRegex(ValueError, '不安全|多余'):
            import_card(package, store=self.store)
        self.assertFalse((self.base / 'escape.wav').exists())

    def test_import_rejects_hash_mismatch(self):
        package = self.base / 'role.zip'
        export_card(self.card['id'], package, store=self.store)
        buffer = io.BytesIO()
        sf.write(buffer, np.full(2400, 0.1, dtype=np.float32), 24000,
                 format='WAV', subtype='FLOAT')
        replacement = buffer.getvalue()
        tampered = self.base / 'tampered.zip'
        with zipfile.ZipFile(package) as source, zipfile.ZipFile(tampered, 'w') as target:
            for info in source.infolist():
                data = source.read(info.filename)
                if info.filename == 'reference.wav':
                    data = replacement
                target.writestr(info.filename, data)
        with self.assertRaisesRegex(ValueError, '哈希'):
            import_card(tampered, store=self.store)

    def test_import_without_derived_audio_falls_back_to_original(self):
        package = self.base / 'role.zip'
        export_card(self.card['id'], package, store=self.store)
        stripped = self.base / 'stripped.zip'
        with zipfile.ZipFile(package) as source, zipfile.ZipFile(stripped, 'w') as target:
            for info in source.infolist():
                if info.filename.startswith('audio/'):
                    continue
                target.writestr(info.filename, source.read(info.filename))
        imported = import_card(stripped, store=self.store)
        self.assertIsNone(imported['canonical_audio'])
        self.assertEqual(Path(imported['reference_audio']).read_bytes(),
                         Path(self.card['reference_audio']).read_bytes())

    def test_duplicate_copies_audio_and_history_but_not_prompts(self):
        card_dir = Path(self.card['reference_audio']).parent
        (card_dir / 'prompts').mkdir(exist_ok=True)
        (card_dir / 'prompts' / 'cache.pt').write_bytes(b'tensors')
        self.store.set_clone_prompt(self.card['id'], 'prompts/cache.pt', 'icl')
        copy = duplicate_card(self.card['id'], store=self.store)
        self.assertNotEqual(copy['id'], self.card['id'])
        self.assertIn('副本', copy['name'])
        self.assertIsNone(copy['clone_prompt'])
        self.assertFalse((Path(copy['reference_audio']).parent / 'prompts').exists())
        self.assertEqual(copy['provenance']['duplicated_from'], self.card['id'])
        self.assertEqual(Path(copy['reference_audio']).read_bytes(),
                         Path(self.card['reference_audio']).read_bytes())
        self.assertEqual(copy['audio_edit_chain'], self.card['audio_edit_chain'])
        self.assertEqual(copy['version'], 1)

    def test_package_never_contains_audio_hashes_without_files(self):
        package = self.base / 'role.zip'
        export_card(self.card['id'], package, store=self.store)
        with zipfile.ZipFile(package) as archive:
            names = archive.namelist()
            manifest = json.loads(archive.read('manifest.json'))
        self.assertFalse([name for name in names if name.startswith('prompts')])
        self.assertIn('manifest.json', names)
        self.assertIn('package.json', names)
        self.assertEqual(manifest['clone_prompt'], None)
        # Every stored document output referenced in the package exists.
        for entry in manifest['audio_edit_chain']:
            if 'output_audio' in entry and entry['output_audio']:
                self.assertIn(entry['output_audio'], names)


if __name__ == '__main__':
    unittest.main()

