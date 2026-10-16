import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from timbreforge.characters import CharacterStore
from timbreforge.voice_library import VoiceLibrary, library_view


def make_card(root, name='林澈', seconds=.2):
    wav = Path(root) / f'{name}.wav'
    sf.write(wav, np.full(int(24000 * seconds), .2), 24000, subtype='FLOAT')
    return CharacterStore(Path(root) / 'characters').create(
        name, wav, reference_text='别担心，我会带你回家。', backend='qwen3')


class VoiceLibraryTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.lib = VoiceLibrary(self.root / 'characters')
        self.card = make_card(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_one_character_can_own_several_voices(self):
        group = self.lib.create_character('林澈', '救援队长')
        second = make_card(self.root, '林澈二')
        self.lib.add_voice(group['id'], self.card['id'], description='最初的一条',
                           preview_text='别担心，我会带你回家。', origin={'plan_id': 'a' * 32})
        self.lib.add_voice(group['id'], second['id'], label='备用声线', description='更冷一点')
        stored = self.lib.get(group['id'])
        self.assertEqual([voice['character_id'] for voice in stored['voices']],
                         [self.card['id'], second['id']])
        self.assertEqual(stored['description'], '救援队长')
        self.assertEqual(stored['voices'][1]['label'], '备用声线')
        self.assertEqual(stored['voices'][0]['origin']['plan_id'], 'a' * 32)

    def test_duplicate_voice_and_unknown_character_are_rejected(self):
        group = self.lib.create_character('林澈')
        self.lib.add_voice(group['id'], self.card['id'])
        with self.assertRaisesRegex(ValueError, '已经在该角色名下'):
            self.lib.add_voice(group['id'], self.card['id'])
        with self.assertRaisesRegex(ValueError, '找不到该角色'):
            self.lib.add_voice('f' * 32, self.card['id'])
        with self.assertRaisesRegex(ValueError, '角色名'):
            self.lib.create_character('   ')
        with self.assertRaises(ValueError):
            self.lib.get('../escape')

    def test_dangling_card_is_not_presented_as_a_voice(self):
        group = self.lib.create_character('林澈')
        self.lib.add_voice(group['id'], self.card['id'])
        # A card that disappears must not be shown as if the voice still existed.
        card_dir = self.card['reference_audio']
        import shutil
        shutil.rmtree(Path(card_dir).parent)
        view = library_view_with(self.lib.root)
        self.assertTrue(view['ok'])
        self.assertEqual(view['characters'][0]['voices'], [])
        self.assertEqual(self.lib.get(group['id'])['voices'][0]['character_id'], self.card['id'])

    def test_remove_voice_drops_empty_character_only(self):
        group = self.lib.create_character('林澈')
        second = make_card(self.root, '林澈二')
        self.lib.add_voice(group['id'], self.card['id'])
        self.lib.add_voice(group['id'], second['id'])
        self.lib.remove_voice(group['id'], self.card['id'])
        self.assertEqual(len(self.lib.get(group['id'])['voices']), 1)
        self.lib.remove_voice(group['id'], second['id'])
        self.assertIsNone(self.lib.get(group['id']))
        with self.assertRaisesRegex(ValueError, '找不到该角色'):
            self.lib.remove_voice(group['id'], second['id'])

    def test_corrupt_index_and_unsafe_entries(self):
        self.lib.root.mkdir(parents=True, exist_ok=True)
        self.lib.path.write_text('{broken', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, '损坏'):
            self.lib.load_all()
        self.lib.path.write_text(json.dumps(dict(schema_version=99, characters=[])), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, '版本不兼容'):
            self.lib.load_all()
        self.lib.path.unlink()
        good = self.lib.create_character('林澈')
        broken = dict(good, id='not-an-id')
        data = json.loads(self.lib.path.read_text(encoding='utf-8'))
        data['characters'].append(broken)
        data['characters'].append(dict(good))  # duplicate id
        self.lib.path.write_text(json.dumps(data), encoding='utf-8')
        groups = self.lib.load_all()
        self.assertEqual([group['id'] for group in groups], [good['id']])

    def test_atomic_write_failure_keeps_previous_index(self):
        group = self.lib.create_character('林澈')
        before = self.lib.path.read_text(encoding='utf-8')
        with patch('timbreforge.voice_library.os.replace', side_effect=OSError('disk error')):
            with self.assertRaises(OSError):
                self.lib.add_voice(group['id'], self.card['id'])
        self.assertEqual(self.lib.path.read_text(encoding='utf-8'), before)
        self.assertEqual([item.name for item in self.lib.root.iterdir() if item.name.startswith('.')], [])


def library_view_with(root):
    with patch('timbreforge.voice_library.get_config',
               return_value=type('C', (), {'characters_dir': root})()):
        return library_view()


if __name__ == '__main__':
    unittest.main()
