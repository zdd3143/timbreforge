"""New pages build without model I/O; key helpers stay pure and testable."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from timbreforge import admin_ui, character_ui, clone_prompts, library_ui, ui_tasks, voice_ui
from timbreforge.characters import CharacterStore
from timbreforge.model_registry import ModelRegistry
from timbreforge.voice_recipe import empty_recipe, normalize


def tone(path, *, seconds=0.4, sample_rate=24000, frequency=180.0):
    time_axis = np.arange(int(seconds * sample_rate)) / sample_rate
    sf.write(str(path), 0.2 * np.sin(2 * np.pi * frequency * time_axis), sample_rate,
             format='WAV', subtype='FLOAT')
    return Path(path)


class HelperTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.store = CharacterStore(self.base / 'characters')
        self.source = tone(self.base / 'voice.wav')

    def test_create_character_requires_transcript_and_audio(self):
        with self.assertRaises(ValueError):
            character_ui.create_character('角色', '描述', '', '', 'default', '', '', str(self.source),
                                          store=self.store)
        with self.assertRaises(ValueError):
            character_ui.create_character('角色', '描述', '台词。', '', 'default', '', '', None,
                                          store=self.store)
        result = character_ui.create_character(
            '角色', '描述', '准确台词。', '冷淡的二次元少女音，语速偏慢。', 'auto',
            '二次元, 冷淡', '备注', str(self.source), store=self.store)
        card = result['card']
        self.assertEqual(card['name'], '角色')
        self.assertEqual(card['reference_text'], '准确台词。')
        self.assertEqual(card['backend'], 'qwen3')
        self.assertEqual(card['source'], 'import/manual_audio')
        self.assertEqual(card['tags'], ['二次元', '冷淡'])
        self.assertEqual(card['voice_recipe']['identity']['style'], 'anime')
        self.assertIn('关键词', result['note'])
        self.assertIn('Clone Prompt', character_ui.character_markdown(card, store=self.store))
        self.assertIn('实验性', admin_ui.vc_markdown())

    def test_recipe_builder_and_labels(self):
        self.assertEqual(voice_ui.build_recipe({}), empty_recipe())
        recipe = voice_ui.build_recipe({'pitch': 'high', 'pace': 'slow', 'emotion': 'cold'})
        self.assertEqual(recipe['timbre']['pitch'], 'high')
        self.assertEqual(recipe['delivery']['pace'], 'slow')
        self.assertEqual(recipe['delivery']['emotion'], 'cold')
        with self.assertRaises(ValueError):
            voice_ui.build_recipe({'pitch': 'ultra'})
        with self.assertRaises(ValueError):
            voice_ui.build_recipe({'not_a_field': 'x'})
        self.assertEqual(voice_ui.field_choices('pitch')[0][0], '中等')
        self.assertIn('配方', voice_ui.recipe_markdown(normalize({})))

    def test_library_export_duplicate_import_flow(self):
        card = self.store.create('库角色', self.source, reference_text='台词。', backend='qwen3')
        original_snapshot = (lambda: self.store.list())
        with patch.object(library_ui, '_store', side_effect=lambda: self.store):
            exported = library_ui.export_role(card['id'])
            self.assertTrue(Path(exported['path']).is_file())
            duplicate = library_ui.duplicate_role(card['id'])
            self.assertNotEqual(duplicate['id'], card['id'])
            imported = library_ui.import_role(exported['path'], name='导入角色')
            self.assertEqual(imported['name'], '导入角色')
            cards, choices, value, selected = library_ui.snapshot(card['id'])
            self.assertEqual(len(cards), 3)
            self.assertEqual(selected['id'], card['id'])
            self.assertIn('库角色', library_ui._card_html(cards))
            self.assertIn('角色卡', library_ui._details(selected))
        self.assertEqual(len(library_ui.CharacterStore(self.store.root).list()), 3)
        self.assertEqual(len(original_snapshot()), 3)

    def test_library_export_never_overwrites(self):
        card = self.store.create('重复', self.source, reference_text='台词。', backend='qwen3')
        target = self.base / 'fixed.zip'
        target.write_bytes(b'keep')
        with patch.object(library_ui, '_store', side_effect=lambda: self.store):
            with self.assertRaises(FileExistsError):
                library_ui.export_role(card['id'], target)
        self.assertEqual(target.read_bytes(), b'keep')

    def test_model_management_helpers(self):
        model = self.base / 'model'
        model.mkdir()
        (model / 'config.json').write_text('{}')
        (model / 'model.safetensors').write_bytes(b'weights')
        registry = ModelRegistry(self.base / 'registry')
        with patch.object(admin_ui, 'registry', side_effect=lambda: registry):
            message = admin_ui.register_model('测试模型', str(model), 'v1', 'tts_base')
            self.assertIn('已登记', message)
            rows = admin_ui.model_rows()
            self.assertEqual(len(rows), 1)
            entry_id = admin_ui.entry_choices()[0][1]
            self.assertIn('完整', admin_ui.verify_model(entry_id))
            self.assertIn('移除', admin_ui.remove_model(entry_id))
            self.assertEqual(admin_ui.model_rows(), [])
        self.assertIn('本机路径', admin_ui.settings_markdown())
        self.assertIn('实验性', admin_ui.vc_markdown())


class UiHandlerTests(unittest.TestCase):
    """Handlers are exercised with an immediate heavy-task runner, never a model."""

    class _Context:
        def __init__(self):
            import threading
            self.cancel_event = threading.Event()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.store = CharacterStore(self.base / 'characters')
        self.source = tone(self.base / 'voice.wav')
        self.card = self.store.create('处理器角色', self.source, reference_text='台词。',
                                      backend='qwen3')
        self.context = self._Context()

    def runner(self, kind, task, **kwargs):
        return task(self.context)

    def test_synthesize_wires_generation_result(self):
        result = {'file': 'a.wav', 'metadata_file': 'a.json', 'duration': 1.2,
                  'sample_rate': 24000, 'text': '台词。', 'seed': 3,
                  'role_id': self.card['id'], 'role_name': '处理器角色', 'role_version': 1,
                  'model_name': 'Qwen3-TTS-12Hz-0.6B-Base', 'params': {'seed': 3},
                  'clone_prompt': {'status': 'valid', 'reason': ''},
                  'warnings': ['提示'], 'audio': {'sha256': 'a' * 64}}
        with patch.object(ui_tasks, 'run_heavy', side_effect=self.runner), \
             patch.object(voice_ui.generation, 'generate_for_character', return_value=result):
            audio, details = voice_ui.synthesize(self.card['id'], '台词。', 3, 'cuda:0',
                                                  store=self.store)
        self.assertEqual(audio, 'a.wav')
        self.assertIn('处理器角色', details)
        self.assertIn('seed', details)

    def test_long_text_wires_report(self):
        payload = {'file': 'long.wav', 'report_file': 'long.json',
                   'segments': [{'text': 'a'}, {'text': 'b'}],
                   'analysis': {'problems': []}, 'warnings': ['注意衔接']}
        with patch.object(ui_tasks, 'run_heavy', side_effect=self.runner), \
             patch('timbreforge.longform.generate_longform', return_value=payload):
            audio, details = voice_ui.generate_long_text(self.card['id'], '长文本。', 60,
                                                          25, 'cuda:0', store=self.store)
        self.assertEqual(audio, 'long.wav')
        self.assertIn('分段数：2', details)
        self.assertIn('注意衔接', details)

    def test_apply_preset_handler_reports_metrics(self):
        payload = {'card': self.card, 'document': {}, 'preview': {'path': 'preview.ogg'},
                   'metrics': {'lufs_in': -23.0, 'lufs_out': -16.0,
                               'peak_dbfs_out': -1.0, 'clipped_samples_out': 0},
                   'notes': ['处理说明'], 'summary': '清晰人声', 'output_path': 'canonical.wav'}
        with patch.object(ui_tasks, 'run_heavy', side_effect=self.runner), \
             patch.object(character_ui.character_audio, 'render_preset', return_value=payload):
            audio, message = character_ui.apply_preset(self.card['id'], 'clean_voice',
                                                        store=self.store)
        self.assertEqual(audio, 'preview.ogg')
        self.assertIn('清晰人声', message)
        self.assertIn('-16.0 LUFS', message)

    def test_make_clone_prompt_handler_reports_status(self):
        report = {'ok': True, 'status': 'valid', 'reasons': [], 'metadata': {},
                  'path': 'p.pt', 'card': self.card}
        with patch.object(ui_tasks, 'run_heavy', side_effect=self.runner), \
             patch.object(clone_prompts, 'create_clone_prompt',
                          return_value={'created': True}), \
             patch.object(clone_prompts, 'validate_clone_prompt', return_value=report):
            message = character_ui.make_clone_prompt(self.card['id'], store=self.store)
        self.assertIn('已创建', message)
        self.assertIn('哈希', message)


class UiBuildTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('gradio'), 'Gradio not installed')
    def test_new_pages_build_without_model_calls(self):
        from timbreforge import webui
        with (patch('timbreforge.backend_qwen.design', side_effect=AssertionError('no model')),
              patch('timbreforge.backend_qwen.clone', side_effect=AssertionError('no model')),
              patch('timbreforge.clone_prompts.create_clone_prompt',
                    side_effect=AssertionError('no model'))):
            demo = webui.build_ui()
        config = demo.get_config_file()
        ids = {component.get('props', {}).get('elem_id') for component in config['components']}
        self.assertTrue({'character_library_cards', 'library_audition', 'library_export',
                         'library_import_file', 'create_audio', 'create_transcript',
                         'audio_preset', 'audio_apply', 'tuning_generate', 'tuning_compare',
                         'tuning_promote', 'generate_text', 'long_text', 'long_run',
                         'models_table', 'models_register', 'training_manifest',
                         'training_prepare'} <= ids)
        tabs = [component['props']['label'] for component in config['components']
                if component['type'] == 'tabitem']
        for label in ('角色库', '创建角色', '音频修整', '声线微调', '文字生成', '长文本生成',
                      '模型管理', '设置与任务', '高级训练'):
            self.assertIn(label, tabs)
        demo.close()

    def test_ui_task_rows_do_not_require_a_model(self):
        rows = ui_tasks.job_rows(limit=5)
        self.assertIsInstance(rows, list)
        self.assertIn('内存', ui_tasks.memory_summary())


if __name__ == '__main__':
    unittest.main()

