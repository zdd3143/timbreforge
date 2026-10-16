"""UI services are mocked: no model, dataset, server or download needed."""
from __future__ import annotations
import importlib.util
import sys
import types
import unittest
from unittest.mock import Mock, patch
from timbreforge import creation_ui as ui


class CreationCallbackTests(unittest.TestCase):
    def setUp(self):
        self.store = Mock(root='projects')
        self.project = {'id': 'project-id', 'name': 'Scene', 'bindings': {}, 'lines': [
            {'id': 'line-stable', 'line_number': 1, 'speaker': '甲', 'text': '你好',
             'selected_candidate_id': 'candidate-a', 'candidates': [
                 {'id': 'candidate-a', 'file': 'candidates/a.wav', 'seed': 42,
                  'favorite': True, 'issues': ['噪声/失真'], 'note': 'listen',
                  'scores': {'speaker_similarity': None, 'content_accuracy': {'verified': False}, 'quality': None}},
                 {'id': 'candidate-b', 'file': 'candidates/b.wav', 'seed': 43}]}]}
        self.store.list.return_value = [self.project]
        self.store.load.return_value = self.project
        self.patch = patch.object(ui, 'project_store', return_value=self.store)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_choices_are_stable_ids_not_rows(self):
        self.assertEqual(ui.project_choices(), [('Scene · project-id', 'project-id')])
        self.assertEqual(ui.load_project('project-id'), self.project)
        self.store.load.assert_called_once_with('project-id')

    def test_generation_requires_consent_and_only_dispatches_selected_line(self):
        with self.assertRaisesRegex(ValueError, '授权'):
            ui.generate_line('project-id', 'line-stable', confirmed=False)
        self.store.generate_line.assert_not_called()
        ui.generate_line('project-id', 'line-stable', 4, 42, True)
        self.store.generate_line.assert_called_once_with('project-id', 'line-stable', count=4, seed=42, confirmed=True, device='cuda:0')
        for invalid in (0, 5, 1.5, True):
            with self.assertRaises(ValueError):
                ui.generate_line('p', 'l', invalid, 42, True)

    def test_qwen_cpu_device_explicit_and_old_root_position_preserved(self):
        ui.generate_line('p', 'l', 1, 42, True, 'legacy-root', device='cpu')
        ui.project_store.assert_called_with('legacy-root')
        self.store.generate_line.assert_called_once_with('p', 'l', count=1, seed=42, confirmed=True, device='cpu')
        self.store.reset_mock()
        with self.assertRaises(ValueError):
            ui.generate_line('p', 'l', confirmed=True, device='auto')
        self.store.generate_line.assert_not_called()

    def test_review_preserves_stable_candidate_and_manual_fields(self):
        ui.review_candidate('project-id', 'line-stable', 'candidate-a', True, True, ['噪声/失真'], 'note')
        self.store.review.assert_called_once_with('project-id', 'line-stable', 'candidate-a',
            selected=True, favorite=True, issues=['噪声/失真'], note='note')

    def test_assess_is_explicit_and_empty_transcript_unverified(self):
        ui.assess_candidate('project-id', 'line-stable', 'candidate-a')
        self.store.assess.assert_called_once_with('project-id', 'line-stable', 'candidate-a',
            observed_text=None, run_speaker=False, run_quality=False)

    def test_compare_uses_relative_candidate_paths_and_returns_listening_files(self):
        module = types.ModuleType('timbreforge.candidates')
        module.compare_audio = Mock(return_value={'a': 'listen-a.wav', 'b': 'listen-b.wav'})
        with patch.dict(sys.modules, {'timbreforge.candidates': module}):
            result = ui.compare_candidates(self.project, 'line-stable', 'candidate-a', 'candidate-b', True)
        args, kwargs = module.compare_audio.call_args
        self.assertTrue(args[0].replace('\\', '/').endswith('projects/project-id/candidates/a.wav'))
        self.assertTrue(args[1].replace('\\', '/').endswith('projects/project-id/candidates/b.wav'))
        self.assertEqual(kwargs, {'match_volume': True})
        self.assertEqual(result['a'], 'listen-a.wav')

    def test_candidate_paths_are_reloaded_and_contained(self):
        stale = {'id': 'project-id', 'lines': [{'id': 'line-stable', 'candidates': [
            {'id': 'candidate-a', 'file': '../../tampered.wav'}]}]}
        path = ui.candidate_path(stale, 'line-stable', 'candidate-a')
        self.assertTrue(path.replace('\\', '/').endswith('/project-id/candidates/a.wav'))
        self.project['lines'][0]['candidates'][0]['file'] = '../../escaped.wav'
        with self.assertRaisesRegex(ValueError, '项目目录'):
            ui.candidate_path(stale, 'line-stable', 'candidate-a')

    def test_reference_processing_needs_explicit_confirmation(self):
        module = types.ModuleType('timbreforge.reference_assistant')
        module.prepare_confirmed_reference = Mock(return_value={'output_path': 'new.wav'})
        with patch.dict(sys.modules, {'timbreforge.reference_assistant': module}):
            with self.assertRaises(ValueError):
                ui.prepare_reference('raw.wav')
            module.prepare_confirmed_reference.assert_not_called()
            result = ui.prepare_reference('raw.wav', True, 1, 3, True, 'exact text', True)
        module.prepare_confirmed_reference.assert_called_once_with('raw.wav', confirmed=True,
            start_seconds=1, end_seconds=3, mono=True, output_dir=None,
            transcript='exact text', transcript_confirmed=True)
        self.assertEqual(result['output_path'], 'new.wav')

    @unittest.skipUnless(importlib.util.find_spec('gradio'), 'Gradio not installed')
    def test_build_is_lazy_and_controls_use_separate_score_axes(self):
        import gradio as gr
        with patch.object(ui, 'project_store', side_effect=AssertionError('no project read')), \
             patch.object(ui, 'character_choices', side_effect=AssertionError('no character read')), \
             patch.object(ui, 'inspect_reference', side_effect=AssertionError('no audio read')), \
             patch('requests.sessions.Session.request', side_effect=AssertionError('no network')), \
             patch('httpx.Client.send', side_effect=AssertionError('no network')):
            with gr.Blocks() as demo:
                ui.build_creation_tabs(gr, {'concurrency_limit': 1, 'concurrency_id': 'timbreforge-synthesis'})
        config = demo.get_config_file()
        labels = [c['props'].get('label', '') for c in config['components']]
        self.assertIn('多角色项目 / 候选比较', labels)
        self.assertIn('参考音频助手', labels)
        for axis in ('声线相似度（独立轴）', '内容准确度（独立轴）', '音频质量（独立轴）'):
            self.assertIn(axis, labels)
        self.assertNotIn('综合分', labels)
        components = {c['props'].get('elem_id'): c for c in config['components']}
        self.assertEqual(components['create_script']['props']['value'], ui.DEFAULT_SCRIPT)
        self.assertEqual(components['create_ref']['props']['sources'], ['upload'])
        self.assertIsNone(components['create_ref']['props'].get('format'))
        self.assertFalse(components['create_ref']['props']['editable'])
        self.assertIn('create_play_a', components)
        self.assertIn('create_play_b', components)


if __name__ == '__main__':
    unittest.main()
