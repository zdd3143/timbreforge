"""Argument routing without importing model services or running subprocesses."""
from __future__ import annotations
import contextlib
import io
import json
import sys
import types
import unittest
from unittest.mock import Mock, patch
from timbreforge.creation_cli import main


class CreationCLITests(unittest.TestCase):
    def setUp(self):
        self.store = Mock()
        for method in ('create', 'load', 'bind', 'generate_line', 'review', 'assess', 'export'):
            getattr(self.store, method).return_value = {'id': 'p'}
        self.store.list.return_value = [{'id': 'p'}]
        self.projects = types.ModuleType('timbreforge.projects')
        self.projects.ProjectStore = Mock(return_value=self.store)
        self.reference = types.ModuleType('timbreforge.reference_assistant')
        self.reference.inspect_reference = Mock(return_value={'duration_seconds': 2})
        self.reference.prepare_confirmed_reference = Mock(return_value={'output_path': 'new.wav'})
        self.modules = patch.dict(sys.modules, {'timbreforge.projects': self.projects,
            'timbreforge.reference_assistant': self.reference})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def run_cli(self, args):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(args), 0)
        return json.loads(output.getvalue())

    def test_create_list_show_and_root_on_either_side(self):
        self.run_cli(['--root', 'chosen', 'create', '--name', 'Scene', '--script', '甲：你好'])
        self.projects.ProjectStore.assert_called_with('chosen')
        self.store.create.assert_called_once_with('Scene', '甲：你好')
        self.assertEqual(self.run_cli(['list', '--root', 'chosen']), [{'id': 'p'}])
        self.run_cli(['show', '--project', 'stable-id'])
        self.store.load.assert_called_with('stable-id')

    def test_config_processed_before_side_effects(self):
        with patch('timbreforge.config.set_config_path') as set_path, patch('timbreforge.config.get_config') as get_config:
            self.run_cli(['--config', 'portable.toml', 'list'])
        set_path.assert_called_once_with('portable.toml')
        get_config.assert_called_once_with()

    def test_generate_requires_rights_before_store_constructed(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main(['generate', '--project', 'p', '--line', 'stable-line'])
        self.assertEqual(error.exception.code, 1)
        self.projects.ProjectStore.assert_not_called()
        self.run_cli(['generate', '--project', 'p', '--line', 'stable-line', '--count', '4', '--seed', '12', '--confirm-rights'])
        self.store.generate_line.assert_called_once_with('p', 'stable-line', count=4, seed=12, confirmed=True, device='cuda:0')

    def test_generate_passes_explicit_cpu_device(self):
        self.run_cli(['generate', '--project', 'p', '--line', 'l', '--device', 'cpu', '--confirm-rights'])
        self.store.generate_line.assert_called_once_with('p', 'l', count=1, seed=42, confirmed=True, device='cpu')
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(['generate', '--project', 'p', '--line', 'l', '--device', 'auto', '--confirm-rights'])

    def test_review_and_assess_preserve_stable_ids_and_explicit_flags(self):
        self.run_cli(['review', '--project', 'p', '--line', 'l', '--candidate', 'c',
                      '--selected', '--no-favorite', '--issue', 'noise', '--note', 'retry'])
        self.store.review.assert_called_once_with('p', 'l', 'c', selected=True, favorite=False, issues=['noise'], note='retry')
        self.run_cli(['assess', '--project', 'p', '--line', 'l', '--candidate', 'c'])
        self.store.assess.assert_called_once_with('p', 'l', 'c', observed_text=None, run_speaker=False, run_quality=False)

    def test_bind_export_and_clear_issues(self):
        self.run_cli(['bind', '--project', 'p', '--speaker', '甲', '--character', 'card-id'])
        self.store.bind.assert_called_once_with('p', '甲', 'card-id')
        self.run_cli(['export', '--project', 'p', '--gap-seconds', '.4'])
        self.store.export.assert_called_once_with('p', gap_seconds=.4)
        self.run_cli(['review', '--project', 'p', '--line', 'l', '--candidate', 'c', '--clear-issues'])
        self.assertEqual(self.store.review.call_args.kwargs['issues'], [])

    def test_reference_commands_are_local_and_preparation_is_explicit(self):
        self.run_cli(['inspect-reference', '--path', 'raw.wav', '--backend', 'cosyvoice3', '--transcript', 'words'])
        self.reference.inspect_reference.assert_called_once_with('raw.wav', transcript='words', backend='cosyvoice3', instruction='')
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(['prepare-reference', '--path', 'raw.wav'])
        self.reference.prepare_confirmed_reference.assert_not_called()
        self.run_cli(['prepare-reference', '--path', 'raw.wav', '--confirm-processing', '--start-seconds', '1',
                      '--end-seconds', '3', '--mono', '--transcript', 'words', '--confirm-transcript'])
        self.reference.prepare_confirmed_reference.assert_called_once_with('raw.wav', confirmed=True,
            start_seconds=1., end_seconds=3., mono=True, output_dir=None, transcript='words', transcript_confirmed=True)

    def test_service_errors_are_clean_without_traceback(self):
        self.store.load.side_effect = ValueError('unknown project')
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
            main(['show', '--project', 'missing'])
        self.assertEqual(error.exception.code, 1)
        self.assertIn('ValueError: unknown project', stderr.getvalue())
        self.assertNotIn('Traceback', stderr.getvalue())


if __name__ == '__main__':
    unittest.main()
