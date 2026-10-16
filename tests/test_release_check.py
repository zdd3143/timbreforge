"""Release guards use synthetic credentials and temporary files only."""
from pathlib import Path
import tempfile
import unittest
import zipfile

from tools import release_check as release


class ReleaseCheckTests(unittest.TestCase):
    def test_private_asset_paths_and_safe_sources(self):
        for name in ('.local-data/plan.json', '.env', 'timbreforge.local.toml', 'model.safetensors',
                     'tests/fixtures/private.wav', 'credentials.json', 'voice.docx', 'secret.pem'):
            self.assertTrue(release.forbidden(name), name)
        for name in ('timbreforge/config.py', 'timbreforge.example.toml', 'README.md', '.env.example'):
            self.assertFalse(release.forbidden(name), name)

    def test_credential_guard_never_displays_matched_value(self):
        secret = 'ghp_' + 'a' * 36
        errors = release.inspect_text('source.py', secret.encode())
        self.assertEqual(len(errors), 1)
        self.assertNotIn(secret, errors[0])
        self.assertIn('GitHub credential', errors[0])
        self.assertTrue(release.inspect_text('binary.py', b'\xff\x00'))

    def test_markdown_links_are_relative_and_exist(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            source = root / 'README.md'
            (root / 'guide.md').touch()
            source.write_text('[valid](guide.md#anchor) [web](https://example.com) [missing](absent.md)', encoding='utf-8')
            errors = release.check_links(source, root)
            self.assertEqual(len(errors), 1)
            self.assertIn('absent.md', errors[0])
            source.write_text('[escape](../elsewhere.md)', encoding='utf-8')
            self.assertTrue(release.check_links(source, root))

    def test_archive_requires_workers_and_rejects_private_assets(self):
        with tempfile.TemporaryDirectory() as td:
            archive = Path(td) / 'example.whl'
            workers = ('qwen_worker', 'planner_worker', 'cosyvoice_worker', 'mos_infer', 'campplus_sim')
            with zipfile.ZipFile(archive, 'w') as stream:
                for name in workers:
                    stream.writestr(f'timbreforge/workers/{name}.py', '# test\n')
            self.assertEqual(release.check_archive(archive)[0], [])
            with zipfile.ZipFile(archive, 'a') as stream:
                stream.writestr('.local-data/profile.json', '{}')
            self.assertIn('private asset', release.check_archive(archive)[0][0])


if __name__ == '__main__':
    unittest.main()
