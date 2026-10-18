"""Release guards use synthetic credentials and temporary files only."""
from pathlib import Path
import io
import tempfile
import tarfile
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

    def test_sdist_allows_public_binary_assets_under_version_prefix(self):
        with tempfile.TemporaryDirectory() as td:
            archive = Path(td) / 'example.tar.gz'
            workers = ('qwen_worker', 'planner_worker', 'cosyvoice_worker', 'mos_infer', 'campplus_sim')
            with tarfile.open(archive, 'w:gz') as stream:
                for name in workers:
                    payload = b'# test\n'
                    info = tarfile.TarInfo(f'timbreforge-0.6.0/{"timbreforge/workers/" + name + ".py"}')
                    info.size = len(payload)
                    stream.addfile(info, io.BytesIO(payload))
                payload = b'\x89PNG\r\n\x00\xff'
                info = tarfile.TarInfo('timbreforge-0.6.0/assets/studio-background-v2.png')
                info.size = len(payload)
                stream.addfile(info, io.BytesIO(payload))
                for name, payload in {
                    'README.md': b'# test\n', 'CHANGELOG.md': b'# test\n',
                    'backend-lock.json': b'{}', 'qwen-backend-lock.json': b'{}',
                    'planner-backend-lock.json': b'{}',
                    'timbreforge.example.toml': b'[paths]\n',
                    '启动声线工作室.cmd': b'@echo off\r\n',
                    '安装并启动声线工作室.cmd': b'@echo off\r\n',
                }.items():
                    info = tarfile.TarInfo(f'timbreforge-0.6.0/{name}')
                    info.size = len(payload)
                    stream.addfile(info, io.BytesIO(payload))
            self.assertEqual(release.check_archive(archive)[0], [])


if __name__ == '__main__':
    unittest.main()

