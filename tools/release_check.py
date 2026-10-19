"""Check Git source candidates and optional wheel/sdist contents before publishing.

This is a high-confidence credential/asset guard, not a complete security audit.
It prints file names and rule names only, never matching secret values.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
import subprocess
import tarfile
from urllib.parse import unquote, urlparse
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = {'timbreforge', 'tests', 'tools', 'docs', 'examples', 'patches', '.github', 'assets'}
ROOT_FILES = {'.gitignore', '.gitattributes', 'README.md', 'CHANGELOG.md', 'LICENSE',
              'CONTRIBUTING.md', 'SECURITY.md', 'THIRD_PARTY_NOTICES.md', 'MANIFEST.in',
              'pyproject.toml', 'backend-lock.json', 'qwen-backend-lock.json',
              'planner-backend-lock.json', 'timbreforge.example.toml', '启动声线工作室.cmd',
              '安装并启动声线工作室.cmd', 'DEEPSEEK_TASKS.md'}
MAX_BYTES = 2 * 1024 * 1024
PUBLIC_BINARY_ASSETS = {
    'assets/studio-background.png',
    'assets/studio-background-v2.png',
    'assets/social-preview.jpg',
}
PRIVATE_DIRS = {'.git', '.local-data', '.dsh', '.codex', '.agents', '.venv', 'venv', 'private', '__pycache__'}
ASSET_SUFFIXES = {'.wav', '.flac', '.mp3', '.ogg', '.m4a', '.mp4', '.pt', '.pth', '.onnx',
                  '.safetensors', '.ckpt', '.bin', '.doc', '.docx', '.pem', '.key', '.pfx', '.p12', '.log'}
SECRET_PATTERNS = {
    'GitHub credential': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})\b'),
    'API credential': re.compile(r'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,}\b'),
    'AWS access key': re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    'private key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'),
    'personal Windows path': re.compile(r'[A-Za-z]:[\\/]+Users[\\/]+(?!Public\b|your\b|user\b)[^\s\\/]+', re.I),
}
LINK = re.compile(r'!?\[[^\]\n]*\]\((?:<([^>]+)>|([^\s)]+))(?:\s+"[^"]*")?\)')


def forbidden(name):
    path = Path(name)
    return (path.is_absolute() or '..' in path.parts or bool(set(path.parts) & PRIVATE_DIRS) or path.suffix.lower() in ASSET_SUFFIXES
            or path.name.startswith('.env') and path.name != '.env.example'
            or path.name in {'timbreforge.local.toml', 'timbreforge.toml'}
            or path.name.endswith('.local.toml') or path.name.startswith('credentials'))


def inspect_text(name, raw):
    errors = []
    if len(raw) > MAX_BYTES:
        return [f'{name}: exceeds 2 MiB source limit']
    try:
        text = raw.decode('utf-8-sig')
    except UnicodeDecodeError:
        return [f'{name}: non-UTF-8/binary file needs explicit review']
    for label, pattern in SECRET_PATTERNS.items():
        if pattern.search(text):
            errors.append(f'{name}: {label} detected')
    return errors


def archive_source_name(name):
    """Map wheel/sdist members back to source-relative paths.

    Source distributions wrap files under a top-level ``package-version``
    directory, while the public binary allowlist is intentionally source
    relative (for example ``assets/studio-background-v2.png``).
    """
    parts = Path(name).as_posix().split('/')
    if len(parts) > 1 and parts[0].startswith('timbreforge-'):
        return '/'.join(parts[1:])
    return '/'.join(parts)


def check_links(path, root=ROOT):
    errors = []
    for match in LINK.finditer(path.read_text(encoding='utf-8-sig')):
        href = match.group(1) or match.group(2)
        parsed = urlparse(href)
        if parsed.scheme or href.startswith(('#', '//')):
            continue
        target = (path.parent / unquote(parsed.path)).resolve()
        if not target.is_relative_to(root) or not target.exists():
            errors.append(f'{path.relative_to(root)}: missing/unsafe relative link {href}')
    return errors


def source_files(root=ROOT):
    result = subprocess.run(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'],
        cwd=root, capture_output=True, check=True)
    return sorted(set(result.stdout.decode('utf-8').split('\0')) - {''})


def check_source(root=ROOT):
    errors, total = [], 0
    files = source_files(root)
    for name in files:
        path = root / name
        if Path(name).parts[0] not in SOURCE_DIRS and name not in ROOT_FILES:
            errors.append(f'{name}: unexpected source candidate; review before publishing')
            continue
        if forbidden(name) or path.is_symlink() or not path.resolve().is_relative_to(root):
            errors.append(f'{name}: private asset or unsafe path')
            continue
        if not path.is_file():
            errors.append(f'{name}: missing source file')
            continue
        raw = path.read_bytes()
        total += len(raw)
        if name in PUBLIC_BINARY_ASSETS:
            if len(raw) > MAX_BYTES:
                errors.append(f'{name}: exceeds 2 MiB source limit')
            continue
        errors.extend(inspect_text(name, raw))
        if path.suffix.lower() == '.md':
            errors.extend(check_links(path, root))
    return errors, len(files), total


def check_archive(path):
    errors, names = [], []
    if path.suffix == '.whl':
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                names.append(info.filename)
                if forbidden(info.filename):
                    errors.append(f'{info.filename}: private asset in archive')
                if info.file_size > MAX_BYTES:
                    errors.append(f'{info.filename}: oversized archive member')
                elif archive_source_name(info.filename) in PUBLIC_BINARY_ASSETS:
                    continue
                else:
                    errors.extend(inspect_text(info.filename, archive.read(info)))
    else:
        with tarfile.open(path, 'r:*') as archive:
            for member in archive.getmembers():
                if member.isdir():
                    continue
                names.append(member.name)
                if not member.isfile() or forbidden(member.name):
                    errors.append(f'{member.name}: unsafe/private archive member')
                elif member.size > MAX_BYTES:
                    errors.append(f'{member.name}: oversized archive member')
                elif archive_source_name(member.name) in PUBLIC_BINARY_ASSETS:
                    continue
                else:
                    errors.extend(inspect_text(member.name, archive.extractfile(member).read()))
    required = [f'timbreforge/workers/{name}.py' for name in
                ('qwen_worker', 'planner_worker', 'cosyvoice_worker', 'mos_infer', 'campplus_sim')]
    if path.suffix != '.whl':
        required += ['README.md', 'CHANGELOG.md', 'backend-lock.json', 'qwen-backend-lock.json',
                     'planner-backend-lock.json', 'timbreforge.example.toml', '启动声线工作室.cmd',
                     '安装并启动声线工作室.cmd']
    for name in required:
        if not any(entry == name or entry.endswith('/' + name) for entry in names):
            errors.append(f'{path.name}: missing {name}')
    return errors, len(names)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', action='append', type=Path, default=[])
    args = parser.parse_args(argv)
    errors, count, total = check_source()
    print(f'Source: {count} files, {total / 1024:.1f} KiB; checked assets, credentials and relative links')
    for path in args.archive:
        problems, members = check_archive(path)
        errors.extend(problems)
        print(f'Archive: {path.name}, {members} files')
    for error in errors:
        print('FAIL:', error)
    print('PASS' if not errors else f'FAILED: {len(errors)} issue(s)')
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())

