"""Read-only SHA-256 verification of separately installed backend assets.

Never imports a backend, downloads weights, or applies patches. Paths in the lock
are relative to explicitly supplied local roots, not private developer paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(root: Path, entries: list[dict]) -> list[str]:
    errors = []
    for entry in entries:
        relative = PurePosixPath(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative) or ":" in str(relative):
            errors.append(f"unsafe manifest path: {relative}")
            continue
        # Explicitly named root junctions/caches may resolve outside root. The
        # manifest still names only relative asset paths, never junction targets.
        path = root.joinpath(*relative.parts)
        if not path.is_file():
            errors.append(f"missing: {relative}")
        elif path.stat().st_size != entry["bytes"]:
            errors.append(f"size mismatch: {relative}")
        elif sha256(path) != entry["sha256"]:
            errors.append(f"SHA256 mismatch: {relative}")
    return errors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=Path(__file__).resolve().parents[1] / "backend-lock.json")
    parser.add_argument("--index-node", type=Path)
    parser.add_argument("--index-model", type=Path)
    parser.add_argument("--cosy-source", type=Path)
    parser.add_argument("--cosy-model", type=Path)
    args = parser.parse_args(argv)
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    index, cosy = lock["backends"]["indextts25"], lock["backends"]["cosyvoice3"]
    groups = [("release patch", args.lock.resolve().parent, [index["patch"]]),
              ("Index modified source", args.index_node, index["modified_source_files"]),
              ("Index model assets", args.index_model, index["model"]["files"]),
              ("Cosy source snapshot", args.cosy_source, cosy["source_files"]),
              ("Cosy model assets", args.cosy_model, cosy["model"]["files"])]
    failures = 0
    for label, root, entries in groups:
        if root is None:
            print(f"SKIP {label}: supply its local root explicitly")
            continue
        errors = verify(root, entries)
        failures += len(errors)
        print(f"{'FAIL' if errors else 'PASS'} {label}: {len(entries)} pinned files")
        for error in errors:
            print(f"  {error}")
    print("Byte identity only: not proof of upstream commit, dependency resolution, live service reload, or inference quality.")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Audit failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
