from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from timbreforge.characters import CharacterStore


class CharacterStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "characters"
        self.source = Path(self.temp.name) / "source.wav"
        sf.write(self.source, np.full(2400, 0.08, dtype=np.float32), 24000,
                 subtype="FLOAT")

    def test_create_roundtrip_copies_exact_bytes_and_metadata(self):
        original = self.source.read_bytes()
        store = CharacterStore(self.root)
        record = store.create(
            "Ada",
            self.source,
            reference_text="Hello",
            description="test character",
            backend="cosyvoice3",
            instruction="speak warmly",
            params={"temperature": 0.8, "tags": ["warm"]},
            provenance={"source": "unit-test"},
        )
        cid = record["id"]
        self.assertRegex(cid, r"^[0-9a-f]{32}$")
        self.assertEqual(record["schema_version"], 1)
        self.assertEqual(record["backend"], "cosyvoice3")
        self.assertEqual(Path(record["reference_audio"]).read_bytes(), original)
        self.assertEqual(record["reference_sha256"], hashlib.sha256(original).hexdigest())
        self.assertEqual(store.load(cid), record)
        self.assertEqual(store.list(), [record])

    def test_copy_survives_source_mutation_and_deletion(self):
        expected = self.source.read_bytes()
        store = CharacterStore(self.root)
        record = store.create("Persisted", self.source)
        self.source.write_bytes(b"changed")
        self.source.unlink()
        loaded = store.load(record["id"])
        self.assertEqual(Path(loaded["reference_audio"]).read_bytes(), expected)

    def test_defaults_and_validation(self):
        store = CharacterStore(self.root)
        record = store.create("Default", self.source)
        self.assertEqual(record["backend"], "indextts")
        for kwargs in (
            {"name": ""},
            {"name": "x" * 121},
            {"name": "x", "backend": "other"},
            {"name": "x", "params": {"bad": float("nan")}},
            {"name": "x", "provenance": {"bad": float("inf")}},
        ):
            kwargs.setdefault("reference_audio", self.source)
            with self.assertRaises(ValueError):
                store.create(**kwargs)

    def test_rejects_silent_and_too_long_audio(self):
        silent = Path(self.temp.name) / "silent.wav"
        sf.write(silent, np.zeros(2400, dtype=np.float32), 24000, subtype="FLOAT")
        with self.assertRaisesRegex(ValueError, "silent"):
            CharacterStore(self.root).create("Silent", silent)
        long_audio = Path(self.temp.name) / "long.wav"
        sf.write(long_audio, np.full(31 * 24000, 0.1, dtype=np.float32), 24000,
                 subtype="FLOAT")
        with self.assertRaisesRegex(ValueError, "30"):
            CharacterStore(self.root).create("Long", long_audio)

    def test_load_rejects_invalid_ids_schema_hash_and_unsafe_paths(self):
        store = CharacterStore(self.root)
        record = store.create("Checked", self.source)
        cid = record["id"]
        for invalid in ("", "../" + cid, cid.upper(), "0" * 31, "g" * 32):
            with self.assertRaises((ValueError, FileNotFoundError)):
                store.load(invalid)

        manifest = self.root / cid / "character.json"
        original = json.loads(manifest.read_text(encoding="utf-8"))
        original["schema_version"] = 99
        manifest.write_text(json.dumps(original), encoding="utf-8")
        with self.assertRaises(ValueError):
            store.load(cid)
        original["schema_version"] = 1
        original["reference_sha256"] = "0" * 64
        manifest.write_text(json.dumps(original), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash"):
            store.load(cid)

        original["reference_sha256"] = record["reference_sha256"]
        original["reference_audio"] = "../outside.wav"
        manifest.write_text(json.dumps(original), encoding="utf-8")
        with self.assertRaises(ValueError):
            store.load(cid)

    def test_load_rejects_reference_symlink_escape_and_list_skips_bad_entry(self):
        store = CharacterStore(self.root)
        record = store.create("Unsafe", self.source)
        cid = record["id"]
        ref = Path(record["reference_audio"])
        outside = Path(self.temp.name) / "outside.wav"
        outside.write_bytes(self.source.read_bytes())
        ref.unlink()
        try:
            ref.symlink_to(outside)
        except (OSError, NotImplementedError):
            # Simulate an unsafe resolved reference path on hosts where creating
            # symlinks is not permitted, preserving the load-path assertion.
            with patch.object(Path, "is_symlink", autospec=True,
                              side_effect=lambda path: path == ref):
                with self.assertRaisesRegex(ValueError, "unsafe"):
                    store.load(cid)
            self.assertEqual(store.list(), [])
            return
        with self.assertRaisesRegex(ValueError, "unsafe"):
            store.load(cid)
        self.assertEqual(store.list(), [])

    def test_atomic_failure_does_not_leave_temporary_directories(self):
        store = CharacterStore(self.root)
        with patch("timbreforge.characters.os.rename", side_effect=OSError("publish failed")):
            with self.assertRaisesRegex(OSError, "publish failed"):
                store.create("Bad", self.source)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_invalid_floating_point_empty_and_boundary_audio(self):
        store = CharacterStore(self.root)
        for value in (float("nan"), float("inf"), -float("inf")):
            sf.write(self.source, np.array([0.1, value]), 24000, subtype="FLOAT")
            with self.assertRaisesRegex(ValueError, "finite"):
                store.create("Invalid", self.source)
        sf.write(self.source, np.zeros(0), 24000, subtype="FLOAT")
        with self.assertRaisesRegex(ValueError, "nonempty"):
            store.create("Empty", self.source)
        sf.write(self.source, np.full(30 * 8000, 0.1), 8000, subtype="FLOAT")
        self.assertEqual(store.create("Boundary", self.source)["name"], "Boundary")

    def test_manifest_uses_relative_filename_and_listing_skips_corruption(self):
        store = CharacterStore(self.root)
        valid = store.create("Valid", self.source)
        corrupt = store.create("Corrupt", self.source)
        manifest = self.root / corrupt["id"] / "character.json"
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(metadata["reference_audio"], "reference.wav")
        Path(corrupt["reference_audio"]).write_bytes(b"mutated")
        with self.assertRaisesRegex(ValueError, "hash"):
            store.load(corrupt["id"])
        self.assertEqual(store.list(), [valid])

    def test_json_values_must_not_be_coerced_and_returned_values_are_isolated(self):
        store = CharacterStore(self.root)
        for value in ({1: "integer-key"}, {"tuple": (1, 2)}, {"set": {1}}):
            with self.assertRaises(ValueError):
                store.create("Bad JSON", self.source, params=value)
        cyclic = {}
        cyclic["self"] = cyclic
        with self.assertRaises(ValueError):
            store.create("Cycle", self.source, params=cyclic)
        params = {"nested": {"value": 1}}
        record = store.create("Isolated", self.source, params=params)
        params["nested"]["value"] = 2
        record["params"]["nested"]["value"] = 3
        self.assertEqual(store.load(record["id"])["params"], {"nested": {"value": 1}})

    def test_uuid_collision_never_overwrites_existing_record(self):
        store = CharacterStore(self.root)
        record = store.create("Existing", self.source)
        fixed_uuid = type("FixedUUID", (), {"hex": record["id"]})()
        with patch("timbreforge.characters.uuid.uuid4", return_value=fixed_uuid):
            with self.assertRaises(FileExistsError):
                store.create("Collision", self.source)
        self.assertEqual(store.load(record["id"]), record)
        self.assertEqual(len(list(self.root.iterdir())), 1)


if __name__ == "__main__":
    unittest.main()
