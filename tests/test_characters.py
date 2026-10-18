from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
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
        self.assertEqual(record["schema_version"], 2)
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
        original["schema_version"] = 2
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


class ExtendedCharacterCardTests(unittest.TestCase):
    """Schema 2: VoiceRecipe, audio history, clone prompt pointer and versions."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "characters"
        self.source = Path(self.temp.name) / "source.wav"
        sf.write(self.source, np.full(24000, 0.08, dtype=np.float32), 24000, subtype="FLOAT")
        self.store = CharacterStore(self.root)

    def card_dir(self, card):
        return Path(card["reference_audio"]).parent

    def test_schema_v1_card_is_readable_and_migrated_in_memory(self):
        record = self.store.create("Legacy", self.source, reference_text="旧台词",
                                   backend="qwen3",
                                   params={"qwen": {"seed": 123}},
                                   provenance={"kind": "qwen_voice_design",
                                               "design_prompt": "冷淡女声"})
        manifest = self.card_dir(record) / "character.json"
        v1_fields = {"schema_version", "id", "name", "reference_audio", "reference_sha256",
                     "reference_text", "description", "backend", "instruction", "params",
                     "provenance", "created_at"}
        data = json.loads(manifest.read_text(encoding="utf-8"))
        legacy = {key: data[key] for key in v1_fields}
        legacy["schema_version"] = 1
        manifest.write_text(json.dumps(legacy), encoding="utf-8")

        card = self.store.load(record["id"])
        self.assertEqual(card["schema_version"], 2)
        self.assertIsNone(card["canonical_audio"])
        self.assertIsNone(card["clone_prompt"])
        self.assertEqual(card["clone_mode"], "icl")
        self.assertEqual(card["version"], 1)
        self.assertEqual(card["updated_at"], card["created_at"])
        self.assertEqual(card["voice_design_prompt"], "冷淡女声")
        self.assertEqual(card["seed"], 123)
        self.assertEqual(card["source"], "qwen_voice_design")
        self.assertEqual(card["model_name"], "Qwen3-TTS-12Hz-0.6B-Base")
        from timbreforge.voice_recipe import empty_recipe
        self.assertEqual(card["voice_recipe"], empty_recipe())
        self.assertEqual(card["audio_edit_chain"], [])

    def test_extended_fields_round_trip(self):
        from timbreforge.voice_recipe import RECIPE_EXAMPLES
        recipe = RECIPE_EXAMPLES["cold_anime_girl"]["recipe"]
        card = self.store.create(
            "冷音", self.source, reference_text="测试台词。", backend="qwen3",
            voice_design_prompt="高音、轻薄、冷淡。", voice_recipe=recipe,
            source="voice_design/simple_generation", tags=["二次元", "冷淡", "二次元"],
            notes="本机测试角色。", model_name="Qwen3-TTS-12Hz-0.6B-Base",
            model_revision="local-2026-10", seed=12345)
        loaded = self.store.load(card["id"])
        self.assertEqual(loaded["voice_design_prompt"], "高音、轻薄、冷淡。")
        self.assertEqual(loaded["voice_recipe"], recipe)
        self.assertEqual(loaded["tags"], ["二次元", "冷淡"])
        self.assertEqual(loaded["notes"], "本机测试角色。")
        self.assertEqual(loaded["seed"], 12345)
        self.assertEqual(loaded["source"], "voice_design/simple_generation")
        self.assertEqual(loaded["model_revision"], "local-2026-10")
        self.assertEqual(loaded["version"], 1)
        self.assertEqual(len(loaded["reference_text_sha256"]), 64)
        self.assertEqual(loaded, self.store.load(card["id"]))

    def test_invalid_extended_fields_are_rejected(self):
        with self.assertRaises(ValueError):
            self.store.create("Bad recipe", self.source,
                              voice_recipe={"timbre": {"pitch": "ultra"}})
        with self.assertRaises(ValueError):
            self.store.create("Bad seed", self.source, seed=1.5)
        with self.assertRaises(ValueError):
            self.store.create("Bad tags", self.source, tags=[""])
        with self.assertRaises(ValueError):
            self.store.create("Bad mode", self.source, clone_mode="magic")

    def test_clone_prompt_pointer_and_voice_changes_invalidate_it(self):
        card = self.store.create("缓存", self.source, reference_text="原始台词。",
                                 backend="qwen3", clone_mode="xvector")
        directory = self.card_dir(card)
        (directory / "prompts").mkdir()
        (directory / "prompts" / "clone-xvector.pt").write_bytes(b"tensors")
        card = self.store.set_clone_prompt(card["id"], "prompts/clone-xvector.pt", "xvector")
        self.assertEqual(Path(card["clone_prompt"]).name, "clone-xvector.pt")
        self.assertEqual(card["version"], 1)
        with self.assertRaises(ValueError):
            self.store.set_clone_prompt(card["id"], "reference.wav", "icl")

        updated = self.store.update(card["id"], reference_text="改过的台词。")
        self.assertIsNone(updated["clone_prompt"])
        self.assertEqual(updated["version"], 2)
        reasons = updated["provenance"]["clone_prompt_invalidations"]
        self.assertEqual(len(reasons), 1)
        self.assertIn("逐字稿", reasons[0]["reason"])

    def test_clone_mode_switch_bumps_version_and_invalidates(self):
        card = self.store.create("模式", self.source, backend="qwen3", clone_mode="icl")
        (self.card_dir(card) / "prompts").mkdir()
        (self.card_dir(card) / "prompts" / "a.pt").write_bytes(b"x")
        card = self.store.set_clone_prompt(card["id"], "prompts/a.pt", "icl")
        switched = self.store.update(card["id"], clone_mode="xvector")
        self.assertEqual(switched["clone_mode"], "xvector")
        self.assertEqual(switched["version"], 2)
        self.assertIsNone(switched["clone_prompt"])

    def test_notes_and_tags_do_not_bump_voice_version(self):
        card = self.store.create("备注", self.source)
        updated = self.store.update(card["id"], notes="补充说明", tags=["测试"])
        self.assertEqual(updated["version"], 1)
        self.assertEqual(updated["notes"], "补充说明")
        self.assertNotEqual(updated["updated_at"], card["updated_at"])

    def test_canonical_audio_document_and_revert(self):
        from timbreforge.audio_edit_schema import make_edit_document
        card = self.store.create("编辑", self.source, reference_text="台词。")
        directory = self.card_dir(card)
        (directory / "audio").mkdir()
        canonical = directory / "audio" / "canonical-test.wav"
        shutil.copyfile(self.source, canonical)
        digest = hashlib.sha256(canonical.read_bytes()).hexdigest()
        document = make_edit_document(
            source_audio="reference.wav", source_hash=card["reference_sha256"],
            operations=[{"name": "normalize"}], output_audio="audio/canonical-test.wav",
            output_hash=digest, preset="clean_voice", sample_rate=24000, channels=1,
            duration_in=1.0, duration_out=1.0)
        edited = self.store.apply_audio_document(card["id"], document, preset="clean_voice")
        self.assertEqual(Path(edited["canonical_audio"]).name, "canonical-test.wav")
        self.assertEqual(edited["canonical_sha256"], digest)
        self.assertEqual(edited["version"], 2)
        self.assertEqual(len(edited["audio_edit_chain"]), 1)
        self.assertEqual(edited["audio_edit_chain"][0]["preset"], "clean_voice")

        reverted = self.store.revert_audio(card["id"])
        self.assertIsNone(reverted["canonical_audio"])
        self.assertEqual(reverted["version"], 3)
        self.assertEqual(reverted["audio_edit_chain"][-1]["event"], "revert_to_original")
        # The original recording is never touched by either operation.
        self.assertEqual(Path(reverted["reference_audio"]).read_bytes(),
                         self.source.read_bytes())

    def test_canonical_hash_mismatch_is_rejected(self):
        card = self.store.create("校验", self.source)
        directory = self.card_dir(card)
        (directory / "audio").mkdir()
        shutil.copyfile(self.source, directory / "audio" / "canonical.wav")
        manifest = directory / "character.json"
        data = json.loads(manifest.read_text(encoding="utf-8"))
        data["canonical_audio"] = "audio/canonical.wav"
        data["canonical_sha256"] = "0" * 64
        manifest.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash"):
            self.store.load(card["id"])

    def test_missing_clone_prompt_file_is_stale_state_not_corruption(self):
        card = self.store.create("丢失", self.source, backend="qwen3")
        directory = self.card_dir(card)
        (directory / "prompts").mkdir()
        (directory / "prompts" / "gone.pt").write_bytes(b"x")
        card = self.store.set_clone_prompt(card["id"], "prompts/gone.pt", "icl")
        (directory / "prompts" / "gone.pt").unlink()
        loaded = self.store.load(card["id"])
        self.assertIsNone(loaded["clone_prompt"])
        self.assertEqual(loaded["version"], card["version"])

    def test_delete_removes_derived_files_and_audio_history(self):
        from timbreforge.audio_edit_schema import make_edit_document
        card = self.store.create("删除", self.source)
        directory = self.card_dir(card)
        (directory / "audio").mkdir()
        (directory / "prompts").mkdir()
        shutil.copyfile(self.source, directory / "audio" / "canonical.wav")
        digest = hashlib.sha256((directory / "audio" / "canonical.wav").read_bytes()).hexdigest()
        document = make_edit_document(
            source_audio="reference.wav", source_hash=card["reference_sha256"],
            operations=[{"name": "trim_silence"}], output_audio="audio/canonical.wav",
            output_hash=digest, sample_rate=24000, channels=1,
            duration_in=1.0, duration_out=0.9)
        self.store.apply_audio_document(card["id"], document)
        (directory / "prompts" / "c.pt").write_bytes(b"x")
        self.store.set_clone_prompt(card["id"], "prompts/c.pt", "icl")
        removed = self.store.delete(card["id"])
        self.assertEqual(removed["name"], "删除")
        self.assertEqual(list(self.root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()

