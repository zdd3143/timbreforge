"""Persistent, model-independent storage for voice characters.

Each character owns an exact byte-for-byte copy of its validated reference
recording and a JSON manifest. No embeddings or model state are created here.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from timbreforge.config import get_config

_SCHEMA_VERSION = 1
_BACKENDS = frozenset({"indextts", "cosyvoice3", "qwen3"})
_ID_RE = re.compile(r"[0-9a-f]{32}\Z")
_HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
_MANIFEST_NAME = "character.json"
_METADATA_FIELDS = frozenset({
    "schema_version", "id", "name", "reference_audio", "reference_sha256",
    "reference_text", "description", "backend", "instruction", "params",
    "provenance", "created_at",
})


def _validate_json_value(value: Any, path: str) -> None:
    """Reject values JSON would silently coerce or encode non-portably."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite numbers")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} keys must be strings")
            _validate_json_value(item, f"{path}.{key}")
        return
    raise ValueError(f"{path} contains a non-JSON value: {type(value).__name__}")


def _validate_audio(path: Path) -> None:
    """Ensure the recording is decodable, finite, audible, and at most 30 s."""
    try:
        with sf.SoundFile(str(path)) as recording:
            if recording.frames <= 0 or recording.channels <= 0:
                raise ValueError("reference_audio must contain nonempty audio samples")
            if recording.samplerate <= 0:
                raise ValueError("reference_audio has an invalid sample rate")
            if recording.frames / recording.samplerate > 30.0:
                raise ValueError("reference_audio must be no longer than 30 seconds")
            frames_read = 0
            peak = 0.0
            for audio in recording.blocks(blocksize=65536, dtype="float64", always_2d=True):
                if not np.isfinite(audio).all():
                    raise ValueError("reference_audio samples must all be finite")
                frames_read += len(audio)
                if audio.size:
                    peak = max(peak, float(np.max(np.abs(audio))))
            if frames_read != recording.frames:
                raise ValueError("reference_audio is incomplete or truncated")
            # A small floor rejects digital silence but permits quiet recordings.
            if frames_read == 0 or peak <= 1e-8:
                raise ValueError("reference_audio is silent or empty")
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"reference_audio is not a readable audio file: {exc}") from exc


def _safe_reference_name(name: str) -> bool:
    """A manifest reference must be exactly one ordinary filename component."""
    return re.fullmatch(r"reference\.[A-Za-z0-9]{1,10}", name) is not None


class CharacterStore:
    """Store and verify reusable voice-character definitions.

    The default directory comes from the portable configuration. References
    are limited to 30 seconds for conservative compatibility across both
    backends, including CosyVoice3. ``list`` returns verified records in ID order
    and skips malformed/corrupted entries; ``load`` reports those entries with
    a ``ValueError`` instead. Hashes detect reference-byte mutation, not malicious
    edits that replace both the reference and its unsigned manifest hash.
    """

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root if root is not None else get_config().characters_dir).expanduser().resolve()

    def create(
        self,
        name: str,
        reference_audio: str | Path,
        reference_text: str = "",
        description: str = "",
        backend: str = "indextts",
        instruction: str = "",
        params: dict | None = None,
        provenance: dict | None = None,
    ) -> dict:
        """Validate and persist a new character, copying the source bytes intact."""
        if not isinstance(name, str) or not name.strip() or len(name) > 120:
            raise ValueError("name must be a nonempty string of at most 120 characters")
        for field, value in (
            ("reference_text", reference_text),
            ("description", description),
            ("instruction", instruction),
        ):
            if not isinstance(value, str):
                raise ValueError(f"{field} must be a string")
        if not isinstance(backend, str) or backend not in _BACKENDS:
            raise ValueError("backend must be 'indextts', 'cosyvoice3' or 'qwen3'")
        if params is None:
            params = {}
        if provenance is None:
            provenance = {}
        if not isinstance(params, dict) or not isinstance(provenance, dict):
            raise ValueError("params and provenance must be dictionaries")
        try:
            _validate_json_value(params, "params")
            _validate_json_value(provenance, "provenance")
        except RecursionError as exc:
            raise ValueError("params and provenance must not contain cycles or excessive nesting") from exc

        source = Path(reference_audio).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"reference_audio does not exist: {source}")
        _validate_audio(source)

        suffix = source.suffix
        if not re.fullmatch(r"\.[A-Za-z0-9]{1,10}", suffix or ""):
            suffix = ".audio"
        reference_name = f"reference{suffix}"

        self.root.mkdir(parents=True, exist_ok=True)
        # Resolve once more after creation in case a parent component was a symlink.
        self.root = self.root.resolve()
        temp_path: Path | None = None
        try:
            temp_path = Path(tempfile.mkdtemp(prefix=".character-tmp-", dir=self.root))
            reference_path = temp_path / reference_name
            shutil.copyfile(source, reference_path)
            # Validate the exact bytes that will be committed (the source could
            # have changed during the copy), without re-encoding the recording.
            _validate_audio(reference_path)
            digest = hashlib.sha256(reference_path.read_bytes()).hexdigest()
            character_id = uuid.uuid4().hex
            target = self.root / character_id
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"character ID already exists: {character_id}")

            metadata = {
                "schema_version": _SCHEMA_VERSION,
                "id": character_id,
                "name": name,
                "reference_audio": reference_name,
                "reference_sha256": digest,
                "reference_text": reference_text,
                "description": description,
                "backend": backend,
                "instruction": instruction,
                "params": params,
                "provenance": provenance,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            manifest_path = temp_path / _MANIFEST_NAME
            with manifest_path.open("x", encoding="utf-8", newline="\n") as manifest:
                json.dump(metadata, manifest, ensure_ascii=False, allow_nan=False,
                          indent=2, sort_keys=True)
                manifest.write("\n")
            # Check the destination a second time immediately before publication.
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"character ID already exists: {character_id}")
            os.rename(temp_path, target)
            temp_path = None
            return self.load(character_id)
        finally:
            if temp_path is not None:
                self._remove_owned_temp(temp_path)

    def load(self, character_id: str) -> dict:
        """Load one record, rejecting malformed metadata, paths, or audio hashes."""
        if not isinstance(character_id, str) or not _ID_RE.fullmatch(character_id):
            raise ValueError("character_id must be a lowercase 32-character UUID hex string")
        if uuid.UUID(hex=character_id).hex != character_id:
            raise ValueError("invalid character_id")

        try:
            root = self.root.resolve(strict=True)
        except FileNotFoundError:
            raise FileNotFoundError(f"character store does not exist: {self.root}") from None
        directory = root / character_id
        if directory.is_symlink():
            raise ValueError("character directory must not be a symlink")
        if not directory.is_dir():
            raise FileNotFoundError(f"character not found: {character_id}")
        resolved_directory = directory.resolve(strict=True)
        if resolved_directory.parent != root:
            raise ValueError("character directory escapes the store root")

        manifest_path = directory / _MANIFEST_NAME
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError("character manifest is missing or unsafe")
        try:
            metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError, RecursionError) as exc:
            raise ValueError(f"character manifest is unreadable: {exc}") from exc
        if not isinstance(metadata, dict) or set(metadata) != _METADATA_FIELDS:
            raise ValueError("character manifest fields do not match schema version 1")
        if type(metadata["schema_version"]) is not int or metadata["schema_version"] != _SCHEMA_VERSION:
            raise ValueError("unsupported character schema_version")
        if metadata["id"] != character_id:
            raise ValueError("character manifest ID does not match its directory")
        for field in ("name", "reference_text", "description", "instruction", "created_at"):
            if not isinstance(metadata[field], str):
                raise ValueError(f"character manifest field {field} must be a string")
        if not metadata["name"].strip() or len(metadata["name"]) > 120:
            raise ValueError("character manifest has an invalid name")
        if not isinstance(metadata["backend"], str) or metadata["backend"] not in _BACKENDS:
            raise ValueError("character manifest has an unsupported backend")
        if not isinstance(metadata["params"], dict) or not isinstance(metadata["provenance"], dict):
            raise ValueError("character params and provenance must be dictionaries")
        try:
            _validate_json_value(metadata["params"], "params")
            _validate_json_value(metadata["provenance"], "provenance")
        except RecursionError as exc:
            raise ValueError("character metadata is excessively nested") from exc
        try:
            created_at = datetime.fromisoformat(metadata["created_at"])
        except ValueError as exc:
            raise ValueError("character created_at must be an ISO timestamp") from exc
        if created_at.tzinfo is None:
            raise ValueError("character created_at must include a timezone")

        reference_name = metadata["reference_audio"]
        if not isinstance(reference_name, str) or not _safe_reference_name(reference_name):
            raise ValueError("character reference_audio must be a contained filename")
        reference_path = directory / reference_name
        if reference_path.is_symlink() or not reference_path.is_file():
            raise ValueError("character reference audio is missing or unsafe")
        resolved_reference = reference_path.resolve(strict=True)
        if resolved_reference.parent != resolved_directory:
            raise ValueError("character reference audio escapes its directory")
        digest = metadata["reference_sha256"]
        if not isinstance(digest, str) or not _HASH_RE.fullmatch(digest):
            raise ValueError("character reference_sha256 is invalid")
        actual_digest = hashlib.sha256(reference_path.read_bytes()).hexdigest()
        if actual_digest != digest:
            raise ValueError("character reference audio hash mismatch")

        result = dict(metadata)
        result["reference_audio"] = str(resolved_reference)
        return result

    def list(self) -> list[dict]:
        """Return verified records in stable ID order; ignore invalid entries."""
        try:
            root = self.root.resolve(strict=True)
        except FileNotFoundError:
            return []
        records: list[dict] = []
        try:
            entries = sorted(root.iterdir(), key=lambda item: item.name)
        except OSError:
            return records
        for entry in entries:
            if not _ID_RE.fullmatch(entry.name):
                continue
            try:
                records.append(self.load(entry.name))
            except (OSError, ValueError):
                # Corrupt or unsafe records are deliberately omitted from listings.
                continue
        return records

    def delete(self, character_id: str) -> dict:
        """Delete one verified character card and its private reference audio."""
        card = self.load(character_id)
        directory = (self.root / character_id).resolve()
        if directory.parent != self.root.resolve():
            raise ValueError("character directory escapes the store root")
        shutil.rmtree(directory)
        # Keep the optional multi-voice index free of dangling references.
        try:
            from timbreforge.voice_library import VoiceLibrary
            VoiceLibrary(self.root).remove_card_references(character_id)
        except (OSError, ValueError, TypeError):
            # The card has already been removed; a damaged optional index must
            # not turn a successful card deletion into a reported failure.
            pass
        return card

    def _remove_owned_temp(self, path: Path) -> None:
        """Remove only the temporary directory created directly under this root."""
        try:
            if path.name.startswith(".character-tmp-") and path.parent.resolve() == self.root.resolve():
                if path.is_symlink():
                    path.unlink()
                elif path.exists():
                    shutil.rmtree(path)
        except FileNotFoundError:
            pass
