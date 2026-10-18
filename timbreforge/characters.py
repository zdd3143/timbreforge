"""Persistent, model-independent storage for voice characters.

Each character owns an exact byte-for-byte copy of its validated reference
recording, an optional derived canonical recording, an optional cached Qwen
clone prompt, a structured VoiceRecipe and the full audio-edit history. Schema
version 1 cards from earlier releases remain readable; they are migrated in
memory and re-saved as version 2 on the next update.

No embeddings or model state are created here: the clone prompt is a derived
file owned by :mod:`timbreforge.clone_prompts` and can always be regenerated
from the reference audio and its transcript.
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

from timbreforge.audio_edit_schema import ChainError, validate_chain_entry
from timbreforge.config import get_config
from timbreforge.voice_recipe import RecipeError, empty_recipe, normalize as normalize_recipe

_SCHEMA_VERSION = 2
_SUPPORTED_SCHEMA_VERSIONS = (1, 2)
_BACKENDS = frozenset({"indextts", "cosyvoice3", "qwen3"})
_CLONE_MODES = ('icl', 'xvector')
_ID_RE = re.compile(r"[0-9a-f]{32}\Z")
_HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
_MANIFEST_NAME = "character.json"
_MAX_CHAIN_ENTRIES = 64
_MAX_TAGS = 24
_UNSET = object()

_V1_FIELDS = frozenset({
    "schema_version", "id", "name", "reference_audio", "reference_sha256",
    "reference_text", "description", "backend", "instruction", "params",
    "provenance", "created_at",
})
_V2_FIELDS = _V1_FIELDS | frozenset({
    "canonical_audio", "canonical_sha256", "voice_design_prompt", "clone_prompt",
    "clone_mode", "voice_recipe", "audio_edit_chain", "model_name",
    "model_revision", "seed", "source", "tags", "notes", "version", "updated_at",
})
_DEFAULT_MODELS = {
    'qwen3': 'Qwen3-TTS-12Hz-0.6B-Base',
    'indextts': '',
    'cosyvoice3': '',
}
_DERIVED_DIRS = {'canonical_audio': 'audio', 'clone_prompt': 'prompts'}


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

    # ------------------------------------------------------------- creation
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
        *,
        voice_design_prompt: str = "",
        voice_recipe: dict | None = None,
        source: str = "",
        tags: list | None = None,
        notes: str = "",
        model_name: str = "",
        model_revision: str = "",
        seed: int | None = None,
        clone_mode: str = "icl",
    ) -> dict:
        """Validate and persist a new character, copying the source bytes intact."""
        if not isinstance(name, str) or not name.strip() or len(name) > 120:
            raise ValueError("name must be a nonempty string of at most 120 characters")
        for field, value in (
            ("reference_text", reference_text),
            ("description", description),
            ("instruction", instruction),
            ("voice_design_prompt", voice_design_prompt),
            ("source", source),
            ("notes", notes),
            ("model_name", model_name),
            ("model_revision", model_revision),
        ):
            if not isinstance(value, str):
                raise ValueError(f"{field} must be a string")
            if len(value) > 5000:
                raise ValueError(f"{field} must be at most 5000 characters")
        if not isinstance(backend, str) or backend not in _BACKENDS:
            raise ValueError("backend must be 'indextts', 'cosyvoice3' or 'qwen3'")
        if clone_mode not in _CLONE_MODES:
            raise ValueError("clone_mode must be 'icl' or 'xvector'")
        params = {} if params is None else params
        provenance = {} if provenance is None else provenance
        if not isinstance(params, dict) or not isinstance(provenance, dict):
            raise ValueError("params and provenance must be dictionaries")
        try:
            _validate_json_value(params, "params")
            _validate_json_value(provenance, "provenance")
        except RecursionError as exc:
            raise ValueError("params and provenance must not contain cycles or excessive nesting") from exc
        recipe = _validate_recipe(voice_recipe if voice_recipe is not None else empty_recipe())
        tags = _validate_tags(tags)
        seed = _validate_seed(seed)
        model = model_name.strip() or _DEFAULT_MODELS[backend]

        source_path = Path(reference_audio).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"reference_audio does not exist: {source_path}")
        _validate_audio(source_path)

        suffix = source_path.suffix
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
            shutil.copyfile(source_path, reference_path)
            # Validate the exact bytes that will be committed (the source could
            # have changed during the copy), without re-encoding the recording.
            _validate_audio(reference_path)
            digest = hashlib.sha256(reference_path.read_bytes()).hexdigest()
            character_id = uuid.uuid4().hex
            target = self.root / character_id
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"character ID already exists: {character_id}")

            now = datetime.now(timezone.utc).isoformat()
            metadata = {
                "schema_version": _SCHEMA_VERSION,                "id": character_id,
                "name": name,
                "reference_audio": reference_name,
                "reference_sha256": digest,
                "reference_text": reference_text,
                "description": description,
                "backend": backend,
                "instruction": instruction,
                "params": params,
                "provenance": provenance,
                "created_at": now,
                # Extended role card (schema 2). Derived fields start empty: a
                # new card always plays its untouched original recording.
                "canonical_audio": None,
                "canonical_sha256": None,
                "voice_design_prompt": voice_design_prompt.strip(),
                "clone_prompt": None,
                "clone_mode": clone_mode,
                "voice_recipe": recipe,
                "audio_edit_chain": [],
                "model_name": model,
                "model_revision": model_revision.strip(),
                "seed": seed,
                "source": source.strip(),
                "tags": tags,
                "notes": notes,
                "version": 1,
                "updated_at": now,
            }
            # Validate the exact manifest that will be published against the
            # staged copy, so an invalid field can never create a broken card.
            metadata = _validate_metadata(metadata, character_id, temp_path)
            _write_manifest(temp_path, metadata)
            # Check the destination a second time immediately before publication.
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"character ID already exists: {character_id}")
            os.rename(temp_path, target)
            temp_path = None
            return self.load(character_id)
        finally:
            if temp_path is not None:
                self._remove_owned_temp(temp_path)

    # ----------------------------------------------------------------- load
    def load(self, character_id: str) -> dict:
        """Load one record, rejecting malformed metadata, paths, or audio hashes."""
        directory, metadata = self._load_metadata(character_id)
        return self._public_record(directory, metadata)

    def _load_metadata(self, character_id: str):
        """Internal: return ``(directory, version-2 metadata)`` after validation."""
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
        if not isinstance(metadata, dict):
            raise ValueError("character manifest must be an object")
        version = metadata.get("schema_version")
        if type(version) is not int or version not in _SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError("unsupported character schema_version")
        if version == 1:
            metadata = _migrate_v1(metadata)
        metadata = _validate_metadata(metadata, character_id, directory)
        return resolved_directory, metadata

    def _public_record(self, directory: Path, metadata: dict) -> dict:
        result = dict(metadata)
        result["reference_audio"] = str(directory / metadata["reference_audio"])
        result["canonical_audio"] = (str(directory / metadata["canonical_audio"])
                                     if metadata["canonical_audio"] else None)
        result["clone_prompt"] = (str(directory / metadata["clone_prompt"])
                                  if metadata["clone_prompt"] else None)
        result["reference_text_sha256"] = hashlib.sha256(
            metadata["reference_text"].encode("utf-8")).hexdigest()
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

    # --------------------------------------------------------------- update
    def update(
        self,
        character_id: str,
        *,
        name=None,
        description=None,
        reference_text=None,
        voice_design_prompt=None,
        voice_recipe=None,
        tags=None,
        notes=None,
        source=None,
        model_name=None,
        model_revision=None,
        seed=_UNSET,
        clone_mode=None,
        canonical_audio=_UNSET,
        canonical_sha256=None,
        audio_edit_chain=_UNSET,
        clone_prompt=_UNSET,
        invalidate_reason=None,
        provenance_patch=None,
        bump_version=None,
    ) -> dict:
        """Atomically update one card and invalidate derived state when needed.

        Voice-affecting changes (transcript, canonical audio, edit chain, model
        name/revision, clone mode) invalidate the cached clone prompt unless the
        caller explicitly supplies a freshly created one. The role ``version``
        only increments for voice-affecting changes; note/tag edits keep it.
        """
        directory, metadata = self._load_metadata(character_id)
        before = dict(metadata)
        voice_changed = False

        def assign(field, value):
            nonlocal voice_changed
            if value is None:
                return
            if field in {"name", "description", "reference_text", "voice_design_prompt",
                         "notes", "source", "model_name", "model_revision"}:
                if not isinstance(value, str):
                    raise ValueError(f"{field} must be a string")
                if field == "name" and (not value.strip() or len(value) > 120):
                    raise ValueError("name must be a nonempty string of at most 120 characters")
                if len(value) > (120 if field in {"model_name", "model_revision"} else 5000):
                    raise ValueError(f"{field} is too long")
                if metadata[field] != value:
                    metadata[field] = value
                    # Identity inputs that a cached clone prompt depends on.
                    if field in {"reference_text", "model_name", "model_revision"}:
                        voice_changed = True

        assign("name", name)
        assign("description", description)
        assign("reference_text", reference_text)
        assign("voice_design_prompt", voice_design_prompt)
        assign("notes", notes)
        assign("source", source)
        assign("model_name", model_name)
        assign("model_revision", model_revision)
        if voice_recipe is not None:
            metadata["voice_recipe"] = _validate_recipe(voice_recipe)
        if tags is not None:
            metadata["tags"] = _validate_tags(tags)
        if seed is not _UNSET:
            metadata["seed"] = _validate_seed(seed)
        if clone_mode is not None:
            if clone_mode not in _CLONE_MODES:
                raise ValueError("clone_mode must be 'icl' or 'xvector'")
            if clone_mode != metadata["clone_mode"]:
                metadata["clone_mode"] = clone_mode
                voice_changed = True
        if canonical_audio is not _UNSET:
            if canonical_audio is None:
                metadata["canonical_audio"] = None
                metadata["canonical_sha256"] = None
            else:
                relative, digest = _resolve_derived_file(
                    directory, canonical_audio, allowed_dir=_DERIVED_DIRS["canonical_audio"],
                    expected_sha=canonical_sha256, label="canonical_audio")
                metadata["canonical_audio"] = relative
                metadata["canonical_sha256"] = digest
            voice_changed = True
        if audio_edit_chain is not _UNSET:
            metadata["audio_edit_chain"] = _validate_chain(audio_edit_chain)
            voice_changed = True
        if clone_prompt is not _UNSET:
            if clone_prompt is None:
                metadata["clone_prompt"] = None
            else:
                relative, _digest = _resolve_derived_file(
                    directory, clone_prompt, allowed_dir=_DERIVED_DIRS["clone_prompt"],
                    expected_sha=None, label="clone_prompt")
                metadata["clone_prompt"] = relative
        if provenance_patch is not None:
            if not isinstance(provenance_patch, dict):
                raise ValueError("provenance_patch must be a dictionary")
            _validate_json_value(provenance_patch, "provenance_patch")
            metadata["provenance"] = _merge_dict(metadata["provenance"], provenance_patch)

        if invalidate_reason is not None:
            if not isinstance(invalidate_reason, str) or not invalidate_reason.strip():
                raise ValueError("invalidate_reason must be nonempty text")
            _record_invalidation(metadata, invalidate_reason)
            metadata["clone_prompt"] = None
        elif voice_changed and metadata["clone_prompt"] is not None and clone_prompt is _UNSET:
            _record_invalidation(metadata, "参考音频、逐字稿、模型版本、克隆模式或处理链已变化")
            metadata["clone_prompt"] = None

        changed = metadata != before
        if changed:
            metadata["updated_at"] = datetime.now(timezone.utc).isoformat()
            if bump_version is None:
                bump_version = voice_changed
            if bump_version:
                metadata["version"] = int(metadata["version"]) + 1
            metadata = _validate_metadata(metadata, character_id, directory)
            _write_manifest(directory, metadata)
        return self.load(character_id)

    # --------------------------------------------------------- audio history
    def apply_audio_document(self, character_id: str, document: dict,
                             *, preset='') -> dict:
        """Append a validated edit document and point the card at its output."""
        try:
            document = validate_chain_entry(document)
        except ChainError as exc:
            raise ValueError(str(exc)) from exc
        if "event" in document:
            raise ValueError("apply_audio_document requires an edit document, not an event")
        directory, metadata = self._load_metadata(character_id)
        output = document["output_audio"]
        relative, digest = _resolve_derived_file(
            directory, output, allowed_dir=_DERIVED_DIRS["canonical_audio"],
            expected_sha=document["output_hash"], label="canonical_audio")
        if digest != document["output_hash"]:
            raise ValueError("audio edit document output hash does not match its file")
        chain = list(metadata["audio_edit_chain"]) + [document]
        if len(chain) > _MAX_CHAIN_ENTRIES:
            raise ValueError(f"audio_edit_chain must contain at most {_MAX_CHAIN_ENTRIES} entries")
        return self.update(
            character_id, canonical_audio=relative, canonical_sha256=digest,
            audio_edit_chain=chain,
            invalidate_reason=f"参考音频已重新渲染（{preset or '自定义处理链'}）")

    def revert_audio(self, character_id: str, *, note: str = '') -> dict:
        """Restore the untouched original as the current reference audio."""
        from timbreforge.audio_edit_schema import make_event
        directory, metadata = self._load_metadata(character_id)
        if not metadata["canonical_audio"]:
            return self.load(character_id)
        current = directory / metadata["canonical_audio"]
        event = make_event(event='revert_to_original',
                           source_audio=metadata["canonical_audio"],
                           source_hash=metadata["canonical_sha256"],
                           note=note or '恢复为原始参考音频；派生文件保留在处理历史中。')
        chain = list(metadata["audio_edit_chain"]) + [event]
        if len(chain) > _MAX_CHAIN_ENTRIES:
            chain = chain[-_MAX_CHAIN_ENTRIES:]
        return self.update(character_id, canonical_audio=None, canonical_sha256=None,
                           audio_edit_chain=chain,
                           invalidate_reason='参考音频已恢复为原始版本')

    def set_clone_prompt(self, character_id: str, path, mode: str) -> dict:
        """Record a freshly created clone prompt; callers verify its contents."""
        if mode not in _CLONE_MODES:
            raise ValueError("clone_mode must be 'icl' or 'xvector'")
        return self.update(character_id, clone_prompt=path, clone_mode=mode,
                           provenance_patch={'clone_prompt_updated_at':
                                             datetime.now(timezone.utc).isoformat()})

    def clear_clone_prompt(self, character_id: str, reason: str) -> dict:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be nonempty text")
        return self.update(character_id, clone_prompt=None, invalidate_reason=reason)

    # --------------------------------------------------------------- delete
    def delete(self, character_id: str) -> dict:
        """Delete one verified character card and all of its derived files."""
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


# ---------------------------------------------------------------- validation
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


def _validate_audio(path: Path, label: str = "reference_audio") -> None:
    """Ensure the recording is decodable, finite, audible, and at most 30 s."""
    try:
        with sf.SoundFile(str(path)) as recording:
            if recording.frames <= 0 or recording.channels <= 0:
                raise ValueError(f"{label} must contain nonempty audio samples")
            if recording.samplerate <= 0:
                raise ValueError(f"{label} has an invalid sample rate")
            if recording.frames / recording.samplerate > 30.0:
                raise ValueError(f"{label} must be no longer than 30 seconds")
            frames_read = 0
            peak = 0.0
            for audio in recording.blocks(blocksize=65536, dtype="float64", always_2d=True):
                if not np.isfinite(audio).all():
                    raise ValueError(f"{label} samples must all be finite")
                frames_read += len(audio)
                if audio.size:
                    peak = max(peak, float(np.max(np.abs(audio))))
            if frames_read != recording.frames:
                raise ValueError(f"{label} is incomplete or truncated")
            # A small floor rejects digital silence but permits quiet recordings.
            if frames_read == 0 or peak <= 1e-8:
                raise ValueError(f"{label} is silent or empty")
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"{label} is not a readable audio file: {exc}") from exc


def _safe_reference_name(name: str) -> bool:
    """A manifest reference must be exactly one ordinary filename component."""
    return re.fullmatch(r"reference\.[A-Za-z0-9]{1,10}", name) is not None


def _validate_recipe(value):
    try:
        return normalize_recipe(value)
    except RecipeError as exc:
        raise ValueError(f"voice_recipe invalid: {exc}") from exc


def _validate_tags(tags):
    if tags is None:
        return []
    if not isinstance(tags, list) or len(tags) > _MAX_TAGS:
        raise ValueError(f"tags must be a list of at most {_MAX_TAGS} strings")
    result, seen = [], set()
    for tag in tags:
        if not isinstance(tag, str) or not tag.strip() or len(tag) > 40:
            raise ValueError("each tag must be a nonempty string of at most 40 characters")
        value = tag.strip()
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _validate_seed(seed):
    if seed is None:
        return None
    if type(seed) is not int or not 0 <= seed <= 0x7FFFFFFFFFFFFFFF:
        raise ValueError("seed must be an integer in [0, 2^63-1] or null")
    return seed


def _validate_chain(chain):
    if not isinstance(chain, list):
        raise ValueError("audio_edit_chain must be a list")
    if len(chain) > _MAX_CHAIN_ENTRIES:
        raise ValueError(f"audio_edit_chain must contain at most {_MAX_CHAIN_ENTRIES} entries")
    result = []
    for entry in chain:
        try:
            result.append(validate_chain_entry(entry))
        except ChainError as exc:
            raise ValueError(f"audio_edit_chain invalid: {exc}") from exc
    return result


def _migrate_v1(metadata: dict) -> dict:
    """Fill schema-2 fields from a version-1 card without reinterpreting audio."""
    if set(metadata) != _V1_FIELDS:
        raise ValueError("character manifest fields do not match schema version 1")
    provenance = metadata.get("provenance") or {}
    params = metadata.get("params") or {}
    design_prompt = provenance.get("design_prompt", "") if isinstance(provenance, dict) else ""
    kind = provenance.get("kind", "") if isinstance(provenance, dict) else ""
    qwen = params.get("qwen", {}) if isinstance(params, dict) else {}
    seed = qwen.get("seed") if isinstance(qwen, dict) else None
    migrated = dict(metadata)
    migrated.update({
        "schema_version": _SCHEMA_VERSION,
        "canonical_audio": None,
        "canonical_sha256": None,
        "voice_design_prompt": design_prompt if isinstance(design_prompt, str) else "",
        "clone_prompt": None,
        "clone_mode": "icl",
        "voice_recipe": empty_recipe(),
        "audio_edit_chain": [],
        "model_name": _DEFAULT_MODELS.get(metadata.get("backend"), ""),
        "model_revision": "",
        "seed": seed if type(seed) is int else None,
        "source": kind if isinstance(kind, str) else "",
        "tags": [],
        "notes": "",
        "version": 1,
        "updated_at": metadata["created_at"],
    })
    return migrated


def _validate_metadata(metadata: dict, character_id: str, directory: Path) -> dict:
    """Validate a version-2 manifest and return a normalized copy."""
    if set(metadata) != _V2_FIELDS:
        raise ValueError("character manifest fields do not match schema version 2")
    if type(metadata["schema_version"]) is not int or metadata["schema_version"] != _SCHEMA_VERSION:
        raise ValueError("unsupported character schema_version")
    if metadata["id"] != character_id:
        raise ValueError("character manifest ID does not match its directory")
    for field in ("name", "reference_text", "description", "instruction", "created_at",
                  "voice_design_prompt", "model_name", "model_revision", "source", "notes",
                  "updated_at"):
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
    for field in ("created_at", "updated_at"):
        if not isinstance(metadata[field], str):
            raise ValueError(f"character {field} must be an ISO timestamp")
        try:
            parsed = datetime.fromisoformat(metadata[field])
        except ValueError as exc:
            raise ValueError(f"character {field} must be an ISO timestamp") from exc
        if parsed.tzinfo is None:
            raise ValueError(f"character {field} must include a timezone")
    if type(metadata["version"]) is not int or metadata["version"] < 1:
        raise ValueError("character version must be a positive integer")
    if metadata["clone_mode"] not in _CLONE_MODES:
        raise ValueError("character clone_mode must be 'icl' or 'xvector'")
    _validate_seed(metadata["seed"])
    metadata["tags"] = _validate_tags(metadata["tags"])
    metadata["voice_recipe"] = _validate_recipe(metadata["voice_recipe"])
    metadata["audio_edit_chain"] = _validate_chain(metadata["audio_edit_chain"])

    reference_name = metadata["reference_audio"]
    if not isinstance(reference_name, str) or not _safe_reference_name(reference_name):
        raise ValueError("character reference_audio must be a contained filename")
    reference_path = directory / reference_name
    if reference_path.is_symlink() or not reference_path.is_file():
        raise ValueError("character reference audio is missing or unsafe")
    resolved_reference = reference_path.resolve(strict=True)
    if resolved_reference.parent != directory.resolve():
        raise ValueError("character reference audio escapes its directory")
    digest = metadata["reference_sha256"]
    if not isinstance(digest, str) or not _HASH_RE.fullmatch(digest):
        raise ValueError("character reference_sha256 is invalid")
    actual_digest = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    if actual_digest != digest:
        raise ValueError("character reference audio hash mismatch")

    canonical = metadata["canonical_audio"]
    if canonical is None:
        if metadata["canonical_sha256"] is not None:
            raise ValueError("character canonical_sha256 must be null without canonical_audio")
    else:
        if metadata["canonical_sha256"] is None:
            raise ValueError("character canonical_sha256 is required with canonical_audio")
        relative, actual = _resolve_derived_file(
            directory, canonical, allowed_dir=_DERIVED_DIRS["canonical_audio"],
            expected_sha=metadata["canonical_sha256"], label="canonical_audio")
        metadata["canonical_audio"] = relative
        metadata["canonical_sha256"] = actual
        _validate_audio(directory / relative, label="canonical_audio")

    prompt = metadata["clone_prompt"]
    if prompt is None:
        pass
    else:
        if not isinstance(prompt, str):
            raise ValueError("character clone_prompt must be a relative path or null")
        try:
            metadata["clone_prompt"] = _resolve_derived_file(
                directory, prompt, allowed_dir=_DERIVED_DIRS["clone_prompt"],
                expected_sha=None, label="clone_prompt")[0]
        except FileNotFoundError:
            # A missing derived prompt is stale state, not a broken character:
            # the next generation falls back to an ordinary clone and rebuilds it.
            metadata["clone_prompt"] = None
    return metadata


def _resolve_derived_file(directory: Path, value, *, allowed_dir: str,
                          expected_sha=None, label: str):
    """Resolve a contained derived file and verify its bytes when asked."""
    if not isinstance(value, (str, Path)):
        raise ValueError(f"{label} must be a relative path")
    candidate = Path(value)
    if candidate.is_absolute():
        try:
            relative = candidate.resolve().relative_to(Path(directory).resolve())
        except ValueError as exc:
            raise ValueError(f"{label} escapes the character directory") from exc
    else:
        relative = candidate
    if '..' in relative.parts or not relative.parts or relative.parts[0] != allowed_dir:
        raise ValueError(f"{label} must stay inside {allowed_dir}/")
    if relative.is_absolute() or relative.name in {'', '.', '..'}:
        raise ValueError(f"{label} must be a plain file inside {allowed_dir}/")
    resolved_directory = Path(directory).resolve()
    path = resolved_directory / relative
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} is missing or unsafe: {relative.as_posix()}")
    resolved = path.resolve(strict=True)
    if resolved.parent != resolved_directory / allowed_dir:
        raise ValueError(f"{label} escapes its directory")
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    if expected_sha is not None:
        if not isinstance(expected_sha, str) or not _HASH_RE.fullmatch(expected_sha):
            raise ValueError(f"{label} sha256 is invalid")
        if digest != expected_sha:
            raise ValueError(f"{label} hash mismatch")
    return relative.as_posix(), digest


def _record_invalidation(metadata: dict, reason: str) -> None:
    provenance = metadata.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}
        metadata["provenance"] = provenance
    history = provenance.get("clone_prompt_invalidations")
    if not isinstance(history, list):
        history = []
    history.append({"at": datetime.now(timezone.utc).isoformat(), "reason": reason})
    provenance["clone_prompt_invalidations"] = history[-20:]


def _merge_dict(base, patch):
    result = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def _write_manifest(directory: Path, metadata: dict) -> None:
    """Atomically publish the manifest; fsync so a crash cannot truncate it."""
    directory = Path(directory)
    temporary = directory / ('.manifest-' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(metadata, stream, ensure_ascii=False, allow_nan=False,
                      indent=2, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, directory / _MANIFEST_NAME)
    finally:
        temporary.unlink(missing_ok=True)


def audio_path(card: dict) -> str:
    """Current audio for generation: the canonical derivative or the original."""
    return card.get('canonical_audio') or card['reference_audio']

