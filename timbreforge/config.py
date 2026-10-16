"""Portable TimbreForge configuration.

Configuration is read from TOML (``TIMBREFORGE_CONFIG`` or an explicit path),
then overridden by environment variables.  Relative TOML paths are relative to
the TOML file; environment paths are relative to the current working directory.
No model or runtime is imported while loading configuration.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
import tomllib
from urllib.parse import urlparse

_PACKAGE = Path(__file__).resolve().parent
_PROJECT = _PACKAGE.parent

@dataclass(frozen=True)
class Config:
    comfy_url: str
    comfy_input_dir: Path
    comfy_output_dir: Path
    cosyvoice_repo: Path
    cosyvoice_python: Path
    cosyvoice_model: Path
    outputs_dir: Path
    characters_dir: Path
    scorer_python: Path
    mos_models_dir: Path
    indextts_repo: Path
    campplus_checkpoint: Path
    qwen_python: Path
    qwen_design_model: Path
    qwen_base_model: Path
    planner_python: Path
    planner_model: Path

    @property
    def config_path(self) -> Path | None:
        return _config_path

_config_path: Path | None = None
_cached: Config | None = None


def set_config_path(path: str | Path | None) -> None:
    """Select an explicit config for subsequent :func:`get_config` calls."""
    global _config_path, _cached
    _config_path = Path(path).expanduser().resolve() if path is not None else None
    _cached = None


def _env(name: str, *aliases: str) -> str | None:
    for key in (name, *aliases):
        value = os.environ.get(key)
        if value is not None and value.strip():
            return value.strip()
    return None


def _path(value: str | Path, base: Path) -> Path:
    p = Path(value).expanduser()
    return (base / p).resolve() if not p.is_absolute() else p.resolve()


def _toml(path: Path | None) -> tuple[dict, Path]:
    if path is None:
        raw = _env("TIMBREFORGE_CONFIG")
        path = Path(raw).expanduser().resolve() if raw else None
    if path is None:
        return {}, _PROJECT
    if not path.is_file():
        raise FileNotFoundError(f"TimbreForge config file not found: {path}")
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Invalid TimbreForge TOML config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"TimbreForge config must be a TOML table: {path}")
    return data, path.parent


def _pick(data: dict, section: str, key: str, *aliases: str):
    table = data.get(section, {})
    if not isinstance(table, dict):
        raise ValueError(f"Config section [{section}] must be a table")
    for k in (key, *aliases):
        if k in table:
            return table[k]
    for k in (key, *aliases):
        if k in data:
            return data[k]
    return None


def load_config(path: str | Path | None = None) -> Config:
    """Load TOML, environment overrides, and portable defaults."""
    explicit = Path(path).expanduser().resolve() if path is not None else _config_path
    data, config_base = _toml(explicit)
    # project-relative tools are intentional defaults; mutable data lives in user data.
    user = Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".local" if sys.platform != "win32" else os.environ.get("LOCALAPPDATA") or Path.home())
    user = Path(user) / "TimbreForge"
    defaults = {
        "comfy_url": "http://127.0.0.1:8188",
        "comfy_input_dir": user / "comfy" / "input",
        "comfy_output_dir": user / "comfy" / "output",
        "cosyvoice_repo": _PROJECT / "tools" / "cosyvoice3",
        "cosyvoice_python": _PROJECT / "tools" / "cosyvoice3" / "upy" / ("python.exe" if os.name == "nt" else "bin/python"),
        "cosyvoice_model": _PROJECT / "tools" / "cosyvoice3" / "pretrained_models" / "Fun-CosyVoice3-0.5B",
        "outputs_dir": user / "outputs",
        "characters_dir": user / "characters",
        "scorer_python": user / "comfy" / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python"),
        "mos_models_dir": user / "models_mos",
        "indextts_repo": _PROJECT / "tools" / "indextts",
        "campplus_checkpoint": user / "models" / "campplus_cn_common.bin",
        "qwen_python": _PROJECT / "tools" / "qwen3-tts" / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python"),
        "qwen_design_model": _PROJECT / "tools" / "qwen3-tts" / "models" / "VoiceDesign-1.7B",
        "qwen_base_model": _PROJECT / "tools" / "qwen3-tts" / "models" / "Base-0.6B",
        "planner_python": _PROJECT / "tools" / "qwen3-tts" / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python"),
        "planner_model": _PROJECT / "tools" / "voice-planner" / "Qwen3-1.7B",
    }
    specs = {
        "comfy_url": ("comfy", "url", ("TIMBREFORGE_COMFY_URL",)),
        "comfy_input_dir": ("comfy", "input_dir", ("TIMBREFORGE_COMFY_INPUT", "TIMBREFORGE_COMFY_INPUT_DIR")),
        "comfy_output_dir": ("comfy", "output_dir", ("TIMBREFORGE_COMFY_OUTPUT", "TIMBREFORGE_COMFY_OUTPUT_DIR")),
        "cosyvoice_repo": ("cosyvoice", "repo", ("TIMBREFORGE_COSYVOICE_REPO",)),
        "cosyvoice_python": ("cosyvoice", "python", ("TIMBREFORGE_COSYVOICE_PYTHON",)),
        "cosyvoice_model": ("cosyvoice", "model", ("TIMBREFORGE_COSYVOICE_MODEL",)),
        "outputs_dir": ("paths", "outputs_dir", ("TIMBREFORGE_OUTPUTS_DIR",)),
        "characters_dir": ("paths", "characters_dir", ("TIMBREFORGE_CHARACTERS_DIR",)),
        "scorer_python": ("scorer", "python", ("TIMBREFORGE_SCORER_PYTHON",)),
        "mos_models_dir": ("scorer", "mos_models", ("TIMBREFORGE_MOS_MODELS", "TIMBREFORGE_MOS_MODELS_DIR")),
        "indextts_repo": ("scorer", "indextts_repo", ("TIMBREFORGE_INDEXTTS_REPO",)),
        "campplus_checkpoint": ("scorer", "campplus_checkpoint", ("TIMBREFORGE_CAMPPLUS_CHECKPOINT",)),
        "qwen_python": ("qwen", "python", ("TIMBREFORGE_QWEN_PYTHON",)),
        "qwen_design_model": ("qwen", "design_model", ("TIMBREFORGE_QWEN_DESIGN_MODEL",)),
        "qwen_base_model": ("qwen", "base_model", ("TIMBREFORGE_QWEN_BASE_MODEL",)),
        "planner_python": ("planner", "python", ("TIMBREFORGE_PLANNER_PYTHON",)),
        "planner_model": ("planner", "model", ("TIMBREFORGE_PLANNER_MODEL",)),
    }
    result = {}
    for field, (section, key, envs) in specs.items():
        env_value = _env(*envs)
        config_value = _pick(data, section, key)
        value = env_value if env_value is not None else config_value
        if field == "comfy_url":
            value = value if value is not None else defaults[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError("[comfy].url must be a non-empty HTTP(S) URL")
            normalized_url = value.strip()
            parsed = urlparse(normalized_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("[comfy].url must be a valid HTTP(S) URL")
            result[field] = normalized_url.rstrip("/")
            continue
        if value is None:
            value = defaults[field]
            base = _PROJECT if isinstance(value, Path) and _PROJECT in value.parents else Path.cwd()
        else:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Configuration value for {field} must be a non-empty path string")
            base = Path.cwd() if env_value is not None else config_base
        result[field] = _path(value, base)
    return Config(**result)


def get_config(path: str | Path | None = None) -> Config:
    global _cached
    if path is not None:
        return load_config(path)
    if _cached is None:
        _cached = load_config()
    return _cached


def reset_config() -> None:
    global _cached
    _cached = None
