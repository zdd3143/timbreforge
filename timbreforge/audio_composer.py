"""Small, local-only helpers for joining reference and generated audio.

The composer deliberately creates a new FLOAT WAV and never edits any input.
It reuses the same validated concatenation path as long-form export while
presenting a focused API for the reference audio assistant.
"""
from __future__ import annotations

from pathlib import Path


def compose_audio(audio_paths, *, gap_ms=80, crossfade_ms=25, output_dir=None):
    """Join two or more audio files and return the generated file/report.

    ``audio_paths`` may be a Gradio ``Files`` value (paths or file-like
    objects).  Inputs are decoded read-only and must share sample rate and
    channel count.  The output is always a fresh FLOAT WAV.
    """
    if not isinstance(audio_paths, (list, tuple)):
        raise ValueError("请至少选择一个音频文件。")
    paths = []
    for item in audio_paths:
        value = getattr(item, "name", item)
        if isinstance(value, dict):
            value = value.get("path") or value.get("name")
        if value:
            paths.append(str(value))
    if len(paths) < 2:
        raise ValueError("组合音频至少需要选择两个文件。")

    from timbreforge.config import get_config
    from timbreforge.longform import concatenate

    directory = Path(output_dir or (get_config().outputs_dir / "reference_assistant"))
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "composed_reference.wav"
    # concatenate protects against collisions only when overwrite is explicit;
    # a unique name keeps concurrent browser sessions isolated.
    import uuid
    destination = directory / f"composed_{uuid.uuid4().hex}.wav"
    return concatenate(paths, destination, crossfade_ms=crossfade_ms,
                       gap_ms=gap_ms, overwrite=False)

