"""CPU-only, consent-driven reference inspection and explicit preparation.

No ASR, speech/music/speaker detector, model loading, enhancement or auto-crop.
Reports contain only JSON-safe values. See docs/REFERENCE_ASSISTANT.md.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
import uuid

import numpy as np
import soundfile as sf

CLIPPING_THRESHOLD = 0.999
SILENCE_RMS_THRESHOLD = 1e-5
SILENT_PEAK_THRESHOLD = 1e-6
FRAME_SECONDS = 0.020


def _load(path):
    source = Path(path).expanduser().resolve(strict=True)
    raw = source.read_bytes()
    audio, sr = sf.read(io.BytesIO(raw), dtype="float64", always_2d=True)
    if not audio.size:
        raise ValueError("Reference audio is empty")
    if not np.isfinite(audio).all():
        raise ValueError("Reference audio contains nonfinite samples")
    return source, raw, audio, int(sr)


def _consumption(backend, transcript, instruction):
    name = str(backend).strip().lower()
    if name in {"indextts", "indextts2", "index"}:
        return dict(backend=name, mode="indextts", audio_consumed=True,
                    transcript_consumed=False, instruction_consumed=bool(instruction),
                    explanation="IndexTTS consumes reference audio, not its transcript. In Studio instruct mode, the instruction is emotional text only, not general voice identity or guaranteed performance control; clone mode rejects instructions.")
    if name in {'qwen3', 'qwen_base'}:
        return dict(backend=name, mode='voice_clone', audio_consumed=True,
                    transcript_consumed=bool(transcript), instruction_consumed=False,
                    explanation='Qwen Base consumes fixed reference audio and requires its exact transcript in this adapter. Independent instructions are not supported; VoiceDesign prompts only apply when creating a new voice, not during identity reuse.')
    if name in {"cosyvoice", "cosyvoice3", "cosy"}:
        if instruction:
            mode, text_used, instruction_used = "instruct2", False, True
            explanation = "CosyVoice instruct2 consumes reference audio and instruction; reference transcript is ignored."
        elif transcript:
            mode, text_used, instruction_used = "zero_shot", True, False
            explanation = "CosyVoice zero_shot consumes reference audio and supplied transcript; correspondence is not verified."
        else:
            mode, text_used, instruction_used = "cross_lingual", False, False
            explanation = "CosyVoice cross_lingual consumes reference audio without a reference transcript."
        return dict(backend=name, mode=mode, audio_consumed=True,
                    transcript_consumed=text_used, instruction_consumed=instruction_used,
                    explanation=explanation)
    return dict(backend=name, mode="unknown", audio_consumed=None,
                transcript_consumed=None, instruction_consumed=None,
                explanation="Unknown backend: input consumption is unverified.")


def _report(source, audio, sr, transcript, backend, instruction):
    # Pool squared samples across channels, never average waveforms for analysis.
    hop = max(1, int(sr * FRAME_SECONDS))
    starts = np.arange(0, len(audio), hop)
    # Scale first to keep the RMS computation finite for large FLOAT WAV samples.
    peak = float(np.max(np.abs(audio)))
    if peak:
        power = np.mean((audio / peak) ** 2, axis=1)
        sums = np.add.reduceat(power, starts)
        counts = np.minimum(hop, len(audio) - starts)
        rms = np.sqrt(sums / counts) * peak
    else:
        rms = np.zeros(len(starts))
    duration = len(audio) / sr
    warnings = []
    if duration > 30:
        warnings.append("Reference is longer than 30 seconds; consider an explicitly selected shorter excerpt.")
    consumption = _consumption(backend, transcript, instruction)
    if consumption["mode"] == "indextts" and duration > 15:
        warnings.append("IndexTTS reference is longer than 15 seconds.")
    clipping = float(np.mean(np.abs(audio) >= CLIPPING_THRESHOLD))
    if clipping:
        warnings.append("Near-full-scale samples detected; this is a clipping risk signal, not proof of clipping.")
    silence = float(np.mean(rms <= SILENCE_RMS_THRESHOLD))
    if silence:
        warnings.append("Low-energy frames detected; this heuristic does not identify speech or perceptual silence.")
    if peak <= SILENT_PEAK_THRESHOLD:
        warnings.append("Reference has no usable signal above the silent-peak threshold.")
    if consumption["mode"] == "unknown":
        warnings.append("Backend input consumption is unknown.")
    return dict(
        path=str(source), duration_seconds=float(duration), channels=int(audio.shape[1]),
        samplerate=sr, peak=peak, clipping_fraction=clipping,
        silence_frame_proportion=silence,
        thresholds=dict(clipping_absolute_sample_gte=CLIPPING_THRESHOLD,
                        silence_frame_rms_lte=SILENCE_RMS_THRESHOLD,
                        silent_peak_lte=SILENT_PEAK_THRESHOLD,
                        frame_seconds=FRAME_SECONDS, frame_samples=hop,
                        silence_method="Non-overlapping frames; RMS pooled over samples and channels; final partial frame uses actual samples, no padding."),
        warnings=warnings,
        unverified=dict(music="unknown: no music detector is run",
                        multiple_speakers="unknown: no speaker detector is run",
                        transcript_correspondence="unverified: no ASR or alignment is run"),
        transcript=transcript, backend_consumption=consumption,
    )


def inspect_reference(path, *, transcript='', backend='indextts', instruction=''):
    """Inspect decoded samples without modifying any file or verifying speech content."""
    if not isinstance(transcript, str) or not isinstance(instruction, str):
        raise TypeError("transcript and instruction must be strings")
    source, _, audio, sr = _load(path)
    return _report(source, audio, sr, transcript, backend, instruction)


def prepare_confirmed_reference(path, *, confirmed=False, start_seconds=0,
                                end_seconds=None, mono=False, output_dir=None,
                                transcript='', transcript_confirmed=False):
    """Create a unique FLOAT WAV only after explicit consent.

    Transcript confirmation asserts that the supplied text matches the *output*
    excerpt. It is a human assertion, never a machine correspondence check.
    """
    if confirmed is not True:
        raise ValueError("Explicit confirmation is required before preparing reference audio")
    if not isinstance(mono, bool) or not isinstance(transcript_confirmed, bool):
        raise TypeError("mono and transcript_confirmed must be booleans")
    if not isinstance(transcript, str):
        raise TypeError("transcript must be a string")
    source, raw, audio, sr = _load(path)
    duration = len(audio) / sr
    try:
        if isinstance(start_seconds, bool) or isinstance(end_seconds, bool):
            raise ValueError
        start = float(start_seconds)
        end = duration if end_seconds is None else float(end_seconds)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Trim range must contain finite numeric seconds") from exc
    if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end <= duration:
        raise ValueError("Trim range must satisfy 0 <= start < end <= duration")
    # Floor requested boundaries to sample indices; report exact applied indices.
    first = int(start * sr)
    last = len(audio) if end_seconds is None or end == duration else int(end * sr)
    if first >= last:
        raise ValueError("Trim range contains no audio samples")
    trimmed = first != 0 or last != len(audio)
    selected = audio[first:last]
    if mono:
        selected = np.mean(selected, axis=1, keepdims=True)
    if float(np.max(np.abs(selected))) <= SILENT_PEAK_THRESHOLD:
        raise ValueError("Selected audio is silent (or mono conversion cancelled its channels)")
    if np.max(np.abs(selected)) > np.finfo(np.float32).max:
        raise ValueError("Selected samples cannot be represented in FLOAT WAV")
    output = selected.astype(np.float32)
    effective_transcript = transcript if not trimmed or transcript_confirmed else ''
    status = ("user_confirmed" if transcript.strip() and transcript_confirmed else
              "invalidated_by_trim" if trimmed and transcript else
              "unverified" if transcript else "not_supplied")
    if output_dir is None:
        from timbreforge.config import get_config
        output_dir = get_config().outputs_dir / 'reference_assistant'
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents overwriting originals or any earlier output,
    # including in the extremely unlikely event of an identifier collision.
    while True:
        destination = directory / f"reference_{uuid.uuid4().hex}.wav"
        try:
            handle = destination.open('xb')
            break
        except FileExistsError:
            continue
    with handle:
        sf.write(handle, output, sr, format='WAV', subtype='FLOAT')
    report = _report(destination, output.astype(np.float64), sr,
                     effective_transcript, 'indextts', '')
    manifest = dict(
        version=1, original_path=str(source), original_sha256=hashlib.sha256(raw).hexdigest(),
        output_path=str(destination), output_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        format='WAV', subtype='FLOAT', samplerate=sr,
        source_frames=len(audio), source_channels=int(audio.shape[1]),
        start_frame=first, end_frame_exclusive=last,
        requested_start_seconds=start, requested_end_seconds=end,
        applied_start_seconds=first / sr, applied_end_seconds=last / sr,
        trimmed=trimmed, mono=mono, mono_method='arithmetic_channel_mean' if mono else None,
        gain=1.0, enhancement=False, automatic_cropping=False,
        confirmed=True, transcript_status=status,
        transcript_confirmed=bool(transcript.strip() and transcript_confirmed),
    )
    manifest_path = destination.with_suffix('.processing.json')
    with manifest_path.open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
    report.update(processing_manifest_file=str(manifest_path), output_path=str(destination), original_path=str(source),
                  original_sha256=manifest['original_sha256'], processing_manifest=manifest,
                  transcript_status=status, transcript_confirmed=manifest['transcript_confirmed'])
    if status == 'invalidated_by_trim':
        report['warnings'].append("Trimming invalidated the transcript; supply and explicitly confirm text matching the new excerpt before zero_shot use.")
    report['warnings'].append("Preparation does not select a synthesis backend; backend_consumption describes the default IndexTTS inspection only.")
    return report
