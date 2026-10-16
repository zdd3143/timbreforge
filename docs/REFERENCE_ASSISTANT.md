# Reference assistant: inspect first, prepare only with consent

The `timbreforge.reference_assistant` module uses CPU NumPy and SoundFile only.
It never loads models, downloads resources, runs ASR, identifies speakers, or
silently enhances, normalizes, resamples, trims, or crops audio. Inspection reads
the source without modifying it. Preparation is a separate, explicitly confirmed
operation. This module does not call the older automatic `prepare_reference` helper.

## Inspection API

```python
from timbreforge.reference_assistant import inspect_reference

report = inspect_reference(
    "reference.wav", transcript="The words actually spoken.",
    backend="cosyvoice3", instruction="",
)
```

Signature:

```python
inspect_reference(path, *, transcript='', backend='indextts', instruction='') -> dict
```

Both APIs return JSON-safe dictionaries (including with `json.dumps(...,
allow_nan=False)`). Paths are expanded and resolved, and returned as strings.
Missing/unreadable/unsupported files raise filesystem or SoundFile errors;
empty audio or nonfinite decoded samples raise `ValueError`.

Inspection fields:

| Field | Meaning |
| --- | --- |
| `path` | Absolute resolved audio path. |
| `duration_seconds` | Decoded frame count divided by native sample rate. |
| `channels`, `samplerate` | Native decoded channel count and sample rate. |
| `peak` | Maximum absolute decoded sample over every channel. FLOAT WAV can exceed 1. |
| `clipping_fraction` | Fraction of **channel samples** with absolute value >= 0.999. A near-full-scale risk signal, **not certainty that clipping occurred**. |
| `silence_frame_proportion` | Fraction of non-overlapping 20 ms frames with pooled RMS <= 0.00001 (approximately -100 dBFS). A low-energy heuristic, not semantic/perceptual silence detection. |
| `thresholds` | Exact numerical thresholds, actual frame sample count, and RMS method. |
| `warnings` | Human-readable strings; advisory, not automatic editing instructions. |
| `unverified` | Explicit unknown/unverified music, multiple-speaker and transcript-correspondence results. |
| `transcript` | Supplied text, not an ASR result. |
| `backend_consumption` | Backend, selected mode, consumed input booleans, explanation. |

Frame length is `max(1, int(samplerate * 0.020))` samples. RMS pools squared
samples over **all channels**, not their averaged waveform. An antiphase stereo
signal is therefore not mistakenly classified as silence. The last partial
frame uses only its actual samples (no zero padding) and counts as one frame.
The separate unusable-signal threshold for preparation is peak <= 0.000001.

Warnings are emitted for duration **strictly greater than 30 seconds** for any
backend and, additionally, **strictly greater than 15 seconds** for IndexTTS.
Exactly 15 or 30 seconds does not cross the respective threshold. Near-full-scale
samples and low-energy frames also produce explicitly qualified warnings.

### Backend input consumption

These modes reflect the existing CosyVoice worker, including instruction priority:

| Backend/mode | Reference audio | Reference transcript | Instruction |
| --- | --- | --- | --- |
| IndexTTS | Consumed | **Ignored** | Studio instruct mode uses emotional text only; not general identity/performance control. Clone rejects an instruction. |
| CosyVoice `zero_shot` (transcript, no instruction) | Consumed | Consumed, correspondence unverified | Not supplied |
| CosyVoice `cross_lingual` (neither text input) | Consumed | Not consumed | Not consumed |
| CosyVoice `instruct2` (instruction present) | Consumed | **Ignored even if supplied** | Consumed |

Accepted aliases: `indextts`, `indextts2`, `index`; `cosyvoice3`, `cosyvoice`,
`cosy`. Unknown backend consumption is explicitly unknown (`None` booleans).
These inputs are reference inputs, not the target synthesis text.

**Music, multiple speakers and audio/transcript correspondence are always
unknown/unverified.** Numerical energy, clipping and duration measurements do
not justify conclusions about those properties. Users should listen, inspect
and confirm the material themselves. No optional third-party analysis is accepted.

## Explicit preparation API

```python
from timbreforge.reference_assistant import prepare_confirmed_reference

prepared = prepare_confirmed_reference(
    "reference.wav",
    confirmed=True,
    start_seconds=1.0,
    end_seconds=8.0,
    mono=False,
    transcript="Words in the newly selected excerpt.",
    transcript_confirmed=True,
)
new_reference = prepared["output_path"]
```

Signature:

```python
prepare_confirmed_reference(
    path, *, confirmed=False, start_seconds=0, end_seconds=None,
    mono=False, output_dir=None, transcript='', transcript_confirmed=False,
) -> dict
```

`confirmed` must be the literal boolean `True`: otherwise it raises `ValueError`
before reading the source or creating directories. This confirms **all** supplied
processing choices, not merely the existence of an upload.

- Start/end must be finite and satisfy `0 <= start < end <= duration`.
  `None` end means EOF. Boundaries are floored to sample indices; a selection
  containing no samples is rejected. The manifest records requested times,
  exact applied times and frame indices (`end_frame_exclusive`).
- Only the explicit selection is used. Defaults preserve the entire audio,
  native sample rate, channel count and gain.
- `mono=True` explicitly requests an arithmetic channel mean. This can cancel
  antiphase channels; if that produces silent output it is rejected, **not**
  silently replaced with a chosen channel. Inspection alone never downmixes.
- Empty/nonfinite audio, selected peak <= 0.000001 and samples outside FLOAT
  WAV's finite representable range are rejected. Silent input may be inspected
  but cannot be prepared. No normalization is applied to peaks above full scale.
- Output is always a **new**, uniquely named `reference_<uuid>.wav`, explicitly
  WAV/FLOAT, created exclusively so collisions cannot overwrite an existing
  file. Every call creates a new output, never a cache hit or in-place edit.
  Files are immutable by this API: it never reopens them for mutation (this is
  not an OS read-only flag or a claim that external software cannot edit them).
- Default destination: `get_config().outputs_dir / 'reference_assistant'`.
  A supplied `output_dir` is a directory, not a replacement source filename.

Preparation returns all inspection fields for the **output**, plus:

| Field | Meaning |
| --- | --- |
| `output_path` | Unique new FLOAT WAV path, also exposed as `path`. |
| `original_path`, `original_sha256` | Resolved source path and SHA-256 of source bytes read for processing. Original bytes are never written. |
| `processing_manifest` | Version, original/output paths and SHA-256 hashes, format/subtype/sample rate, source frame/channel counts, requested/applied ranges, explicit mono method, gain=1, enhancement=false, automatic_cropping=false, consent and transcript status. |
| `transcript_status` | `not_supplied`, `unverified`, `invalidated_by_trim`, or `user_confirmed`. |
| `transcript_confirmed` | True only for nonblank supplied text explicitly confirmed by the user for this output. |

Preparation has no backend argument. Its inspection fields describe the default
IndexTTS backend; integrations must call `inspect_reference(output_path,
backend=..., transcript=prepared['transcript'], instruction=...)` to inspect
consumption for their selected backend. The returned manifest is JSON-safe but
is not automatically persisted as a sidecar; callers can store it with their
project metadata.

### Trimming and transcript validity

Any actual frame removal invalidates the previous transcript. If trimming is
applied without `transcript_confirmed=True`, the effective returned `transcript`
is cleared, the status is `invalidated_by_trim` when text was supplied, and a
warning tells the user to supply and confirm matching text. Do not reuse the
original text implicitly for zero-shot synthesis.

A caller may supply the **new excerpt's** transcript and explicitly set
`transcript_confirmed=True` in the same request. This is a human assertion,
not machine verification; `unverified.transcript_correspondence` remains
unverified. The module never invents a transcript. Full-length copying retains
supplied text but leaves it `unverified` unless explicitly confirmed.

## Tests

```console
python -m unittest tests.test_reference_assistant -v
```

Tests cover consent, native properties, strict duration boundaries, qualified
near-full-scale reporting, low-energy frames and final partial frames, antiphase
stereo and explicit mono cancellation, backend modes, range rejection,
nonfinite/empty/silent input, Unicode/space paths, unique FLOAT WAV creation,
original SHA invariance, exact applied sample selection, default output config,
JSON safety and trimmed transcript invalidation/reconfirmation.
