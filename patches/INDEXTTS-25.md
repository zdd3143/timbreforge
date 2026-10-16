# IndexTTS 2.5 local integration patch

## Exact base and scope

- Repository: <https://github.com/joyfoxai/ComfyUI-Index-TTS-25.git>
- Base commit: `f4c544925005565a539b0d9177bfc25d4ba1eef2`.
- Patch: [index-tts-25-f4c5449-local.patch](<index-tts-25-f4c5449-local.patch>).
- Raw UTF-8, no BOM, LF; 10,671 bytes; SHA256 `e3a82c08c4c4d71aebfd7fd3c96ef55b7fefb567aef7f522470279f8b41722af`.
- This is the **entire observed tracked diff**, not a minimal decoder-only patch: 2 modified files, 83 insertions / 12 deletions. No untracked assets or audio included. The lock records modified-source SHA256 separately from Git blob identities.

## Audited changes (including older local modifications)

1. `index-tts/indextts/infer_v2_5.py`: consume `diffusion_steps` (default 25, range 1–100) and `inference_cfg_rate` (default 0.7, range 0–2) from generation kwargs instead of replacing them with constants in the decoder; forward caller `do_sample` instead of hard-coded `True`.
2. `nodes.py`: expose optional decoder controls and forward them **only for model version 2.5**, preserving older 2.0 call compatibility.
3. Older local change: make QwenEmotion loading conditional, include the flag in model-cache identity, expose `need_emo_text` on loader/synthesis, and re-resolve the model for text-emotion modes. This can rebuild the model when switching modes. It is not a general GPU memory manager; loader and synthesis flags can cause separate cache resolutions.
4. Older local change: replace temporary-reference WAV save/output WAV load with SoundFile to avoid a local TorchCodec/FFmpeg dependency problem. Channel transpose is preserved. **`sf.write` does not specify a subtype, so temporary WAV writing defaults to PCM_16; do not call this lossless float reference passthrough.** The original input file remains unchanged; this conversion is in the external node's temporary conditioning pipeline.
5. Older local change: add warning-level model/cache/inference timing and CUDA memory logging. Existing comments claiming approximately 7× speed or 1.1 GB memory differences are historical local observations, not performance promises or results established by this release audit.

The exact source diff has two existing trailing-whitespace lines in `nodes.py` (installed lines 448 and 538). `git diff --check` therefore reports them (exit 2). They are intentionally preserved, rather than silently changing the observed source snapshot. Numeric range checks are simple comparisons, not complete non-finite/type validation.

## Reproduce in a separate clean checkout (operator action)

Do **not** apply this to an already modified installation. Save an absolute path to this release's patch before changing directories, then use a new checkout. The following commands are instructions, not actions the audit ran against live sources:

```powershell
$patch = (Resolve-Path ./patches/index-tts-25-f4c5449-local.patch).Path
git clone https://github.com/joyfoxai/ComfyUI-Index-TTS-25.git IndexTTS25-clean
Set-Location IndexTTS25-clean
git checkout --detach f4c544925005565a539b0d9177bfc25d4ba1eef2
git status --short  # must be clean
git apply --check $patch
# Only after reviewing the full diff and checking the base:
git apply $patch
git diff --stat
git apply --reverse --check $patch
```

Do not use `--whitespace=error` with this exact snapshot; see the preserved-whitespace finding. Do not automatically use `--reject`, force an incompatible base, install dependencies, or download weights merely to make a check pass. Review node/source and model licenses separately; TimbreForge's license does not relicense them.

## Audit evidence and boundaries

- Installed node HEAD and origin were read and matched the exact base/repository above; `git status --short` showed only the two named modified files.
- Full patch was read and inspected.
- `git apply --reverse --check` against the installed modified source passed. This is a read-only compatibility check; **no live apply/reverse, reset, checkout, clean, install, or model download was run**.
- A clean-base forward check uses `git show SHA:path` to materialize just the two untouched base files inside a temporary directory in this release repository and runs `git apply --check`. This checks patch context, not a full dependency build.
- Opt-in source integration tests parse the external files with Python AST, without importing ComfyUI, torch, or IndexTTS. They check source plumbing, not model execution, service reload, audio quality, reproducibility, or GPU memory behavior.
- Restart the operator-managed ComfyUI process after applying a patch to a **separately prepared** installation. Editing disk files does not reload an existing Python process. The Studio adapter checks the live node schema for decoder controls, but that is not cryptographic attestation of loaded source.

For byte pins and limitations see [backend lock](<../backend-lock.json>); for setup and isolated runtimes see [backend guide](<../docs/BACKENDS.md>). The audit tool accepts explicit local roots and only reads files; it never downloads or patches an installation.
