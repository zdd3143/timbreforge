# Dialogue creation projects

Dialogue creation is a local, explicit-choice workflow. It does not automatically
choose the loudest candidate, combine similarity/MOS/content into a ranking, run
ASR, or export a partially selected script. GPU/backend inference remains in the
existing Studio; project persistence and export use CPU NumPy/soundfile only.

## Python API

```python
from timbreforge.projects import ProjectStore

store = ProjectStore()  # configured outputs_dir / "projects"
p = store.create("深夜", "旁白：夜已经很深了。\n角色A〔压低声音〕：你听见了吗？\n角色B〔迟疑〕：可能只是风。")
p = store.bind(p["id"], "旁白", narrator_character_id)
p = store.bind(p["id"], "角色A", character_a_id)
p = store.bind(p["id"], "角色B", character_b_id)
line = p["lines"][0]
p = store.generate_line(p["id"], line["id"], count=2, seed=42, confirmed=True)
# Listen to both original files before explicitly choosing one.
line = p["lines"][0]
candidate = line["candidates"][0]
raw_path = store.root / p["id"] / candidate["file"]
p = store.review(p["id"], line["id"], candidate["id"], selected=True,
                 favorite=True, issues=[], note="人工试听选择")
# Repeat generation/listening/selection for EVERY line before export.
# result = store.export(p["id"], gap_seconds=.25)
# result["zip"] is the downloadable archive; result["file"] is the mix.
```

`confirmed=True` is mandatory on every generation call: the caller must have
explicitly obtained rights/consent for the voice and script. It is not a substitute
for legal permission. Count must be an integer from 1 to 4. Seeds are `seed+i`;
the request is `task='character'`, with the speaker's saved character ID, the
line's exact text, its instruction, and a seed override. CharacterStore validates
bindings when binding and again when generating. Rebinding a speaker does not
rewrite earlier candidates or selections.

Each nonblank script line must be `speaker：text` or `speaker〔instruction〕：text`.
ASCII `:` is also accepted. Blank lines are skipped; malformed input reports its
original one-based line number. Speaker, instruction, and text are stripped of
outer whitespace. Limits are 500 dialogue lines, 120-character speakers,
1000-character instructions, 5000-character line text, and 2,600,000 total script
characters. Line IDs are stable random UUID32 strings, independent of
speaker names and line text.

Public methods:

- `ProjectStore(root=None)`
- `create(name, script) -> dict`
- `load(project_id) -> dict`
- `list() -> list[dict]` (verified projects only; corrupt entries skipped)
- `bind(project_id, speaker, character_id) -> dict`
- `generate_line(project_id, line_id, *, count=1, seed=42, confirmed=False, character_store=None) -> dict`
- `review(project_id, line_id, candidate_id, *, selected=None, favorite=None, issues=None, note=None) -> dict`
- `assess(project_id, line_id, candidate_id, *, observed_text=None, run_speaker=False, run_quality=False, character_store=None) -> dict`
- `export(project_id, *, gap_seconds=.25) -> dict`

All project mutators return the full updated project. Generation appends, never
replaces or automatically selects. An explicit `selected=True` replaces that
line's manual selection; `selected=False` clears it only if that candidate is
currently selected. `None` leaves a review field unchanged. Empty `issues=[]`
and `note=''` explicitly clear those fields.

## Persisted schema 1

Each UUID32 directory has `project.json`, project-local `candidates/*.wav`, and
explicitly created exports. The JSON project has these exact top-level keys:

```json
{
  "schema_version": 1,
  "id": "<uuid32>",
  "name": "scene",
  "script": "A〔quiet〕:one",
  "created_at": "<timezone-aware ISO timestamp>",
  "updated_at": "<timezone-aware ISO timestamp>",
  "bindings": {"A": "<character uuid32>"},
  "lines": [{
    "id": "<line uuid32>",
    "line_number": 1,
    "speaker": "A",
    "text": "one",
    "instruction": "quiet",
    "selected_candidate_id": null,
    "candidates": [{
      "id": "<candidate uuid32>",
      "file": "candidates/<candidate uuid32>.wav",
      "sha256": "<64 lowercase hex>",
      "seed": 42,
      "provenance": {"rights_confirmed": true, "character_id": "<original character uuid32>", "...": "full Studio metadata"},
      "favorite": false,
      "issues": [],
      "note": "",
      "scores": {"speaker_similarity": null, "content_accuracy": null, "quality": null}
    }]
  }]
}
```

The example uses placeholders, not literal valid hashes/IDs. Candidate `file`
always stays relative to its owning project; never resolve it against the current
working directory. Provenance contains the original Studio metadata, including
its diagnostic source/reference paths; these are not candidate file locations
and are not trusted for reference lookup. The original raw waveform is copied
without normalization and SHA-256 checked whenever the project loads. Studio
may also leave its original generation WAV/metadata in the candidates directory;
these diagnostic assets are not treated as candidates until explicitly committed
into the manifest.

Generated candidates must decode as finite, nonempty, nonsilent FLOAT WAVs;
reference hashes and original character provenance must match the bound card.
Review calls append explicit preference events to `provenance.review_history`;
assessment calls append evidence snapshots to `provenance.assessment_history`.
Current favorite/issues/note/scores are included in exported dialogue/manifest
records alongside the history and original generation provenance.

A successful render is atomically committed before starting the next render.
If candidate two fails, candidate one and every prior selection remain available;
the exception is reported, and retry appends new candidates. Project JSON uses
atomic replacement with fsync. A shared reentrant lock serializes same-process
edits across store instances; multiple independent writer processes are **not**
coordinated. Stop other writers before moving/copying a live project store.

IDs, schema, script/line consistency, candidate ownership, selected IDs, relative
paths, symlinks, and file hashes are checked. Corrupt loads fail rather than
silently dropping a line. Hashes are integrity checks, not signatures: an attacker
who can replace both JSON and waveform hashes can forge provenance.

## Independent auxiliary evidence

`assess` calls the candidate helper only on explicit request. `observed_text` is
a **user-provided transcript**, not ASR. Content evidence reports raw Unicode
character error rate including punctuation/spacing. Optional CAM++ speaker and
MOS quality scorers can be unavailable and do not determine human selection.
After assessment each axis is a dictionary with `status`, `value`, and `method`;
there are also independent `signal` metrics and `warnings`. Unmeasured initial
axes are JSON null. Previously measured axes are retained if a later assessment
does not request them. Selection, favorites, notes, and issues are never changed
by assessment.

Speaker comparison reloads the **original candidate provenance character ID**,
not the current speaker binding, and verifies its reference against the original
generation SHA-256. If that card/reference no longer exists or changed, speaker
comparison explicitly fails instead of comparing against a different voice.
No untrusted stored absolute reference path is opened for assessment.

## Exports and numerical behavior

Export is only invoked by an explicit API/UI/CLI action. Every line must have a
manual selected candidate. There are no silent omissions or automatic fallbacks.
Each call creates a fresh UUID export directory under the project, containing:

- byte-for-byte original selected WAV copies in ordinal `lines/line_0001.wav`;
- UTF-8 CSV and JSON dialogue lists, with order and precise start/end positions;
- one full-length aligned stem per speaker, with silence during other speakers;
- a mixed WAV and a transparent conversion/provenance manifest;
- a ZIP with all those files (not a recursive copy of the ZIP itself).

Speaker-controlled names never become paths: stems use ordinal filenames such
as `speaker_001.wav`. Raw WAV copies retain their source channels/sample rate;
only aligned stems and mix use canonical **24,000 Hz mono**. Multichannel input
is arithmetic-mean downmixed. Other sample rates use `numpy.interp` linear
interpolation, rounded canonical frame counts, and endpoint extension where
needed. The manifest records each conversion and original rate/channels/frames.
Linear interpolation is transparent and lightweight, not a studio-quality
anti-aliasing resampler; retain the raw copies for professional post-production.

Silence gaps are inserted between lines, not after the final line. Gaps are
rounded to the nearest canonical frame and must be finite within 0–60 seconds.
Stem and mixed accumulation use float64 and WAV subtype FLOAT, without peak
normalization, limiting, or clipping: samples above full scale are preserved and
may require gain adjustment downstream. Original candidate bytes are untouched.
A 30-minute canonical duration limit and aggregate sample/memory budgets reject
oversized exports; split long scenes. A failed export may leave an incomplete
unique directory, never a falsely successful ZIP or overwritten raw recording.

Export returns absolute paths as `directory`, `zip`, `file`/`mixed_audio`,
`manifest`, `csv`, `dialogue`, plus `stems` (`speaker`, `file`), `line_audio`, and
`timeline`. Timeline records include IDs, speaker, text, instruction, raw-copy
relative file, seed, hash, full provenance, and exact `start_frame`, `end_frame`,
`start_seconds`, `end_seconds` (end exclusive).

## 中文操作步骤与 CLI

网页进入“多角色项目 / 候选比较”：创建剧本 → 刷新角色卡并逐个绑定说话人 →
选择单条台词 → 勾选声音使用授权 → 生成 1–4 个候选。再次生成只追加候选。
A/B 两个播放器和“试听 A / 试听 B”按钮可独立对照；快捷按钮会暂停另一播放器。
默认不匹配音量；可明确启用 RMS 振幅匹配试听副本（不是 LUFS 响度匹配），原始候选不变。
在 A 上勾选“选用”、收藏、问题标签并填写备注后，必须点击保存。
每句都有人工作出选择后才可导出 ZIP 和对白清单。刷新项目保留当前有效 ID。

需要辅助证据时，填写人工听写的实际内容，再点击“评估并保存 A 的独立评分轴”。
CAM++ / MOS 仅在分别勾选时运行，可能需要已就绪的本地评分环境；不要将任一数值当作自动排名。
内容留空表示未核验，不会自动识别语音。初次加载界面不扫描项目、加载模型或访问网络。

安装后的 `timbreforge-create` 是直接入口；也可以运行 `python -m timbreforge.creation_cli`。
现有主入口用 `python -m timbreforge create <操作>` 转交，因此创建项目写成 `create create`。
`--config` 放在操作之前；`--root` 可放在操作之前或之后。以下 `<...>` 都是需要替换的占位符，
ID 来自 create/show 的 JSON，不是表格行号：

```console
timbreforge-create --config timbreforge.example.toml create --name 深夜 --script "旁白：夜已经很深了。"
timbreforge-create list
timbreforge-create show --project <项目ID>
timbreforge-create bind --project <项目ID> --speaker 旁白 --character <角色卡ID>
timbreforge-create generate --project <项目ID> --line <台词ID> --count 2 --seed 42 --confirm-rights
timbreforge-create review --project <项目ID> --line <台词ID> --candidate <候选ID> --selected --favorite --note "人工试听选择"
timbreforge-create assess --project <项目ID> --line <台词ID> --candidate <候选ID> --observed-text "夜已经很深了。"
timbreforge-create export --project <项目ID> --gap-seconds 0.25
```

Qwen 角色复用页和逐句生成页提供显式 GPU/CPU 设备选择；其他后端忽略此选择，不自动回退。CPU 会采用 Qwen float32 路径，速度较慢。CLI 逐句生成可添加 `--device cpu`（默认 `cuda:0`）；Python `ProjectStore.generate_line(..., device='cpu')` 同样传递到 Studio。

多行剧本可在 PowerShell 变量 `$script` 中构建，再传 `--script $script`；该参数接收正文，不读取文件名。
`review --no-selected` 只取消当前候选的选用状态；`--no-favorite` 取消收藏；
`--issue` 可重复，`--clear-issues` 清空问题标签，`--note ""` 清空备注。省略的审听字段保持原值。
`assess --run-speaker` / `--run-quality` 分别明确启用两个本地评分器，默认都不启用。

“参考音频助手”只接受上传原音频，不静默录音或转码。先检查后端消费说明与警告；如需裁切，
填写秒数和单声道选项，再明确勾选另存处理确认。转录是否与处理后片段一致必须由用户独立确认。
处理后的音频/文件只作为输出供主动选用，不会自动填入合成控件、不自动上传合成服务，也不覆盖原文件。
CLI 等价示例：

```console
timbreforge-create inspect-reference --path original.wav --backend cosyvoice3 --transcript "对应的准确转录"
timbreforge-create prepare-reference --path original.wav --start-seconds 1 --end-seconds 8 --mono --confirm-processing --transcript "裁切后准确转录" --confirm-transcript
python -m timbreforge create list
python -m timbreforge --config timbreforge.example.toml create create --name 深夜 --script "旁白：夜已经很深了。"
```

未提供 `--confirm-rights` 的生成和未提供 `--confirm-processing` 的处理会明确报错且不调用服务。
确认旗标不替代实际获得授权或核对内容。CLI 的业务错误以简洁错误信息和非零退出码报告，无模型下载。

## CPU verification

```console
python -m unittest tests.test_projects -v
```

Tests mock generation and character lookup, use real local FLOAT WAVs, and cover
parsing, rights/count/seed validation, request provenance, append/retry persistence,
manual selection, independent transcript evidence, raw integrity, resampling,
stem/mix alignment, gaps, unique exports, safe speaker filenames, schema/ID/path/
selection corruption, and duration caps. No GPU or model/scorer download is used.
