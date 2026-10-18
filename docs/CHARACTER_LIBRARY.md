# 角色库与角色卡

角色库是 TimbreForge 的唯一角色数据源。生成、声线微调、长文本、多角色项目和角色复用都从角色卡读取声音与逐字稿，不会各自创建孤立的临时声线。

## 目录结构

默认位于 `%LOCALAPPDATA%/TimbreForge/characters`（由 `timbreforge.local.toml` 的 `characters_dir` 决定）。

```text
characters/
  _voice-library.json          # 可选：角色分组索引，只保存声线卡引用
  <character-id>/
    character.json             # 角色卡清单（schema 2）
    reference.wav              # 原始参考音频，逐字节复制，永不覆盖
    audio/
      canonical-<hash>.wav     # 当前标准音频（无处理时不存在）
      promoted-<hash>.wav      # 声线微调采用的候选
      previews/                # 可删除的压缩试听副本
    prompts/
      clone_prompt.json        # Clone Prompt 缓存元数据
      clone-<mode>-<hash>.pt   # Qwen 条件张量（可重建）
```

删除角色会删除整个角色目录，包括原始音频、派生音频、预览和 Clone Prompt；界面会先显示确认提示。

## 角色卡字段（schema 2）

```json
{
  "schema_version": 2,
  "id": "32 位小写十六进制",
  "name": "角色名称",
  "description": "角色设定原文",
  "voice_design_prompt": "声线设计提示词",
  "reference_audio": "reference.wav",
  "reference_sha256": "原始音频 SHA-256",
  "canonical_audio": "audio/canonical-….wav",
  "canonical_sha256": "当前标准音频 SHA-256",
  "reference_text": "参考音频逐字稿",
  "clone_prompt": "prompts/clone-icl-….pt",
  "clone_mode": "icl",
  "voice_recipe": {"identity": {}, "timbre": {}, "delivery": {}, "text_style": {}},
  "audio_edit_chain": [],
  "model_name": "Qwen3-TTS-12Hz-0.6B-Base",
  "model_revision": "local",
  "seed": 12345,
  "source": "voice_design/simple_generation/imported",
  "tags": [],
  "notes": "",
  "backend": "qwen3",
  "instruction": "",
  "params": {"qwen": {}},
  "provenance": {},
  "version": 1,
  "created_at": "ISO 时间",
  "updated_at": "ISO 时间"
}
```

- `version`（角色版本）只在声音相关字段变化时递增：参考音频、处理链、逐字稿、模型名称/版本、克隆模式。只改备注或标签不会增加版本号。
- `audio_edit_chain` 按时间顺序保存每次编辑文档，以及 `revert_to_original`、`promote_candidate` 等事件。每个编辑文档都有输入/输出相对路径与 SHA-256。
- `clone_prompt` 只是派生缓存指针。文件缺失会被视为“缓存失效”，不会让整张角色卡失效；`canonical_audio` 缺失则视为角色卡损坏，因为那会改变实际使用的声音。

## 向后兼容

早期版本的 `schema_version: 1` 角色卡仍可读取。读取时会迁移到内存中的 schema 2：`canonical_audio` 为空、`voice_recipe` 使用默认配方、`model_name` 按后端填默认值、`audio_edit_chain` 为空。下一次通过界面修改时才会写回 schema 2；迁移不会重写原始音频或猜测逐字稿。

## 复制、导出与重新导入

- **复制角色卡**：在角色库中选择「复制角色卡」。音频与处理链逐字节复制，但生成新的角色 ID；Clone Prompt 不复制（它与模型、主机相关），需要重新创建。
- **导出角色包**：ZIP 内包含 `manifest.json`、`package.json`、`README.txt`、原始音频、当前标准音频，以及在 64 MiB 预算内尽量多的历史派生音频。超过预算的历史文件会记录在 `package.json` 的 `omitted_history`。
- **重新导入**：导入会分配新的角色 ID，保留配方、标签、模型信息和处理历史，但 `clone_prompt` 清空、角色版本从 1 开始，并在 `provenance` 写入 `imported_from_package`。导入包会检查 ZIP 条目路径、解压大小、音频格式与 SHA-256；原始音频哈希不一致会被拒绝。

## 验收检查

```powershell
python -m unittest tests.test_characters tests.test_character_audio_package tests.test_voice_library
```

这些测试覆盖：创建/读取/删除、损坏与符号链接拒绝、旧 schema 迁移、编辑历史与版本号、原始音频不被覆盖、复制/导出/重新导入、删除后无孤立派生文件。

