# TimbreForge 0.6.0a1 实施报告（2026-10-18）

对应任务书：[DEEPSEEK_TASKS.md](../DEEPSEEK_TASKS.md)（工作区根目录另有同内容的 `TIMBREFORGE_NEXT_TASKS.md`）。

## 1. 结论

P0、P1 全部落地，P2、P3 以可运行骨架落地并明确边界：

- 角色卡升级为 schema 2，旧 schema 1 只读兼容、按需迁移；
- 音频处理链非破坏性，原始音频在任何路径下都不会被覆盖；
- Qwen Clone Prompt 有完整缓存、校验、失效和回退；
- 统一生成入口记录完整溯源，失败/取消不发布半成品；
- 资源管理器串行调度、可取消、失败清理，并按顺序释放低优先级资源；
- VoiceRecipe、单维候选 A/B、长文本分段与固定测试集评估可用；
- 统一模型适配器、本地模型注册表、实验性 VC 外部后端与离线训练骨架可用；
- 角色库、创建角色、音频修整、声线微调、文字生成、长文本、模型管理、设置与任务、高级训练页面已接入主界面。

## 2. 本轮新增文件

### 核心模块

| 文件 | 作用 |
| --- | --- |
| [audio_edit_schema.py](../timbreforge/audio_edit_schema.py) | 处理步骤、参数范围、处理文档与历史事件校验 |
| [audio_edit_pipeline.py](../timbreforge/audio_edit_pipeline.py) | 去静音/去点击/降噪/EQ/压缩/响度、哈希与原子写入 |
| [audio_presets.py](../timbreforge/audio_presets.py) | 四个预设 |
| [audio_preview.py](../timbreforge/audio_preview.py) | 压缩试听副本与清理 |
| [character_audio.py](../timbreforge/character_audio.py) | 按角色应用预设、回退、试听、历史、状态 |
| [character_package.py](../timbreforge/character_package.py) | 角色卡复制、ZIP 导出与重新导入 |
| [clone_prompts.py](../timbreforge/clone_prompts.py) | Clone Prompt 创建/加载/校验/失效 |
| [generation.py](../timbreforge/generation.py) | 统一生成入口与溯源元数据 |
| [job_queue.py](../timbreforge/job_queue.py) | 持久化任务状态机 |
| [resource_manager.py](../timbreforge/resource_manager.py) | 串行重型任务、取消、清理、内存压力释放 |
| [voice_recipe.py](../timbreforge/voice_recipe.py) | 结构化配方、提示词、单维候选、文本风格 |
| [voice_candidates.py](../timbreforge/voice_candidates.py) | 微调会话、A/B、保存为新版本 |
| [longform.py](../timbreforge/longform.py) | 长文本切分、逐段检查与淡化拼接 |
| [voice_eval.py](../timbreforge/voice_eval.py) | 固定测试集与辅助指标 |
| [model_registry.py](../timbreforge/model_registry.py) | 本地模型目录/版本登记、切换、回滚 |
| [backends.py](../timbreforge/backends.py) | 统一 TTS 适配器与注册表 |
| [vc_backend.py](../timbreforge/vc_backend.py) | 实验性 VC 外部后端 |
| [training.py](../timbreforge/training.py) | 数据集准备、训练任务编排、评估、登记与回滚 |

### UI 与页面

| 文件 | 页面 |
| --- | --- |
| [library_ui.py](../timbreforge/library_ui.py) | 角色库（升级：摘要、模型/角色版本、Clone Prompt 状态、试听、复制、导出、导入、删除） |
| [character_ui.py](../timbreforge/character_ui.py) | 创建角色、音频修整 |
| [voice_ui.py](../timbreforge/voice_ui.py) | 声线微调、文字生成、长文本生成 |
| [admin_ui.py](../timbreforge/admin_ui.py) | 模型管理、设置与任务、高级训练 |
| [ui_tasks.py](../timbreforge/ui_tasks.py) | UI 到资源管理器的适配、清理回调 |

### 修改的既有文件

- [characters.py](../timbreforge/characters.py)：schema 2、`update`/版本、音频链、Clone Prompt 指针、原子写。
- [backend_qwen.py](../timbreforge/backend_qwen.py)：`clone_prompt` 模式、缓存克隆、取消、统一结果校验。
- [workers/qwen_worker.py](../timbreforge/workers/qwen_worker.py)：条件张量保存/加载（`weights_only=True`）、脚本校验、缓存克隆。
- [webui.py](../timbreforge/webui.py)：接入 9 个新/升级页面并注册低优先级清理。
- [__init__.py](../timbreforge/__init__.py)、[pyproject.toml](../pyproject.toml)：版本 0.6.0a1。
- [README.md](../README.md)、[CHANGELOG.md](../CHANGELOG.md)：功能、数据布局、更新记录。
- [tools/release_check.py](../tools/release_check.py)：将 `DEEPSEEK_TASKS.md` 纳入根文件白名单。
- 测试：[test_characters.py](../tests/test_characters.py)、[test_qwen_backend.py](../tests/test_qwen_backend.py)、[test_role_planner.py](../tests/test_role_planner.py)、[test_simple_ui.py](../tests/test_simple_ui.py)。
- 文档：[TESTING.md](TESTING.md)、[RELEASING.md](RELEASING.md)。

### 新增文档

- [CHARACTER_LIBRARY.md](CHARACTER_LIBRARY.md)
- [AUDIO_EDITING.md](AUDIO_EDITING.md)
- [CLONE_PROMPTS.md](CLONE_PROMPTS.md)
- [VOICE_RECIPE.md](VOICE_RECIPE.md)
- [TUNING_AND_EVALUATION.md](TUNING_AND_EVALUATION.md)
- [MODELS_AND_VC.md](MODELS_AND_VC.md)
- [TRAINING.md](TRAINING.md)

### 新增测试

`tests/test_audio_editing.py`、`test_character_audio_package.py`、`test_clone_prompts.py`、`test_generation.py`、`test_jobs.py`、`test_voice_candidates.py`、`test_longform.py`、`test_voice_eval.py`、`test_backends.py`、`test_vc_backend.py`、`test_training.py`、`test_extended_ui.py`。

## 3. 行为变化

- 新角色卡写入 schema 2；打开旧卡不会重写文件，修改时才迁移。
- 音频修整后 `canonical_audio` 指向新派生文件，旧 Clone Prompt 自动失效；`reference.wav` 字节不变。
- 生成一律落在 `outputs/generation/<role-id>/<run-id>.wav/.json`，原始模型输出作为中间文件被删除。
- 生成失败/取消不发布音频或元数据；资源管理器任务状态在清理完成后才变为终态。
- 角色模型切换会递增角色版本并使 Clone Prompt 失效；`rollback` 恢复上一模型。
- 删除角色会删除整个角色目录，并清理可选声线索引中的引用。
- 新页面在本地 Qwen 文件齐备时通过统一适配器调用；缺文件时明确报错，不静默换模型。
- `DEEPSEEK_TASKS.md` 现在被发布检查视为可审阅根文件，不再报“意外源码候选”。

## 4. 验证记录

### 基线（修改前）

`python -m unittest discover -s tests -p "test_*.py"`：197 项，2 失败、1 跳过。

- `test_role_planner`：测试自身用默认编码读取请求文件，在中文 Windows 下失败；改为显式 UTF-8。
- `test_simple_ui`：保存确认复选框加入后旧断言已过时；更新为只允许该确认复选框，并修正失败状态取值位置。

### 最终结果

```powershell
$env:PYTHONUTF8='1'
python -m unittest discover -s tests -p "test_*.py"
# Ran 425 tests in ~27s
# OK (skipped=1)

python tools/release_check.py
# Source: 164 files, 3392.5 KiB; checked assets, credentials and relative links
# PASS

python -m timbreforge.launcher --check
# 独立 Python、VoiceDesign、Base 文件齐备（不加载模型）

python tools/backend_audit.py
# PASS release patch: 1 pinned files；其余外部源码检查按设计跳过

python -m timbreforge.webui --config timbreforge.local.toml --port 7872
# HTTP 200；/config 返回 17 个页面；角色库 HTML 渲染 4 张旧角色卡
# 旧卡 manifest 仍为 schema_version 1，文件修改时间未变（只读迁移）
```

## 5. 已知限制

- 本轮未在真实 GPU 上重跑克隆、Clone Prompt 张量创建、微调候选、长文本拼接、VC 与训练；这些路径由 mock 后端、假 worker/脚本、合成音频和临时目录覆盖。真实听感、显存峰值和跨平台行为仍需单独验收。
- DSP 为短参考音频设计：LUFS 为 BS.1770 风格的工程近似，降噪/去点击/EQ 不是母带级处理；强降噪可能损失气声。
- Qwen Base 不支持独立表演指令；微调候选通过 VoiceDesign 生成新参考，试听台词会成为新的逐字稿，必须人工确认逐字一致。
- 长文本使用边界淡化而非重叠交叉淡化，语气衔接仍需试听；自动检查只覆盖漏句、重复、加速、音量、亮度漂移与点击等粗粒度问题。
- 评估指标是辅助证据：CER 依赖用户转写（不做 ASR），WER 对中文不适用，说话人相似度可选且可能不可用。
- VC 不包含模型；外部脚本被视为操作者可信组件，未做沙箱；可能改变情绪、语速和咬字。
- 训练不包含训练循环，需要自备外部训练器和足够显存；检查点必须包含 `config.json` 才能登记；显存数字是启发式估算。
- 页面已做构建与 HTTP 冒烟，未做完整浏览器点击回归；Windows Gradio 仍输出非致命 event-loop ResourceWarning。
- 一次全量运行出现过一次未复现的候选提升测试错误；随后三次全量运行均为 425 项通过（1 跳过）。若再次出现，优先检查测试环境文件句柄与临时目录清理。

