# TimbreForge 后续开发任务与实现方案

本文档用于交给 DeepSeek，目标是继续完善 TimbreForge 的音频编辑、声线微调、角色库和模型生态接入功能。

## 一、总体目标

将项目完善为以“角色库”为核心的角色声线工作台：用户先导入或创建角色设定，系统推演声线并生成参考音频；参考音频、逐字稿、声线配方和模型信息全部保存到角色库；后续文字生成、声线设计、角色复用和长文本生成均从角色库选择角色。

核心原则：原始数据不可破坏；每次生成可追溯；角色库是唯一数据源；重型模型任务串行执行；所有后端通过统一接口接入；内存不足时优先关闭空闲模型和预览资源。

## 二、优先级

### P0：先完成

- 扩展角色卡：原始参考音频、标准音频、逐字稿、声线提示词、VoiceRecipe、模型版本、随机种子、来源、处理链。
- 音频非破坏性编辑：去首尾静音、音量标准化、基础降噪、EQ、去点击声。
- 接入 Qwen3-TTS 可复用 Clone Prompt 缓存。
- 所有生成和复用页面统一从角色库选择角色。
- 生成异常时释放子进程、显存、共享内存、临时文件和空闲模型。

### P1：核心体验

- 增加结构化 VoiceRecipe。
- 将音色和表达方式分开控制。
- 一次生成多个候选并支持 A/B 试听。
- 保存候选的 prompt、seed、模型版本和参数。
- 长文本分段生成和拼接。
- 增加固定测试文本与基础稳定性评估。

### P2：生态接入

- 增加 Voice Conversion 后端，用于保留已有音频内容和情绪、替换目标音色。
- 增加统一模型适配器，便于接入其他 TTS/VC 模型。
- 增加本地模型目录和版本管理。

### P3：训练能力

- 增加数据集准备、校对、切分和质量筛选。
- 优先支持 LoRA/Adapter，不直接覆盖完整模型。
- 增加离线训练任务、模型注册表、评估报告和回滚。
- 训练功能放在高级页面，不影响普通生成流程。

## 三、角色库数据结构

角色卡至少包含：

```json
{
  "id": "unique-role-id",
  "name": "角色名称",
  "description": "角色设定原文",
  "voice_design_prompt": "声线设计提示词",
  "reference_audio": "audio/original.wav",
  "canonical_audio": "audio/canonical.wav",
  "reference_text": "参考音频逐字稿",
  "clone_prompt": "prompts/qwen_clone_prompt.bin",
  "clone_mode": "icl",
  "voice_recipe": {},
  "audio_edit_chain": [],
  "model_name": "Qwen3-TTS-1.7B-Base",
  "model_revision": "local-version",
  "seed": 12345,
  "source": "voice_design/simple_generation/imported",
  "tags": [],
  "notes": ""
}
```

保存原始文件、当前标准文件和历史派生文件之间的关系。删除角色时清理其派生文件并提示用户。所有页面使用角色库中的角色，不允许各自创建孤立的临时声线。

## 四、音频编辑层

建议增加：

```text
audio_edit_pipeline.py
audio_presets.py
audio_edit_schema.py
audio_preview.py
```

处理流程：

```text
原始音频 -> 校验 -> 去静音 -> 降噪/去点击声 -> 音量标准化 -> EQ/压缩 -> 导出派生 WAV -> 保存处理链 -> 重新生成 Clone Prompt
```

要求：

- 绝不覆盖原始音频。
- 所有处理参数写入 JSON。
- 主文件使用 WAV，预览文件可以使用压缩格式。
- 修改参考音频后，旧 Clone Prompt 必须失效。
- 提供恢复到原始版本的按钮。
- 先提供 `clean_voice`、`warm_voice`、`bright_voice`、`low_noise` 四个预设。

建议元数据：

```json
{
  "source_audio": "character_original.wav",
  "operations": [
    {"name": "trim_silence", "threshold_db": -42},
    {"name": "normalize", "target_lufs": -16}
  ],
  "output_audio": "character_clean.wav",
  "source_hash": "...",
  "output_hash": "..."
}
```

## 五、VoiceRecipe 声线配方

不要只保存一段无法解析的自然语言。增加结构化配置：

```json
{
  "identity": {"age": "young", "gender_expression": "feminine", "style": "anime"},
  "timbre": {"pitch": "high", "brightness": "soft", "body": "light", "breathiness": "low", "roughness": "low"},
  "delivery": {"emotion": "cold", "energy": "low", "pace": "slow", "pauses": "clear", "sentence_ending": "flat", "emphasis": "restrained"},
  "text_style": {"sentence_length": "short", "avoid_excessive_exclamation": true, "avoid_overacting": true}
}
```

必须区分：

- 音色：音高、厚度、明亮度、气声、粗糙度、鼻腔感。
- 表达：语速、停顿、情绪强度、能量、重音、句尾、咬字。
- 文本：句子长度、标点、台词情绪和上下文。

“冷淡的二次元萝莉音”应拆成高音、轻薄、低能量、少气声、慢速、句尾平直、避免夸张感叹等可检查字段。

## 六、Qwen Clone Prompt 缓存

实现：

```text
create_clone_prompt(role_id)
load_clone_prompt(role_id)
invalidate_clone_prompt(role_id)
validate_clone_prompt(role_id)
```

缓存记录角色 ID、模型名称和版本、参考音频 SHA-256、参考文本 SHA-256、克隆模式和创建时间。以下情况必须失效缓存：音频变化、逐字稿变化、模型版本变化、ICL/x-vector 模式切换、处理链变化。

默认使用 ICL 以获得更好的相似度和表达控制；显存不足或需要轻量运行时再使用 x-vector-only。缓存失效时回退到普通克隆流程，并向用户说明原因。

## 七、候选生成和微调工作流

```text
选择角色 -> 选择 VoiceRecipe -> 输入台词 -> 生成 4 到 8 个候选 -> 试听 A/B -> 选择一个 -> 保存为角色新版本 -> 重新创建 Clone Prompt
```

候选生成时一次只改变一个维度，例如：

```text
A：冷淡、低能量、句尾平直
B：冷淡、低能量、语速更慢
C：冷淡、低能量、气声更少
D：冷淡、低能量、咬字更清晰
```

每个候选都必须保存 prompt、seed、模型版本、生成参数和角色 ID。

## 八、内存和任务调度

重型任务统一经过资源管理器：

```text
等待 -> 排队 -> 准备模型 -> 生成 -> 保存结果 -> 清理临时资源 -> 卸载空闲模型 -> 完成
```

建议增加或统一：

```text
resource_manager.py
model_worker.py
job_queue.py
```

规则：

- 同时只允许一个重型模型任务占用 GPU。
- 生成失败必须执行 finally 清理。
- 关闭浏览器窗口时自动停止后台程序。
- 页面刷新不能重复启动模型。
- 任务取消后释放显存、共享内存、临时音频和子进程。
- 内存不足时按“正在生成 > 当前试听 > 页面缓存 > 历史预览 > 空闲模型”释放资源。
- 所有错误显示具体原因和恢复建议。

## 九、长文本

不要一次把整篇文本发送给模型：

```text
长文本 -> 按段落和句子切分 -> 使用同一个 Clone Prompt 逐段生成 -> 检查异常 -> 淡化拼接 -> 输出
```

检查漏句、重复、突发加速、音量跳变、音色漂移和点击声。每段都保留角色 ID、段落序号、文本和生成参数。

## 十、自动评估

为每个角色准备固定测试集：低情绪短句、疑问句、命令句、长句、数字和专有名词、高情绪句、冷淡低能量句，以及覆盖不同声母和韵母的中文句子。

记录：

```text
WER/CER、说话人相似度、时长、平均语速、首尾静音、峰值/LUFS、削波数量、音色漂移、用户 A/B 评分
```

同时保存平均分和最差候选，避免只看平均效果。

## 十一、Voice Conversion

作为独立后端，不与普通 TTS 混为一个流程：

```text
已有音频 -> 保留内容和情绪 -> 转换目标角色音色 -> 音频修整 -> 保存派生结果
```

接口建议：

```python
class VoiceConversionBackend:
    def convert(self, source_audio, target_role_id, preserve_prosody=True, **kwargs):
        ...
```

第一阶段只做实验性后端，先检查许可证、显存、延迟和音质。VC 可能改变情绪、语速和咬字，不能默认等同于完美音色替换。

## 十二、微调训练

训练功能必须是离线高级功能：

```text
导入有授权录音 -> 自动切分 -> 识别逐字稿 -> 人工校对 -> 去噪筛选 -> 训练/验证集 -> LoRA/Adapter -> 固定测试集评估 -> 注册模型版本 -> 绑定角色卡
```

要求：

- 只使用有合法使用权的录音。
- 逐字稿准确，去除噪声、混响和多人说话。
- 训练集和验证集分离。
- 不覆盖 Base 模型。
- 检查短句、长句、语速、音色、性别感和过拟合。
- 检查点使用临时文件后原子替换，避免中断损坏。
- 训练失败不能破坏已有角色。
- 模型可切换和回滚。

优先在低显存条件下验证 0.6B Base；1.7B 需要更高显存和主机内存。训练页面要显示基础模型、训练日期、数据摘要、参数、评估结果和显存需求。

## 十三、建议的页面

1. 角色库
2. 创建/导入角色
3. 音频修整
4. 声线微调
5. 文字生成
6. 长文本生成
7. 高级训练
8. 设置和模型管理

所有页面先选择角色。角色库卡片显示角色名、声线摘要、试听按钮、模型版本、角色版本、Clone Prompt 状态和最后修改时间。

## 十四、实施顺序

1. 扩展角色卡字段并兼容旧数据。
2. 完成非破坏性音频处理链和四个基础预设。
3. 接入 Clone Prompt 创建、保存、加载、校验和失效。
4. 增加角色库统一入口、A/B 试听和处理历史。
5. 修复生成失败后的资源清理和内存回收。
6. 增加 VoiceRecipe 和候选生成。
7. 增加长文本分段和自动评估。
8. 评估并接入 Voice Conversion。
9. 最后增加 LoRA/Adapter 训练。

## 十五、验收标准

- 创建、导入、删除、复制、导出和重新导入角色都正常。
- 删除角色不会留下孤立派生文件。
- 原始音频永远不会被覆盖。
- 每次处理都有处理链和输入输出哈希。
- 修改音频或逐字稿后旧 Clone Prompt 会失效。
- 每个候选可追溯到 prompt、seed、模型版本、参数和角色 ID。
- 生成失败后模型、临时进程和内存会释放。
- 内存不足时会清理低优先级资源并给出提示。
- 长文本没有明显漏句、重复、突发加速和音色漂移。
- 训练模型和 Base 模型可独立切换并回滚。

## 十六、给 DeepSeek 的执行要求

1. 先阅读项目结构、README、角色库、Qwen 后端、任务调度和 UI。
2. 先运行现有测试和启动流程，记录基线。
3. 不要一次重写整个项目，先完成 P0，再完成 P1。
4. 每次修改前检查是否破坏已有角色数据。
5. 新模型必须通过统一后端接口。
6. 每个生成结果必须保存来源、参数、模型版本和角色 ID。
7. 所有重型任务使用统一资源管理器，异常时释放资源。
8. 不要自动下载大模型，不要上传角色录音或角色文件。
9. 每个阶段完成后报告修改文件、行为变化、测试命令、测试结果和已知限制。
10. 如需求与现有实现冲突，优先保持角色库为唯一数据源，并说明冲突。

## 参考资料

- Qwen3-TTS：https://github.com/QwenLM/Qwen3-TTS
- 官方推理实现：https://github.com/QwenLM/Qwen3-TTS/blob/main/qwen_tts/inference/qwen3_tts_model.py
- 官方 Demo：https://github.com/QwenLM/Qwen3-TTS/blob/main/qwen_tts/cli/demo.py
- Issues：https://github.com/QwenLM/Qwen3-TTS/issues
- Voice Prompt 讨论：https://github.com/QwenLM/Qwen3-TTS/discussions/279
- Technical Report：https://arxiv.org/abs/2601.15621
- Long-form TTS：https://arxiv.org/abs/2609.16989
- EZ-VC：https://arxiv.org/abs/2505.16691

