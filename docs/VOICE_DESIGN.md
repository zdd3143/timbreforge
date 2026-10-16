# Qwen3-TTS 声线设计

VoiceDesign 是真正的文字条件声线生成，不是从声线库检索，也不是把振荡器参数叫作声音身份。

## 从提示词到可复用角色

1. 启动配置好的 WebUI。已有角色资料可用首页「角色设定 → 声线推演」；只想直接输入声线描述可用「简化版 · 一句话生成」。若需手动参数、候选对比或保存角色，打开「文字声线设计 / Qwen VoiceDesign」。
2. 描述年龄感、性别表达、音高、沙哑/气声/明亮等音色、口音和表演风格。可选属性框只帮助拼接可见文字，没有隐藏数值映射。
3. 填写短而完整的试听台词，先生成一个候选，再少量追加；候选使用起始种子依次递增。
4. 用 A/B 播放器试听，必要时生成 RMS 音量匹配的试听副本。它不是 LUFS 测量，也不修改原始输出。
5. 明确选择想要的候选，保存收藏和备注；没有自动最高分赢家。
6. **听取输出并确认实际词句与试听台词一致**，再把所选候选保存为角色。保存参考原音、已确认转录、设计描述、实际参数与来源记录。
7. 角色复用和多角色脚本使用 **Qwen Base 0.6B 的参考克隆**，不为每句重新执行设计提示词。固定参考比重复提示词更适合作为持续的身份锚点，但仍不保证每次完全一致。

原始设计参考超过 30 秒时保存会明确失败，不静默裁切。请用更短完整台词重新生成，或在参考助手中明确选择片段并重新核对其转录。

## CLI

```powershell
python -m timbreforge --config timbreforge.local.toml design create --prompt "低沉温暖的成年男声，清晰自然，平静克制。" --text "你好，很高兴与你见面。" --seed 42 --count 1 --confirm-design
python -m timbreforge --config timbreforge.local.toml design list
```

用 `design select --session ID --candidate ID` 明确选定，再用 `design save-character --session ID --candidate ID --name 名称 --confirm-transcript` 保存。安装入口也提供 `timbreforge-design`。`design append` 使用当前命令参数追加，既有描述/试听台词不变。

## 参数：像创作面板，但不假装是 SD

Qwen3-TTS 使用自回归音频 token 生成，不是扩散图像模型。实际暴露的控制为：

- `seed`：在隔离 worker 中设置 Python、NumPy、PyTorch 和 CUDA RNG，模型加载后、实际生成前重新设定。不是无效的 `generate(seed=...)` 参数。
- `do_sample`、`temperature`、`top_p`、`top_k`：主 talker 采样。
- `repetition_penalty`：重复惩罚，过大不保证好听或准确。
- `max_new_tokens`：主 talker token 上限，是资源边界，不是扩散步数；过小可能截断，过大增加耗时和内存。项目默认比模型原始 8192 上限更保守。
- `subtalker_dosample`、`subtalker_temperature`、`subtalker_top_p`、`subtalker_top_k`：残余声码器预测的单独采样。
- `language`：实际模型语言选择。

预设仅是可以继续编辑的提示词起点；随机种子按钮只生成新的真实 RNG 种子。加载会话会恢复人工所选候选的实际参数与设备（未选择时恢复初始配方），避免看着当前控件却误以为是历史配方。

要关闭随机采样，应同时关闭主 talker 和 subtalker 的采样。固定种子在同一环境有助于复现，不保证跨 GPU、库版本、attention 实现或设备逐采样一致。

**没有负面提示词、CFG、扩散 steps 控件。** 官方核心只消费白名单参数，不应将任意参数转发后就称其生效。提示词属性也不保证能分别精确控制：音色、年龄、节奏、口音可能互相影响，需要试听迭代。

## 隔离环境与配置

模型与观察到的独立环境版本记录在[Qwen 后端锁文件](<../qwen-backend-lock.json>)，不含本机路径或音频；它不是跨平台通用依赖锁。

本机首次真实设计已成功（5.84 秒、RTX4060 8GB、FP16 SDPA，用户已核对台词）；具体资源和验收边界见[测试说明](<TESTING.md>)。不将该单样本结果泛化为所有提示词都符合或身份完全稳定。

核心包不导入 torch，也不自动下载权重。模型栈位于单独 Python 环境，通过包内 file-JSON worker 启动。配置示例：

```toml
[qwen]
python = "../external/qwen3-tts/.venv/Scripts/python.exe"
design_model = "../external/qwen3-tts/models/VoiceDesign-1.7B"
base_model = "../external/qwen3-tts/models/Base-0.6B"
```

对应环境变量为 `TIMBREFORGE_QWEN_PYTHON`、`TIMBREFORGE_QWEN_DESIGN_MODEL`、`TIMBREFORGE_QWEN_BASE_MODEL`。相对 TOML 路径相对配置文件位置解析。

适配基线为 `qwen-tts==0.1.1`，官方依赖 `transformers==4.57.3`、`accelerate==1.12.0`；torch 与 CUDA 安装必须单独验证匹配，不把源码声明等同于已安装可运行环境。

完整固定本地模型快照需要 main safetensors、文字 tokenizer/vocab/merges、generation/preprocessor 配置，以及附带 `speech_tokenizer`。不要仅下载主权重，也不要仅依赖 `revision=`，因为部分官方内部加载调用不完整转发 revision。

| 模型 | 官方仓库 | 固定 revision |
| --- | --- | --- |
| VoiceDesign 1.7B | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | `5ecdb67327fd37bb2e042aab12ff7391903235d3` |
| Base 0.6B | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | `5d83992436eae1d760afd27aff78a71d676296fc` |

两个完整快照约 7.04 GB（十进制），各包含约 0.68 GB 的 speech tokenizer；不随 TimbreForge 分发。模型和依赖许可仍分别适用。

GPU 默认 FP16 + SDPA，不要求在 Windows 编译 FlashAttention。8GB 显存必须单模型串行运行：每个隔离 worker 完成后退出，不同时驻留 Design 和 Base。仍可能 OOM；不会自动卸载其他程序模型。CPU 明确选择 float32，速度与内存需求可能很高，不是自动静默兜底。

## 来源与验收边界

官方实现依据：[Qwen3-TTS 仓库](https://github.com/QwenLM/Qwen3-TTS)，研究源码 commit `022e286b98fbec7e1e916cb940cdf532cd9f488e`。

- [官方设计后再克隆工作流](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/README.md)
- [官方包装接口](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/qwen_tts/inference/qwen3_tts_model.py)
- [实际核心生成参数](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/qwen_tts/core/models/modeling_qwen3_tts.py)

CPU 单测使用临时音频和模拟后端，只证明请求、参数、存储、错误传播及 UI 连线，不证明声线设计音质。是否完成本机真实模型推理、耗时、显存与可懂度检查应以本轮实际验收记录为准；本文不预先宣称通过。
