# 第三方材料与许可边界

TimbreForge 原始项目代码由所有者明确选择以 Apache-2.0 发布，见[LICENSE](<LICENSE>)。这不是对模型、数据、参考录音、声音身份或上游仓库的重新许可。以下为依赖/集成索引，不是完整法律意见或第三方许可全文替代品。

本发行包不包含第三方模型权重、完整 ComfyUI/IndexTTS/CosyVoice 仓库、声线数据集、私有参考录音、自动下载缓存或 Helm 参考资产。安装依赖时，应同时保留相应分发包的版权与许可文件。

| 材料 | 本项目关系 | 发布/使用注意 |
| --- | --- | --- |
| [NumPy](https://numpy.org/) | CPU 核心依赖 | NumPy 的 BSD 系许可，具体以安装发行版的 LICENSE 为准。 |
| [SoundFile](https://github.com/bastibe/python-soundfile) / [libsndfile](https://libsndfile.github.io/libsndfile/) | 音频读写依赖及其底层库 | Python 包与底层库许可不同；分别检查 BSD 与 LGPL 等相关许可及具体分发方式。 |
| [Gradio](https://github.com/gradio-app/gradio) | 可选界面依赖 | Apache-2.0；兼容基线 6.30.0，不随核心强制安装。 |
| [ComfyUI](https://github.com/comfyanonymous/ComfyUI) | 外部 HTTP 服务 | 独立安装与许可；本项目的 Apache-2.0 不覆盖 ComfyUI 或其 custom nodes。 |
| [IndexTTS](https://github.com/index-tts/index-tts) / ComfyUI IndexTTS 2.5 节点 | 外部推理模型与节点适配 | 源码、模型、节点与补丁可能有不同条款；按锁定版本核验许可、NOTICE 及模型使用约束。见[后端说明](<docs/BACKENDS.md>)和[补丁清单](<patches/INDEXTTS-25.md>)。 |
| [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) / VoiceDesign 1.7B / Base 0.6B | 独立 Python 环境中的真实设计与参考克隆，权重不随核心包分发 | 固定 qwen-tts 0.1.1 与模型完整本地 revision；源码/模型使用须分别查对应许可证、模型卡与适用限制，见[设计指南](<docs/VOICE_DESIGN.md>)。不得将声线设计能力用于冒充真人。 |
| [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) / Fun-CosyVoice3-0.5B | 隔离环境中的外部模型栈 | 源码与具体模型/附属 tokenizer/声码器许可应分别检查；不能只凭适配器许可推断商用授权。 |
| [TTS-AGI/moss-character-reference-voices](https://huggingface.co/datasets/TTS-AGI/moss-character-reference-voices) | 可选旧检索数据源，未随包分发 | 历史适配器标注 CC-BY-4.0；使用前核验当前数据卡、署名要求、用途限制与来源。数据源“全合成音”的标签不是本项目独立审计结论，不等于无须核验声音权利。 |
| CAM++ / UTMOS / DNSMOS 及其社区移植版 | 可选实验评分器 | 本包不分发权重；源码、模型和训练数据许可各自适用，不将社区包装器许可等同于全部资产许可。 |
| AutoSynth / Helm 参考资产 | 实验导出代码历史依据；资产未包含 | 缺资产时不可用；在纳入任何资产或代码前单独核验来源、版本、归属与许可。尚未承诺真实 Helm 加载验收。 |

## 分发者检查清单

- 锁定实际使用的提交、模型版本和权重来源，保存对应许可与 NOTICE。
- 上游补丁发布时注明来源、修改范围和适用版本；不要将完整第三方源码搬入本项目并改挂 Apache-2.0。
- 加入音频/数据/demo 之前核验明确的再分发权、说话者授权和署名；私有验收许可不允许公开上传。
- 核心依赖安装包可能携带自己的第三方组件，随实际分发方式逐项检查。

发现归属或许可遗漏时，请通过维护者联系渠道报告材料名称、来源和具体文件；在核验前不应扩大分发相关资产。
# 角色声线推演语言模型

可选本地后端使用 Qwen/Qwen3-1.7B（Apache-2.0），官方模型页：https://huggingface.co/Qwen/Qwen3-1.7B 。版本信息见 planner-backend-lock.json。权重不随本包分发，模型下载目录保留上游 LICENSE。角色设定导入与模型推演使用 Transformers / PyTorch，沿用隔离运行环境。
