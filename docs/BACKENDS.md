# 外部后端准备与来源记录（Alpha）

TimbreForge 不包含、下载或安装模型、ComfyUI、CosyVoice 源码或它们的 Python/CUDA 运行环境。先完成 CPU 核心安装，再由操作者在独立目录和环境准备后端，审核许可及来源，最后配置本机路径。不要把重模型依赖安装进核心/UI 环境；不要把 Gradio 的 UI 基线版本套用到 CosyVoice 环境。

## 锁定范围与已知缺口

[backend-lock.json](<../backend-lock.json>) 是**本地实际字节的 SHA256 清单**，不是完整环境 lock，不是上游模型 commit 的替代品，也不证明资产可信、许可完备或推理可用。清单只包含相对路径、长度、哈希和公开仓库地址，不包含私人绝对路径、音频或模型字节。

| 项目 | 可核验身份 | 限制 |
| --- | --- | --- |
| IndexTTS ComfyUI 节点 | <https://github.com/joyfoxai/ComfyUI-Index-TTS-25.git>，commit `f4c544925005565a539b0d9177bfc25d4ba1eef2` + 已捕获的完整本地补丁 | 补丁不是仅包含新解码参数；含旧的 SoundFile I/O、Qwen 按需加载及日志修改。 |
| IndexTTS-2.5 模型 | 34 个本地模型/辅助文件的逐文件 SHA256；约 10.97 GB（十进制） | 上游模型 revision **unknown**。包含展开的 `hf_cache` 目录联接内容；原始模型卡说明辅助权重来自多个仓库。不能把节点 commit 当模型 commit。 |
| CosyVoice3 源码 | <https://github.com/FunAudioLLM/CosyVoice.git> 为本地 README 所指来源；96 个源码、声明依赖及许可文件的原始字节 SHA256 | 实际源码目录没有 `.git`；archive revision / commit **unknown**。Matcha-TTS 的 `.git` 指针缺失目标；不能编造 submodule commit。 |
| CosyVoice3 模型 | 本地模型卡及下载 helper 指向 <https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512>；17 个本地资产的 SHA256；约 5.43 GB（十进制） | 上游模型 revision **unknown**。helper 使用可变的 `main`，没有恢复出不可变 revision；这是基本推理子集，不含 RL 权重或 TensorRT estimator。 |

CosyVoice 原始 [requirements.txt 指纹](<../backend-lock.json>) 为 `5d62cb5592118886df90af8639ff5c1a34e935ba47d8f2ed6c5cd03e263c2db4`。这是声明依赖文件的指纹，**不是已安装包的 pip freeze**。该声明包含 Python 3.10 时代的 `torch==2.3.1`、`torchaudio==2.3.1`、`transformers==4.51.3`、`numpy==1.26.4`，不能据此声称当前运行环境逐项符合声明、兼容核心 Python 3.11/3.12 或可在其他 GPU 上重现。

另以隔离 Python 的 `importlib.metadata` 仅读取实际安装发行版元数据，未导入模型栈：Windows AMD64 / CPython **3.10.11**，158 个发行版版本记录在 lock 的 `observed_runtime`。其中 torch / torchaudio 均为 **2.3.1+cu121**，transformers 4.51.3、NumPy 1.26.4、SoundFile 0.12.1、ONNX Runtime 1.18.0（CPU 包；未安装 `onnxruntime-gpu`）、ModelScope 1.20.0、HyperPyYAML 1.2.3、Gradio 5.4.0。这只是已观察版本清单：未锁 wheel 哈希、解释器字节或 GPU driver；`+cu121` 是分发包版本信息，未探测 CUDA 执行能力，不是可直接安装的完整环境锁。

来源调查只读取了相关本地 helper、模型卡、源码 README 及日志/元数据候选，没有运行下载脚本。未找到能证明 Cosy 源码 archive revision 的不可变元数据。另一个有 Git 的同名/相邻 checkout 的 HEAD 不代表实际使用的无 Git 源码目录；不要借用它填 commit。没有恢复出原始源码 archive；本清单选定的逐文件指纹不能重建整个源码树。要达到严格 commit 可复现，后续应由操作者取得可信的版本化来源或确切 release/commit，记录 revision、匹配本地源文件并审查差异后重新测试。本次明确保留此缺口。

## IndexTTS / ComfyUI

1. 独立准备并审核 ComfyUI、上面指定的节点仓库、其依赖及单独许可的 IndexTTS 模型与辅助模型。本项目不锁定完整 ComfyUI/CUDA 环境。
2. 按[补丁说明](<../patches/INDEXTTS-25.md>)，在**新建干净 checkout** 中 `git checkout --detach` 精确 SHA，先读完整 diff，执行 `git apply --check` 后才自行应用。不要在已修改的正在使用的源码目录再应用。
3. 将模型与辅助资产准备完整；`hf_cache` 可以是操作者配置的目录或链接，但公共 lock 不记录链接的私人目标。确保任何链接均由自己控制，缺少文件时应明确失败，不要默认为允许联网补齐。
4. 由操作者启动/重启 ComfyUI，配置 URL、input/output 目录。重启后核查节点 schema，再用有权使用的参考音频进行单条私有验收。**磁盘改动不等于服务已重载**。

Studio 使用 `LoadAudio → IndexTTS25ModelLoader → IndexTTS25Synthesize → SaveAudio` 的 ComfyUI API 图。2.5 解码控制为 `diffusion_steps` 默认 25 / 1–100，`inference_cfg_rate` 默认 0.7 / 0–2；`do_sample` 应传入推理。文字指令进入情感文本路径，不等于通用声线身份设计。QwenEmotion 按需加载可能导致模式切换时重新构建模型；历史性能注释不是本次性能保证。

**音频边界**：Studio 校验并保留原始参考，不自动裁切或响度归一化；但是外部节点的旧 SoundFile 临时 WAV 写入默认 PCM_16，不能称整条模型条件处理路径为 float 无损透传。IndexTTS 上游可能只采用参考前约 15 秒。相关变化、已知空白行尾与风险见[完整审计](<../patches/INDEXTTS-25.md>)。

## CosyVoice3 隔离 worker

1. 操作者独立提供经过来源/许可核验的 CosyVoice 源码、Matcha-TTS 依赖、Python/CUDA 运行环境及本地模型。此发行包不提供“安装即复现”的 Cosy commit 保证。
2. 配置源码根目录、隔离 Python 可执行文件及模型目录，见[配置说明](<CONFIGURATION.md>)。worker 随 TimbreForge 包分发，模型栈不导入核心进程。
3. 模型目录须包含 `cosyvoice3.yaml`、`llm.pt`、`flow.pt`、`hift.pt`、`campplus.onnx`、`speech_tokenizer_v3.onnx` 及 `CosyVoice-BlankEN` 配置/tokenizer/权重。缺失文件不由适配器下载。
4. worker 设置 HF/Transformers 离线环境，禁用 TRT/vLLM，使用 CUDA + fp16、`stream=False`、固定 `speed=1.0`。这不是完整网络沙箱，也不证明任意第三方依赖绝不联网；请自行隔离运行环境。

模式必须区分：

- 参考音频 + 准确完整转录：`zero_shot`；不自动裁切参考，以免破坏转录对齐。
- 参考音频无转录：`cross_lingual`，不是自动识别出转录。
- 参考音频 + 表演指令：`instruct2`；该轮不消费参考转录，适配器记录警告。

CosyVoice 不消费 IndexTTS 的 beams、温度/采样组合、情感向量、扩散或时长参数；查看实际元数据 `unsupported_params`，不能将两后端参数名做语义等价映射。种子不能保证跨环境逐样本一致。一次只驻留一个重后端，操作员负责释放其他进程模型；Studio 锁不会卸载 ComfyUI 模型或管理全系统显存。

## 只读核对命令

在源码仓库根目录运行[backend_audit.py](<../tools/backend_audit.py>)；下列路径均为占位，替换为自己控制的本机资产根目录：

```powershell
python -B tools/backend_audit.py `
  --index-node ../external/ComfyUI-Index-TTS-25 `
  --index-model ../models/IndexTTS-2.5 `
  --cosy-source ../external/CosyVoice3 `
  --cosy-model ../models/Fun-CosyVoice3-0.5B
```

不传某项根目录会显示 `SKIP`，不是那项验收通过；工具始终核对 release patch。核对仅读取文件，完整模型哈希会产生较多磁盘 I/O。它不导入 torch、不执行 Git apply、不启动服务、不下载；文件匹配也不是安全认证。manifest 中没有收录的额外文件不在这个检查范围。

本次只读审计实际结果：补丁 1 / Index 修改源码 2 / Index 资产 34 / Cosy 源码 96 / Cosy 资产 17 均核对通过；反向 `git apply --reverse --check` 通过；隔离目录中的两个精确 base 文件前向 `git apply --check` 通过；3 项 Index 源码 AST 检查通过。**后端审计阶段未应用补丁到 live 源码、未下载模型。** 随后在已有隔离环境完成 CosyVoice3 私有短句和重启角色复用验收，见[测试说明](<TESTING.md>)；这不补齐未知源码 revision，也不是全新环境安装复现。IndexTTS 的 live schema 仍未重载，未宣称该后端本轮真实推理通过。未发布音频演示。
