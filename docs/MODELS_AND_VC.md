# 模型管理、统一适配器与 Voice Conversion

## 本地模型注册表

实现：`timbreforge/model_registry.py`，默认文件为 `<本地数据目录>/models/registry.json`。

```python
from timbreforge.model_registry import ModelRegistry

registry = ModelRegistry()
entry = registry.register('Qwen3-TTS-12Hz-0.6B-Base', r'D:\models\Base-0.6B',
                          'local-2026-10', kind='tts_base')
registry.verify(entry['id'])
registry.activate(role_id, entry['id'])
registry.rollback(role_id)
registry.remove(entry['id'])
```

- `register` 只登记**已经存在**的本地目录：要求 `config.json`，TTS/VC 模型还要求至少一个 `*.safetensors`。不下载、不复制权重。
- 记录名称、版本标记、类型、绝对路径、`config.json` SHA-256、目录体积与登记时间；同一名称+版本指向不同目录时必须显式 `replace=True`。
- `verify` 重新计算配置哈希并检查权重是否缺失，不信任注册表记录。
- `activate` 更新角色卡的模型名称/版本，递增角色版本并使 Clone Prompt 失效；每次切换都写入 `provenance.model_activations`。
- `rollback` 恢复上一次切换前的模型版本；没有历史时明确报错。
- `remove` 只删除注册表记录，**不删除磁盘模型**。

模型管理页面还显示内存状态、最近任务和清理入口。切换模型不会自动加载模型。

## 统一模型适配器

实现：`timbreforge/backends.py`。

```python
from timbreforge.backends import BackendRequest, default_registry

backend = default_registry().get('qwen3')
request = BackendRequest(mode='clone', text='新台词', params=qwen_params,
                         ref_audio=r'…\reference.wav', ref_text='逐字稿')
result = backend.synthesize(request)
```

- `TTSBackend` 抽象接口：`spec`、`available()`、`synthesize()`、`create_clone_prompt()`。
- `BackendSpec` 声明名称、类型、能力（`design` / `clone` / `clone_prompt` / `longform` / `vc`）、模型名称/版本与说明。
- `BackendRegistry` 按名称注册，拒绝重复名称；`require(name, capability)` 会检查本地文件是否齐备并给出中文原因。
- `Qwen3Backend` 把请求映射到现有 Qwen 适配器：设计走 `voice_design`，克隆走 `voice_clone`，并原样传递 Clone Prompt 与参考音频哈希。`available()` 只检查文件，不加载模型。
- `BackendResult` 统一返回音频路径、seed、参数、模型信息、克隆模式、时长与原始结果。

新模型必须实现该接口，并遵守“同时只运行一个重型任务、失败清理临时文件、不覆盖原始音频”的约束。生成页面在本地 Qwen 文件齐备时会通过该适配器调用；文件缺失时明确报错，不会静默换模型。

## Voice Conversion（实验性）

实现：`timbreforge/vc_backend.py`。VC 是独立后端，不与普通 TTS 混为一个流程。

```text
已有音频 -> 保留内容与情绪 -> 转换目标角色音色 -> 音频修整 -> 保存派生结果
```

```python
from timbreforge.vc_backend import ExternalVoiceConversionBackend

backend = ExternalVoiceConversionBackend(
    python=r'D:\envs\vc\python.exe',
    script=r'D:\vc\convert.py',
    model_dir=r'D:\models\vc-model',
)
backend.convert(r'D:\audio\source.wav', role_id,
                preserve_prosody=True, output_dir=r'D:\audio\vc-out')
```

外部脚本使用文件 JSON 协议：读取 `--request`，写出 `--result`，并在 `output` 路径生成 FLOAT WAV。请求包含源音频、目标角色当前参考音频、双方 SHA-256、`preserve_prosody`、模型目录与输出路径。返回结果必须声明 `file`、`sample_rate`、`duration`、`source_sha256`、`target_sha256`、`preserve_prosody` 与 `model_dir`。

安全与验证：

- 目标角色必须来自角色库；目标音频使用 `canonical_audio`（存在时）或原始参考。
- 源音频必须可解码、有限、非静音且不超过 300 秒。
- 转换前后重新计算源/目标哈希，任一方变化即拒绝结果。
- 输出必须是 FLOAT WAV、有限、非静音、采样率 8–192 kHz、不超过 300 秒；失败会删除本次部分输出并给出日志路径。
- 取消任务会终止子进程并删除部分输出。
- 只在本机读写，不上传音频，不自动下载模型。

**能力边界**：VC 可能改变情绪、语速和咬字，不是完美的音色替换；第一阶段只做本机实验，不接入正式生成流程，也不保证跨环境一致。使用前请自行确认模型许可证、声音授权与显存需求。

## 测试

```powershell
python -m unittest tests.test_backends tests.test_vc_backend
```

覆盖注册表重复/缺失/能力检查、Qwen 适配器请求映射与 Clone Prompt 互斥、可用性原因、外部 VC 协议、哈希变化拒绝、PCM/非有限输出拒绝、取消与日志提示。

