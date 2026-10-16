# 配置说明（Alpha）

配置只声明本机路径和外部环境，不下载模型、不安装依赖、不启动服务。默认值适合核心测试和无资产安装，但不代表任何重模型已就绪。

## 配置文件

通过 `TIMBREFORGE_CONFIG` 指定 TOML 文件，或由支持配置的 CLI 使用 `--config PATH`。配置文件按以下表组织：

```toml
[qwen]
python = "../external/qwen3-tts/.venv/Scripts/python.exe"
design_model = "../external/qwen3-tts/models/VoiceDesign-1.7B"
base_model = "../external/qwen3-tts/models/Base-0.6B"

[comfy]
url = "http://127.0.0.1:8188"
input_dir = "../ComfyUI/input"
output_dir = "../ComfyUI/output"

[cosyvoice]
repo = "../external/CosyVoice3"
python = "../external/CosyVoice3/upy/python.exe"
model = "../external/CosyVoice3/pretrained_models/Fun-CosyVoice3-0.5B"

[paths]
outputs_dir = "./outputs"
characters_dir = "./characters"

[scorer]
python = "../ComfyUI/venv/Scripts/python.exe"
mos_models = "../models_mos"
indextts_repo = "../external/index-tts"
campplus_checkpoint = "../models/campplus_cn_common.bin"
```

表和字段均可省略；当前实现忽略未识别字段，因此拼写错误可能静默回落默认值，修改配置后务必核查实际路径。不要把字段当作可拼接的命令。相对路径在 TOML 文件位置解析；环境变量里的相对路径按当前工作目录解析。建议使用绝对路径，尤其是多个工作目录或服务账户场景。

当前字段与代码 API 对应：

| 表 | 字段 | 环境变量 | 用途 |
| --- | --- | --- | --- |
| `qwen` | `python` | `TIMBREFORGE_QWEN_PYTHON` | 独立 Qwen Python，不将模型栈导入核心环境。 |
| `qwen` | `design_model` | `TIMBREFORGE_QWEN_DESIGN_MODEL` | 完整固定 revision 的 VoiceDesign 本地快照。 |
| `qwen` | `base_model` | `TIMBREFORGE_QWEN_BASE_MODEL` | 完整固定 revision 的 Base 本地快照，用于固定设计身份复用。 |
| `comfy` | `url` | `TIMBREFORGE_COMFY_URL` | IndexTTS/ComfyUI HTTP base URL。 |
| `comfy` | `input_dir` | `TIMBREFORGE_COMFY_INPUT_DIR` | 参考音频复制目标；不要指向不可信共享目录。 |
| `comfy` | `output_dir` | `TIMBREFORGE_COMFY_OUTPUT_DIR` | 后端输出目录。 |
| `cosyvoice` | `repo` | `TIMBREFORGE_COSYVOICE_REPO` | 隔离 CosyVoice 仓库根目录。 |
| `cosyvoice` | `python` | `TIMBREFORGE_COSYVOICE_PYTHON` | 隔离环境 Python 可执行文件。 |
| `cosyvoice` | `model` | `TIMBREFORGE_COSYVOICE_MODEL` | 本地模型目录。 |
| `paths` | `outputs_dir` | `TIMBREFORGE_OUTPUTS_DIR` | Studio/本地生成输出根目录。 |
| `paths` | `characters_dir` | `TIMBREFORGE_CHARACTERS_DIR` | 角色卡与原始参考副本目录。 |
| `scorer` | `python` | `TIMBREFORGE_SCORER_PYTHON` | CAM++/MOS 隔离 Python。 |
| `scorer` | `mos_models` | `TIMBREFORGE_MOS_MODELS_DIR` | 本地 MOS 权重目录。 |
| `scorer` | `indextts_repo` | `TIMBREFORGE_INDEXTTS_REPO` | 可选 IndexTTS 源码/集成位置；不自动检出。 |
| `scorer` | `campplus_checkpoint` | `TIMBREFORGE_CAMPPLUS_CHECKPOINT` | 可选 CAM++ 权重；不自动下载。 |

代码接口为 `get_config()` 读取合并后配置，`set_config_path(path)` 固定当前配置路径；配置模块应优先遵循环境覆盖，再使用 TOML，再使用平台默认。`TIMBREFORGE_CONFIG` 只选择文件，不应被误解为把任意字符串当作路径字段。

`indextts_repo` 与 `campplus_checkpoint` 是本配置模块当前字段；评分/后端脚本是否在每个发行版本使用它们，仍以对应 adapter 行为与后端锁定文档为准。

## 默认数据位置

未显式配置时，用户数据默认：

- Windows：`%LOCALAPPDATA%\TimbreForge`
- Linux/POSIX：`~/.local/TimbreForge`

默认运行时模型/外部仓库路径不是自动下载目的地。若操作者手工提供项目本地 `tools/` 外部资产，适配器可以使用，但发行包不携带这些资产，也不保证默认可用。不要把 `data_ref`、模型、角色卡或私有音频提交到仓库。

配置改变后重启使用它的 CLI/进程；已经运行的 ComfyUI 不会因 TOML 或磁盘补丁自动重载节点。Studio 的进程锁只串行本进程请求，不负责卸载其他进程模型。

## 安全与检查

配置文件可能包含本机路径、账号挂载点和私有仓库位置，不要公开分享；环境变量同样可能出现在诊断/进程列表中。只将路径指向自己控制的目录，核验 Python 可执行文件和 worker 来源，避免路径劫持。设置模型/外部 Python 不等于授予这些代码安全性。

安装核心包后可先运行：

```bash
python -m timbreforge.studio_cli capabilities
```

这个命令只做本地文件状态检查，不探测硬件、不联网、不下载。真实后端启动、模型加载和可用性要在隔离环境单独验证。
