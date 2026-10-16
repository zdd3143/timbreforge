# Studio 使用指南（Alpha）

## 任务边界

- `clone`：使用参考音频生成新台词；`instruction` 应留空。
- `instruct`：在参考身份条件上增加表演指令。IndexTTS 仅将文字传给情感模型，不保证方言、语速或新身份；CosyVoice3 使用 instruct2，参考转录不参与该模式。
- `character`：读取角色卡的后端、原始参考、转录和参数，生成新台词；不是训练。
- `design`：未接入，明确报错。文字描述只作角色元数据，不是声音设计条件。

## 参考音频

只使用获得授权的参考；建议短、清晰、单说话者、无背景音乐。Studio 要求可解码、非空、非静音、采样有限且最长 30 秒，不自动裁切或改响度。上传音频与转录必须对应原始完整文件。IndexTTS 上游可能只使用约前 15 秒；其接口不消费参考转录，但角色卡仍会保存。

CosyVoice3 clone 在有准确转录时使用 zero_shot；无转录时使用 cross_lingual，不等于自动识别出转录。instruct2 不消费转录。IndexTTS 的 beams、扩散步数、CFG、时长等参数不能当作 CosyVoice 通用控制；查阅实际返回的 warnings / unsupported_params。

## 保存与复用角色

以下指令不会训练模型；第一个命令只校验并复制音频、写入角色元数据。占位音频路径须自行替换。

```bash
timbreforge-studio save-character --name "已授权角色甲" --reference authorized-reference.wav --ref-text "真实完整转录。" --backend cosyvoice3 --confirm-rights
timbreforge-studio characters
```

从结果复制角色 ID，再调用外部后端生成：

```bash
timbreforge-studio generate --task character --character REPLACE_WITH_CHARACTER_ID --text "角色的新台词。" --confirm-rights
```

角色 ID 是稳定标识，角色名可以重复。角色模式不要同时提供新的参考音频或转录。未传 `--seed` 时保留已保存种子；指定种子只覆盖种子，不应重置其他采样参数。留空指令保留已保存指令。

也可以在成功生成后用 `--save-name "角色名"` 保存角色。失败生成不应被描述为创建成功。

## 输出与可追溯性

Studio 对后端输出进行解码验证后保存 FLOAT WAV 与对应 JSON 元数据；不会把 FLAC 字节直接改名为 WAV，不做峰值/响度处理。元数据包含台词、参考路径/哈希、转录、参数、警告、后端结果等，可能包含敏感信息。不要把目录或日志无审查公开。

角色保存原始音频的逐字节副本以及 JSON 清单；哈希用于发现意外修改，**不是签名、防伪认证或授权证明**。不要加载不可信角色目录；不要让不可信用户写入角色库、模型或可执行脚本目录。

## 常见问题

| 现象 | 检查 |
| --- | --- |
| ComfyUI 无响应 | 在自己的外部环境中确认服务、配置 URL、input/output 映射；本项目不自动启动服务。 |
| 节点缺少 diffusion_steps / inference_cfg_rate | 按[补丁说明](<../patches/INDEXTTS-25.md>)确认兼容版本、补丁与重启。磁盘源码变化不代表运行节点已更新。 |
| CosyVoice 资产缺失 | 核验[配置](<CONFIGURATION.md>)中的 Python、repo、模型目录及实际必需文件；不自动下载。 |
| OOM / 长时间等待 | 一次仅一个重模型；释放其他后端；先用短台词、单 beam 基线。不要宣称 8 GB 必定够用。 |
| 同种子声音不同 | 外部模型版本、设备和随机算子可能影响结果；种子并非波形一致性保证。 |
| 检索库为空 | 旧数据集是可选工作流；不要为了 Studio 克隆而自动下载数据。 |
| 角色未在列表出现 | 刷新列表；无效/损坏角色会被跳过，直接 load 才显示具体错误；备份后核查清单与参考。 |

## 实验能力独立使用

实验合成器入口 `timbreforge-synth` / `python -m timbreforge.cli` 生成程序化音色，非自然语音；其元音“人声”也不是可说台词的模型。旧 `timbreforge-voicegen` 是检索/网格入口，不具有 Studio 任务的全部校验与存储语义。评分模型仅作实验排序，不能替代真实听测。
