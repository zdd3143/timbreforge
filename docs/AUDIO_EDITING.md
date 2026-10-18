# 参考音频修整

音频修整只处理**角色的参考音频**，绝不覆盖原始录音。所有操作生成新的派生 WAV，处理参数、输入与输出哈希写入角色卡的 `audio_edit_chain`。

## 处理流程

```text
原始音频 -> 校验 -> 去静音 -> 去点击声 -> 降噪 -> EQ/压缩 -> 响度标准化
        -> 导出派生 WAV -> 生成压缩预览 -> 记录处理链 -> 旧 Clone Prompt 失效
```

实现模块：

| 模块 | 作用 |
| --- | --- |
| `timbreforge/audio_edit_schema.py` | 验证处理步骤、参数范围、处理链与历史事件 |
| `timbreforge/audio_edit_pipeline.py` | 解码、DSP、响度测量、哈希与原子写入 |
| `timbreforge/audio_presets.py` | `clean_voice`、`warm_voice`、`bright_voice`、`low_noise` |
| `timbreforge/audio_preview.py` | OGG/FLAC 压缩试听副本与清理 |
| `timbreforge/character_audio.py` | 按角色执行预设、回退、试听、历史与状态 |

## 四个内置预设

| 预设 | 用途 | 主要步骤 |
| --- | --- | --- |
| `clean_voice` 清晰人声 | 大多数干净录音 | 去静音、去点击、轻度降噪、高通 70 Hz + 3.2 kHz 微提、-16 LUFS |
| `warm_voice` 温暖声线 | 偏冷或偏薄的录音 | 去静音、降噪、低频搁架 +1.5 dB、高频搁架 -1.2 dB、-16 LUFS |
| `bright_voice` 明亮声线 | 闷、远、偏暗的录音 | 去静音、降噪、高通 90 Hz、4 kHz 微提、7 kHz 搁架 +2 dB、-16 LUFS |
| `low_noise` 强降噪 | 底噪明显 | 更激进降噪与去点击，可能损失气声与尾音，必须试听 |

预设是**起点**，不是音质保证。每次选择内置预设都会从角色卡的原始 `reference.wav` 重新处理，不会把上一次预设继续叠加到下一次；处理后请试听换气、齿音、尾音和情绪是否仍自然。

## 非破坏性保证

- **原始音频永不覆盖**：派生文件名带处理链与输入音频摘要；同名文件不会被覆盖。
- **处理过程先写临时文件**：重新校验尺寸与哈希后才 `os.replace` 发布。
- **处理后再次校验源哈希**：源文件在处理期间被修改时，派生文件会被删除并报错。
- **恢复原始版本**：写入 `revert_to_original` 事件并把当前标准音频指回 `reference.wav`；历史派生文件保留在角色目录中。
- **重复应用不同预设**：各生成独立文件，可在处理历史中查看。

## 处理链示例

```json
{
  "schema_version": 1,
  "source_audio": "reference.wav",
  "source_hash": "…",
  "operations": [
    {"name": "trim_silence", "threshold_db": -42, "keep_leading_ms": 50, "keep_trailing_ms": 80},
    {"name": "de_click", "threshold": 5.5, "window": 5},
    {"name": "denoise", "strength": 0.4, "window_ms": 32, "noise_seconds": 0.35},
    {"name": "eq", "bands": [
      {"type": "highpass", "freq": 70, "gain_db": 0, "q": 0.707},
      {"type": "peaking", "freq": 3200, "gain_db": 0.8, "q": 1.0}
    ]},
    {"name": "normalize", "target_lufs": -16, "ceiling_dbfs": -1}
  ],
  "output_audio": "audio/canonical-….wav",
  "output_hash": "…",
  "preset": "clean_voice",
  "sample_rate": 24000,
  "channels": 1,
  "duration_in": 3.2,
  "duration_out": 2.4,
  "created_at": "ISO 时间",
  "notes": ["去静音：保留 …", "响度标准化：…"]
}
```

允许的步骤与参数范围由 `audio_edit_schema.py` 定义；未知参数、越界值、重复的单实例步骤（去静音/降噪/去点击/标准化）都会被拒绝。`eq` 与 `compress` 可以出现多次。

## 测量口径与限制

- 响度按 ITU-R BS.1770 的 K 加权、400 ms 块、绝对 -70 LUFS 与相对 -10 LU 门限估算，输出 `LUFS`；这是工程近似，不是认证响度计。
- EQ 与降噪使用 NumPy 实现的 RBJ 双二阶频响与简化谱减，适合短参考音频，不是录音棚级母带处理。
- 输出超过 0 dBFS 时会整体衰减并写入说明，不会硬削波。
- WAV 主文件使用 FLOAT；预览使用 OGG/Vorbis，编码器不可用时退回 FLAC 或 PCM WAV。

## 与 Clone Prompt 的关系

任何音频编辑都会把角色卡 `canonical_audio` 指向新文件，并使旧 Clone Prompt 失效。下一次生成会回退到普通克隆，或由用户在「音频修整」页重新创建缓存。详见 [Clone Prompt 缓存](CLONE_PROMPTS.md)。

## 运行记录清理

角色卡目录是持久数据。服务关闭时，`outputs_dir` 下的设计会话、候选试听、生成记录、任务日志、Qwen 中间文件和临时目录会自动清空；如果配置把角色库放在输出目录内，程序会拒绝清理并提示先分离两个目录。

## 测试

```powershell
python -m unittest tests.test_audio_editing tests.test_character_audio_package
```

覆盖预设校验、去静音边界、响度目标、去点击、降噪、EQ 增益、压缩、原子写入、源文件不被修改、哈希一致性与预览编码。

