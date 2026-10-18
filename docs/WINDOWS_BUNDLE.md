# Windows 小白整合包

下载并解压源码包后，双击根目录的 `安装并启动声线工作室.cmd`。脚本会自动创建隔离环境、安装界面依赖、创建本机配置并启动浏览器工作室。

模型权重没有放进整合包。首次启动后打开 `timbreforge.local.toml`，填写 `[qwen]` 下的 Qwen Python、VoiceDesign 模型和 Base 模型路径；没有模型时仍可打开界面查看功能和角色库。

开发者可在仓库根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File tools/build_windows_bundle.ps1
```

脚本会先运行发布检查，再生成 `dist/TimbreForge-Portable-0.6.0a1.zip`。它只包含源码、文档、公开 UI 素材、安装脚本和示例配置，不包含 `.local-data`、角色录音、日志、密钥、模型或外部后端目录。

首次生成音频仍需准备 Qwen3-TTS 环境和模型，具体见 [Qwen3-TTS 声线设计](VOICE_DESIGN.md)。

