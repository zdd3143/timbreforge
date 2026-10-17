# AutoDL 部署

AutoDL 适合承担本项目的 Qwen 推演、VoiceDesign 和 Base 克隆。当前仓库没有可直接访问你的 AutoDL 实例的连接，因此这里提供可复现脚本；拿到实例 SSH 命令后，再按本页执行实际安装和 GPU 验收。

如果要我代为执行安装，请提供 AutoDL 控制台显示的 SSH 命令中的主机、端口和用户名，并确保实例已经开机；不要发送密码、私钥或令牌。

## 建议配置

建议选择 Linux、Python 3.10/3.11/3.12、CUDA 与 PyTorch 匹配的 PyTorch 镜像，并预留模型磁盘空间。VoiceDesign 1.7B、Base 0.6B 和 Qwen3-1.7B 需要分别保存完整快照；如果磁盘有限，先只部署 Base 和 VoiceDesign，角色设定推演可以稍后加入。

Qwen 官方建议在新的隔离环境中安装 `qwen-tts`；本项目固定的兼容基线是 `qwen-tts==0.1.1`、`transformers==4.57.3`。模型权重不随仓库分发。[Qwen3-TTS 官方安装说明](https://github.com/QwenLM/Qwen3-TTS)显示，FlashAttention 2 可降低显存，但不是本项目启动的必要条件。

## 安装

```bash
git clone https://github.com/zdd3143/timbreforge.git
cd timbreforge
bash tools/autodl_install.sh
```

脚本会：

1. 创建 `.autodl-venv`，安装核心包、Gradio 和固定 Qwen 运行依赖。
2. 检查 CUDA、GPU 名称和计算能力。
3. 写入 `timbreforge.autodl.toml`。
4. 默认不下载模型权重，避免未经确认地消耗 AutoDL 磁盘和流量。

确认模型许可、磁盘和费用后，再执行：

```bash
bash tools/autodl_install.sh --download-models
```

生产环境建议在 AutoDL 数据盘保存 `models/` 和 `.local-data/`，不要放在会随实例释放的临时目录。

## 启动与访问

```bash
source .autodl-venv/bin/activate
python -m timbreforge.launcher --config timbreforge.autodl.toml --host 0.0.0.0 --port 6006
```

AutoDL 自定义服务通常使用容器 `6006` 端口；也可以从本地建立 SSH 隧道，把实例的 6006 转到本机：

```bash
ssh -CNg -L 6006:127.0.0.1:6006 root@<AutoDL-SSH-Host> -p <AutoDL-SSH-Port>
```

然后访问 `http://127.0.0.1:6006`。不要把服务绑定到公网而不设置访问保护；工作室是面向本机可信用户的 Alpha，没有多用户鉴权。

## 验收

```bash
python -m timbreforge.launcher --config timbreforge.autodl.toml --check
python -m unittest discover -s tests -p 'test_*.py' -v
nvidia-smi
```

先用「简化版」短句测试，再测试角色设定推演，最后测试参考克隆。AutoDL 生成的模型日志和音频仍属于你的本地数据；不要上传到公开 issue 或仓库。

如果生成时显存不足，先只保留一个 Qwen worker，缩短台词，使用 Qwen Base 的默认 BF16，并确认 ComfyUI、评分模型和其它 notebook 已释放。工作室会串行模型进程并清理已配置的闲置 ComfyUI 缓存，但不会结束你在 AutoDL 上运行的其它程序。
