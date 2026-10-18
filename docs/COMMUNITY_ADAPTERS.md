# 社区适配器

TimbreForge 支持导入本地 PEFT LoRA 适配器，并将它绑定到 Qwen3 角色卡。当前入口只接受已经下载到本机的文件夹或 ZIP，不会自动执行仓库脚本，也不会静默下载模型。

## 适配器目录

一个可导入的 Qwen3-TTS LoRA 至少包含：

```text
my-lora/
├── adapter_config.json
└── adapter_model.safetensors
```

不接受 `adapter_model.bin` 等 pickle 权重。`adapter_config.json` 必须声明 `peft_type: LORA`、`r`、`lora_alpha` 和 `target_modules`。

## 使用流程

1. 打开「模型管理 → 社区适配器」。
2. 导入 ZIP，或填写已经解压的适配器目录。
3. 填写作者、来源 URL、许可证和 Base 版本。
4. 点击「导入并校验」。
5. 选择 Qwen3 角色和 LoRA scale，点击「绑定适配器」。
6. 生成时系统会在独立 Qwen worker 中加载 Base + LoRA，并把适配器哈希和 scale 写入生成记录。

建议先试听 `0.2`、`0.3`、`0.35`、`0.5` 四个 scale。不同适配器不保证同一 scale 产生相同强度。

## 失败行为

以下情况会拒绝绑定或生成：

- `adapter_config.json` 或 safetensors 缺失。
- 文件哈希与注册记录不一致。
- 适配器声明的 Base 模型与角色模型不匹配。
- Qwen 环境缺少 `peft`。
- 当前模型结构没有可适配的 `talker.model`。

系统不会自动退回 Base 模型，因为那会让用户误以为 LoRA 已生效。可以先解除角色绑定，再使用 Base 生成。

## 角色卡保存内容

角色卡只保存适配器 ID、scale、Base 名称和文件哈希，不复制权重：

```json
{
  "adapter": {
    "id": "...",
    "name": "角色 LoRA",
    "scale": 0.35,
    "artifact_sha256": "...",
    "config_sha256": "...",
    "base_model": "Qwen3-TTS-12Hz-0.6B-Base"
  }
}
```

## 兼容边界

当前版本只实现 Qwen3-TTS PEFT LoRA。CosyVoice、F5-TTS、合并后的完整模型和 Full SFT checkpoint 需要各自的加载器，不能直接放入这个目录冒充 LoRA。

适配器来自社区时，需要同时确认 Base 模型版本、训练代码版本、采样率、许可证和声音使用授权。适配器文件本身不代表录音或角色声音拥有公开授权。

