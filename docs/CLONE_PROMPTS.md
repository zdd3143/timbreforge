# Qwen Clone Prompt 缓存

Qwen Base 的声音条件由参考音频与其逐字稿编码得到。重复编码会浪费显存和时间，因此 TimbreForge 把条件张量缓存在角色目录的 `prompts/` 下，并在使用前验证它与角色卡完全一致。

## 缓存内容

```text
prompts/
  clone_prompt.json                  # 元数据（角色 ID、模型、哈希、模式、时间）
  clone-icl-<hash>.pt                # ICL 模式条件张量
  clone-xvector-<hash>.pt            # x-vector-only 模式条件张量
```

`clone_prompt.json` 记录：

- 角色 ID 与角色版本；
- 克隆模式（`icl` / `xvector`）；
- 模型名称、版本标记、目录、`config.json` SHA-256、`qwen-tts` 版本；
- 参考音频相对路径与 SHA-256、逐字稿 SHA-256、处理链 SHA-256；
- Prompt 文件名、大小与 SHA-256、张量形状；
- 创建时间、设备与精度。

`.pt` 文件由 worker 以 `torch.save` 写成纯张量字典；加载时使用 `weights_only=True`，并检查文件哈希、格式版本、克隆模式与逐字稿是否匹配。

## 公共接口

```python
from timbreforge import clone_prompts

clone_prompts.create_clone_prompt(role_id, device='cuda:0')
clone_prompts.load_clone_prompt(role_id)
clone_prompts.validate_clone_prompt(role_id)
clone_prompts.invalidate_clone_prompt(role_id, '原因')
clone_prompts.status_text(report)
```

- `create_clone_prompt` 会复用仍然有效的缓存；`force=True` 才重建。旧文件在重建前删除，避免留下与角色卡不匹配的张量。
- `validate_clone_prompt` 返回 `{'ok', 'status', 'reasons', 'metadata', 'path', 'card'}`；`status` 为 `valid`、`missing`、`stale`、`corrupt`、`unsupported` 之一。
- `load_clone_prompt` 仅在有效时返回路径与元数据，否则抛出 `ClonePromptMissing` / `ClonePromptStale`。
- `invalidate_clone_prompt` 删除 `prompts/` 内由本工具生成的文件并清空角色卡指针，可重复调用。

## 缓存失效条件

以下任一变化都会让缓存失效：

- 参考音频字节变化（含重新应用音频预设、采用微调候选）；
- 逐字稿变化；
- 角色卡登记的模型名称或版本变化；
- 实际 `config.json` 内容变化（本地模型被替换）；
- `qwen-tts` 包版本变化；
- ICL / x-vector-only 模式切换；
- 音频处理链变化；
- 角色卡不再指向该缓存文件。

角色卡更新会自动清空指针并把原因追加到 `provenance.clone_prompt_invalidations`（最多 20 条）。`validate_clone_prompt` 会逐项比较并返回中文原因。

## 回退流程

生成入口 `timbreforge.generation.generate_for_character` 在缓存缺失、失效或损坏时：

1. 记录 `clone_prompt.status = fallback` 与具体原因；
2. 删除已证明不匹配的张量；
3. 使用角色卡当前音频与逐字稿执行普通克隆；
4. 把原因写入生成元数据和界面提示。

不会静默使用旧缓存，也不会因为缓存损坏而让角色卡不可用。

## 模式选择

- 默认使用 **ICL**，相似度与表达控制更好，需要逐字稿和非空参考音频。
- 显存紧张或只需轻量运行时切换为 **x-vector-only**：只使用说话人嵌入，不要求逐字稿参与条件。角色卡 `clone_mode` 切换会使缓存失效。
- 两种模式都不支持“用一句表演指令改变情绪”；Qwen Base 没有独立表演指令。

## 限制

- 缓存与模型、`qwen-tts` 版本和主机相关，不应跨机器复制；角色包也刻意不包含它。
- 张量缓存不加密，保存在本机角色目录；请像保护录音一样保护该目录。
- 创建缓存需要加载一次模型；这不是训练，也不会改变声音身份。

## 测试

```powershell
python -m unittest tests.test_clone_prompts tests.test_qwen_backend tests.test_generation
```

测试使用临时角色卡、伪模型目录与假 worker，覆盖创建、复用、强制重建、逐字稿/音频/模式/模型变化导致的失效、文件损坏、指针缺失、非法路径与回退原因。

