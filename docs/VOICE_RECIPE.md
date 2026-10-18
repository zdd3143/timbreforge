# VoiceRecipe 声线配方

VoiceRecipe 把“冷淡的二次元萝莉音”拆成可检查、可比较、可复现的字段，而不是只保存一段无法解析的自然语言。

## 三段分离

| 分段 | 作用 | 字段 |
| --- | --- | --- |
| `identity` | 角色身份与风格 | 年龄感、性别表达、风格 |
| `timbre` | 音色本身 | 音高、明亮度、厚度、气声、粗糙度、鼻腔共鸣 |
| `delivery` | 这一条的表达方式 | 情绪、能量、语速、停顿、句尾、重音、咬字 |
| `text_style` | 台词文本风格 | 句子长度、避免过多感叹号、避免夸张表演 |

音色与表达分开后，声线微调可以只改一个表达字段而保持音色不变；文本风格只影响标点与提示，不会改写台词内容。

## 示例

“冷淡的二次元萝莉音”对应：

```json
{
  "identity": {"age": "young", "gender_expression": "feminine", "style": "anime"},
  "timbre": {"pitch": "high", "brightness": "soft", "body": "light",
             "breathiness": "low", "roughness": "low", "nasality": "low"},
  "delivery": {"emotion": "cold", "energy": "low", "pace": "slow",
               "pauses": "clear", "sentence_ending": "flat",
               "emphasis": "restrained", "articulation": "clear"},
  "text_style": {"sentence_length": "short",
                 "avoid_excessive_exclamation": true,
                 "avoid_overacting": true}
}
```

内置示例配方：`RECIPE_EXAMPLES['cold_anime_girl']`、`RECIPE_EXAMPLES['warm_narrator']`。

## 公共接口

```python
from timbreforge import voice_recipe

voice_recipe.empty_recipe()
voice_recipe.normalize(recipe)            # 严格校验并补齐默认值
voice_recipe.merge(recipe, patch)         # 局部修改
voice_recipe.describe(recipe)             # 一句话中文摘要
voice_recipe.details(recipe)              # 逐字段中文表
voice_recipe.timbre_prompt(recipe)        # 只含音色
voice_recipe.delivery_prompt(recipe)      # 只含表达
voice_recipe.design_prompt(recipe)        # VoiceDesign 指令
voice_recipe.variants(recipe, limit=6)     # 单维候选
voice_recipe.suggest_from_prompt(text)     # 关键词草稿（需确认）
voice_recipe.apply_text_style(text, recipe)
voice_recipe.text_style_violations(text, recipe)
```

## 校验规则

- 字段与取值来自固定词表；未知分组、未知字段、越界取值都会被拒绝，避免“看起来可复现但实际被静默丢弃”。
- 布尔字段必须是真正的布尔值。
- 配方存储于角色卡 `voice_recipe`；旧角色卡迁移时使用默认配方。

## 单维候选

`variants()` 只改变一个表达或文本字段，音色与身份不变：

```text
A：语速 偏慢 → 很慢
B：情绪 冷淡 → 平静
C：句尾 平直 → 轻柔
D：咬字 清晰 → 更利落
```

每个候选记录 `dimension`、`from`、`to` 与中文标签；生成时同时记录提示词、seed、模型与参数。音色字段不进入候选维度，避免把“微调”变成“换角色”。

## 文本风格

`apply_text_style` 只做标点归一化与重复感叹号压缩，**不会改写台词文字**。句子过长时只给出提示，由用户决定是否拆分；`text_style_violations` 可用于固定测试集检查。

## 关键词推断的边界

`suggest_from_prompt` 是关键词草稿，不是模型推理。界面会展示匹配到的字段与提示，保存前需要用户确认；未匹配字段保持默认值。

## 与生成流程的关系

- VoiceDesign 使用 `design_prompt()`（音色 + 表达 + 文本风格）。
- 克隆生成不支持表演指令；配方用于设计新参考音频、微调候选与评估记录，不会伪造“用指令控制 Qwen Base 情绪”。
- 角色包、调音会话和生成元数据都会保存候选当时的完整配方。

## 测试

```powershell
python -m unittest tests.test_voice_candidates
```

覆盖配方校验、单维候选、候选溯源、A/B 副本、保存为新版本与 Clone Prompt 重建。

