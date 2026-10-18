"""Structured VoiceRecipe: timbre, delivery and text style as separate fields.

The recipe exists so a voice can be described, diffed and re-generated without
depending on one opaque natural-language sentence. Timbre defines *what the
voice is*; delivery defines *how this take performs*; text style adjusts the
script. Every field has a closed vocabulary, so a saved recipe is checkable and
a candidate can change exactly one dimension at a time.
"""
from __future__ import annotations

import math

# field -> ordered allowed values. Defaults are the first value of each field.
VOCABULARY = {
    'identity': {
        'age': ('young', 'teen', 'child', 'mature', 'elder'),
        'gender_expression': ('neutral', 'feminine', 'masculine', 'androgynous'),
        'style': ('natural', 'anime', 'narrative', 'character', 'radio_drama'),
    },
    'timbre': {
        'pitch': ('mid', 'very_low', 'low', 'high', 'very_high'),
        'brightness': ('neutral', 'dark', 'soft', 'bright', 'sparkling'),
        'body': ('medium', 'light', 'full', 'thick'),
        'breathiness': ('low', 'none', 'medium', 'high'),
        'roughness': ('low', 'none', 'medium', 'high'),
        'nasality': ('low', 'none', 'medium', 'high'),
    },
    'delivery': {
        'emotion': ('neutral', 'cold', 'warm', 'gentle', 'angry', 'sad',
                    'happy', 'tense', 'playful'),
        'energy': ('medium', 'very_low', 'low', 'high'),
        'pace': ('medium', 'very_slow', 'slow', 'fast', 'very_fast'),
        'pauses': ('clear', 'none', 'few', 'frequent'),
        'sentence_ending': ('flat', 'rising', 'soft', 'firm'),
        'emphasis': ('restrained', 'balanced', 'strong'),
        'articulation': ('standard', 'relaxed', 'clear', 'crisp'),
    },
    'text_style': {
        'sentence_length': ('medium', 'short', 'long'),
        'avoid_excessive_exclamation': (True, False),
        'avoid_overacting': (True, False),
    },
}

LABELS = {
    'age': {'young': '年轻', 'teen': '少年', 'child': '孩童', 'mature': '成熟', 'elder': '年长'},
    'gender_expression': {'neutral': '中性', 'feminine': '女性化', 'masculine': '男性化',
                          'androgynous': '雌雄莫辨'},
    'style': {'natural': '自然口语', 'anime': '二次元动画', 'narrative': '旁白叙述',
              'character': '角色扮演', 'radio_drama': '广播剧'},
    'pitch': {'mid': '中等', 'very_low': '极低', 'low': '偏低', 'high': '偏高',
              'very_high': '极高'},
    'brightness': {'neutral': '中性', 'dark': '偏暗', 'soft': '柔和', 'bright': '明亮',
                   'sparkling': '清亮'},
    'body': {'medium': '适中', 'light': '轻薄', 'full': '厚实', 'thick': '厚重'},
    'breathiness': {'low': '少', 'none': '无', 'medium': '适中', 'high': '多'},
    'roughness': {'low': '低', 'none': '无', 'medium': '适中', 'high': '高'},
    'nasality': {'low': '弱', 'none': '无', 'medium': '适中', 'high': '强'},
    'emotion': {'neutral': '平静', 'cold': '冷淡', 'warm': '温暖', 'gentle': '温柔',
                'angry': '愤怒', 'sad': '悲伤', 'happy': '愉快', 'tense': '紧张',
                'playful': '俏皮'},
    'energy': {'medium': '中等', 'very_low': '很低', 'low': '低', 'high': '高'},
    'pace': {'medium': '中等', 'very_slow': '很慢', 'slow': '偏慢', 'fast': '偏快',
             'very_fast': '很快'},
    'pauses': {'clear': '清晰', 'none': '不明显', 'few': '少', 'frequent': '频繁'},
    'sentence_ending': {'flat': '平直', 'rising': '上扬', 'soft': '轻柔', 'firm': '坚定'},
    'emphasis': {'restrained': '克制', 'balanced': '均衡', 'strong': '强烈'},
    'articulation': {'standard': '标准', 'relaxed': '放松', 'clear': '清晰', 'crisp': '咬字利落'},
    'sentence_length': {'medium': '中等', 'short': '简短', 'long': '较长'},
}

GROUPS = tuple(VOCABULARY)
FIELD_LABELS = {
    'age': '年龄感', 'gender_expression': '性别表达', 'style': '风格',
    'pitch': '音高', 'brightness': '明亮度', 'body': '厚度',
    'breathiness': '气声', 'roughness': '粗糙度', 'nasality': '鼻腔共鸣',
    'emotion': '情绪', 'energy': '能量', 'pace': '语速', 'pauses': '停顿',
    'sentence_ending': '句尾', 'emphasis': '重音', 'articulation': '咬字',
    'sentence_length': '句子长度',
    'avoid_excessive_exclamation': '避免过多感叹号',
    'avoid_overacting': '避免夸张表演',
}

DEFAULT_VALUES = {group: {field: values[0] for field, values in fields.items()}
                  for group, fields in VOCABULARY.items()}

# "冷淡的二次元萝莉音" is not saved as one sentence but as these fields.
RECIPE_EXAMPLES = {
    'cold_anime_girl': {
        'label': '冷淡的二次元萝莉音',
        'recipe': {
            'identity': {'age': 'young', 'gender_expression': 'feminine', 'style': 'anime'},
            'timbre': {'pitch': 'high', 'brightness': 'soft', 'body': 'light',
                       'breathiness': 'low', 'roughness': 'low', 'nasality': 'low'},
            'delivery': {'emotion': 'cold', 'energy': 'low', 'pace': 'slow',
                         'pauses': 'clear', 'sentence_ending': 'flat',
                         'emphasis': 'restrained', 'articulation': 'clear'},
            'text_style': {'sentence_length': 'short',
                           'avoid_excessive_exclamation': True,
                           'avoid_overacting': True},
        },
    },
    'warm_narrator': {
        'label': '温暖旁白',
        'recipe': {
            'identity': {'age': 'mature', 'gender_expression': 'neutral', 'style': 'narrative'},
            'timbre': {'pitch': 'low', 'brightness': 'soft', 'body': 'full',
                       'breathiness': 'low', 'roughness': 'low', 'nasality': 'none'},
            'delivery': {'emotion': 'warm', 'energy': 'low', 'pace': 'slow',
                         'pauses': 'clear', 'sentence_ending': 'soft',
                         'emphasis': 'balanced', 'articulation': 'clear'},
            'text_style': {'sentence_length': 'medium',
                           'avoid_excessive_exclamation': True,
                           'avoid_overacting': True},
        },
    },
}


class RecipeError(ValueError):
    """Raised when a recipe contains unknown fields or values."""


def empty_recipe():
    """Return a fully populated recipe with documented defaults."""
    return {group: dict(values) for group, values in DEFAULT_VALUES.items()}


def normalize(recipe):
    """Validate a complete recipe and return a normalized copy.

    Unknown groups, unknown fields and out-of-vocabulary values are rejected:
    silently dropping a typo would make a saved voice look reproducible when it
    is not.
    """
    if not isinstance(recipe, dict):
        raise RecipeError('VoiceRecipe 必须是对象。')
    unknown_groups = set(recipe) - set(GROUPS)
    if unknown_groups:
        raise RecipeError(f'VoiceRecipe 含未知分组：{sorted(unknown_groups)}。')
    normalized = empty_recipe()
    for group in GROUPS:
        values = recipe.get(group, {})
        if not isinstance(values, dict):
            raise RecipeError(f'VoiceRecipe.{group} 必须是对象。')
        unknown = set(values) - set(VOCABULARY[group])
        if unknown:
            raise RecipeError(f'VoiceRecipe.{group} 含未知字段：{sorted(unknown)}。')
        for field, allowed in VOCABULARY[group].items():
            value = values.get(field, DEFAULT_VALUES[group][field])
            if not _allowed(value, allowed):
                raise RecipeError(
                    f'VoiceRecipe.{group}.{field} 不支持取值 {value!r}；'
                    f'允许：{", ".join(map(str, allowed))}。')
            normalized[group][field] = value
    return normalized


def _allowed(value, allowed):
    if isinstance(value, bool):
        return any(value is item for item in allowed if isinstance(item, bool))
    if not isinstance(value, str) or value not in allowed:
        return False
    return True


def merge(base, patch):
    """Apply a partial nested patch, then normalize the result."""
    if patch is None:
        return normalize(base or empty_recipe())
    if not isinstance(patch, dict):
        raise RecipeError('VoiceRecipe 补丁必须是对象。')
    combined = normalize(base or empty_recipe()) if base else empty_recipe()
    for group, values in patch.items():
        if group not in VOCABULARY:
            raise RecipeError(f'VoiceRecipe 补丁含未知分组：{group!r}。')
        if not isinstance(values, dict):
            raise RecipeError(f'VoiceRecipe 补丁 {group} 必须是对象。')
        for field, value in values.items():
            if field not in VOCABULARY[group]:
                raise RecipeError(f'VoiceRecipe 补丁含未知字段：{group}.{field}。')
            combined[group][field] = value
    return normalize(combined)


def is_default(recipe):
    """True when every field still holds its default value."""
    return normalize(recipe) == empty_recipe()


def describe(recipe):
    """One-line Chinese summary used on cards and in provenance."""
    value = normalize(recipe)
    identity = value['identity']
    timbre = value['timbre']
    delivery = value['delivery']
    return (f"{LABELS['age'][identity['age']]}"
            f"{LABELS['gender_expression'][identity['gender_expression']]}声线"
            f"（{LABELS['style'][identity['style']]}），"
            f"音高{LABELS['pitch'][timbre['pitch']]}、"
            f"{LABELS['brightness'][timbre['brightness']]}、"
            f"{LABELS['body'][timbre['body']]}、气声{LABELS['breathiness'][timbre['breathiness']]}，"
            f"情绪{LABELS['emotion'][delivery['emotion']]}、"
            f"能量{LABELS['energy'][delivery['energy']]}、"
            f"语速{LABELS['pace'][delivery['pace']]}、"
            f"句尾{LABELS['sentence_ending'][delivery['sentence_ending']]}")


def details(recipe):
    """Field-by-field ``(group, label, value)`` triples for a settings table."""
    value = normalize(recipe)
    rows = []
    for group in GROUPS:
        for field in VOCABULARY[group]:
            item = value[group][field]
            label = FIELD_LABELS.get(field, field)
            if isinstance(item, bool):
                rendered = '是' if item else '否'
            else:
                rendered = LABELS.get(field, {}).get(item, item)
            rows.append((group, label, rendered))
    return rows


def timbre_prompt(recipe):
    """Voice identity only: pitch, weight, colour, breath and resonance."""
    value = normalize(recipe)
    identity, timbre = value['identity'], value['timbre']
    return (f"{LABELS['age'][identity['age']]}"
            f"{LABELS['gender_expression'][identity['gender_expression']]}声线，"
            f"风格{LABELS['style'][identity['style']]}；"
            f"音高{LABELS['pitch'][timbre['pitch']]}，"
            f"音色{LABELS['brightness'][timbre['brightness']]}，"
            f"厚度{LABELS['body'][timbre['body']]}，"
            f"气声{LABELS['breathiness'][timbre['breathiness']]}，"
            f"粗糙度{LABELS['roughness'][timbre['roughness']]}，"
            f"鼻腔共鸣{LABELS['nasality'][timbre['nasality']]}。")


def delivery_prompt(recipe):
    """Performance only: emotion, energy, pace, pauses, endings, emphasis."""
    value = normalize(recipe)['delivery']
    return (f"情绪{LABELS['emotion'][value['emotion']]}，"
            f"能量{LABELS['energy'][value['energy']]}，"
            f"语速{LABELS['pace'][value['pace']]}，"
            f"停顿{LABELS['pauses'][value['pauses']]}，"
            f"句尾{LABELS['sentence_ending'][value['sentence_ending']]}，"
            f"重音{LABELS['emphasis'][value['emphasis']]}，"
            f"咬字{LABELS['articulation'][value['articulation']]}。")


def text_style_prompt(recipe):
    value = normalize(recipe)['text_style']
    parts = [f"句子{LABELS['sentence_length'][value['sentence_length']]}"]
    if value['avoid_excessive_exclamation']:
        parts.append('避免过多感叹号')
    if value['avoid_overacting']:
        parts.append('避免夸张表演')
    return '文本风格：' + '，'.join(parts) + '。'


def design_prompt(recipe, *, include_text_style=True):
    """Compose the VoiceDesign instruction from separated sections."""
    value = normalize(recipe)
    parts = [timbre_prompt(value), delivery_prompt(value)]
    if include_text_style:
        parts.append(text_style_prompt(value))
    return ''.join(parts)


def variants(base, *, dimensions=None, limit=6):
    """Candidates that each change exactly one delivery/text dimension.

    Timbre and identity are deliberately excluded: a tuning candidate should be
    comparable to the others, not turn into a different character.
    """
    recipe = normalize(base)
    if type(limit) is not int or not 1 <= limit <= 8:
        raise RecipeError('候选数量必须在 1–8 之间。')
    fields = [('delivery', 'pace'), ('delivery', 'emotion'), ('delivery', 'energy'),
              ('delivery', 'pauses'), ('delivery', 'sentence_ending'),
              ('delivery', 'emphasis'), ('delivery', 'articulation'),
              ('text_style', 'sentence_length')]
    if dimensions is not None:
        requested = set(dimensions)
        unknown = requested - {field for _group, field in fields}
        if unknown:
            raise RecipeError(f'不支持作为候选维度的字段：{sorted(unknown)}。')
        fields = [item for item in fields if item[1] in requested]
    result = []
    for group, field in fields:
        current = recipe[group][field]
        alternatives = [value for value in VOCABULARY[group][field] if value != current]
        for value in alternatives[:1]:
            candidate = normalize(recipe)
            candidate[group][field] = value
            label = (f"{FIELD_LABELS[field]}："
                     f"{_render(field, current)} → {_render(field, value)}")
            result.append({'label': label, 'recipe': candidate,
                           'dimension': field, 'from': current, 'to': value})
            break
        if len(result) >= limit:
            break
    if not result:
        raise RecipeError('当前配方没有可用的单维候选；请先调整音色或表达字段。')
    return result


def _render(field, value):
    return LABELS.get(field, {}).get(value, value)


_KEYWORDS = {
    'pitch': {'极高': 'very_high', '偏高': 'high', '高音': 'high', '低沉': 'low',
              '偏低': 'low', '极低': 'very_low', '萝莉': 'high'},
    'brightness': {'明亮': 'bright', '清亮': 'sparkling', '沙哑': 'dark', '柔和': 'soft',
                   '闷': 'dark'},
    'body': {'轻薄': 'light', '厚实': 'full', '厚重': 'thick', '纤细': 'light'},
    'breathiness': {'气声': 'medium', '气息': 'medium'},
    'roughness': {'粗糙': 'medium', '沙哑': 'medium'},
    'nasality': {'鼻腔': 'medium', '鼻音': 'medium'},
    'emotion': {'冷淡': 'cold', '冷漠': 'cold', '温暖': 'warm', '温柔': 'gentle',
                '愤怒': 'angry', '悲伤': 'sad', '愉快': 'happy', '俏皮': 'playful',
                '紧张': 'tense'},
    'energy': {'低能量': 'low', '有气无力': 'very_low', '高能量': 'high', '亢奋': 'high'},
    'pace': {'语速慢': 'slow', '缓慢': 'very_slow', '语速快': 'fast', '急促': 'very_fast'},
    'sentence_ending': {'句尾平直': 'flat', '上扬': 'rising', '句尾上扬': 'rising'},
    'emphasis': {'克制': 'restrained', '重音强': 'strong'},
    'articulation': {'咬字清晰': 'clear', '吐字清晰': 'crisp', '含糊': 'relaxed'},
    'age': {'萝莉': 'young', '少女': 'young', '少年': 'teen', '孩童': 'child',
            '成熟': 'mature', '年长': 'elder'},
    'style': {'二次元': 'anime', '动画': 'anime', '旁白': 'narrative', '广播剧': 'radio_drama'},
    'gender_expression': {'女声': 'feminine', '男声': 'masculine', '中性': 'neutral'},
}


def suggest_from_prompt(text):
    """Heuristic draft recipe from a natural-language description.

    This is a *suggestion*: the UI must show which fields were inferred and ask
    the user to confirm before saving, because keyword matching is not a model
    inference.
    """
    if not isinstance(text, str) or len(text) > 4000:
        raise RecipeError('声线描述必须是不超过 4000 字的文本。')
    recipe = empty_recipe()
    matches = []
    for field, mapping in _KEYWORDS.items():
        for keyword, value in mapping.items():
            if keyword in text:
                group = next(name for name, fields in VOCABULARY.items() if field in fields)
                if recipe[group][field] != value:
                    recipe[group][field] = value
                    matches.append((field, keyword, value))
                break
    if '冷淡' in text or '冷' in text:
        recipe['text_style']['avoid_excessive_exclamation'] = True
        recipe['text_style']['avoid_overacting'] = True
    if '短句' in text or '句子短' in text:
        recipe['text_style']['sentence_length'] = 'short'
    return {'recipe': normalize(recipe), 'matched': matches,
            'note': '关键词推断只是草稿，请逐项确认后再保存；未匹配的字段保持默认值。'}


_PUNCTUATION = str.maketrans({'！': '！', '!': '！', '~': '～', '～': '～'})


def apply_text_style(text, recipe):
    """Normalize punctuation for the selected text style; never rewrite words.

    Returns ``(text, notes)``. The function deliberately does not paraphrase or
    split sentences: altering the script would change what the character says.
    """
    if not isinstance(text, str):
        raise RecipeError('台词必须是文本。')
    style = normalize(recipe)['text_style']
    notes = []
    result = text.translate(_PUNCTUATION)
    while '！！' in result:
        result = result.replace('！！', '！')
    while '！？' in result:
        result = result.replace('！？', '？')
    while '～～' in result:
        result = result.replace('～～', '～')
    if style['avoid_excessive_exclamation']:
        marks = result.count('！')
        if marks:
            kept, rebuilt = 0, []
            for character in result:
                if character == '！':
                    kept += 1
                    # Keep one exclamation mark per sentence-sized clause.
                    if kept % 2 == 0:
                        rebuilt.append('。')
                        continue
                rebuilt.append(character)
            result = ''.join(rebuilt)
            notes.append('按文本风格压缩了重复感叹号；语义与台词文字未改动。')
    if style['sentence_length'] == 'short':
        longest = max((len(part) for part in result.replace('。', '。\n').splitlines()), default=0)
        if longest > 60:
            notes.append('文本风格要求短句，但存在超过 60 字的句子；请人工拆分，程序不会改写台词。')
    return result, notes


def text_style_violations(text, recipe):
    """Checkable text-style rules used by the fixed evaluation set."""
    style = normalize(recipe)['text_style']
    violations = []
    if style['avoid_excessive_exclamation'] and text.count('！') > max(1, text.count('。')):
        violations.append('感叹号多于句号，可能显得夸张。')
    if style['sentence_length'] == 'short' and max(
            (len(part) for part in text.replace('。', '。\n').splitlines()), default=0) > 60:
        violations.append('存在超过 60 字的句子，不符合短句风格。')
    return violations

