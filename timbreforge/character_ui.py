"""Create/import a character card and condition its reference audio."""
from __future__ import annotations

from pathlib import Path

from timbreforge import character_audio, clone_prompts, ui_tasks
from timbreforge.audio_presets import get_preset, preset_choices, preset_names
from timbreforge.characters import CharacterStore
from timbreforge.voice_recipe import (RECIPE_EXAMPLES, describe, details,
                                      empty_recipe, suggest_from_prompt)

RECIPE_CHOICES = [('默认（未指定）', 'default')] + [
    (item['label'], key) for key, item in RECIPE_EXAMPLES.items()
] + [('从声线描述自动推断', 'auto')]


def role_choices(store=None):
    cards = (store or CharacterStore()).list()
    return [(f"{card['name']} · v{card.get('version', 1)}", card['id']) for card in cards]


def _recipe_for(preset, voice_prompt):
    if preset == 'default':
        return empty_recipe(), '使用默认配方；可稍后在「声线微调」逐项调整。'
    if preset == 'auto':
        suggestion = suggest_from_prompt(voice_prompt or '')
        return suggestion['recipe'], suggestion['note']
    if preset in RECIPE_EXAMPLES:
        return RECIPE_EXAMPLES[preset]['recipe'], f"已套用示例配方：{RECIPE_EXAMPLES[preset]['label']}。"
    raise ValueError('未知声线配方选项。')


def parse_tags(text):
    if text is None:
        return []
    if not isinstance(text, str):
        raise ValueError('标签必须是文本。')
    parts = [part.strip() for part in text.replace('，', ',').replace('、', ',').split(',')]
    return [part for part in parts if part]


def create_character(name, description, transcript, voice_prompt, recipe_choice, tags_text,
                     notes, audio_path, *, store=None) -> dict:
    """Validate and persist a manually imported role card."""
    if not audio_path:
        raise ValueError('请先上传参考音频（WAV/FLAC/OGG/MP3，建议 30 秒以内）。')
    path = Path(audio_path)
    if not path.is_file():
        raise ValueError(f'参考音频不存在：{path}')
    if path.suffix.lower() not in {'.wav', '.flac', '.ogg', '.mp3', '.m4a'}:
        raise ValueError('参考音频格式不受支持；请使用 WAV/FLAC/OGG/MP3/M4A。')
    if not transcript or not transcript.strip():
        raise ValueError('请填写参考音频的准确逐字稿；它是声音身份的一部分。')
    recipe, recipe_note = _recipe_for(recipe_choice, voice_prompt)
    card = (store or CharacterStore()).create(
        name, path, reference_text=transcript.strip(), description=description or '',
        backend='qwen3', instruction='', params={'qwen': {'seed': 42}},
        provenance={'kind': 'manual_import', 'recipe_note': recipe_note},
        voice_design_prompt=voice_prompt or '', voice_recipe=recipe,
        source='import/manual_audio', tags=parse_tags(tags_text), notes=notes or '',
        model_name='Qwen3-TTS-12Hz-0.6B-Base', model_revision='local', seed=None,
        clone_mode='icl')
    return {'card': card, 'note': recipe_note}


def character_markdown(card, store=None):
    if not card:
        return '选择一张角色卡查看状态。'
    recipe = card.get('voice_recipe') or {}
    try:
        rows = details(recipe)
        recipe_text = '；'.join(f'{label} {value}' for _group, label, value in rows[:8])
        summary = describe(recipe)
    except (ValueError, KeyError, TypeError):
        recipe_text, summary = '未设置', '未设置'
    try:
        status = clone_prompts.validate_clone_prompt(card['id'], store=store or CharacterStore())
        prompt_text = clone_prompts.status_text(status)
    except (OSError, ValueError, KeyError):
        prompt_text = '暂时无法读取 Clone Prompt 状态。'
    return (f"### {card['name']}\n\n"
            f"- 当前参考音频：`{Path(card.get('canonical_audio') or card['reference_audio']).name}`\n"
            f"- 原始音频哈希：`{card['reference_sha256'][:16]}…`（永远保留）\n"
            f"- 角色版本：v{card.get('version', 1)} · 模型 `{card.get('model_name') or '未登记'}`\n"
            f"- Clone Prompt：{prompt_text}\n"
            f"- 声线摘要：{summary}\n"
            f"- 配方前八项：{recipe_text}")


def apply_preset(role_id, preset_name, *, store=None):
    store = store or CharacterStore()
    if not role_id:
        raise ValueError('请先选择角色。')
    if preset_name not in preset_names():
        raise ValueError('请选择可用的音频预设。')

    def task(context):
        result = character_audio.render_preset(role_id, preset_name, store=store)
        return {'card': result['card'], 'document': result['document'],
                'preview': result['preview'], 'metrics': result['metrics'],
                'notes': result['notes'], 'summary': result['summary'],
                'output_path': result['output_path']}

    payload = ui_tasks.run_heavy('audio_edit', task, role_id=role_id,
                                 payload={'preset': preset_name})
    if not payload:
        raise RuntimeError('音频处理没有返回结果。')
    return _apply_view(payload)


def apply_custom_cleanup(role_id, strength, max_gain_db, *, store=None):
    """Render the conservative chain with two safe user-controlled limits."""
    if not role_id:
        raise ValueError('请先选择角色。')
    if type(strength) not in (int, float) or not 0 <= float(strength) <= 1:
        raise ValueError('降噪强度必须在 0–1 之间。')
    if type(max_gain_db) not in (int, float) or not 0 <= float(max_gain_db) <= 12:
        raise ValueError('响度补偿上限必须在 0–12 dB 之间。')
    operations = get_preset('clean_voice')['operations']
    for operation in operations:
        if operation['name'] == 'denoise':
            operation['strength'] = float(strength)
        elif operation['name'] == 'normalize':
            operation['max_gain_db'] = float(max_gain_db)

    def task(context):
        result = character_audio.render_custom(role_id, operations, preset='custom_cleanup', store=store)
        return {'card': result['card'], 'document': result['document'],
                'preview': result['preview'], 'metrics': result['metrics'],
                'notes': result['notes'], 'summary': result['summary'],
                'output_path': result['output_path']}

    payload = ui_tasks.run_heavy('audio_edit', task, role_id=role_id,
                                 payload={'preset': 'custom_cleanup', 'strength': strength,
                                          'max_gain_db': max_gain_db})
    if not payload:
        raise RuntimeError('音频处理没有返回结果。')
    return _apply_view(payload)


def _apply_view(payload):
    notes = '；'.join(payload.get('notes') or []) or '完成。'
    metrics = payload.get('metrics') or {}
    met = (f"输入 {metrics.get('lufs_in', 0):.1f} LUFS / 输出 "
           f"{metrics.get('lufs_out', 0):.1f} LUFS；峰值 "
           f"{metrics.get('peak_dbfs_out', 0):.1f} dBFS；削波采样 "
           f"{metrics.get('clipped_samples_out', 0)}") if metrics else ''
    status = (f"已应用预设并生成新的标准参考音频（角色版本 v{payload['card'].get('version', 1)}）。"
              f"原音频未被覆盖。\n\n- 处理链：{payload.get('summary', '')}\n- {notes}\n- {met}\n"
              '- 旧 Clone Prompt 已失效，需要时可点击「创建 Clone Prompt」。')
    preview = (payload.get('preview') or {}).get('path') or payload.get('output_path')
    return preview, status


def revert(role_id, *, store=None):
    if not role_id:
        raise ValueError('请先选择角色。')
    result = character_audio.revert(role_id, store=store)
    return result['audio'], (f"已恢复为原始参考音频（角色版本 v{result['card'].get('version', 1)}）。"
                             '历史派生文件仍保留在处理记录中。')


def make_clone_prompt(role_id, device='cuda:0', *, store=None):
    if not role_id:
        raise ValueError('请先选择角色。')
    store = store or CharacterStore()
    card = store.load(role_id)
    mode = card.get('clone_mode') or 'icl'
    task = lambda context: clone_prompts.create_clone_prompt(
        role_id, store=store, device=device,
        dtype='float32' if device == 'cpu' else 'float16')
    result = ui_tasks.run_heavy('clone_prompt', task, role_id=role_id,
                                payload={'mode': mode, 'device': device})
    status = clone_prompts.validate_clone_prompt(role_id, store=store)
    created = '已创建' if result and result.get('created') else '已复用'
    return (f'{created} Clone Prompt（{mode}）：{clone_prompts.status_text(status)}\n\n'
            f'缓存记录包含参考音频哈希、逐字稿哈希、模型版本、克隆模式与处理链哈希。')


def history_rows(role_id, *, store=None):
    rows = []
    for entry in character_audio.history(role_id, store=store):
        rows.append([entry['index'], entry['kind'], entry['summary'], entry['at']])
    return rows


def build_character_tabs(gr, include_audio=True):
    """Create/import and optional audio-conditioning pages; no model loads at build.

    The main studio keeps audio conditioning out of the primary navigation.  A
    caller that needs the legacy page (and its stable component IDs) can still
    request it explicitly.
    """
    initial_choices = role_choices()
    initial_value = initial_choices[0][1] if initial_choices else None
    initial_card = None
    if initial_value:
        try:
            initial_card = CharacterStore().load(initial_value)
        except (OSError, ValueError):
            initial_card = None

    with gr.Tab('创建角色', id='create_character'):
        gr.Markdown(
            '## 创建角色\n导入有授权的参考录音并填写**准确逐字稿**，再描述声线。'
            '角色卡会保存原始音频、标准音频、配方和处理历史；本页不训练模型。')
        with gr.Row():
            with gr.Column(scale=5):
                name = gr.Textbox(label='角色名称', elem_id='create_name')
                description = gr.Textbox(label='角色设定/描述', lines=4,
                                         elem_id='create_description')
                voice_prompt = gr.Textbox(
                    label='声线描述（可留空）', lines=3,
                    placeholder='例如：冷淡的二次元少女音，音高偏高、轻薄、气声少、语速偏慢、句尾平直。',
                    elem_id='create_voice_prompt')
                recipe = gr.Dropdown(label='结构化声线配方', choices=RECIPE_CHOICES,
                                     value='default', elem_id='create_recipe')
                tags = gr.Textbox(label='标签（逗号分隔）', elem_id='create_tags')
                notes = gr.Textbox(label='备注', lines=2, elem_id='create_notes')
            with gr.Column(scale=5):
                audio = gr.Audio(label='参考音频（30 秒以内，必须有权使用）', type='filepath',
                                 elem_id='create_audio')
                transcript = gr.Textbox(
                    label='参考音频逐字稿（必须与录音逐字一致）', lines=4,
                    placeholder='把参考录音里说出的每个字都写下来，包括标点。',
                    elem_id='create_transcript')
                create = gr.Button('创建角色卡', variant='primary', elem_id='create_submit')
                status = gr.Markdown('填写后点击创建；参考音频会被逐字节复制，原始文件不会被修改。',
                                     elem_id='create_status')

        def do_create(name_value, description_value, transcript_value, prompt_value,
                      recipe_value, tags_value, notes_value, audio_value):
            try:
                result = create_character(name_value, description_value, transcript_value,
                                          prompt_value, recipe_value, tags_value,
                                          notes_value, audio_value)
            except (OSError, ValueError, RuntimeError, TypeError) as exc:
                return f'创建失败：{exc}'
            card = result['card']
            return (f"已创建角色「{card['name']}」（ID `{card['id']}`）。\n\n"
                    f"- {result['note']}\n"
                    f"- 原始音频：`{Path(card['reference_audio']).name}`，哈希 `{card['reference_sha256'][:16]}…`\n"
                    '- 下一步：到「音频修整」处理参考音频，或到「文字生成」直接试听。')

        create.click(do_create,
                     [name, description, transcript, voice_prompt, recipe, tags, notes, audio],
                     status, api_name='character_create')

    audio_container = gr.Tab('音频修整', id='audio_conditioning', visible=include_audio)
    with audio_container:
        gr.Markdown(
            '## 音频修整\n所有处理都生成**新的派生 WAV**，原始音频永不覆盖；'
            '处理参数、输入/输出哈希写入处理链，旧 Clone Prompt 自动失效。')
        with gr.Row():
            picker = gr.Dropdown(label='选择角色', choices=initial_choices,
                                 value=initial_value, scale=5, elem_id='audio_role')
            refresh = gr.Button('刷新角色', elem_id='audio_refresh')
        info = gr.Markdown(character_markdown(initial_card), elem_id='audio_info')
        with gr.Row():
            current_audio = gr.Audio(
                value=(initial_card.get('canonical_audio') or initial_card['reference_audio'])
                if initial_card else None,
                label='当前参考音频', type='filepath', interactive=False,
                elem_id='audio_current')
            result_audio = gr.Audio(label='处理结果试听', type='filepath', interactive=False,
                                    elem_id='audio_result')
        preset = gr.Dropdown(label='内置预设', choices=preset_choices(),
                             value='clean_voice', elem_id='audio_preset')
        with gr.Accordion('高级修整（保守模式）', open=False):
            custom_strength = gr.Slider(0, 1, value=.45, step=.05,
                                         label='降噪强度（越高越容易损失气声）',
                                         elem_id='audio_custom_strength')
            custom_gain = gr.Slider(0, 12, value=6, step=.5,
                                     label='最大响度补偿（dB）', elem_id='audio_custom_gain')
            custom_button = gr.Button('按当前参数生成派生音频', elem_id='audio_custom_apply')
        with gr.Row():
            apply_button = gr.Button('应用预设并生成标准音频', variant='primary',
                                     elem_id='audio_apply')
            revert_button = gr.Button('恢复原始版本', elem_id='audio_revert')
            prompt_button = gr.Button('创建 Clone Prompt', elem_id='audio_prompt')
        status = gr.Markdown('预设只是起点；应用后务必试听，确认没有换气、齿音或尾音损失。',
                             elem_id='audio_status')
        history = gr.Dataframe(headers=['#', '类型', '处理内容', '时间'], interactive=False,
                               label='处理历史（最新在前）', elem_id='audio_history')

        def view(role_id):
            try:
                card = CharacterStore().load(role_id) if role_id else None
            except (OSError, ValueError) as exc:
                return None, None, f'读取角色失败：{exc}', []
            if card is None:
                return None, None, '请先创建或选择一个角色。', []
            path = card.get('canonical_audio') or card['reference_audio']
            return path, None, character_markdown(card), history_rows(card['id'])

        refresh.click(view, picker, [current_audio, result_audio, info, history],
                      api_name='audio_refresh')
        picker.change(view, picker, [current_audio, result_audio, info, history],
                      api_name='audio_select')

        def do_apply(role_id, preset_name):
            try:
                audio_path, message = apply_preset(role_id, preset_name)
            except (OSError, ValueError, RuntimeError, TypeError) as exc:
                return None, None, f'处理失败：{exc}', gr.update()
            card = CharacterStore().load(role_id)
            return audio_path, audio_path, message, history_rows(role_id)

        apply_button.click(do_apply, [picker, preset], [current_audio, result_audio, status, history],
                           api_name='audio_apply')

        def do_custom(role_id, strength, max_gain):
            try:
                audio_path, message = apply_custom_cleanup(role_id, strength, max_gain)
            except (OSError, ValueError, RuntimeError, TypeError) as exc:
                return None, None, f'处理失败：{exc}', gr.update()
            return audio_path, audio_path, message, history_rows(role_id)

        custom_button.click(do_custom, [picker, custom_strength, custom_gain],
                            [current_audio, result_audio, status, history],
                            api_name='audio_custom_apply')

        def do_revert(role_id):
            try:
                audio_path, message = revert(role_id)
            except (OSError, ValueError, RuntimeError, TypeError) as exc:
                return None, f'恢复失败：{exc}', gr.update()
            return audio_path, message, history_rows(role_id)

        revert_button.click(do_revert, picker, [current_audio, status, history],
                            api_name='audio_revert')

        def do_prompt(role_id):
            try:
                message = make_clone_prompt(role_id)
            except (OSError, ValueError, RuntimeError, TypeError) as exc:
                return f'创建 Clone Prompt 失败：{exc}'
            return message

        prompt_button.click(do_prompt, picker, status, api_name='audio_prompt')

