"""Import character settings, review a voice plan, then render and keep takes."""
from datetime import datetime

from timbreforge.role_planner import (import_profile, infer_profile, generate_from_plan,
                                      list_plans, restore_plan, list_generations, save_voice,
                                      library_characters)

_NUMERALS = ('一', '二', '三', '四', '五', '六', '七', '八', '九')


def _clock(value):
    try:
        return datetime.fromisoformat(value).astimezone().strftime('%H:%M')
    except (TypeError, ValueError):
        return ''


def record_label(index, created_at=''):
    """记录一 / 记录二 / 记录三, newest first; time keeps them unambiguous."""
    name = f'记录{_NUMERALS[index]}' if index < len(_NUMERALS) else f'记录{index + 1}'
    stamp = _clock(created_at)
    return f'{name} · {stamp}' if stamp else name


def record_choices(plan_id):
    """(choices, current value) for usable takes; never raises into the UI."""
    try:
        takes = list_generations(plan_id) if plan_id else []
    except (OSError, ValueError, KeyError, TypeError):
        return [], None
    choices = [(record_label(index, take.get('created_at', '')), take['id'])
               for index, take in enumerate(takes)]
    return choices, (choices[0][1] if choices else None)


def library_choices():
    """(character choices, message) for the saved-voice library."""
    try:
        view = library_characters()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return [], None, f'声线库无法读取：{exc}'
    if not view.get('ok'):
        return [], None, f"声线库无法读取：{view.get('error', '未知错误')}"
    groups = view['characters']
    choices = [(f"{group['name']}（{len(group['voices'])} 条声线）", group['id']) for group in groups]
    if not choices:
        return [], None, '声线库还是空的。生成满意的一条后，点击「保存这条声线到声线库」。'
    return choices, choices[0][1], f'声线库共 {len(choices)} 个角色。'


def _truncate(text, limit=24):
    text = (text or '').strip().replace('\n', ' ')
    return text if len(text) <= limit else text[:limit] + '…'


def voice_choices(group):
    """Labels for one character's voices, oldest first (声线一 is the first saved)."""
    voices = list(group.get('voices') or [])
    choices = []
    for index, voice in enumerate(voices):
        name = voice.get('label') or (f'声线{_NUMERALS[index]}' if index < len(_NUMERALS) else f'声线{index + 1}')
        stamp = _clock(voice.get('created_at', ''))
        preview = _truncate(voice.get('preview_text', ''))
        label = ' · '.join(part for part in (name, stamp, preview) if part)
        choices.append((label, voice['character_id']))
    return choices


def voice_markdown(group, voice):
    lines = [f"**{group['name']}**"]
    if group.get('description'):
        lines.append(f"角色描述：{group['description']}")
    if voice is None:
        lines.append('选择一条声线可试听。')
        return '\n\n'.join(lines)
    lines.append(f"声线备注：{voice.get('label') or '（未命名）'}")
    lines.append(f"声线描述：{voice.get('description') or '（未填写）'}")
    lines.append(f"试听台词：{voice.get('preview_text') or '（无）'}")
    origin = voice.get('origin') or {}
    if origin.get('voice_prompt'):
        lines.append(f"保存时使用的提示词：{origin['voice_prompt']}")
    stamp = _clock(voice.get('created_at', ''))
    if stamp:
        lines.append(f'保存时间：{stamp}')
    lines.append('这条声线可以作为「角色复用」和「多角色项目」的参考声音。')
    return '\n\n'.join(lines)


def build_role_tab(gr, synthesis_options=None):
    options = dict(synthesis_options or {})
    with gr.Tab('角色设定 → 声线推演'):
        record_state = gr.State(None)
        take_state = gr.State(None)
        with gr.Row():
            gr.Markdown('## 从角色设定，推演他的声音\n导入人物背景、性格和经历，模型给出声线建议与完整提示词。'
                        '你可以修改后试听、换种子重生成，并把满意的一条存进声线库。', scale=5)
            library_open = gr.Button('声线库', scale=1, min_width=120, elem_id='role_library_open')

        with gr.Group(visible=False) as library_panel:
            with gr.Row():
                # NOTE: never name a local component after an imported function here;
                # a shadowing name silently breaks the service call inside this scope.
                library_picker = gr.Dropdown(choices=[], label='已保存角色', scale=4,
                                             elem_id='role_library_characters')
                library_refresh = gr.Button('刷新', elem_id='role_library_refresh')
                library_close = gr.Button('收起', elem_id='role_library_close')
            library_voices = gr.Radio(choices=[], label='该角色的声线（一个角色可以有多条）',
                                      elem_id='role_library_voices')
            library_audio = gr.Audio(type='filepath', interactive=False, label='声线试听',
                                     elem_id='role_library_audio')
            library_info = gr.Markdown(elem_id='role_library_info')

        with gr.Row():
            history = gr.Dropdown(choices=[], label='已保存的角色推演', elem_id='role_history', scale=3)
            refresh = gr.Button('刷新角色记录', elem_id='role_refresh')
            load = gr.Button('恢复角色与试听', elem_id='role_load')
        with gr.Row():
            with gr.Column():
                upload = gr.File(label='导入角色设定（Word / 文本 / JSON）',
                                 file_types=['.docx', '.txt', '.md', '.json'], type='filepath',
                                 elem_id='role_upload')
                profile = gr.Textbox(label='角色设定（可直接粘贴或修改导入内容）', lines=12,
                                     placeholder='角色名、年龄感、性格、背景、身份、说话习惯……也可以加入你希望保留的声音特点。',
                                     elem_id='role_profile')
                analyze = gr.Button('根据设定推演声线', variant='primary', elem_id='role_analyze')
            with gr.Column():
                summary = gr.Textbox(label='角色概括', interactive=False, elem_id='role_summary')
                traits = gr.Dataframe(headers=['维度', '声线建议', '设定依据 / 创作建议'],
                                      datatype=['str', 'str', 'str'], interactive=False, wrap=True,
                                      elem_id='role_traits')
                uncertainties = gr.Textbox(label='待确认的创作选择', interactive=False, lines=2,
                                           elem_id='role_uncertainties')
        status = gr.Markdown('支持单个角色、最多 8000 字。Word 导入正文和表格文字，图片内容不会自动识别。',
                             elem_id='role_status')
        prompt = gr.Textbox(label='声线提示词（实际用于生成，可编辑）', lines=4, elem_id='role_prompt')
        preview = gr.Textbox(label='角色试听台词（可编辑）', lines=2, elem_id='role_preview')
        generate = gr.Button('用这段提示词生成角色声音', interactive=False, variant='primary',
                             elem_id='role_generate')
        audio = gr.Audio(label='角色声线试听', type='filepath', interactive=False, elem_id='role_audio')
        with gr.Row():
            reroll = gr.Button('↻ 换个种子重新生成', interactive=False, elem_id='role_reroll')
            save_open = gr.Button('保存这条声线到声线库', interactive=False, elem_id='role_save_open')
        records = gr.Radio(choices=[], label='生成记录（最多保留 3 条，点选可切回试听）',
                           elem_id='role_records')

        with gr.Group(visible=False) as save_panel:
            gr.Markdown('### 保存声线到声线库\n保存的是这条试听音频和它的台词，作为可复用参考声音；'
                        '不是重新运行提示词，也不是训练模型。')
            save_name = gr.Textbox(label='角色名（必填）', elem_id='role_save_name')
            save_description = gr.Textbox(label='角色描述（选填）', lines=2,
                                          elem_id='role_save_description')
            save_target = gr.Dropdown(choices=[], label='保存到', elem_id='role_save_target')
            save_confirm = gr.Checkbox(value=False,
                                       label='我已试听，并确认这段音频与试听台词一致，可以保存为参考声音。',
                                       elem_id='role_save_confirm')
            with gr.Row():
                save_do = gr.Button('保存', variant='primary', elem_id='role_save_do')
                save_cancel = gr.Button('取消', elem_id='role_save_cancel')

        gr.Markdown('推演是创作建议。参数会自动配置；推演成功及试听生成成功时自动存档。'
                    '刷新页面后，可从上方恢复记录；尚未生成的文字修改不会自动保存。')

    # Clear all derived values when the source changes; never reuse a stale plan.
    derived = [record_state, summary, traits, uncertainties, prompt, preview, generate,
               audio, take_state, records, save_open, reroll]
    controls = [upload, profile, analyze, history, refresh, load]

    def enabled(value):
        return tuple(gr.update(interactive=value) for _ in controls)

    def cleared(editable_text=True):
        return (None, '', [], '', gr.update(value='', interactive=editable_text),
                gr.update(value='', interactive=editable_text), gr.update(interactive=False), None,
                None, gr.update(choices=[], value=None), gr.update(interactive=False),
                gr.update(interactive=False))

    def plan_view(record, instruction=None, text=None, path=None):
        plan = record['plan']
        rows = [[t['dimension'], t['suggestion'], t['basis']] for t in plan['traits']]
        choices, value = record_choices(record['id'])
        return (record, plan['character_name'] + '：' + plan['summary'], rows,
                '\n'.join(plan['uncertainties']) or '暂无额外待确认项。',
                gr.update(value=instruction or plan['voice_prompt'], interactive=True),
                gr.update(value=text or plan['preview_text'], interactive=True),
                gr.update(interactive=True), path, value, gr.update(choices=choices, value=value),
                gr.update(interactive=bool(value)), gr.update(interactive=bool(value)))

    def history_choices(current):
        records = list_plans()
        choices = [(f"{r['plan']['character_name']} · {datetime.fromisoformat(r['created_at']).astimezone():%m-%d %H:%M} · {r['id'][:8]}",
                    r['id']) for r in records]
        value = current if current in [r['id'] for r in records] else (records[0]['id'] if records else None)
        return gr.update(choices=choices, value=value), f'找到 {len(records)} 条可恢复的角色记录。'

    refresh.click(history_choices, history, [history, status], api_name='role_history')

    def restore(plan_id):
        try:
            restored = restore_plan(plan_id)
            message = restored['warning'] or ('已恢复角色、实际生成提示词和最近一次试听。'
                                              if restored['audio'] else '已恢复角色推演，可修改提示词后生成试听。')
            values = plan_view(restored['record'], restored['prompt'], restored['preview_text'],
                               restored['audio'])
            # Keep the restored take selected so saving targets the audio on screen.
            choices = values[9]['choices']
            value = restored.get('generation_id')
            if value not in [item[1] for item in choices]:
                value = choices[0][1] if choices else None
            values = values[:8] + (value, gr.update(choices=choices, value=value)) + values[10:]
            return (restored['record']['profile'], None, *values, message)
        except (OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
            return (*(gr.skip() for _ in [profile, upload] + derived), f'恢复未完成：{exc} 当前编辑内容已保留。')

    load.click(restore, history, [profile, upload] + derived + [status], api_name='role_restore',
               **options)

    def import_file(path):
        try:
            text = import_profile(path)
            return (text, '导入完成。检查左侧设定后，点击“根据设定推演声线”。', *cleared())
        except (OSError, ValueError) as exc:
            return ('', f'导入未完成：{exc}', *cleared())
    # Programmatic resets during history restore must not trigger a new import.
    upload.upload(import_file, upload, [profile, status] + derived, api_name='role_import')

    def profile_edited():
        return (*cleared(), '角色设定已修改，请重新推演声线。')
    profile.input(profile_edited, outputs=derived + [status], queue=False)

    def analyze_profile(text, progress=gr.Progress()):
        yield (*cleared(False), *enabled(False), '正在分析角色设定并推演声线，请稍候…')
        try:
            record = infer_profile(text, progress)
            message = '推演完成。下方是模型给出的完整提示词，可直接修改后生成试听。'
            notes = record['plan'].get('_notes') or []
            if notes:
                message += ' 注意：' + ' '.join(notes)
            yield (*plan_view(record), *enabled(True), message)
        except (OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
            yield (*cleared(), *enabled(True), f'推演未完成：{exc}')
    analyze.click(analyze_profile, profile, derived + controls + [status], api_name='role_analyze',
                  **options)

    render_outputs = [audio, status, records, take_state, save_panel, save_open, reroll,
                      generate, prompt, preview] + controls

    def _editable(value):
        return (gr.update(interactive=value), gr.update(interactive=value),
                gr.update(interactive=value), *enabled(value))

    def _render(record, source, instruction, text, progress, note):
        yield (None, note, gr.update(), None, gr.update(visible=False),
               gr.update(interactive=False), gr.update(interactive=False), *_editable(False))
        try:
            path, info = generate_from_plan(record, source, instruction, text, progress)
            plan_id = (record or {}).get('id')
            choices, value = record_choices(plan_id)
            take = info.get('generation_id') if isinstance(info, dict) else None
            if take not in [item[1] for item in choices]:
                value = choices[0][1] if choices else None
                take = value
            kept = len(choices)
            pruned = (info or {}).get('pruned') or []
            extra = f'；已删除最早的 {len(pruned)} 条旧记录' if pruned else ''
            message = (f'生成完成并已存档（当前保留 {kept} 条记录{extra}）。'
                       '试听满意后点击「保存这条声线到声线库」。')
            yield (path, message, gr.update(choices=choices, value=value), take,
                   gr.update(visible=False), gr.update(interactive=bool(take)),
                   gr.update(interactive=True), *_editable(True))
        except (OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
            yield (None, f'生成未完成：{exc}', gr.update(), None, gr.update(visible=False),
                   gr.update(interactive=False), gr.update(interactive=True), *_editable(True))

    def render_generate(record, source, instruction, text, progress=gr.Progress()):
        yield from _render(record, source, instruction, text, progress,
                           '正在按照当前提示词生成角色声音…')

    def render_reroll(record, source, instruction, text, progress=gr.Progress()):
        yield from _render(record, source, instruction, text, progress,
                           '正在用新的随机种子重新生成，请稍候…')

    generate.click(render_generate, [record_state, profile, prompt, preview], render_outputs,
                   api_name='role_generate', **options)
    reroll.click(render_reroll, [record_state, profile, prompt, preview], render_outputs,
                 api_name='role_reroll', **options)

    def select_take(record, generation_id):
        """Switch playback to an older take without overwriting the current text."""
        if not generation_id:
            return None, None, gr.update(visible=False), '请先选择一条生成记录。', gr.update(interactive=False)
        try:
            restored = restore_plan((record or {}).get('id'), generation_id)
        except (OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
            return None, None, gr.update(visible=False), f'切换记录失败：{exc}', gr.update(interactive=False)
        if not restored['audio']:
            return (None, generation_id, gr.update(visible=False),
                    restored['warning'] or '这条记录没有可用音频。', gr.update(interactive=False))
        message = (f"已切回该记录（台词：{_truncate(restored['preview_text'], 30)}）。"
                   '输入框保持你当前的文字，没有被这条记录覆盖。')
        return (restored['audio'], restored['generation_id'], gr.update(visible=False), message,
                gr.update(interactive=True))

    records.change(select_take, [record_state, records],
                   [audio, take_state, save_panel, status, save_open], api_name='role_select_take')

    def open_save(record, generation_id):
        if not generation_id:
            return (gr.update(visible=False), gr.update(), gr.update(), gr.update(value=False),
                    '请先生成一条试听，再保存声线。')
        choices, _, message = library_choices()
        target = [('新建角色', '')] + choices
        default_name = ((record or {}).get('plan') or {}).get('character_name', '')
        if default_name in ('未命名角色', 'None'):
            default_name = ''
        return (gr.update(visible=True), gr.update(value=default_name),
                gr.update(choices=target, value=''), gr.update(value=False),
                '填写角色名后保存。' + ('' if message.startswith('声线库共') else ' ' + message))

    save_open.click(open_save, [record_state, take_state],
                    [save_panel, save_name, save_target, save_confirm, status],
                    api_name='role_save_open')

    def close_save():
        return gr.update(visible=False), '已取消保存，试听仍然保留。'
    save_cancel.click(close_save, outputs=[save_panel, status], queue=False)

    def do_save(record, generation_id, name, description, target, confirmed):
        try:
            result = save_voice((record or {}).get('id'), generation_id, name,
                                description=description, target_character_id=target or None,
                                confirmed=bool(confirmed))
        except (OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
            return (gr.update(visible=True), f'保存未完成：{exc}', gr.update(), gr.update(), gr.update(), gr.update())
        action = '新建角色并保存' if result['created_character'] else '加入已有角色'
        message = f"{action}成功：{result['character']['name']}，当前共 {len(result['character']['voices'])} 条声线。"
        groups = _library_groups()
        choices = [(f"{item['name']}（{len(item['voices'])} 条声线）", item['id']) for item in groups]
        group = next((item for item in groups if item['id'] == result['character']['id']), None)
        value = group['id'] if group else (choices[0][1] if choices else None)
        voices = voice_choices(group) if group else []
        latest = ((group or {}).get('voices') or [None])[-1]
        info = voice_markdown(group, latest) if group else message
        return (gr.update(visible=False), message + ' 可在右上角「声线库」查看。',
                gr.update(visible=True), gr.update(choices=choices, value=value),
                gr.update(choices=voices, value=voices[-1][1] if voices else None), info)

    save_do.click(do_save, [record_state, take_state, save_name, save_description, save_target, save_confirm],
                  [save_panel, status, library_panel, library_picker, library_voices, library_info],
                  api_name='role_save_voice', **options)

    def _library_snapshot():
        """(groups, error). A broken index must be reported, not shown as empty."""
        try:
            view = library_characters()
        except (OSError, ValueError, KeyError) as exc:
            return [], f'声线库无法读取：{exc}'
        if not view.get('ok'):
            return [], f"声线库无法读取：{view.get('error', '未知错误')}"
        return view.get('characters', []), ''

    def _library_groups():
        return _library_snapshot()[0]

    def open_library():
        groups, error = _library_snapshot()
        choices = [(f"{group['name']}（{len(group['voices'])} 条声线）", group['id']) for group in groups]
        value = choices[0][1] if choices else None
        message = (error or (f'声线库共 {len(choices)} 个角色。' if choices
                             else '声线库还是空的。生成满意的一条后，点击「保存这条声线到声线库」。'))
        group = next((item for item in groups if item['id'] == value), None)
        voices = voice_choices(group) if group else []
        return (gr.update(visible=True), gr.update(choices=choices, value=value),
                gr.update(choices=voices, value=voices[0][1] if voices else None), message, None)

    library_open.click(open_library, outputs=[library_panel, library_picker, library_voices,
                                              library_info, library_audio], api_name='role_library')
    library_refresh.click(open_library, outputs=[library_panel, library_picker, library_voices,
                                                 library_info, library_audio])

    def collapse_library():
        return gr.update(visible=False)
    library_close.click(collapse_library, outputs=[library_panel], queue=False)

    def pick_character(character_id):
        if not character_id:
            # Setting the dropdown value programmatically also fires `change`; an
            # empty value must not wipe the voices that the same response filled in.
            return gr.update(), gr.update(), gr.update()
        group = next((item for item in _library_groups() if item['id'] == character_id), None)
        if group is None:
            return gr.update(choices=[], value=None), None, '请选择一个角色。'
        voices = voice_choices(group)
        return (gr.update(choices=voices, value=voices[0][1] if voices else None), None,
                voice_markdown(group, None))
    library_picker.change(pick_character, library_picker,
                          [library_voices, library_audio, library_info])

    def play_voice(character_id, card_id):
        if not card_id:
            # Same programmatic-update race as above: never blank the panel.
            return gr.update(), gr.update()
        group = next((item for item in _library_groups() if item['id'] == character_id), None)
        if group is None:
            return None, '请选择角色与声线。'
        voice = next((item for item in group['voices'] if item['character_id'] == card_id), None)
        if voice is None:
            return None, '这条声线的记录已不存在。'
        try:
            from timbreforge.role_planner import library_audio
            path = library_audio(character_id, card_id)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return None, f'无法读取这条声线：{exc}'
        return path, voice_markdown(group, voice)
    library_voices.change(play_voice, [library_picker, library_voices],
                          [library_audio, library_info])
