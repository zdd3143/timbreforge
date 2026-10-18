"""Voice tuning (A/B candidates), text generation and long-form generation."""
from __future__ import annotations

from pathlib import Path

from timbreforge import generation, ui_tasks, voice_candidates
from timbreforge.characters import CharacterStore
from timbreforge.voice_recipe import (FIELD_LABELS, LABELS, VOCABULARY, describe,
                                      empty_recipe, normalize)

DEVICES = [('自动（cuda:0）', 'cuda:0'), ('CPU（float32，较慢）', 'cpu')]


def role_choices(store=None):
    cards = (store or CharacterStore()).list()
    return [(f"{card['name']} · v{card.get('version', 1)}", card['id']) for card in cards]


def field_choices(field):
    """``(Chinese label, value)`` pairs keeping the documented vocabulary order."""
    group = next(name for name, fields in VOCABULARY.items() if field in fields)
    values = VOCABULARY[group][field]
    return [(LABELS.get(field, {}).get(value, value), value) for value in values]


def build_recipe(values, base=None):
    """Build a validated recipe from ``{field: value}`` UI selections."""
    if not isinstance(values, dict):
        raise ValueError('声线配方控件值无效。')
    recipe = normalize(base) if base is not None else empty_recipe()
    for field, value in values.items():
        group = next((name for name, fields in VOCABULARY.items() if field in fields), None)
        if group is None:
            raise ValueError(f'未知声线配方字段：{field}')
        recipe[group][field] = value
    try:
        return normalize(recipe)
    except ValueError as exc:
        raise ValueError(f'声线配方无效：{exc}') from exc


def empty_values():
    return {field: defaults for group, fields in VOCABULARY.items()
            for field, defaults in fields.items()}


def recipe_markdown(recipe):
    try:
        return f'**当前配方**：{describe(recipe)}\n\n' + '；'.join(
            f'{FIELD_LABELS.get(field, field)}：{LABELS.get(field, {}).get(value, value)}'
            for group, fields in VOCABULARY.items()
            for field, value in normalize(recipe)[group].items())
    except (ValueError, KeyError, TypeError) as exc:
        return f'配方无效：{exc}'


def start_tuning(role_id, text, values, count, seed, device, *, store=None):
    """Generate single-dimension candidates through the serial resource manager."""
    if not role_id:
        raise ValueError('请先选择角色。')
    recipe = build_recipe(values)
    if type(count) is not int or not 1 <= count <= 8:
        raise ValueError('候选数量必须在 1–8 之间。')
    if device not in ('cuda:0', 'cpu'):
        raise ValueError('请选择生成设备。')

    def task(context):
        session = voice_candidates.create_session(
            role_id, text, store=store, count=count, recipe=recipe, seed=int(seed or 42),
            device=device, dtype='float32' if device == 'cpu' else 'float16')
        root = voice_candidates.CandidateStore().root
        return {'session': session, 'root': str(root)}

    payload = ui_tasks.run_heavy('voice_tuning', task, role_id=role_id,
                                 payload={'count': count, 'device': device})
    if not payload:
        raise RuntimeError('声线微调没有返回结果。')
    session = payload['session']
    return session, _candidate_choices(session), _candidate_path(payload['root'], session, 0)


def _candidate_choices(session):
    return [(f"{index + 1}. {candidate['changed']['label']} · seed {candidate['seed']}",
             candidate['id']) for index, candidate in enumerate(session['candidates'])]


def _candidate_path(root, session, index):
    try:
        candidate = session['candidates'][index]
    except (IndexError, KeyError, TypeError):
        return None
    return str(Path(root) / session['id'] / candidate['file'])


def candidate_status(session):
    lines = [voice_candidates.status_text(session), '',
             '| # | 变化维度 | seed | 提示词摘要 |', '| --- | --- | --- | --- |']
    for index, candidate in enumerate(session.get('candidates', []), 1):
        lines.append(f"| {index} | {candidate['changed']['label']} | {candidate['seed']} | "
                     f"{candidate['prompt'][:60]}… |")
    return '\n'.join(lines)


def load_candidate(session_id, candidate_id):
    tuning = voice_candidates.CandidateStore()
    session = tuning.load(session_id)
    for candidate in session['candidates']:
        if candidate['id'] == candidate_id:
            return session, str(tuning._directory(session_id) / candidate['file'])
    raise ValueError('候选不在该会话中。')


def select_candidate(session_id, candidate_id):
    session = voice_candidates.select_candidate(session_id, candidate_id)
    return session, candidate_status(session)


def compare_candidates(session_id, a_id, b_id, match_volume):
    if not a_id or not b_id:
        raise ValueError('请选择两个候选进行 A/B 试听。')
    result = voice_candidates.compare(session_id, a_id, b_id, match_volume=bool(match_volume))
    return result['a'], result['b'], (f"A：{result['a_meta']['label']}\n\nB：{result['b_meta']['label']}\n\n"
                                      f"增益：A {result['gain_a_db']:+.1f} dB / "
                                      f"B {result['gain_b_db']:+.1f} dB（{result['method']}）")


def promote_candidate(session_id, candidate_id, confirmed, *, store=None):
    if confirmed is not True:
        raise ValueError('保存新版本前，请确认候选音频与逐字稿逐字一致。')
    if not candidate_id:
        raise ValueError('请先选中一个候选。')
    tuning = voice_candidates.CandidateStore()
    session = tuning.load(session_id)
    role_id = session['role_id']

    def task(context):
        result = voice_candidates.promote_candidate(
            session_id, candidate_id, confirmed=True, create_prompt=True, store=store)
        card = result['card']
        return {'status': 'ok', 'role_id': card['id'], 'role_version': card.get('version'),
                'prompt': result['clone_prompt'], 'warnings': result['warnings'],
                'label': result['candidate']['changed']['label']}

    payload = ui_tasks.run_heavy('promote_candidate', task, role_id=role_id,
                                 payload={'session_id': session_id})
    if not payload:
        raise RuntimeError('保存候选没有返回结果。')
    message = (f"已采用候选「{payload['label']}」为新版本（角色版本 v{payload['role_version']}）。\n\n"
               f"Clone Prompt：{payload['prompt'].get('status')} "
               f"{payload['prompt'].get('reason', '')}")
    if payload.get('warnings'):
        message += '\n\n' + '\n'.join(payload['warnings'])
    return message


def synthesize(role_id, text, seed, device, *, store=None):
    if not role_id:
        raise ValueError('请先选择角色。')
    if not isinstance(text, str) or not text.strip():
        raise ValueError('请输入台词。')

    def task(context):
        result = generation.generate_for_character(
            role_id, text, seed=None if seed in (None, '') else int(seed),
            device=device, dtype='float32' if device == 'cpu' else 'float16',
            store=store, cancel=context.cancel_event,
            backend=generation.available_backend())
        return {key: value for key, value in result.items()
                if key in {'file', 'metadata_file', 'duration', 'sample_rate', 'text', 'seed',
                           'role_id', 'role_name', 'role_version', 'model_name', 'params',
                           'clone_prompt', 'warnings', 'reference', 'audio'}}

    payload = ui_tasks.run_heavy('text_generation', task, role_id=role_id,
                                 payload={'seed': seed, 'device': device})
    if not payload:
        raise RuntimeError('生成没有返回结果。')
    details = (f"### 生成记录\n\n"
               f"- 角色：{payload['role_name']}（`{payload['role_id']}`，v{payload['role_version']}）\n"
               f"- 模型：`{payload['model_name']}` · seed `{payload['seed']}`\n"
               f"- 时长：{payload['duration']:.2f} 秒 · {payload['sample_rate']} Hz\n"
               f"- Clone Prompt：{payload['clone_prompt']['status']} "
               f"{payload['clone_prompt'].get('reason', '')}\n"
               f"- 音频 SHA-256：`{payload['audio']['sha256'][:16]}…`\n"
               f"- 记录文件：`{Path(payload['metadata_file']).name}`\n\n"
               + '\n'.join(f'- {warning}' for warning in payload.get('warnings', [])))
    return payload['file'], details


def generate_long_text(role_id, text, max_chars, crossfade, device, *, store=None):
    if not role_id:
        raise ValueError('请先选择角色。')
    if not isinstance(text, str) or not text.strip():
        raise ValueError('请输入长文本。')
    from timbreforge import longform

    def task(context):
        result = longform.generate_longform(
            role_id, text, store=store, max_chars=int(max_chars),
            crossfade_ms=int(crossfade), device=device,
            dtype='float32' if device == 'cpu' else 'float16',
            cancel=context.cancel_event)
        return {key: value for key, value in result.items()
                if key in {'file', 'report_file', 'segments', 'analysis', 'warnings'}}

    payload = ui_tasks.run_heavy('long_text', task, role_id=role_id,
                                 payload={'chars': len(text), 'max_chars': max_chars})
    if not payload:
        raise RuntimeError('长文本生成没有返回结果。')
    analysis = payload.get('analysis') or {}
    problems = analysis.get('problems') or []
    details = [f"### 长文本生成\n", f"- 分段数：{len(payload.get('segments', []))}",
               f"- 输出：`{Path(payload['file']).name}`",
               f"- 检查记录：`{Path(payload.get('report_file', '')).name}`",
               f"- 自动检查：{'未发现明显异常' if not problems else '；'.join(problems)}", '']
    for warning in payload.get('warnings', []):
        details.append(f'- {warning}')
    return payload['file'], '\n'.join(details)


def build_voice_tabs(gr):
    """Tuning, text and long-form tabs; no model work at build time."""
    initial = role_choices()
    initial_value = initial[0][1] if initial else None
    with gr.Tab('声线微调', id='voice_tuning'):
        gr.Markdown(
            '## 声线微调\n每个候选只改变**一个**表达维度，音色与身份保持不变。'
            '生成 4–8 个候选试听 A/B，选中后保存为角色新版本并重建 Clone Prompt。')
        with gr.Row():
            picker = gr.Dropdown(label='选择角色', choices=initial, value=initial_value,
                                 scale=5, elem_id='tuning_role')
            refresh = gr.Button('刷新角色', elem_id='tuning_refresh')
        with gr.Row():
            text = gr.Textbox(label='试听台词（保存后会成为新的参考逐字稿）', lines=2,
                              value='再说一遍，我没有听清。', elem_id='tuning_text')
            count = gr.Slider(1, 8, value=4, step=1, label='候选数量', elem_id='tuning_count')
            seed = gr.Number(value=42, precision=0, label='随机种子', elem_id='tuning_seed')
            device = gr.Dropdown(label='设备', choices=DEVICES, value='cuda:0',
                                 elem_id='tuning_device')
        with gr.Accordion('VoiceRecipe 配方编辑（音色 / 表达 / 文本）', open=True):
            value_components = {}
            for group in ('identity', 'timbre', 'delivery', 'text_style'):
                with gr.Accordion({'identity': '身份与风格', 'timbre': '音色',
                                   'delivery': '表达方式', 'text_style': '文本风格'}[group],
                                  open=group in ('timbre', 'delivery')):
                    with gr.Row():
                        for field, defaults in VOCABULARY[group].items():
                            if isinstance(defaults[0], bool):
                                value_components[field] = gr.Checkbox(
                                    label=FIELD_LABELS.get(field, field), value=defaults[0],
                                    elem_id=f'tuning_{field}')
                            else:
                                value_components[field] = gr.Dropdown(
                                    label=FIELD_LABELS.get(field, field),
                                    choices=field_choices(field), value=defaults[0],
                                    elem_id=f'tuning_{field}')
            recipe_preview = gr.Markdown('', elem_id='tuning_recipe_preview')
        with gr.Row():
            generate = gr.Button('生成候选', variant='primary', elem_id='tuning_generate')
            preview = gr.Button('预览配方文字', elem_id='tuning_preview')
        status = gr.Markdown('生成会串行占用本机模型；失败会自动清理临时文件。',
                             elem_id='tuning_status')
        with gr.Row():
            candidates = gr.Dropdown(label='候选', choices=[], elem_id='tuning_candidates')
            play = gr.Button('试听所选候选', elem_id='tuning_play')
            select = gr.Button('选中该候选', elem_id='tuning_select')
        with gr.Row():
            candidate_audio = gr.Audio(label='候选试听', type='filepath', interactive=False,
                                       elem_id='tuning_audio')
            played_audio = gr.Audio(label='播放器', type='filepath', interactive=False,
                                    elem_id='tuning_played')
        with gr.Accordion('A/B 试听与保存', open=True):
            with gr.Row():
                a_choice = gr.Dropdown(label='A', choices=[], elem_id='tuning_a')
                b_choice = gr.Dropdown(label='B', choices=[], elem_id='tuning_b')
                match_volume = gr.Checkbox(label='A/B 响度对齐（仅播放副本）', value=True,
                                           elem_id='tuning_match')
            with gr.Row():
                compare = gr.Button('A/B 对比', elem_id='tuning_compare')
                confirm = gr.Checkbox(label='我确认候选音频与试听台词逐字一致，可保存为新版本',
                                      value=False, elem_id='tuning_confirm')
                promote = gr.Button('保存为新角色版本', variant='primary',
                                    elem_id='tuning_promote')
            with gr.Row():
                audio_a = gr.Audio(label='A', type='filepath', interactive=False,
                                   elem_id='tuning_audio_a')
                audio_b = gr.Audio(label='B', type='filepath', interactive=False,
                                   elem_id='tuning_audio_b')
            promote_status = gr.Markdown('', elem_id='tuning_promote_status')
        session_state = gr.State(None)
        root_state = gr.State(None)

        def refresh_roles(current=None):
            choices = role_choices()
            ids = [item[1] for item in choices]
            value = current if current in ids else (ids[0] if ids else None)
            return gr.update(choices=choices, value=value)

        refresh.click(refresh_roles, picker, picker, api_name='tuning_refresh')

        def preview_recipe(*values):
            try:
                recipe = build_recipe(dict(zip(value_components.keys(), values)))
            except (ValueError, TypeError) as exc:
                return f'配方无效：{exc}'
            return recipe_markdown(recipe)

        preview.click(preview_recipe, list(value_components.values()), recipe_preview,
                      api_name='tuning_recipe')
        for component in value_components.values():
            component.change(preview_recipe, list(value_components.values()), recipe_preview)

        def do_generate(role_id, text_value, *values):
            count_value = values[-3]
            seed_value = values[-2]
            device_value = values[-1]
            field_values = dict(zip(value_components.keys(), values[:-3]))
            try:
                session, choices, first_audio = start_tuning(
                    role_id, text_value, field_values, int(count_value), seed_value, device_value)
            except (OSError, ValueError, RuntimeError, TypeError) as exc:
                return (f'生成失败：{exc}', None, gr.update(choices=[], value=None),
                        gr.update(choices=[], value=None), gr.update(choices=[], value=None),
                        None, None)
            return (candidate_status(session), session['id'], gr.update(choices=choices, value=choices[0][1]),
                    gr.update(choices=choices, value=choices[0][1]),
                    gr.update(choices=choices, value=choices[min(1, len(choices) - 1)][1]),
                    first_audio, None)

        generate.click(
            do_generate,
            [picker, text] + list(value_components.values()) + [count, seed, device],
            [status, session_state, candidates, a_choice, b_choice, candidate_audio, played_audio],
            api_name='tuning_generate')

        def load_candidate_view(session_id, candidate_id):
            if not session_id or not candidate_id:
                return None
            try:
                _session, path = load_candidate(session_id, candidate_id)
            except (OSError, ValueError) as exc:
                return None
            return path

        candidates.change(load_candidate_view, [session_state, candidates], candidate_audio,
                          api_name='tuning_candidate_select')
        play.click(lambda path: path, candidate_audio, played_audio, api_name='tuning_play')

        def do_select(session_id, candidate_id):
            if not session_id or not candidate_id:
                return '请先生成候选。', None
            try:
                session, message = select_candidate(session_id, candidate_id)
            except (OSError, ValueError) as exc:
                return f'选择失败：{exc}', None
            return message, session['id']

        select.click(do_select, [session_state, candidates], [status, session_state],
                     api_name='tuning_select')

        def do_compare(session_id, a_id, b_id, matching):
            if not session_id:
                return None, None, '请先生成候选。'
            try:
                left, right, message = compare_candidates(session_id, a_id, b_id, matching)
            except (OSError, ValueError, RuntimeError) as exc:
                return None, None, f'A/B 对比失败：{exc}'
            return left, right, message

        compare.click(do_compare, [session_state, a_choice, b_choice, match_volume],
                      [audio_a, audio_b, promote_status], api_name='tuning_compare')

        def do_promote(session_id, candidate_id, confirmed):
            if not session_id:
                return '请先生成并选中候选。'
            try:
                return promote_candidate(session_id, candidate_id, confirmed)
            except (OSError, ValueError, RuntimeError, TypeError) as exc:
                return f'保存失败：{exc}'

        promote.click(do_promote, [session_state, candidates, confirm], promote_status,
                      api_name='tuning_promote')

    with gr.Tab('文字生成', id='text_generation'):
        gr.Markdown('## 文字生成\n从角色库选择角色，用参考音频与逐字稿生成新台词；'
                    '每条结果都记录角色版本、模型版本、seed、参数和哈希。')
        with gr.Row():
            gen_role = gr.Dropdown(label='选择角色', choices=initial, value=initial_value,
                                   scale=5, elem_id='generate_role')
            gen_refresh = gr.Button('刷新角色', elem_id='generate_refresh')
        with gr.Row():
            gen_text = gr.Textbox(label='台词', lines=4, elem_id='generate_text')
            gen_seed = gr.Number(value=None, precision=0, label='seed（留空用角色默认）',
                                 elem_id='generate_seed')
            gen_device = gr.Dropdown(label='设备', choices=DEVICES, value='cuda:0',
                                     elem_id='generate_device')
        gen_run = gr.Button('生成', variant='primary', elem_id='generate_run')
        gen_audio = gr.Audio(label='生成结果', type='filepath', interactive=False,
                             elem_id='generate_audio')
        gen_status = gr.Markdown('生成前请先试听确认角色卡声音；不会自动上传任何音频。',
                                 elem_id='generate_status')

        gen_refresh.click(refresh_roles, gen_role, gen_role, api_name='generate_refresh')

        def do_generate_text(role_id, text_value, seed_value, device_value):
            try:
                audio_path, details = synthesize(role_id, text_value, seed_value, device_value)
            except (OSError, ValueError, RuntimeError, TypeError) as exc:
                return None, f'生成失败：{exc}'
            return audio_path, details

        gen_run.click(do_generate_text, [gen_role, gen_text, gen_seed, gen_device],
                      [gen_audio, gen_status], api_name='generate_text')

    with gr.Tab('长文本生成', id='long_text'):
        gr.Markdown('## 长文本生成\n按段落与句子切分，用同一角色逐段生成，检查漏句、突发加速、'
                    '音量跳变、音色漂移与点击声后淡化拼接。')
        with gr.Row():
            long_role = gr.Dropdown(label='选择角色', choices=initial, value=initial_value,
                                    scale=5, elem_id='long_role')
            long_refresh = gr.Button('刷新角色', elem_id='long_refresh')
        long_text = gr.Textbox(label='长文本', lines=12, elem_id='long_text')
        with gr.Row():
            long_chars = gr.Slider(40, 300, value=120, step=10, label='每段最大字数',
                                   elem_id='long_chars')
            long_crossfade = gr.Slider(0, 100, value=25, step=5, label='拼接淡化（毫秒）',
                                       elem_id='long_crossfade')
            long_device = gr.Dropdown(label='设备', choices=DEVICES, value='cuda:0',
                                      elem_id='long_device')
        long_run = gr.Button('分段生成并拼接', variant='primary', elem_id='long_run')
        long_audio = gr.Audio(label='长文本结果', type='filepath', interactive=False,
                              elem_id='long_audio')
        long_status = gr.Markdown('长文本会逐段调用模型，耗时较长；可随时关闭浏览器中断等待。',
                                  elem_id='long_status')

        long_refresh.click(refresh_roles, long_role, long_role, api_name='long_refresh')

        def do_long(role_id, text_value, chars, crossfade, device_value):
            try:
                audio_path, details = generate_long_text(role_id, text_value, chars,
                                                          crossfade, device_value)
            except (OSError, ValueError, RuntimeError, TypeError) as exc:
                return None, f'长文本生成失败：{exc}'
            return audio_path, details

        long_run.click(do_long, [long_role, long_text, long_chars, long_crossfade, long_device],
                       [long_audio, long_status], api_name='long_generate')

