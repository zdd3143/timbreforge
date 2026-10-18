"""Lazy multi-role creation tabs; services are imported only on explicit actions.

All selectors carry persisted IDs. No project, model, dataset or network reads
happen while building these tabs. Reference preparation never feeds synthesis.
"""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_SCRIPT = '旁白：夜已经很深了。\n角色A〔压低声音〕：你听见了吗？\n角色B〔迟疑〕：可能只是风。'
ISSUES = ['漏字/错字', '音色偏移', '噪声/失真', '停顿/节奏', '情绪不符']


def project_store(root=None):
    from timbreforge.projects import ProjectStore
    return ProjectStore(root)


def project_choices(root=None):
    return [(f"{p['name']} · {p['id']}", p['id']) for p in project_store(root).list()]


def character_choices():
    from timbreforge.characters import CharacterStore
    return [(f"{c['name']} · {c['id']}", c['id']) for c in CharacterStore().list()]


def create_project(name, script, root=None):
    return project_store(root).create(name, script)


def load_project(project_id, root=None):
    if not project_id:
        raise ValueError('请先选择项目')
    return project_store(root).load(project_id)


def bind_speaker(project_id, speaker, character_id, root=None):
    if not speaker or not character_id:
        raise ValueError('请选择说话人与角色卡')
    return project_store(root).bind(project_id, speaker, character_id)


def generate_line(project_id, line_id, count=1, seed=42, confirmed=False, root=None, device='cuda:0'):
    # Reject before even constructing the service; a checkbox is never implicit.
    if not confirmed:
        raise ValueError('请确认已获得参考声音使用授权')
    if not line_id:
        raise ValueError('请选择要生成的台词；不会批量生成整个项目')
    if device not in ('cuda:0', 'cpu'):
        raise ValueError('Qwen 设备请选择 cuda:0 或 cpu；不会自动切换')
    if isinstance(count, bool) or int(count) != count or not 1 <= count <= 4:
        raise ValueError('候选数量必须是 1–4 的整数')
    if isinstance(seed, bool) or int(seed) != seed or seed < 0:
        raise ValueError('种子必须是非负整数')
    return project_store(root).generate_line(project_id, line_id, count=int(count),
                                              seed=int(seed), confirmed=True, device=device)


def review_candidate(project_id, line_id, candidate_id, selected, favorite, issues, note, root=None):
    if not candidate_id:
        raise ValueError('请选择候选 A')
    return project_store(root).review(project_id, line_id, candidate_id,
                                      selected=bool(selected), favorite=bool(favorite),
                                      issues=list(issues or []), note=note or '')


def assess_candidate(project_id, line_id, candidate_id, observed_text='', run_speaker=False,
                     run_quality=False, root=None):
    if not candidate_id:
        raise ValueError('请选择候选 A')
    return project_store(root).assess(project_id, line_id, candidate_id,
                                      observed_text=observed_text or None,
                                      run_speaker=bool(run_speaker), run_quality=bool(run_quality))


def _line(project, line_id):
    return next((line for line in (project or {}).get('lines', []) if line['id'] == line_id), None)


def _candidate(project, line_id, candidate_id):
    line = _line(project, line_id)
    return next((c for c in (line or {}).get('candidates', []) if c['id'] == candidate_id), None)


def candidate_path(project, line_id, candidate_id, root=None):
    candidate = _candidate(project, line_id, candidate_id)
    if not candidate:
        return None
    store = project_store(root)
    # State is only a selector hint; authoritative manifest paths come from disk.
    current = store.load(project['id'])
    candidate = _candidate(current, line_id, candidate_id)
    if not candidate:
        raise ValueError('候选已不存在，请重新加载项目')
    directory = (Path(store.root) / current['id']).resolve()
    path = (directory / candidate['file']).resolve()
    if not path.is_relative_to(directory):
        raise ValueError('候选音频路径超出项目目录')
    return str(path)


def compare_candidates(project, line_id, a_id, b_id, match_volume=False, root=None):
    a = candidate_path(project, line_id, a_id, root)
    b = candidate_path(project, line_id, b_id, root)
    if not a or not b:
        raise ValueError('请选择 A 和 B 两个候选')
    from timbreforge.candidates import compare_audio
    # Derived listening files only. The project/raw candidate files are untouched.
    return compare_audio(a, b, match_volume=bool(match_volume))


def export_project(project_id, gap_seconds=.25, root=None):
    return project_store(root).export(project_id, gap_seconds=gap_seconds)


def inspect_reference(path, transcript='', backend='indextts', instruction=''):
    if not path:
        raise ValueError('请先提供参考音频')
    from timbreforge.reference_assistant import inspect_reference as inspect
    return inspect(path, transcript=transcript, backend=backend, instruction=instruction)


def prepare_reference(path, confirmed=False, start_seconds=0, end_seconds=None, mono=False,
                      transcript='', transcript_confirmed=False, output_dir=None):
    if not confirmed:
        raise ValueError('请明确确认裁切/转单声道操作；不会静默处理')
    if not path:
        raise ValueError('请先提供参考音频')
    from timbreforge.reference_assistant import prepare_confirmed_reference
    return prepare_confirmed_reference(path, confirmed=True, start_seconds=start_seconds,
                                       end_seconds=end_seconds, mono=mono, output_dir=output_dir,
                                       transcript=transcript, transcript_confirmed=transcript_confirmed)


def build_creation_tabs(gr, synthesis_options=None):
    """Add tabs without reading stores; share the host's synthesis queue."""
    synthesis_options = synthesis_options or {}
    def safe(fn):
        def call(*args):
            try:
                return fn(*args)
            except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
                raise gr.Error(str(exc)) from exc
        return call

    def pick(choices, current):
        values = [value for _, value in choices]
        return current if current in values else (values[0] if values else None)

    def line_view(project, current):
        choices = [(f"{l['line_number']} · {l['speaker']}：{l['text'][:60]}", l['id'])
                   for l in (project or {}).get('lines', [])]
        return gr.update(choices=choices, value=pick(choices, current))

    def project_view(project, current_line, current_speaker):
        speakers = list(dict.fromkeys(l['speaker'] for l in project['lines']))
        current_speaker = current_speaker if current_speaker in speakers else (speakers[0] if speakers else None)
        return (project, gr.update(choices=[(f"{project['name']} · {project['id']}", project['id'])], value=project['id']),
                line_view(project, current_line), gr.update(choices=speakers, value=current_speaker),
                json.dumps(project.get('bindings', {}), ensure_ascii=False, indent=2))

    def candidate_view(project, line_id, a_id, b_id):
        line = _line(project, line_id) or {}
        choices = [(f"{i + 1} · {c['id']} (seed {c.get('seed', '?')})", c['id'])
                   for i, c in enumerate(line.get('candidates', []))]
        a = pick(choices, a_id)
        b = b_id if b_id in [v for _, v in choices] else (choices[1][1] if len(choices) > 1 else a)
        return gr.update(choices=choices, value=a), gr.update(choices=choices, value=b)

    def review_view(project, line_id, candidate_id):
        candidate = _candidate(project, line_id, candidate_id) or {}
        line = _line(project, line_id) or {}
        scores = candidate.get('scores', {})
        return (candidate_path(project, line_id, candidate_id) if candidate else None,
                bool(candidate_id and line.get('selected_candidate_id') == candidate_id),
                candidate.get('favorite', False), candidate.get('issues', []), candidate.get('note', ''),
                scores.get('speaker_similarity'), scores.get('content_accuracy'), scores.get('quality'))

    with gr.Tab('多角色项目 / 候选比较'):
        gr.Markdown('先在「角色库」创建角色卡，再为每个说话人绑定角色。逐行生成只追加当前台词的候选。')
        state = gr.State(None)
        with gr.Row():
            name = gr.Textbox(label='项目名称', value='雨后重逢', elem_id='create_name')
            create = gr.Button('创建项目', elem_id='create_create')
        script = gr.Textbox(label='剧本：说话人：台词（每行一句）', value=DEFAULT_SCRIPT, lines=5, elem_id='create_script')
        with gr.Row():
            project = gr.Dropdown(label='项目', choices=[], elem_id='create_project')
            refresh = gr.Button('刷新项目（保留当前选择）', elem_id='create_refresh')
            load = gr.Button('加载项目', elem_id='create_load')
        with gr.Row():
            speaker = gr.Dropdown(label='说话人', choices=[], elem_id='create_speaker')
            character = gr.Dropdown(label='从角色库选择角色卡', choices=[], elem_id='create_character')
            refresh_char = gr.Button('刷新角色卡', elem_id='create_refresh_char')
            bind = gr.Button('绑定角色', elem_id='create_bind')
        bindings = gr.Textbox(label='已保存的说话人绑定', interactive=False, elem_id='create_bindings')
        line = gr.Dropdown(label='当前台词（仅生成这一行）', choices=[], elem_id='create_line')
        with gr.Row():
            count = gr.Slider(1, 4, value=1, step=1, label='追加候选数', elem_id='create_count')
            seed = gr.Number(value=42, precision=0, label='起始种子', elem_id='create_seed')
        device = gr.Dropdown(choices=[('GPU / CUDA', 'cuda:0'), ('CPU（较慢）', 'cpu')],
                             value='cuda:0', label='仅 Qwen 逐句生成设备（GPU BF16 / CPU FP32；其他后端忽略；不自动回退）',
                             elem_id='create_device')
        consent = gr.Checkbox(label='我已获得角色参考声音的使用授权', value=False, elem_id='create_consent')
        generate = gr.Button('生成当前台词 / 追加候选', variant='primary', elem_id='create_generate')
        with gr.Row():
            a = gr.Dropdown(label='候选 A（编辑此候选的审听记录）', choices=[], elem_id='create_a')
            b = gr.Dropdown(label='候选 B', choices=[], elem_id='create_b')
        with gr.Row():
            audio_a = gr.Audio(label='A 独立试听', type='filepath', interactive=False, elem_id='create_audio_a')
            audio_b = gr.Audio(label='B 独立试听', type='filepath', interactive=False, elem_id='create_audio_b')
        with gr.Row():
            play_a = gr.Button('试听 A', elem_id='create_play_a')
            play_b = gr.Button('试听 B', elem_id='create_play_b')
        with gr.Row():
            match = gr.Checkbox(label='比较时匹配 RMS 振幅（非 LUFS；只创建试听副本）', value=False, elem_id='create_match')
            compare = gr.Button('准备 A/B 试听', elem_id='create_compare')
        comparison = gr.JSON(label='试听增益与处理说明', elem_id='create_comparison')
        with gr.Row():
            selected = gr.Checkbox(label='选用 A 进入导出', value=False, elem_id='create_selected')
            favorite = gr.Checkbox(label='收藏 A', value=False, elem_id='create_favorite')
        issues = gr.CheckboxGroup(choices=ISSUES, label='A 的问题标签', elem_id='create_issues')
        note = gr.Textbox(label='A 的人工审听备注', elem_id='create_note')
        save = gr.Button('保存 A 的选择 / 收藏 / 问题 / 备注', elem_id='create_save')
        gr.Markdown('以下三个评分轴独立且可为空。空值表示未核验，不代表差评。内容核验不自动运行 ASR。')
        with gr.Row():
            speaker_score = gr.JSON(label='声线相似度（独立轴）', elem_id='create_speaker_score')
            content_score = gr.JSON(label='内容准确度（独立轴）', elem_id='create_content_score')
            quality_score = gr.JSON(label='音频质量（独立轴）', elem_id='create_quality_score')
        observed = gr.Textbox(label='人工听写的实际内容（留空则内容未核验）', elem_id='create_observed')
        with gr.Row():
            run_speaker = gr.Checkbox(label='明确启用本地 CAM++ 声线评估', value=False, elem_id='create_run_speaker')
            run_quality = gr.Checkbox(label='明确启用本地 MOS 质量评估', value=False, elem_id='create_run_quality')
            assess = gr.Button('评估并保存 A 的独立评分轴', elem_id='create_assess')
        with gr.Row():
            gap = gr.Number(value=.25, label='导出台词间隔（秒）', elem_id='create_gap')
            export = gr.Button('导出已选台词', elem_id='create_export')
        export_files = gr.File(label='导出 ZIP / 对白清单 / 音频', file_count='multiple', interactive=False, elem_id='create_export_files')
        export_report = gr.JSON(label='导出明细', elem_id='create_export_report')

    with gr.Tab('参考音频助手'):
        gr.Markdown('仅本地检查，不自动上传任何服务。不自动裁切、不覆盖原始文件。处理结果需要您主动下载或另行选作参考。')
        ref = gr.Audio(label='原始参考音频', type='filepath', sources=['upload'], format=None, editable=False, elem_id='create_ref')
        transcript = gr.Textbox(label='与参考片段对应的转录（不自动识别）', elem_id='create_transcript')
        with gr.Row():
            backend = gr.Dropdown(choices=['indextts', 'cosyvoice3', 'qwen3'], value='qwen3', label='计划使用的后端', elem_id='create_backend')
            instruction = gr.Textbox(label='计划使用的表演指令', elem_id='create_instruction')
        inspect = gr.Button('检查原始音频 / 后端消费说明', elem_id='create_inspect')
        report = gr.JSON(label='检查报告（阈值、警告与未核验项）', elem_id='create_report')
        with gr.Row():
            start = gr.Number(value=0, label='裁切开始（秒）', elem_id='create_start')
            end = gr.Number(value=None, label='裁切结束（留空到结尾）', elem_id='create_end')
            mono = gr.Checkbox(label='转为单声道', value=False, elem_id='create_mono')
        confirmed = gr.Checkbox(label='我明确确认上述裁切/转换操作，另存新文件', value=False, elem_id='create_confirmed')
        text_confirmed = gr.Checkbox(label='我已核对转录与处理后的片段一致', value=False, elem_id='create_text_confirmed')
        prepare = gr.Button('按确认的设置另存参考副本', elem_id='create_prepare')
        processed = gr.Audio(label='处理后参考（不会自动用于合成）', type='filepath', interactive=False, elem_id='create_processed')
        processed_file = gr.File(label='下载副本后自行选择使用', interactive=False, elem_id='create_processed_file')
        processed_report = gr.JSON(label='处理清单与转录确认状态', elem_id='create_processed_report')

    # Browser-local audition only: no file processing and never overlap A/B.
    play_a.click(fn=None, js="() => { const b = document.querySelector('#create_audio_b audio'); if (b) b.pause(); const a = document.querySelector('#create_audio_a audio'); if (a) { a.currentTime = 0; a.play().catch(() => {}); } }")
    play_b.click(fn=None, js="() => { const a = document.querySelector('#create_audio_a audio'); if (a) a.pause(); const b = document.querySelector('#create_audio_b audio'); if (b) { b.currentTime = 0; b.play().catch(() => {}); } }")
    refresh.click(safe(lambda current: gr.update(choices=(choices := project_choices()), value=pick(choices, current))), project, project)
    refresh_char.click(safe(lambda current: gr.update(choices=(choices := character_choices()), value=pick(choices, current))), character, character)
    project_outputs = [state, project, line, speaker, bindings]
    create.click(safe(lambda n, s: project_view(create_project(n, s), None, None)), [name, script], project_outputs)
    load.click(safe(lambda p, l, sp: project_view(load_project(p), l, sp)), [project, line, speaker], project_outputs)
    bind.click(safe(lambda p, sp, c, l: project_view(bind_speaker(p, sp, c), l, sp)), [project, speaker, character, line], project_outputs)
    generate.click(safe(lambda p, l, c, s, ok, sp, dev: project_view(generate_line(p, l, c, s, ok, device=dev), l, sp)),
                   [project, line, count, seed, consent, speaker, device], project_outputs, **synthesis_options)
    state.change(safe(candidate_view), [state, line, a, b], [a, b])
    line.change(safe(candidate_view), [state, line, a, b], [a, b])
    review_outputs = [audio_a, selected, favorite, issues, note, speaker_score, content_score, quality_score]
    a.change(safe(review_view), [state, line, a], review_outputs)
    # A retained dropdown value may not fire change after load/save; refresh its
    # saved review and independent axes explicitly whenever state changes.
    state.change(safe(review_view), [state, line, a], review_outputs)
    b.change(safe(candidate_path), [state, line, b], audio_b)
    state.change(safe(candidate_path), [state, line, b], audio_b)
    save.click(safe(review_candidate), [project, line, a, selected, favorite, issues, note], state)
    assess.click(safe(assess_candidate), [project, line, a, observed, run_speaker, run_quality], state, **synthesis_options).then(
        safe(review_view), [state, line, a], review_outputs)
    def compare_view(p, l, av, bv, matching):
        result = compare_candidates(p, l, av, bv, matching)
        return result['a'], result['b'], result
    compare.click(safe(compare_view), [state, line, a, b, match], [audio_a, audio_b, comparison])
    def export_view(p, seconds):
        result = export_project(p, seconds)
        files = list(dict.fromkeys(str(result[k]) for k in ('zip', 'dialogue', 'csv', 'mixed_audio') if result.get(k)))
        return files, result
    export.click(safe(export_view), [project, gap], [export_files, export_report])
    inspect.click(safe(inspect_reference), [ref, transcript, backend, instruction], report)
    def prepare_view(path, ok, begin, finish, to_mono, text, text_ok, chosen_backend, chosen_instruction):
        result = prepare_reference(path, ok, begin, finish, to_mono, text, text_ok)
        result['selected_backend_inspection'] = inspect_reference(
            result['output_path'], result.get('transcript', ''), chosen_backend, chosen_instruction)
        result['preparation_report_note'] = '默认处理检查与所选后端检查分别保留；请以 selected_backend_inspection 查看所选后端消费说明。'
        return result['output_path'], result['output_path'], result
    prepare.click(safe(prepare_view), [ref, confirmed, start, end, mono, transcript, text_confirmed, backend, instruction],
                  [processed, processed_file, processed_report])
