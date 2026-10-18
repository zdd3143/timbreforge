"""VoiceDesign page: lazy services, explicit human choice, real Qwen controls.

Importing/building this module never inspects models, scans sessions or networks.
"""
from __future__ import annotations

import json

PROMPT_PRESETS = {
    '自定义': '',
    '温暖叙事男声': '成年男性，中低音，温暖厚实而不浑浊，轻微颗粒感，普通话清楚自然，叙事语气平静克制。',
    '冷静成熟御姐': '成年女性，中低音，清冷成熟，带少许沙哑颗粒，吐字清晰，语气沉稳克制，不夸张卖萌。',
    '明亮活力少女': '年轻成年女性，明亮清脆、轻盈自然，中高音，富有活力和笑意，普通话清晰，不尖锐刺耳。',
    '苍老睿智旁白': '年长男性，低沉而有胸腔共鸣，细微沙哑，语速从容，讲述感强，吐字清楚，不虚弱含糊。',
    '轻柔空灵女声': '成年女性，轻柔空灵，少量气声但保留清晰声芯，柔和中高音，近距离自然讲述，避免耳语到听不清。',
}

PARAM_FIELDS = ('seed', 'language', 'do_sample', 'temperature', 'top_p', 'top_k',
                'repetition_penalty', 'max_new_tokens', 'subtalker_dosample',
                'subtalker_temperature', 'subtalker_top_p', 'subtalker_top_k')
VOICE_DESIGN_LANGUAGES = ['Chinese', 'Japanese', 'English']


def design_store(root=None):
    from timbreforge.designs import DesignStore
    return DesignStore(root)


def assemble_prompt(prompt, gender='', age='', timbre='', pitch='', accent='', performance=''):
    """Only append human-readable descriptors; no hidden numeric attribute mapping."""
    base = (prompt or '').strip()
    additions = [f'{label}：{value.strip()}' for label, value in zip(
        ('性别表达', '年龄感', '音色', '音高感', '口音', '表演'),
        (gender, age, timbre, pitch, accent, performance)) if isinstance(value, str) and value.strip()]
    return '；'.join(([base] if base else []) + additions)


def _params(values):
    if not isinstance(values, dict) or set(values) != set(PARAM_FIELDS):
        raise ValueError('请提供完整且已知的 Qwen 采样参数')
    values = dict(values)
    for key in ('seed', 'top_k', 'max_new_tokens', 'subtalker_top_k'):
        value = values[key]
        if isinstance(value, bool) or int(value) != value:
            raise ValueError(f'{key} 必须是整数')
        values[key] = int(value)
    for key in ('do_sample', 'subtalker_dosample'):
        if type(values[key]) is not bool:
            raise ValueError(f'{key} 必须是布尔值')
    from timbreforge.backend_qwen import QwenParams
    params = QwenParams(**values)
    if params.language not in VOICE_DESIGN_LANGUAGES:
        raise ValueError('声线设计只支持中文、日语和英语。')
    return params


def generate_design(prompt, text, values, count=1, device='cuda:0', confirmed=False, root=None):
    if confirmed is not True:
        raise ValueError('请明确确认仅设计新的合成声音，不冒充特定真人')
    if not isinstance(prompt, str) or not prompt.strip() or not isinstance(text, str) or not text.strip():
        raise ValueError('请填写声线描述与试听台词')
    if isinstance(count, bool) or int(count) != count or not 1 <= count <= 4:
        raise ValueError('候选数量必须为 1–4 的整数')
    params = _params(values)
    return design_store(root).create(prompt.strip(), text.strip(), params, count=int(count),
                                      device=device, confirmed=True)


def append_design(session_id, values, count=1, device='cuda:0', confirmed=False, root=None):
    if confirmed is not True:
        raise ValueError('追加候选也需要确认合成声音设计用途')
    if not session_id:
        raise ValueError('请先创建或加载设计会话')
    if isinstance(count, bool) or int(count) != count or not 1 <= count <= 4:
        raise ValueError('候选数量必须为 1–4 的整数')
    return design_store(root).append(session_id, params=_params(values), count=int(count),
                                      device=device, confirmed=True)


def session_choices(root=None, *, library_only=False):
    """Return design records; the UI can limit these to saved character cards."""
    sessions = design_store(root).list()
    labels = {}
    if library_only:
        from timbreforge.characters import CharacterStore
        for card in CharacterStore(root).list():
            session_id = (card.get('provenance') or {}).get('session_id')
            if session_id:
                labels.setdefault(session_id, []).append(card['name'])
        sessions = [session for session in sessions if session['id'] in labels]
    return [(((' / '.join(labels.get(session['id'], [])) + ' · 设计记录 · ')
              if library_only else f"{session['prompt'][:45]} · ") + session['id'], session['id'])
            for session in sessions]


def load_session(session_id, root=None):
    if not session_id:
        raise ValueError('请选择设计会话')
    return design_store(root).load(session_id)


def select_design(session_id, candidate_id, favorite=False, note='', root=None):
    if not session_id or not candidate_id:
        raise ValueError('请选择会话与候选 A')
    if type(favorite) is not bool or not isinstance(note, str):
        raise ValueError('收藏与备注的类型无效')
    return design_store(root).select(session_id, candidate_id, favorite=favorite, note=note)


def save_design_character(session_id, candidate_id, name, confirmed=False, root=None):
    if confirmed is not True:
        raise ValueError('保存角色前请确认合成声音使用范围')
    store = design_store(root)
    session = store.load(session_id)
    if not candidate_id or session.get('selected_candidate_id') != candidate_id:
        raise ValueError('请先明确选用候选 A，再保存该候选为角色')
    if not isinstance(name, str) or not name.strip():
        raise ValueError('请填写角色名称')
    return store.save_character(session_id, candidate_id, name.strip(), confirmed=True)


def save_design_candidate(session_id, candidate_id, name, root=None):
    """Save either visible A/B candidate directly from the save action.

    The old review flow required a separate favourite/select action.  The UI
    now treats pressing "save A" or "save B" as the explicit choice and keeps
    the legacy ``save_design_character`` API available for older callers.
    """
    if not session_id or not candidate_id:
        raise ValueError('请选择要保存的候选。')
    if not isinstance(name, str) or not name.strip():
        raise ValueError('请填写角色名称。')
    store = design_store(root)
    # Persist the same selection metadata used by the legacy store, but do it
    # as part of the save click so the user never has to confirm it separately.
    store.select(session_id, candidate_id, favorite=False, note='')
    return store.save_character(session_id, candidate_id, name.strip(), confirmed=True)


def design_audio(session_id, candidate_id, root=None):
    if not session_id or not candidate_id:
        return None
    return str(design_store(root).audio_path(session_id, candidate_id))


def compare_designs(session_id, a_id, b_id, match_volume=False, root=None):
    if not a_id or not b_id:
        raise ValueError('请选择 A/B 两个候选')
    if type(match_volume) is not bool:
        raise ValueError('试听音量匹配必须为布尔值')
    store = design_store(root)
    from timbreforge.candidates import compare_audio
    return compare_audio(store.audio_path(session_id, a_id), store.audio_path(session_id, b_id),
                         match_volume=match_volume)


def build_design_tab(gr, synthesis_options=None):
    """Build inside existing Tabs; parent places this first. No readiness claims."""
    synthesis_options = dict(synthesis_options or {})
    def safe(fn):
        def call(*args):
            try:
                return fn(*args)
            except (ValueError, RuntimeError, OSError, TypeError, KeyError, OverflowError) as exc:
                from timbreforge.simple_ui import error_message
                raise gr.Error(error_message(exc)) from exc
        return call

    def pick(choices, current):
        return current if current in [value for _, value in choices] else (choices[0][1] if choices else None)

    def update_session(session):
        return (session, gr.update(choices=[(f"{session['prompt'][:45]} · {session['id']}", session['id'])], value=session['id']),
                session['prompt'], session['text'],
                f"会话 {session['id']}：已保存 {len(session['candidates'])} 个候选；最终选择：{session.get('selected_candidate_id') or '尚未人工选择'}。")

    def candidate_choices(session, a_id, b_id):
        choices = [(f"{i+1} · seed {c['seed']} · {c['id']}", c['id']) for i, c in enumerate((session or {}).get('candidates', []))]
        a_value = pick(choices, a_id)
        alternatives = [value for _, value in choices if value != a_value]
        b_value = b_id if b_id in alternatives else next(iter(alternatives), None)
        return (gr.update(choices=choices, value=a_value),
                gr.update(choices=choices, value=b_value))

    def details(session_id, candidate_id):
        if not session_id or not candidate_id:
            return None, None
        session = load_session(session_id)
        candidate = next(c for c in session['candidates'] if c['id'] == candidate_id)
        return design_audio(session_id, candidate_id), dict(
            candidate=candidate, selected=session.get('selected_candidate_id') == candidate_id,
            note='后端元数据不是听感或身份质量保证；请人工选择。')

    with gr.Tab('文字声线设计 / Qwen VoiceDesign'):
        state = gr.State(None)
        gr.Markdown('描述年龄感、音色、口音和说话方式，让你的原创角色拥有自己的声音。')
        with gr.Accordion('本地模型与环境', open=False):
            readiness = gr.Markdown('点击检查本地模型文件。')
            check = gr.Button('检查模型配置', elem_id='design_check')
        with gr.Row():
            with gr.Column(scale=1, min_width=360):
                gr.Markdown('### 01 / 描述声音')
                prompt = gr.Textbox(label='角色听起来是什么样？', lines=3, value='低沉温暖的成年男声，清晰自然，语气平静而克制。', elem_id='design_prompt')
                with gr.Row():
                    preset = gr.Dropdown(choices=list(PROMPT_PRESETS), value='自定义', label='从一个灵感开始', elem_id='design_preset')
                    random_seed = gr.Button('随机种子', elem_id='design_random_seed')
                with gr.Accordion('可选：把结构化描述加入上方提示词', open=False):
                    with gr.Row():
                        gender = gr.Textbox(label='性别表达', elem_id='design_gender')
                        age = gr.Textbox(label='年龄感', elem_id='design_age')
                        timbre = gr.Textbox(label='音色质感', elem_id='design_timbre')
                    with gr.Row():
                        pitch = gr.Textbox(label='音高感', elem_id='design_pitch')
                        accent = gr.Textbox(label='口音', elem_id='design_accent')
                        performance = gr.Textbox(label='表演风格', elem_id='design_performance')
                    assemble = gr.Button('将上述文字明确加入提示词', elem_id='design_assemble')
                text = gr.Textbox(label='让角色说一句话', value='你好，很高兴与你见面。这是我们故事的开始。', lines=3, elem_id='design_text')
                with gr.Row():
                    seed = gr.Number(value=42, precision=0, label='起始种子', elem_id='design_seed')
                    count = gr.Slider(1,4,value=1,step=1,label='本次候选数',elem_id='design_count')
                    device = gr.Dropdown(choices=[('GPU / cuda:0','cuda:0'),('CPU / cpu','cpu')],value='cuda:0',label='设备',elem_id='design_device')
                    language = gr.Dropdown(choices=VOICE_DESIGN_LANGUAGES,value='Chinese',label='语言',elem_id='design_language')
                gr.Markdown('建议先生成 1 个候选，再调整描述或种子。首次生成需要加载模型，请稍候。')
                with gr.Accordion('进阶：采样参数',open=False):
                    do_sample = gr.Checkbox(value=True,label='do_sample',elem_id='design_do_sample')
                    temperature = gr.Slider(.05,2,value=.9,step=.05,label='temperature',elem_id='design_temperature')
                    top_p = gr.Slider(.01,1,value=1,step=.01,label='top_p',elem_id='design_top_p')
                    top_k = gr.Slider(0,100,value=50,step=1,label='top_k',elem_id='design_top_k')
                    repetition = gr.Slider(1,2,value=1.05,step=.01,label='repetition_penalty',elem_id='design_repetition_penalty')
                    max_tokens = gr.Slider(128,4096,value=1024,step=1,label='max_new_tokens',elem_id='design_max_new_tokens')
                    sub_sample = gr.Checkbox(value=True,label='subtalker_dosample',elem_id='design_subtalker_dosample')
                    sub_temp = gr.Slider(.05,2,value=.9,step=.05,label='subtalker_temperature',elem_id='design_subtalker_temperature')
                    sub_p = gr.Slider(.01,1,value=1,step=.01,label='subtalker_top_p',elem_id='design_subtalker_top_p')
                    sub_k = gr.Slider(0,100,value=50,step=1,label='subtalker_top_k',elem_id='design_subtalker_top_k')
                consent = gr.Checkbox(value=False,label='我确认设计新的合成声音，不以冒充特定真人为目的，并有权使用台词内容。',elem_id='design_consent')
                generate = gr.Button('生成新声线候选',variant='primary',interactive=False,elem_id='design_generate')
            with gr.Column(scale=1, min_width=360):
                gr.Markdown('### 02 / 试听与挑选')
                with gr.Row():
                    sessions = gr.Dropdown(choices=[],label='角色库内的设计记录',elem_id='design_sessions')
                    refresh = gr.Button('刷新角色库记录',elem_id='design_refresh')
                    load = gr.Button('加载会话',elem_id='design_load')
                append = gr.Button('追加候选 · 保留原描述和台词',interactive=False,elem_id='design_append')
                status = gr.Markdown('生成后直接试听 A 或 B，填写名称即可保存到角色库。这里不显示未入库的旧草稿。', elem_id='design_status')
                with gr.Row():
                    a = gr.Dropdown(choices=[],label='候选 A',elem_id='design_a')
                    b = gr.Dropdown(choices=[],label='候选 B',elem_id='design_b')
                with gr.Row():
                    audio_a = gr.Audio(type='filepath',interactive=False,label='设计 A',elem_id='design_audio_a')
                    audio_b = gr.Audio(type='filepath',interactive=False,label='设计 B',elem_id='design_audio_b')
                with gr.Row():
                    play_a = gr.Button('试听 A',elem_id='design_play_a')
                    play_b = gr.Button('试听 B',elem_id='design_play_b')
                with gr.Accordion('候选生成记录', open=False):
                    metadata = gr.JSON(label='候选后端记录与人工选择（不自动排名）',elem_id='design_metadata')
                gr.Markdown('### 03 / 保存到角色库\n为喜欢的声音命名；保存成功后它才会进入统一角色库，供角色复用和多角色项目使用。')
                gr.Markdown('保存的是选定设计的原始参考音频与已知台词，不是把提示词当声音身份。后续角色复用使用 Qwen Base 0.6B 参考克隆；'
                            '不重复设计提示词。参考必须 ≤30 秒，超长会报错，不会静默裁切。')
                with gr.Row():
                    character_name_a = gr.Textbox(label='角色名称 A',elem_id='design_character_name_a')
                    save = gr.Button('保存 A 到角色库',elem_id='design_save')
                with gr.Row():
                    character_name_b = gr.Textbox(label='角色名称 B',elem_id='design_character_name_b')
                    save_b = gr.Button('保存 B 到角色库',elem_id='design_save_b')
                # Kept invisible for clients that still call the old API name;
                # the visible workflow never exposes a favourite/confirmation
                # step anymore.
                legacy_confirm = gr.Checkbox(value=True, visible=False,
                                              elem_id='design_save_confirm')
                legacy_select = gr.Button('legacy select', visible=False,
                                          elem_id='design_select')
                card_status = gr.Markdown(elem_id='design_card_status')
                with gr.Accordion('角色存档详情', open=False):
                    card_report = gr.JSON(label='已保存角色（到角色复用页刷新）',elem_id='design_card')

    controls = [seed,language,do_sample,temperature,top_p,top_k,repetition,max_tokens,sub_sample,sub_temp,sub_p,sub_k]
    outputs = [state,sessions,prompt,text,status]
    def create_view(p,t,c,d,ok,*values):
        return update_session(generate_design(p,t,dict(zip(PARAM_FIELDS,values)),c,d,ok))
    def append_view(s,c,d,ok,*values):
        return update_session(append_design(s,dict(zip(PARAM_FIELDS,values)),c,d,ok))
    def choose_preset(name, current):
        return PROMPT_PRESETS.get(name) or current
    def next_seed():
        import secrets
        return secrets.randbelow(2**31)
    from timbreforge.readiness import readiness_markdown
    consent.change(lambda ok: (gr.update(interactive=bool(ok)), gr.update(interactive=bool(ok))),
                   consent, [generate, append], queue=False)
    check.click(safe(readiness_markdown), outputs=readiness, api_name='design_check')
    preset.change(choose_preset, [preset,prompt], prompt)
    random_seed.click(next_seed, outputs=seed)
    assemble.click(safe(assemble_prompt),[prompt,gender,age,timbre,pitch,accent,performance],prompt)
    generate.click(safe(create_view),[prompt,text,count,device,consent]+controls,outputs,api_name='design_generate',**synthesis_options)
    append.click(safe(append_view),[sessions,count,device,consent]+controls,outputs,api_name='design_append',**synthesis_options)
    refresh.click(safe(lambda current: gr.update(choices=(choices:=session_choices(library_only=True)),value=pick(choices,current))),sessions,sessions)
    def load_recipe(sid):
        session = load_session(sid)
        selected = next((c for c in session['candidates'] if c['id'] == session.get('selected_candidate_id')),
                        next(iter(session['candidates']), None))
        values = selected['result'].get('params', session['params']) if selected else session['params']
        candidates = candidate_choices(session, selected['id'] if selected else None, None)
        return (*update_session(session), *(values[key] for key in PARAM_FIELDS),
                selected['result'].get('device', session['device']) if selected else session['device'], *candidates)
    load.click(safe(load_recipe),sessions,outputs+controls+[device,a,b],api_name='design_load')
    state.change(safe(candidate_choices),[state,a,b],[a,b]).then(safe(details),[sessions,a],[audio_a,metadata]).then(safe(design_audio),[sessions,b],audio_b)
    a.change(safe(details),[sessions,a],[audio_a,metadata])
    b.change(safe(design_audio),[sessions,b],audio_b)
    def save_view(sid, cid, name):
        card = save_design_candidate(sid, cid, name)
        return card, '角色已保存。打开「角色复用」，点击刷新并选择角色，即可输入新台词。'
    save.click(safe(save_view),[sessions,a,character_name_a],[card_report,card_status],api_name='design_save',**synthesis_options)
    save_b.click(safe(save_view),[sessions,b,character_name_b],[card_report,card_status],api_name='design_save_b',**synthesis_options)
    legacy_select.click(safe(lambda sid, cid: update_session(select_design(sid, cid, False, ''))),
                        [sessions, a], outputs, api_name='design_select')
    for button,own,other in ((play_a,'a','b'),(play_b,'b','a')):
        button.click(fn=None,js=f"() => {{ document.querySelectorAll('#design_audio_{other} audio').forEach(a => a.pause()); const own=Array.from(document.querySelectorAll('#design_audio_{own} audio')).findLast(a => a.src); if(own) {{ own.currentTime=0; own.play().catch(()=>{{}}); }} }}")

