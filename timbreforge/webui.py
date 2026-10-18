"""TimbreForge 声线工作室（Gradio 界面）· 数据集检索版

对标 AI 生图的交互：
    左侧：声线描述 + 种子 + 台词 + 情绪 + 采样参数
    右侧：候选声线（可试听）-> 选中 -> 合成语音

默认使用授权参考音频；可选检索需要另行准备公开数据集与索引。

启动：
    python -m timbreforge.webui --port 7860
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import threading
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from timbreforge.voicebank import find_voice, get_bank
from timbreforge.voicelib import parse_voice_text

from timbreforge.config import get_config, set_config_path

EMOTION_KEYS = ("happy", "angry", "sad", "fearful",
                "disgusted", "melancholic", "surprised", "calm")
EMO_CN = {"happy": "开心", "angry": "愤怒", "sad": "悲伤", "fearful": "恐惧",
          "disgusted": "厌恶", "melancholic": "忧郁", "surprised": "惊讶",
          "calm": "平静"}

PRESETS = [
    "冰冷沙哑的成熟女声",
    "甜美活泼的少女",
    "睿智沉稳的老者",
    "低沉磁性的大叔",
    "空灵飘渺的女声",
    "清脆嘹亮的少年",
    "气声很重的病弱少女",
    "金属感的机器人声",
]
DEFAULT_TEXT = "你好，我是这个角色的声音。很高兴见到你。"


# ---------------------------------------------------------------- 检索
def _legacy_unavailable(exc):
    return (f"数据集检索不可用：{type(exc).__name__}: {exc}\n\n"
            "此可选功能需要单独准备公开数据集及检索索引；发布包不附带这些资产。"
            "可改用参考克隆或已保存角色。不会在启动时下载数据集。")

def do_search(prompt, seed, num, age, gender):
    """检索候选声线，返回表格数据与状态文本。"""
    if not prompt or not prompt.strip():
        return [], "请先填写声线描述。"
    try:
        bank = get_bank()
        num = int(num)
        e, ranked = find_voice(
            prompt, seed=int(seed), top_k=num,
            age=(float(age) if age is not None and float(age) >= 0 else None),
            gender=(gender if gender and gender != "auto" else None))
    except Exception as exc:
        return [], _legacy_unavailable(exc)

    rows = []
    for i, (x, sc) in enumerate(ranked):
        rows.append([
            "★" if x.cid == e.cid else str(i + 1),
            x.cid, x.name, x.gender, x.age,
            f"{x.duration:.1f}s", f"{sc:.3f}",
            f"{x.axes['rough']:+.2f}/{x.axes['breath']:+.2f}/"
            f"{x.axes['depth']:+.2f}/{x.axes['airy']:+.2f}",
            "、".join(x.tags[:8]),
            x.tagline,
        ])

    parsed = parse_voice_text(prompt)
    st = (f"**库内共 {len(bank)} 条声线**，检索到 {len(ranked)} 个候选。\n\n"
          f"- 命中词：{'、'.join(parsed['hits']) or '（无）'}\n"
          f"- 情感向量：{parsed['emotion'] or '（无）'}\n"
          f"- 选中的声线：`{e.cid}` **{e.name}**（{e.gender}，{e.age}）\n\n"
          f"在表格中挑一个，把它的 cid 填到下方「指定声线 cid」，或直接点「合成语音」"
          f"使用当前选中的 `{e.cid}`。")
    return rows, st


def do_preview(cid):
    """取该声线的原始参考音频（数据集里的 MP3 转 WAV）用于试听。"""
    try:
        bank = get_bank()
        e = bank.get(cid)
        if e is None:
            return None, f"找不到声线 `{cid}`"
        # Serve only a copied preview within the configured output root.
        import uuid
        wav = bank.to_wav(e)
        directory = get_config().outputs_dir / "previews"
        directory.mkdir(parents=True, exist_ok=True)
        preview = directory / f"{uuid.uuid4().hex}.wav"
        shutil.copy2(wav, preview)
        return str(preview), f"**{e.name}**（{e.cid}）\n\n{e.tagline}\n\n音区：{e.register}"
    except Exception as ex:
        return None, _legacy_unavailable(ex)


def do_synth(prompt, cid, seed, text, temperature, top_p, top_k, num_beams,
             rep, dfac, emo_dict, base_url, num_cand, age, gender):
    """合成语音。"""
    try:
        bank = get_bank()
        if cid and cid.strip():
            e = bank.get(cid.strip())
            if e is None:
                return None, f"找不到声线 `{cid}`，请先检索或检查 cid。"
        else:
            e, _ = find_voice(prompt, seed=_studio_seed(seed), top_k=int(num_cand),
                              age=(float(age) if age is not None and float(age) >= 0 else None),
                              gender=(gender if gender and gender != "auto" else None))
        wav = bank.to_wav(e)
    except Exception as exc:
        return None, _legacy_unavailable(exc)
    try:
        from timbreforge.backend_indextts import SynthParams
        from timbreforge.studio import GenerationRequest, generate
        emo = {k: v for k, v in (emo_dict or {}).items() if v > 0}
        parsed = parse_voice_text(prompt) if prompt else {"emotion": {}}
        for k, v in (parsed.get("emotion") or {}).items():
            emo.setdefault(k, v)
        params = SynthParams(
            text=text or DEFAULT_TEXT, seed=_studio_seed(seed), temperature=temperature,
            top_p=top_p, top_k=int(top_k), num_beams=int(num_beams),
            repetition_penalty=rep, duration_factor=dfac, emotion=emo,
            emotion_mode="emotion_vector" if emo else "same_as_speaker",
            max_text_tokens_per_segment=120, max_mel_tokens=1500)
        result = generate(GenerationRequest(
            task="clone", backend="indextts", reference_audio=str(wav),
            text=text or DEFAULT_TEXT, params=params,
            base_url=base_url or get_config().comfy_url,
            description=f"Dataset retrieval: {e.cid}; prompt: {prompt}"))
        audio, note = _studio_result(result)
        return audio, note + f"\n\n- 数据集声线：`{e.cid}` **{e.name}**"
    except Exception as exc:
        return _studio_failure(exc)


# ---------------------------------------------------------------- 工作室（服务与存储延迟导入，不在构建界面时查询硬件）
CONSENT_LABEL = "我确认对参考声音拥有使用权并已获得说话者授权，不用于冒充、欺诈或侵权。"


def _studio_failure(exc):
    from timbreforge.resources import is_memory_error
    if is_memory_error(exc):
        return None, f'⚠️ 内存或显存不足：{exc}\n\n闲置功能会自动释放模型。请等待其他任务结束后重试；已保存的角色与音频仍保留。'
    return None, (f"⚠️ {type(exc).__name__}: {exc}\n\n"
                  "请检查参考音频（非静音、30 秒内）、准确转录、任务/后端及本地依赖；"
                  "IndexTTS 需要 ComfyUI 已启动。不会自动安装、下载或重启服务。")


def _studio_result(result):
    warnings = result.get("warnings") or []
    if isinstance(warnings, str):
        warnings = [warnings]
    lines = ["合成完成。", f"- 任务：{result.get('task')} / {result.get('backend')}",
             f"- 用时：{result.get('elapsed')} s",
             f"- 音频：`{result['file']}`",
             f"- 元数据：`{result.get('metadata_file', '')}`"]
    if result.get("character_id"):
        lines.append(f"- 角色 ID：`{result['character_id']}`（到角色页刷新列表）")
    lines.extend(f"- 提醒：{warning}" for warning in warnings)
    lines.append("相同种子不保证后端最终波形逐样本一致。")
    return str(result["file"]), "\n\n".join(lines)


def _studio_seed(seed):
    if isinstance(seed, bool):
        raise ValueError("种子必须是非负整数")
    value = int(seed)
    if value != float(seed) or not 0 <= value <= 0x7FFFFFFFFFFFFFFF:
        raise ValueError("种子必须是有效范围内的非负整数")
    return value


def do_studio_synth(task, backend, reference_audio, reference_text, instruction,
                    text, seed, save_name, description, consent):
    """Clone/performance callback; pass the original uploaded filepath unchanged."""
    if consent is not True:
        return None, "请先明确确认参考声音的使用权与说话者授权，再合成。"
    try:
        if task not in ("clone", "instruct"):
            raise ValueError("请选择 clone 或 instruct；文字声线设计尚未接入")
        from timbreforge.studio import GenerationRequest, generate
        from timbreforge.backend_indextts import SynthParams
        from timbreforge.backend_qwen import QwenParams
        request = GenerationRequest(
            task=task, backend=backend, text=text,
            reference_audio=reference_audio, reference_text=reference_text or "",
            instruction=instruction or "", params=None if backend == 'qwen3' else SynthParams(seed=_studio_seed(seed)),
            qwen_params=QwenParams(seed=_studio_seed(seed)).to_dict() if backend == 'qwen3' else None,
            save_character_name=save_name or "", description=description or "")
        return _studio_result(generate(request))
    except Exception as exc:
        return _studio_failure(exc)


def studio_character_choices():
    """Gradio-independent choices, with stable IDs even when names repeat."""
    from timbreforge.characters import CharacterStore
    return [(card["name"], card["id"]) for card in CharacterStore().list()]


def do_character_preview(character_id):
    if not character_id:
        return None, "请选择角色卡；新保存的角色请先刷新列表。"
    try:
        from timbreforge.characters import CharacterStore
        card = CharacterStore().load(character_id)
        metadata = {key: card.get(key) for key in (
            "id", "name", "backend", "reference_text", "instruction", "description", "params")}
        return card["reference_audio"], "```json\n" + json.dumps(
            metadata, ensure_ascii=False, indent=2) + "\n```"
    except Exception as exc:
        return _studio_failure(exc)


def do_character_synth(character_id, text, instruction, seed, consent, device='cuda:0'):
    if consent is not True:
        return None, "请先明确确认参考声音的使用权与说话者授权，再合成。"
    try:
        if not character_id:
            raise ValueError("请先选择角色卡")
        if device not in ('cuda:0', 'cpu'):
            raise ValueError('Qwen 设备请选择 cuda:0 或 cpu；不会自动切换')
        from timbreforge.studio import GenerationRequest, generate
        # No default SynthParams here: retain saved backend and sampling controls.
        request = GenerationRequest(
            task="character", character_id=character_id, text=text,
            instruction=instruction or "", params=None,
            seed_override=None if seed is None or seed == "" else _studio_seed(seed), device=device)
        return _studio_result(generate(request))
    except Exception as exc:
        return _studio_failure(exc)


def studio_availability_markdown():
    try:
        from timbreforge.studio import capabilities
        lines = ["### 能力与本地文件状态（不探测硬件或联网）"]
        for card in capabilities():
            if card["backend"] == "indextts":
                state = "已接入（服务运行状态未探测）"
            elif card["available"]:
                state = "本地文件就绪（未执行推理）"
            else:
                state = "不可用"
            lines.append(f"- **{card['label']}**：{state}；任务："
                         f"{', '.join(card['tasks'])}。{card['note']}")
        lines.append("本地文件就绪不代表推理或 GPU 可用；实际生成时才验证运行环境。")
        return "\n\n".join(lines)
    except Exception as exc:
        return f"能力信息不可用：{type(exc).__name__}: {exc}"


def _build_studio_tabs(gr, synthesis_options):
    with gr.Tab("参考克隆 / 表演指令"):
        gr.Markdown("上传原始参考音频（不自动裁切、不做响度处理；最长 30 秒）。"
                    "clone 保留声音；instruct 另加表演指令，不等同于创造新声线。")
        task = gr.Dropdown([("参考克隆", "clone"), ("表演指令", "instruct")],
                           value="clone", label="任务")
        backend = gr.Dropdown([("Qwen Base 0.6B（本地参考克隆）", "qwen3"), ("IndexTTS / ComfyUI", "indextts"),
                               ("CosyVoice3", "cosyvoice3")],
                              value="qwen3", label="后端")
        gr.Markdown('Qwen Base 需要参考音频的准确转录，表演指令请留空。切换功能时自动释放闲置模型；生成结束后模型进程退出并释放内存。')
        reference = gr.Audio(type="filepath", sources=["upload"], format=None,
                             editable=False, label="原始参考音频（不裁切）")
        transcript = gr.Textbox(label="参考音频的准确完整转录", lines=3)
        instruction = gr.Textbox(label="表演指令（仅 instruct；clone 请留空）", lines=2)
        description = gr.Textbox(label="角色描述（元数据，不用于声音设计）", lines=2)
        save_name = gr.Textbox(label="保存为可复用角色（可选名称；合成成功后保存）")
        text = gr.Textbox(label="新台词", value=DEFAULT_TEXT, lines=3)
        seed = gr.Number(value=42, precision=0, label="合成种子（不保证最终波形一致）")
        consent = gr.Checkbox(value=False, label=CONSENT_LABEL)
        button = gr.Button("生成参考克隆 / 表演", variant="primary", interactive=False, elem_id="studio_generate")
        audio = gr.Audio(type="filepath", label="工作室合成音频")
        status = gr.Markdown()
        consent.change(lambda ok: gr.update(interactive=ok is True), consent, button, queue=False)
        button.click(do_studio_synth,
                     inputs=[task, backend, reference, transcript, instruction, text,
                             seed, save_name, description, consent],
                     outputs=[audio, status], api_name="studio_generate", **synthesis_options)
    with gr.Tab("角色复用"):
        gr.Markdown("从「角色库」选择已保存角色，输入新台词即可继续创作。角色库中的删除会立即影响这里的选择。")
        characters = gr.Dropdown(choices=[], value=None, label="角色库中的角色卡")
        refresh = gr.Button("刷新角色列表", elem_id="character_refresh")
        reference = gr.Audio(type="filepath", label="角色原始参考", interactive=False)
        metadata = gr.Markdown()
        text = gr.Textbox(label="角色的新台词", value=DEFAULT_TEXT, lines=3)
        instruction = gr.Textbox(label="覆盖表演指令（留空保留已保存指令）", lines=2)
        seed = gr.Number(value=None, precision=0, label="覆盖种子（留空保留角色种子）")
        device = gr.Dropdown(choices=[('GPU / CUDA', 'cuda:0'), ('CPU（较慢）', 'cpu')],
                             value='cuda:0', label='仅 Qwen 角色设备（GPU BF16 / CPU FP32；其他后端忽略；不自动回退）',
                             elem_id='character_device')
        consent = gr.Checkbox(value=False, label=CONSENT_LABEL)
        button = gr.Button("用角色生成", variant="primary", elem_id="character_generate")
        audio = gr.Audio(type="filepath", label="角色合成音频")
        status = gr.Markdown()

        def _refresh():
            try:
                choices = studio_character_choices()
                return gr.Dropdown(choices=choices, value=None), None, "请选择角色卡。"
            except Exception as exc:
                _, message = _studio_failure(exc)
                return gr.Dropdown(choices=[], value=None), None, message

        refresh.click(_refresh, outputs=[characters, reference, metadata], api_name="character_refresh")
        characters.change(do_character_preview, inputs=[characters], outputs=[reference, metadata])
        button.click(do_character_synth, inputs=[characters, text, instruction, seed, consent, device],
                     outputs=[audio, status], api_name="character_generate", **synthesis_options)
    with gr.Tab("能力状态"):
        gr.Markdown(studio_availability_markdown())


# ---------------------------------------------------------------- 界面
def build_ui(*, include_legacy=False):
    import gradio as gr
    import inspect

    # Gradio 6.30 supports these; introspect for older installed versions too.
    queue_parameters = inspect.signature(gr.Blocks.queue).parameters
    click_parameters = inspect.signature(gr.Button.click).parameters
    synthesis_options = {}
    if "concurrency_limit" in click_parameters:
        synthesis_options["concurrency_limit"] = 1
    if "concurrency_id" in click_parameters:
        synthesis_options["concurrency_id"] = "timbreforge-synthesis"

    cfg = get_config()
    asset_path = (Path(__file__).resolve().parent.parent / 'assets' / 'studio-background.png').resolve()
    asset_url = '/file=' + asset_path.as_posix()
    css = """
    :root { --tf-ink: #e9efff; --tf-muted: #aab9dc; --tf-panel: rgba(9, 18, 48, .78); }
    body { background: #08132f url('__TF_ASSET_URL__') center / cover fixed no-repeat !important; }
    .gradio-container { max-width: 1440px !important; background: rgba(5, 12, 32, .48) !important; }
    .tf-hero { padding: 30px 34px 24px; margin: 14px 0 18px; border: 1px solid rgba(170, 198, 255, .22); border-radius: 24px; background: linear-gradient(120deg, rgba(18, 35, 83, .86), rgba(15, 25, 64, .58)); box-shadow: 0 18px 60px rgba(0, 0, 0, .24); }
    .tf-hero h1 { color: #f4f7ff; letter-spacing: -.03em; margin-bottom: 6px; }
    .tf-hero p { color: var(--tf-muted); margin: 0; font-size: 1.05rem; }
    .tf-card-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin: 8px 0 18px; }
    .tf-card { display: flex; gap: 12px; align-items: center; padding: 14px; min-height: 78px; border: 1px solid rgba(164, 193, 255, .2); border-radius: 16px; background: rgba(20, 38, 85, .7); }
    .tf-card-icon { display: grid; place-items: center; width: 42px; height: 42px; border-radius: 13px; color: #dce8ff; background: linear-gradient(135deg, #526df1, #9a72ed); font-size: 1.15rem; font-weight: 700; }
    .tf-card h3 { color: var(--tf-ink); margin: 0 0 3px; font-size: 1rem; }
    .tf-card p, .tf-card span, .tf-empty { color: var(--tf-muted); margin: 0; font-size: .88rem; }
    .tf-empty { padding: 26px; border: 1px dashed rgba(164, 193, 255, .28); border-radius: 16px; text-align: center; }
    footer { display: none !important; }
    """.replace('__TF_ASSET_URL__', asset_url)
    with gr.Blocks(title="TimbreForge · 角色声线工作室", analytics_enabled=False) as demo:
        gr.Markdown(
            "<div class='tf-hero'><h1>🎙️ TimbreForge</h1>"
            "<p>把角色设定变成可以开口说话的角色。先保存角色卡，再从角色库进入生成、复用和多角色创作。</p></div>")

        with gr.Tabs():
            from timbreforge.role_ui import build_role_tab
            build_role_tab(gr, synthesis_options)
            from timbreforge.simple_ui import build_simple_tab
            build_simple_tab(gr, synthesis_options)
            from timbreforge.design_ui import build_design_tab
            build_design_tab(gr, synthesis_options)
            from timbreforge.library_ui import build_library_tab
            build_library_tab(gr)
            from timbreforge.character_ui import build_character_tabs
            build_character_tabs(gr)
            from timbreforge.voice_ui import build_voice_tabs
            build_voice_tabs(gr)
            _build_studio_tabs(gr, synthesis_options)
            from timbreforge.creation_ui import build_creation_tabs
            build_creation_tabs(gr, synthesis_options)
            from timbreforge.admin_ui import build_admin_tabs
            build_admin_tabs(gr)
            from timbreforge.ui_tasks import register_default_cleanups
            # Low-priority caches may be released under memory pressure; this
            # only registers callbacks and never touches a model or the disk now.
            register_default_cleanups()
            if include_legacy:
                with gr.Tab("数据集检索（原有功能）"):
                    with gr.Row():
                        with gr.Column(scale=5):
                            prompt = gr.Textbox(label="① 声线描述（提示词）", value=PRESETS[0],
                                                lines=2,
                                                placeholder="例如：冰冷沙哑的成熟女声 / 睿智沉稳的老者")
                            gr.Examples(examples=[[p] for p in PRESETS], inputs=[prompt],
                                        label="示例声线")
                            with gr.Row():
                                seed = gr.Number(value=42, label="② 种子", precision=0)
                                dice = gr.Button("🎲", size="sm")
                                num_cand = gr.Slider(3, 20, value=6, step=1, label="候选数")
                            with gr.Row():
                                age_ov = gr.Slider(-1, 1, value=-1, step=0.05,
                                                   label="年龄感覆盖（-1=不覆盖）")
                                gender_ov = gr.Dropdown(["auto", "female", "male", "neutral"],
                                                        value="auto", label="性别覆盖")
                            text = gr.Textbox(label="③ 台词", value=DEFAULT_TEXT, lines=3)

                            with gr.Accordion("④ 情绪（IndexTTS 情感向量）", open=False):
                                emo_sliders = {}
                                for k in EMOTION_KEYS:
                                    emo_sliders[k] = gr.Slider(0.0, 1.0, value=0.0, step=0.05,
                                                               label=EMO_CN[k])
                            with gr.Accordion("⑤ 采样参数（进阶）", open=False):
                                with gr.Row():
                                    temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.1,
                                                            label="temperature")
                                    top_p = gr.Slider(0.0, 1.0, value=0.8, step=0.01, label="top_p")
                                    top_k = gr.Slider(0, 100, value=30, step=1, label="top_k")
                                with gr.Row():
                                    num_beams = gr.Slider(1, 10, value=1, step=1, label="num_beams")
                                    rep = gr.Slider(0.1, 20.0, value=3.0, step=0.1,
                                                    label="repetition_penalty")
                                    dfac = gr.Slider(0.5, 2.0, value=1.0, step=0.01, label="时长系数")
                            base_url = gr.Textbox(value=cfg.comfy_url, visible=False)

                            with gr.Row():
                                btn_search = gr.Button("🔍 检索声线", variant="secondary")
                                btn_synth = gr.Button("🎧 合成语音", variant="primary")

                        with gr.Column(scale=6):
                            status = gr.Markdown(value="可选数据集功能；发布包不附带数据集或索引。填写描述后显式点击检索。")
                            table = gr.Dataframe(
                                headers=["#", "cid", "名称", "性别", "年龄", "时长", "匹配度",
                                         "粗糙/气声/厚度/空灵", "标签", "一句话描述"],
                                datatype=["str"] * 10, interactive=False, wrap=True,
                                label="候选声线（★ 为当前选中）")
                            with gr.Row():
                                cid_in = gr.Textbox(label="指定声线 cid（留空则用检索选中的）",
                                                    placeholder="例如 k378_age4_bg1")
                                btn_preview = gr.Button("▶ 试听这条", size="sm")
                            preview_audio = gr.Audio(label="候选声线原始参考音频", type="filepath")
                            preview_info = gr.Markdown()
                            out_audio = gr.Audio(label="合成语音", type="filepath")

                    def _roll():
                        return random.randint(0, 999999)

                    dice.click(_roll, outputs=[seed])
                    btn_search.click(do_search, inputs=[prompt, seed, num_cand, age_ov, gender_ov],
                                     outputs=[table, status])
                    btn_preview.click(do_preview, inputs=[cid_in],
                                      outputs=[preview_audio, preview_info])

                    def _row_cid(evt: gr.SelectData, data):
                        try:
                            return data.iloc[evt.index[0], 1]
                        except Exception:
                            try:
                                return data[evt.index[0]][1]
                            except Exception:
                                return ""

                    # Resolve local gr explicitly: postponed annotations cannot resolve
                    # the build_ui-local module when Gradio injects SelectData.
                    _row_cid.__annotations__["evt"] = gr.SelectData
                    table.select(_row_cid, inputs=[table], outputs=[cid_in])

                    def _wrap(*a):
                        (p, cid, s, tx, te, tp, tk, nb, rp, df, bu, nc, ag, ge) = a[:14]
                        emos = a[14:14 + len(EMOTION_KEYS)]
                        emo = dict(zip(EMOTION_KEYS, emos))
                        return do_synth(p, cid, s, tx, te, tp, tk, nb, rp, df, emo, bu, nc, ag, ge)

                    btn_synth.click(
                        _wrap,
                        inputs=[prompt, cid_in, seed, text, temperature, top_p, top_k,
                                num_beams, rep, dfac, base_url, num_cand, age_ov, gender_ov]
                        + [emo_sliders[k] for k in EMOTION_KEYS],
                        outputs=[out_audio, status], **synthesis_options)

                    gr.Markdown(
                        "---\n"
                        "**说明**：声线来自公开数据集（全合成音、无真人），描述与种子共同决定选中哪一条，"
                        "因此候选选择可复现；最终波形受后端与运行环境影响，不保证逐样本一致。合成依赖本机 ComfyUI 的 IndexTTS 节点。")

    if "default_concurrency_limit" in queue_parameters:
        demo.queue(default_concurrency_limit=1)
    elif "concurrency_count" in queue_parameters:
        demo.queue(concurrency_count=1)
    # Gradio 6 moved CSS from Blocks construction to launch(). Keep it on the
    # demo object so tests and alternate launchers can still build the UI.
    demo._timbreforge_css = css
    return demo


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="TimbreForge 声线工作室")
    ap.add_argument("--config", help="TOML configuration path")
    ap.add_argument("--legacy-retrieval", action="store_true", help="显示可选数据集检索页（需要另备资产）")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--open-browser", action="store_true", help="启动后打开本机浏览器")
    args = ap.parse_args(argv)
    if args.config:
        set_config_path(args.config)
    cfg = get_config()
    demo = build_ui(include_legacy=args.legacy_retrieval)
    # The launcher opens a local browser by default. In that mode, keep the
    # process tied to the page: a heartbeat stops the old server after the tab
    # is closed, so a later launch does not collide with a stale port.
    auto_exit = bool(args.open_browser and not args.share)
    closed = threading.Event()
    last_heartbeat = [time.monotonic()]
    page_js = None
    if auto_exit:
        page_js = """
        (() => {
          const ping = () => fetch('/__timbreforge_heartbeat', {method: 'POST', keepalive: true}).catch(() => {});
          ping();
          window.__timbreforgeHeartbeat = window.setInterval(ping, 4000);
          window.addEventListener('beforeunload', () => {
            if (window.__timbreforgeHeartbeat) window.clearInterval(window.__timbreforgeHeartbeat);
            navigator.sendBeacon('/__timbreforge_close', 'closed');
          });
        })();
        """
    try:
        demo.launch(server_name=args.host, server_port=args.port,
                    share=args.share, inbrowser=args.open_browser,
                    prevent_thread_lock=auto_exit,
                    css=getattr(demo, '_timbreforge_css', None), js=page_js,
                    allowed_paths=[str(cfg.outputs_dir.resolve()), str(cfg.characters_dir.resolve()),
                                   str((Path(__file__).resolve().parent.parent / 'assets').resolve())])
        if auto_exit:
            def heartbeat():
                last_heartbeat[0] = time.monotonic()
                return {'ok': True}

            def close_page():
                closed.set()
                return {'ok': True}

            # Gradio creates its FastAPI app during launch; routes are attached
            # immediately after the server starts and the page retries heartbeats.
            demo.app.add_api_route('/__timbreforge_heartbeat', heartbeat, methods=['GET', 'POST'])
            demo.app.add_api_route('/__timbreforge_close', close_page, methods=['POST'])
            while not closed.wait(1.0):
                if time.monotonic() - last_heartbeat[0] > 20:
                    break
    finally:
        # Closing the browser or stopping the process must not leave design
        # sessions, playback copies, jobs, Qwen staging files or generated
        # takes behind.  CharacterStore is a separate durable root.
        try:
            demo.close()
        finally:
            from timbreforge.ui_tasks import shutdown_runtime
            shutdown_runtime(clear_records=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

