"""Description-only voice design using the installed model's generation recipe."""
from __future__ import annotations

import json
import os
import secrets
import subprocess

from timbreforge.backend_qwen import QwenParams
from timbreforge.config import get_config
from timbreforge.designs import DesignStore

PREVIEW_TEXT = '你好，很高兴与你见面。这是我们故事的开始。'


def automatic_recipe():
    """Read actual model defaults, with a short-preview token cap and fresh RNG."""
    cfg = get_config()
    path = cfg.qwen_design_model / 'generation_config.json'
    values = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(values, dict):
        raise ValueError('模型 generation_config.json 格式无效')
    defaults = QwenParams().to_dict()
    generation_keys = set(defaults) - {'seed', 'language'}
    for key in generation_keys:
        if key in values:
            defaults[key] = values[key]
    params = QwenParams(**defaults).validate()
    params.max_new_tokens = min(params.max_new_tokens, 1024)
    params.seed = secrets.randbelow(2**31)
    return params.validate()


def automatic_device():
    """Probe the backend environment; do not import torch into the UI process."""
    python = get_config().qwen_python
    if not python.is_file():
        raise FileNotFoundError(f'未找到 Qwen Python：{python}')
    try:
        result = subprocess.run(
            [str(python), '-B', '-c',
             "import torch; print('cuda:0' if torch.cuda.is_available() else 'cpu')"],
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=60,
            env=dict(os.environ, PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1'),
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError('检测生成设备超时，请重试；如持续出现，请检查 Qwen 环境。') from exc
    device = result.stdout.strip()
    if result.returncode != 0 or device not in ('cuda:0', 'cpu'):
        raise RuntimeError('Qwen 环境无法检测生成设备：' + (result.stderr.strip()[-1200:] or device))
    return device


def generate_simple(description, progress=None, *, preview_text=PREVIEW_TEXT):
    if not isinstance(description, str) or not 1 <= len(description.strip()) <= 2000:
        raise ValueError('请填写 1–2000 字的角色声线描述，例如：低沉温暖、语气沉稳的成年男声。')
    if not isinstance(preview_text, str) or not 1 <= len(preview_text.strip()) <= 500:
        raise ValueError('试听台词需为 1–500 字的完整句子。')
    params = automatic_recipe()
    if progress:
        progress(.05, desc='正在自动配置生成设备…')
    device = automatic_device()
    if progress:
        progress(.15, desc='正在生成角色声音…' if device != 'cpu' else '正在使用 CPU 生成，耗时可能较长…')
    store = DesignStore()
    # This action explicitly creates an original synthetic character, no uploaded voice.
    session = store.create(description.strip(), preview_text.strip(), params, count=1,
                           device=device, confirmed=True)
    candidate = session['candidates'][0]
    audio = store.audio_path(session['id'], candidate['id'])
    details = dict(session_id=session['id'], candidate_id=candidate['id'],
                   description=session['prompt'], preview_text=session['text'], device=device,
                   parameters=params.to_dict(),
                   configuration='模型自带 generation_config；短试听上限 1024 tokens；自动随机种子。')
    return audio, details


def save_simple_character(session_id, candidate_id, name, description='', confirmed=False):
    """Promote a one-click design result into the shared character library."""
    if confirmed is not True:
        raise ValueError('请确认已试听，并确认这段声音与试听台词一致后再保存。')
    if not session_id or not candidate_id:
        raise ValueError('请先生成一段角色声音。')
    if not isinstance(name, str) or not name.strip():
        raise ValueError('请填写角色名称。')
    if not isinstance(description, str) or len(description) > 2000:
        raise ValueError('角色描述必须是不超过 2000 字的文本。')
    store = DesignStore()
    # The selected candidate is the exact audio shown in the simple UI.
    store.select(session_id, candidate_id)
    return store.save_character(session_id, candidate_id, name.strip(),
                                description=description.strip() or None, confirmed=True)


def error_message(exc):
    if isinstance(exc, FileNotFoundError):
        return f'本地模型或环境文件缺失：{exc}。请检查 timbreforge.local.toml 中的 Qwen 路径。'
    if isinstance(exc, ValueError):
        return str(exc)
    return f'生成未完成：{exc}'


def build_simple_tab(gr, synthesis_options=None):
    with gr.Tab('简化版 · 一句话生成'):
        gr.Markdown('## 描述一个声音，让角色开口\n只需输入原创角色的声线描述，其余自动配置。')
        description = gr.Textbox(label='你想要什么样的角色声音？', lines=4,
            placeholder='例如：一位年轻的星际领航员，女声清冷而沉稳，略带沙哑，说话自然、让人安心。',
            elem_id='simple_description')
        generate = gr.Button('生成角色声音', variant='primary', elem_id='simple_generate')
        status = gr.Markdown('生成后可以直接试听和下载；再次点击可尝试另一版声音。', elem_id='simple_status')
        audio = gr.Audio(type='filepath', interactive=False, label='角色声线试听', elem_id='simple_audio')
        with gr.Group(visible=False) as save_panel:
            gr.Markdown('### 保存到角色库\n保存后可在「角色库」「角色复用」和多角色项目中使用。')
            with gr.Row():
                save_name = gr.Textbox(label='角色名称', elem_id='simple_save_name')
                save_description = gr.Textbox(label='角色描述（可选）', elem_id='simple_save_description')
            save_confirm = gr.Checkbox(label='我已试听，并确认声音与试听台词一致，可以保存到角色库。',
                                       elem_id='simple_save_confirm')
            save = gr.Button('保存到角色库', variant='secondary', elem_id='simple_save')
        with gr.Accordion('自动配置与试听台词', open=False):
            gr.Markdown('采样参数采用模型自带配置，设备和随机种子自动选择。试听台词：' + PREVIEW_TEXT)
            details = gr.JSON(label='本次生成记录', elem_id='simple_details')
        gr.Markdown('生成满意后，可以直接保存到角色库；其他页面会从角色库读取这张角色卡。')

    def run(description, progress=gr.Progress()):
        yield None, None, gr.update(visible=False), gr.update(value=''), gr.update(value=''), gr.update(value=False), '正在自动配置并生成，请稍候…'
        try:
            audio_path, metadata = generate_simple(description, progress)
            yield audio_path, metadata, gr.update(visible=True), gr.update(value=''), gr.update(value=metadata.get('description', '')), gr.update(value=False), '生成完成。请先试听，再保存到角色库。'
        except (OSError, ValueError, RuntimeError, TypeError, KeyError, OverflowError) as exc:
            yield None, None, gr.update(visible=False), gr.update(value=''), gr.update(value=''), gr.update(value=False), error_message(exc)

    generate.click(run, description, [audio, details, save_panel, save_name, save_description, save_confirm, status], api_name='simple_generate',
                   **dict(synthesis_options or {}))

    def save_run(metadata, name, note, confirmed):
        try:
            card = save_simple_character(metadata.get('session_id') if metadata else None,
                                         metadata.get('candidate_id') if metadata else None,
                                         name, note, confirmed)
            return '已保存到角色库：**%s**。打开「角色库」即可试听和管理。' % card['name'], gr.update(visible=False), gr.update(value=False)
        except (OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
            return error_message(exc), gr.update(visible=True), gr.update(value=False)

    save.click(save_run, [details, save_name, save_description, save_confirm],
               [status, save_panel, save_confirm], api_name='simple_save_character',
               **dict(synthesis_options or {}))
