"""The single character-library surface shared by all voice workflows."""
from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

from timbreforge import character_audio, character_package, clone_prompts
from timbreforge.characters import CharacterStore
from timbreforge.voice_recipe import describe as describe_recipe

AUDIO_SUFFIX = ('.wav', '.flac', '.ogg', '.mp3', '.m4a')


def _store():
    return CharacterStore()


def _choices(cards):
    return [(f"{card['name']} · {_short_model(card)} · v{card.get('version', 1)}", card['id'])
            for card in cards]


def _short_model(card):
    model = card.get('model_name') or card.get('backend') or '未知模型'
    return model if len(model) <= 28 else model[:27] + '…'


def _clock(value):
    try:
        return (datetime.fromisoformat(value).astimezone().strftime('%Y-%m-%d %H:%M')
                if value else '未知')
    except (TypeError, ValueError):
        return '未知'


def _clone_status(card):
    try:
        return clone_prompts.validate_clone_prompt(card['id'], store=_store())
    except (OSError, ValueError):
        return {'status': 'missing', 'reasons': ['无法读取 Clone Prompt 状态'], 'metadata': None}


def _card_html(cards):
    if not cards:
        return ('<div class="tf-empty">还没有角色卡。去「创建角色」导入录音并填写逐字稿，'
                '或使用「一句话生成 / 文字声线设计」保存第一张角色卡。</div>')
    items = []
    for card in cards:
        recipe = card.get('voice_recipe') or {}
        try:
            summary = describe_recipe(recipe)
        except (ValueError, KeyError, TypeError):
            summary = card.get('description') or '暂无声线摘要'
        status = _clone_status(card)
        prompt_state = {'valid': '缓存就绪', 'stale': '缓存已失效', 'corrupt': '缓存损坏',
                        'unsupported': '不支持缓存'}.get(status['status'], '未缓存')
        items.append(
            '<article class="tf-card">'
            f'<div class="tf-card-icon">{html.escape(card["name"][:1])}</div>'
            '<div class="tf-card-body">'
            f'<h3>{html.escape(card["name"])}</h3>'
            f'<p>{html.escape(summary)}</p>'
            f'<span>{html.escape(_short_model(card))} · v{card.get("version", 1)} · '
            f'{html.escape(prompt_state)} · {html.escape(_clock(card.get("updated_at")))}</span>'
            '</div></article>'
        )
    return '<div class="tf-card-grid">' + ''.join(items) + '</div>'


def _details(card):
    if not card:
        return '选择一张角色卡查看详情。'
    status = _clone_status(card)
    history = card.get('audio_edit_chain') or []
    recipe = card.get('voice_recipe') or {}
    try:
        recipe_text = describe_recipe(recipe)
    except (ValueError, KeyError, TypeError):
        recipe_text = '未设置'
    tags = '、'.join(card.get('tags') or []) or '无'
    return (f"### {card['name']}\n\n"
            f"{card.get('description') or '暂无角色描述'}\n\n"
            f"- 声线配方：{recipe_text}\n"
            f"- 模型：`{card.get('model_name') or '未登记'}`"
            f"（版本 `{card.get('model_revision') or '本地'}`）\n"
            f"- 角色版本：v{card.get('version', 1)} · 最后修改 {_clock(card.get('updated_at'))}\n"
            f"- Clone Prompt：{clone_prompts.status_text(status)}\n"
            f"- 参考台词：{card.get('reference_text') or '未填写'}\n"
            f"- 处理历史：{len(history)} 条 · 标签：{tags}\n"
            f"- 来源：`{card.get('source') or '手动创建'}` · 角色 ID：`{card['id']}`\n\n"
            "角色卡是生成、微调、长文本和多角色项目的唯一角色来源。")


def snapshot(current=None):
    cards = _store().list()
    choices = _choices(cards)
    value = current if current in [item[1] for item in choices] else (choices[0][1] if choices else None)
    selected = next((card for card in cards if card['id'] == value), None)
    return cards, choices, value, selected


def audition(character_id, *, store=None):
    return character_audio.audition(character_id, store=store or _store())


def export_role(character_id, destination=None, *, store=None):
    """Export to the default exports directory unless a path is supplied."""
    store = store or _store()
    card = store.load(character_id)
    if destination is None:
        directory = Path(store.root).parent / 'exports'
        destination = directory / f"{_safe_name(card['name'])}-{card['id'][:8]}.zip"
    return character_package.export_card(character_id, destination, store=store)


def _safe_name(name):
    cleaned = ''.join(character for character in name
                      if character.isalnum() or character in '-_（）() ')
    return cleaned.strip()[:40] or 'role'


def duplicate_role(character_id, name=None, *, store=None):
    return character_package.duplicate_card(character_id, store=store or _store(), name=name)


def import_role(path, name=None, *, store=None):
    return character_package.import_card(path, store=store or _store(), name=name)


def build_library_tab(gr):
    """Build the management surface; no model is loaded while constructing UI."""
    initial_cards, initial_choices, initial_value, initial_card = snapshot()
    with gr.Tab('角色库', id='character_library'):
        gr.Markdown('## 角色库\n统一管理已保存的角色卡。生成、微调、长文本和多角色项目都从这里读取角色。')
        cards_view = gr.HTML(value=_card_html(initial_cards), elem_id='character_library_cards')
        with gr.Row():
            picker = gr.Dropdown(label='选择角色卡', choices=initial_choices, value=initial_value,
                                 scale=5, elem_id='library_picker')
            refresh = gr.Button('刷新', elem_id='library_refresh')
            play = gr.Button('试听', elem_id='library_audition')
            export = gr.Button('导出角色包', elem_id='library_export')
            duplicate = gr.Button('复制角色卡', elem_id='library_duplicate')
            delete = gr.Button('删除角色卡', variant='stop', elem_id='library_delete')
        with gr.Row():
            new_name = gr.Textbox(label='复制后的新名称（留空自动加“（副本）”）',
                                  elem_id='library_duplicate_name', scale=4)
            package = gr.File(label='导入角色包（.zip）', type='filepath',
                              file_types=['.zip'], elem_id='library_import_file', scale=3)
            import_button = gr.Button('导入角色包', elem_id='library_import')
        confirm = gr.Checkbox(label='我确认删除这张角色卡及其参考音频、派生音频和 Clone Prompt',
                              visible=False, elem_id='library_delete_confirm')
        with gr.Row():
            audio = gr.Audio(value=initial_card.get('reference_audio') if initial_card else None,
                             label='角色当前参考声音', type='filepath', interactive=False,
                             elem_id='library_audio')
            info = gr.Markdown(_details(initial_card), elem_id='library_info')
        status = gr.Markdown('角色卡只保存在本机角色目录中；删除会同时清理派生文件。',
                             elem_id='library_status')

    def view(current=None):
        try:
            cards, choices, value, card = snapshot(current)
            return (_card_html(cards), gr.update(choices=choices, value=value),
                    card.get('reference_audio') if card else None, _details(card),
                    f'已刷新角色库，共 {len(cards)} 张角色卡。', gr.update(visible=False, value=False))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return ('<div class="tf-empty">角色库读取失败。</div>', gr.update(choices=[], value=None),
                    None, '', f'角色库无法读取：{exc}', gr.update(visible=False, value=False))

    refresh.click(view, picker, [cards_view, picker, audio, info, status, confirm],
                  api_name='library_refresh')
    picker.change(view, picker, [cards_view, picker, audio, info, status, confirm],
                  api_name='library_select')

    def do_audition(current):
        if not current:
            return None, '请先选择一张角色卡。'
        try:
            result = audition(current)
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            return None, f'试听失败：{exc}'
        preview = result.get('preview') or {}
        source = preview.get('path') or result.get('audio')
        note = ('当前播放派生音频（压缩预览）。' if preview else '当前播放原始音频。')
        return source, note

    play.click(do_audition, picker, [audio, status], api_name='library_audition')

    def do_export(current):
        if not current:
            return '请先选择一张角色卡。'
        try:
            result = export_role(current)
        except (OSError, ValueError, RuntimeError) as exc:
            return f'导出失败：{exc}'
        return f"已导出角色包：{result['path']}（{result['bytes'] / 1024:.0f} KiB，不含 Clone Prompt）。"

    export.click(do_export, picker, status, api_name='library_export')

    def do_duplicate(current, name):
        if not current:
            return view(None)[:4] + ('请先选择一张角色卡。', gr.update(visible=False, value=False))
        try:
            card = duplicate_role(current, name or None)
        except (OSError, ValueError, RuntimeError) as exc:
            return (*view(current)[:4], f'复制失败：{exc}', gr.update(visible=False, value=False))
        result = view(card['id'])
        return result[:4] + (f"已复制为「{card['name']}」，这是一张独立角色卡。", result[5])

    duplicate.click(do_duplicate, [picker, new_name],
                    [cards_view, picker, audio, info, status, confirm], api_name='library_duplicate')

    def do_import(path, name):
        if not path:
            return view(None)[:4] + ('请选择要导入的角色包。', gr.update(visible=False, value=False))
        try:
            card = import_role(path, name or None)
        except (OSError, ValueError, RuntimeError) as exc:
            return (*view(None)[:4], f'导入失败：{exc}', gr.update(visible=False, value=False))
        result = view(card['id'])
        return result[:4] + (f"已导入「{card['name']}」；请重新创建 Clone Prompt。", result[5])

    import_button.click(do_import, [package, new_name],
                        [cards_view, picker, audio, info, status, confirm],
                        api_name='library_import')

    def ask_delete(current):
        if not current:
            return gr.update(visible=False), '请先选择一张角色卡。'
        return gr.update(visible=True, value=False), '删除后将移除角色卡、原始与派生音频、Clone Prompt，无法撤销。'

    delete.click(ask_delete, picker, [confirm, status], api_name='library_delete_ask')

    def do_delete(current, ok):
        if not current:
            return view(None)
        if ok is not True:
            return (*view(current)[:5], gr.update(visible=True, value=False))
        try:
            card = _store().delete(current)
            result = view(None)
            return result[:4] + (f'已删除角色卡「{card["name"]}」及其派生文件。', result[5])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return (*view(current)[:5], gr.update(visible=False, value=False))

    confirm.change(do_delete, [picker, confirm],
                   [cards_view, picker, audio, info, status, confirm], api_name='library_delete')

