"""The single character-library surface shared by all voice workflows."""
from __future__ import annotations

import html
from datetime import datetime

from timbreforge.characters import CharacterStore


def _store():
    return CharacterStore()


def _choices(cards):
    return [(f"{card['name']} · {card['backend']}", card['id']) for card in cards]


def _card_html(cards):
    if not cards:
        return '<div class="tf-empty">还没有角色卡。去「角色设定」或「文字声线设计」生成并保存第一张角色卡。</div>'
    items = []
    for card in cards:
        created = card.get('created_at', '')
        try:
            created = datetime.fromisoformat(created).astimezone().strftime('%Y-%m-%d')
        except (TypeError, ValueError):
            created = ''
        items.append(
            '<article class="tf-card">'
            f'<div class="tf-card-icon">{html.escape(card["name"][:1])}</div>'
            f'<div><h3>{html.escape(card["name"])}</h3>'
            f'<p>{html.escape(card.get("description") or "暂无角色描述")}</p>'
            f'<span>{html.escape(card.get("backend", "qwen3"))} · {html.escape(created)}</span></div>'
            '</article>'
        )
    return '<div class="tf-card-grid">' + ''.join(items) + '</div>'


def _details(card):
    if not card:
        return '选择一张角色卡查看详情。'
    return (f"### {card['name']}\n\n"
            f"{card.get('description') or '暂无角色描述'}\n\n"
            f"- 后端：`{card.get('backend', 'qwen3')}`\n"
            f"- 参考台词：{card.get('reference_text') or '未填写'}\n"
            f"- 角色 ID：`{card['id']}`\n\n"
            "这张卡会被角色复用、多角色项目和声线合成页面共同使用。")


def snapshot(current=None):
    cards = _store().list()
    choices = _choices(cards)
    value = current if current in [item[1] for item in choices] else (choices[0][1] if choices else None)
    selected = next((card for card in cards if card['id'] == value), None)
    return cards, choices, value, selected


def build_library_tab(gr):
    """Build the management surface; no model is loaded while constructing UI."""
    initial_cards, initial_choices, initial_value, initial_card = snapshot()
    with gr.Tab('角色库', id='character_library'):
        gr.Markdown('## 角色库\n统一管理已保存的角色卡。生成、复用和多角色项目都从这里读取角色。')
        cards_view = gr.HTML(value=_card_html(initial_cards), elem_id='character_library_cards')
        with gr.Row():
            picker = gr.Dropdown(label='选择角色卡', choices=initial_choices, value=initial_value,
                                 scale=5, elem_id='library_picker')
            refresh = gr.Button('刷新', elem_id='library_refresh')
            delete = gr.Button('删除角色卡', variant='stop', elem_id='library_delete')
        confirm = gr.Checkbox(label='我确认删除这张角色卡及其参考音频', visible=False, elem_id='library_delete_confirm')
        with gr.Row():
            audio = gr.Audio(value=initial_card.get('reference_audio') if initial_card else None,
                             label='参考声音', type='filepath', interactive=False, elem_id='library_audio')
            info = gr.Markdown(_details(initial_card), elem_id='library_info')
        status = gr.Markdown('角色卡只保存在本机角色目录中。', elem_id='library_status')

    def view(current=None):
        try:
            cards, choices, value, card = snapshot(current)
            return (_card_html(cards), gr.update(choices=choices, value=value),
                    card.get('reference_audio') if card else None, _details(card),
                    '已刷新角色库，共 %d 张角色卡。' % len(cards), gr.update(visible=False, value=False))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return ('<div class="tf-empty">角色库读取失败。</div>', gr.update(choices=[], value=None),
                    None, '', f'角色库无法读取：{exc}', gr.update(visible=False, value=False))

    refresh.click(view, picker, [cards_view, picker, audio, info, status, confirm], api_name='library_refresh')
    picker.change(lambda current: view(current), picker,
                  [cards_view, picker, audio, info, status, confirm], api_name='library_select')

    def ask_delete(current):
        if not current:
            return gr.update(visible=False), '请先选择一张角色卡。'
        return gr.update(visible=True, value=False), '删除后将移除角色卡和参考音频，无法撤销。'

    delete.click(ask_delete, picker, [confirm, status], api_name='library_delete_ask')

    def do_delete(current, ok):
        if not current:
            return view(None)
        if ok is not True:
            return (*view(current)[:5], gr.update(visible=True, value=False))
        try:
            card = _store().delete(current)
            result = view(None)
            return result[:4] + (f'已删除角色卡「{card["name"]}」。', result[5])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return (*view(current)[:5], f'删除失败：{exc}', gr.update(visible=True, value=False))

    confirm.change(do_delete, [picker, confirm], [cards_view, picker, audio, info, status, confirm], api_name='library_delete')

