"""Four auditable starting points for reference-audio conditioning.

Presets are *starting points*, not automatic quality guarantees. Each preset is
an ordered chain validated by :mod:`timbreforge.audio_edit_schema`, so a preset
can be inspected, copied into a custom chain and re-rendered at any time.
"""
from __future__ import annotations

from timbreforge.audio_edit_schema import validate_operations

PRESETS = {
    'clean_voice': {
        'label': '清晰人声',
        'description': '保守去静音、去点击和降噪，限制响度补偿，尽量保留气声与尾音。',
        'operations': [
            {'name': 'trim_silence', 'threshold_db': -42, 'keep_leading_ms': 50, 'keep_trailing_ms': 80},
            {'name': 'de_click', 'threshold': 5.5, 'window': 5},
            {'name': 'denoise', 'strength': 0.45, 'window_ms': 32, 'noise_seconds': 0.35},
            {'name': 'eq', 'bands': [
                {'type': 'highpass', 'freq': 70, 'gain_db': 0, 'q': 0.707},
                {'type': 'peaking', 'freq': 3200, 'gain_db': 0.8, 'q': 1.0},
            ]},
            {'name': 'normalize', 'target_lufs': -16, 'ceiling_dbfs': -1, 'max_gain_db': 6},
        ],
    },
    'warm_voice': {
        'label': '温暖声线',
        'description': '在保守降噪后略微加厚低频并收一点高频，避免把底噪推回前景。',
        'operations': [
            {'name': 'trim_silence', 'threshold_db': -42, 'keep_leading_ms': 50, 'keep_trailing_ms': 80},
            {'name': 'de_click', 'threshold': 5.5, 'window': 5},
            {'name': 'denoise', 'strength': 0.4, 'window_ms': 32, 'noise_seconds': 0.35},
            {'name': 'eq', 'bands': [
                {'type': 'highpass', 'freq': 65, 'gain_db': 0, 'q': 0.707},
                {'type': 'lowshelf', 'freq': 180, 'gain_db': 1.5, 'q': 0.707},
                {'type': 'highshelf', 'freq': 6500, 'gain_db': -1.2, 'q': 0.707},
            ]},
            {'name': 'normalize', 'target_lufs': -16, 'ceiling_dbfs': -1, 'max_gain_db': 6},
        ],
    },
    'bright_voice': {
        'label': '明亮声线',
        'description': '提亮高频与清晰度，适合闷、远或录得偏暗的参考音频；齿音重时改用温暖预设。',
        'operations': [
            {'name': 'trim_silence', 'threshold_db': -42, 'keep_leading_ms': 50, 'keep_trailing_ms': 80},
            {'name': 'de_click', 'threshold': 5.5, 'window': 5},
            {'name': 'denoise', 'strength': 0.4, 'window_ms': 32, 'noise_seconds': 0.35},
            {'name': 'eq', 'bands': [
                {'type': 'highpass', 'freq': 90, 'gain_db': 0, 'q': 0.707},
                {'type': 'peaking', 'freq': 4000, 'gain_db': 0.8, 'q': 1.0},
                {'type': 'highshelf', 'freq': 7000, 'gain_db': 2.0, 'q': 0.707},
            ]},
            {'name': 'normalize', 'target_lufs': -16, 'ceiling_dbfs': -1, 'max_gain_db': 6},
        ],
    },
    'low_noise': {
        'label': '强降噪',
        'description': '较强但仍有限幅的降噪；只适合底噪明显的录音，可能损失气声。',
        'operations': [
            {'name': 'trim_silence', 'threshold_db': -50, 'keep_leading_ms': 30, 'keep_trailing_ms': 50},
            {'name': 'de_click', 'threshold': 4.5, 'window': 5},
            {'name': 'denoise', 'strength': 0.9, 'window_ms': 32, 'noise_seconds': 0.6},
            {'name': 'eq', 'bands': [
                {'type': 'highpass', 'freq': 80, 'gain_db': 0, 'q': 0.707},
            ]},
            {'name': 'normalize', 'target_lufs': -16, 'ceiling_dbfs': -1.5, 'max_gain_db': 4},
        ],
    },
}

PRESET_NAMES = tuple(PRESETS)


def preset_names():
    """Ordered preset identifiers."""
    return PRESET_NAMES


def preset_choices():
    """``(label, id)`` pairs for a Gradio dropdown."""
    return [(f"{PRESETS[name]['label']} · {name}", name) for name in PRESET_NAMES]


def get_preset(name):
    """Return a deep-ish copy of one validated preset definition."""
    if name not in PRESETS:
        raise ValueError(f'未知音频预设：{name!r}；可用：{", ".join(PRESET_NAMES)}。')
    preset = PRESETS[name]
    operations = validate_operations(preset['operations'])
    return {'name': name, 'label': preset['label'], 'description': preset['description'],
            'operations': operations, 'summary': _summary(operations)}


def preset_operations(name):
    return get_preset(name)['operations']


def _summary(operations):
    from timbreforge.audio_edit_schema import describe_operations
    return ' → '.join(describe_operations(operations))

