"""Read-only checks for complete local Qwen snapshots; never imports a model."""
from __future__ import annotations

import json
from pathlib import Path

from timbreforge.config import get_config


def _weights(directory):
    """Check single-file or indexed shards without reading large tensors."""
    index = directory / 'model.safetensors.index.json'
    if index.is_file():
        try:
            mapping = json.loads(index.read_text(encoding='utf-8'))['weight_map']
            if not isinstance(mapping, dict) or not mapping:
                return False
            for name in mapping.values():
                if not isinstance(name, str):
                    return False
                path = (directory / name).resolve()
                if directory.resolve() not in path.parents or not path.is_file() or not path.stat().st_size:
                    return False
            return True
        except (OSError, ValueError, KeyError, TypeError):
            return False
    path = directory / 'model.safetensors'
    return path.is_file() and path.stat().st_size > 0


def _snapshot(directory, expected):
    missing = []
    required = ('config.json', 'generation_config.json', 'preprocessor_config.json',
                'tokenizer_config.json', 'vocab.json', 'merges.txt',
                'speech_tokenizer/config.json', 'speech_tokenizer/preprocessor_config.json')
    for name in required:
        path = directory / name
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(name)
    for folder in (directory, directory / 'speech_tokenizer'):
        if not _weights(folder):
            missing.append(str(folder.relative_to(directory) / 'model.safetensors（或完整分片）'))
    try:
        model = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
        if not isinstance(model, dict) or model.get('tts_model_type') != expected:
            missing.append(f'模型类型应为 {expected}')
    except (OSError, ValueError):
        missing.append('可读取的模型 config.json')
    return dict(path=str(directory), files_ready=not missing, missing=missing)


def design_readiness():
    cfg = get_config()
    python_ready = cfg.qwen_python.is_file()
    design = _snapshot(cfg.qwen_design_model, 'voice_design')
    base = _snapshot(cfg.qwen_base_model, 'base')
    return dict(python=str(cfg.qwen_python), python_exists=python_ready,
                design=design, base=base,
                design_files_ready=python_ready and design['files_ready'],
                base_files_ready=python_ready and base['files_ready'])


def readiness_markdown(report=None):
    report = design_readiness() if report is None else report
    lines = ['**本地模型检查**',
             f"- 独立 Python：{'已找到' if report['python_exists'] else '未找到'}"]
    for key, label in (('design', '文字生成声线 / VoiceDesign'), ('base', '保存后换台词 / Base')):
        value = report[key]
        lines.append(f"- {label}：{'文件齐备' if value['files_ready'] else '待配置'}")
        if value['missing']:
            lines.append('  缺失或不匹配：' + '、'.join(value['missing']))
    if not report['design_files_ready'] or not report['base_files_ready']:
        lines.append('请在 timbreforge.local.toml 的 [qwen] 中配置 python、design_model、base_model；详见 docs/VOICE_DESIGN.md。')
    lines.append('文件检查不加载模型；Python 依赖、GPU 和实际生成以运行结果为准。')
    return '\n\n'.join(lines)
