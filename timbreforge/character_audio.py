"""High-level, non-destructive audio conditioning for one character card.

The store keeps the original reference and the edit history; this module is the
only place that knows how to turn a preset or a custom chain into a new
canonical recording, build its audition preview and record the provenance.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import soundfile as sf

from timbreforge.audio_edit_pipeline import render_chain
from timbreforge.audio_edit_schema import (ChainError, chain_digest, describe_operations,
                                           summarize_chain, validate_operations)
from timbreforge.audio_presets import get_preset
from timbreforge.audio_preview import build_preview, preview_path
from timbreforge.characters import CharacterStore, audio_path


def card_directory(card):
    return Path(card['reference_audio']).resolve().parent


def _relative(directory, path):
    return Path(path).resolve().relative_to(Path(directory).resolve()).as_posix()


def _unique_name(directory, digest):
    """Avoid overwriting earlier derivative files with the same chain."""
    stem = f'canonical-{digest[:16]}'
    candidate = directory / f'{stem}.wav'
    index = 2
    while candidate.exists():
        candidate = directory / f'{stem}-{index}.wav'
        index += 1
    return candidate.name


def render_preset(character_id, preset_name, *, store=None) -> dict:
    """Render one of the four built-in presets onto a character reference."""
    store = store or CharacterStore()
    card = store.load(character_id)
    # Presets are alternatives for the same source recording.  Always start
    # from the immutable original so choosing a second preset does not stack
    # the first preset's EQ/noise floor and make every preset sound alike.
    preset = get_preset(preset_name)
    return render_custom(character_id, preset['operations'], preset=preset_name,
                         store=store, source_path=card['reference_audio'])


def render_custom(character_id, operations, *, preset='', store=None,
                  source_path=None) -> dict:
    """Render a validated custom chain; original audio is never modified."""
    store = store or CharacterStore()
    card = store.load(character_id)
    operations = validate_operations(operations)
    if not operations:
        raise ChainError('请至少选择一个处理步骤；恢复原始音频请使用「恢复原始版本」。')
    directory = card_directory(card)
    derived = directory / 'audio'
    derived.mkdir(exist_ok=True)
    source = Path(source_path or audio_path(card)).expanduser().resolve()
    if not source.is_file():
        raise ChainError(f'参考音频不存在：{source}')
    source_name = _relative(directory, source)
    digest = hashlib.sha256((chain_digest(operations) + card['reference_sha256']).encode()).hexdigest()
    output_name = _unique_name(derived, digest)
    result = render_chain(source, operations, dest_dir=derived, output_name=output_name,
                          source_name=source_name, preset=preset,
                          output_label=f'audio/{output_name}')
    card = store.apply_audio_document(character_id, result['document'], preset=preset)
    preview = build_preview(result['output_path'], derived / 'previews')
    return {
        'ok': True,
        'card': card,
        'document': result['document'],
        'output_path': result['output_path'],
        'preview': preview,
        'metrics': result['metrics'],
        'notes': result['notes'],
        'summary': result['summary'],
        'preset': preset,
    }


def revert(character_id, *, store=None, note=''):
    """Restore the original recording and keep the derived history on disk."""
    store = store or CharacterStore()
    card = store.revert_audio(character_id, note=note)
    return {'ok': True, 'card': card, 'audio': audio_path(card),
            'summary': '已恢复为原始参考音频。'}


def audition(character_id, *, store=None) -> dict:
    """Return the current audio and a compressed playback copy."""
    store = store or CharacterStore()
    card = store.load(character_id)
    directory = card_directory(card)
    current = Path(audio_path(card))
    preview = None
    try:
        preview = build_preview(current, directory / 'audio' / 'previews')
    except (OSError, RuntimeError, ValueError):
        # Playback can always fall back to the uncompressed WAV.
        preview = None
    return {'card': card, 'audio': str(current), 'preview': preview,
            'is_canonical': bool(card.get('canonical_audio')),
            'clone_prompt_status': _prompt_status(card)}


def _prompt_status(card):
    if not card.get('clone_prompt'):
        return 'none'
    return 'present' if Path(card['clone_prompt']).is_file() else 'missing'


def history(character_id, *, store=None):
    """Compact, human-readable edit history, newest first."""
    store = store or CharacterStore()
    card = store.load(character_id)
    entries = []
    for index, entry in enumerate(reversed(card.get('audio_edit_chain', [])), 1):
        if 'event' in entry:
            entries.append({'index': index, 'kind': entry['event'],
                            'at': entry['at'], 'summary': entry['note'] or entry['event'],
                            'output': entry.get('output_audio')})
        else:
            entries.append({'index': index, 'kind': 'edit', 'at': entry['created_at'],
                            'summary': '；'.join(describe_operations(entry['operations'])),
                            'output': entry['output_audio'],
                            'source': entry['source_audio'],
                            'preset': entry['preset']})
    return entries


def status(character_id, *, store=None):
    """Card audio status for UI and for the unified generation entry point."""
    store = store or CharacterStore()
    card = store.load(character_id)
    directory = card_directory(card)
    current = Path(audio_path(card))
    with sf.SoundFile(str(current)) as recording:
        duration = recording.frames / recording.samplerate
        sample_rate = recording.samplerate
        channels = recording.channels
    summary = summarize_chain(card.get('audio_edit_chain', []))
    return {'card': card, 'audio': str(current), 'duration': duration,
            'sample_rate': sample_rate, 'channels': channels,
            'is_canonical': bool(card.get('canonical_audio')),
            'preview': preview_path(directory), 'history': summary,
            'clone_prompt': card.get('clone_prompt'),
            'version': card.get('version', 1)}

