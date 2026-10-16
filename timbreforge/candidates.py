"""Independent auxiliary metrics and disposable A/B playback copies.

Never select a winner, normalize originals, or imply that a signal metric is
speaker identity, ASR accuracy, or human preference.
"""
from __future__ import annotations
import hashlib
import math
from pathlib import Path
import uuid
import numpy as np
import soundfile as sf
from timbreforge.config import get_config


def _audio(path):
    p = Path(path).expanduser().resolve()
    x, sr = sf.read(p, dtype='float32', always_2d=True)
    if not x.size or sr <= 0 or not np.isfinite(x).all():
        raise ValueError('Audio must contain finite, nonempty samples')
    return p, x, sr


def _distance(a, b):
    row = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        following = [i]
        for j, right in enumerate(b, 1):
            following.append(min(following[-1] + 1, row[j] + 1,
                                 row[j - 1] + (left != right)))
        row = following
    return row[-1]


def measure_candidate(path, reference=None, expected_text='', observed_text=None,
                      run_speaker=False, run_quality=False):
    """Return distinct axes; optional external scorers are explicit opt-in.

    observed_text is a user-supplied transcript, NOT an automatic ASR result.
    Character error rate is raw Unicode edit distance, including punctuation.
    """
    p, x, sr = _audio(path)
    rms = float(np.sqrt(np.mean(x.astype('float64') ** 2)))
    result = {
        'speaker_similarity': {'status': 'unverified', 'value': None, 'method': 'CAM++ cosine (optional)'},
        'content_accuracy': {'status': 'unverified', 'value': None, 'method': 'user transcript character error rate'},
        'quality': {'status': 'unverified', 'value': None, 'method': 'MOS auxiliary (optional)'},
        'signal': {'duration_seconds': len(x) / sr, 'sample_rate': sr,
                   'peak': float(np.max(np.abs(x))), 'rms': rms,
                   'near_full_scale_fraction': float(np.mean(np.abs(x) >= .999)),
                   'sha256': hashlib.sha256(p.read_bytes()).hexdigest()},
        'warnings': ['Scores are independent auxiliary evidence, not a ranking or automatic selection.']}
    if observed_text is not None:
        if not isinstance(expected_text, str) or not isinstance(observed_text, str) or not expected_text:
            raise ValueError('Content comparison requires nonempty expected text and a text transcript')
        if max(len(expected_text), len(observed_text)) > 5000:
            raise ValueError('Transcript comparison limited to 5000 characters')
        error = _distance(expected_text, observed_text) / len(expected_text)
        result['content_accuracy'] = {'status': 'user_transcript_comparison', 'value': error,
                                     'metric': 'CER', 'lower_is_better': True,
                                     'expected_text': expected_text, 'observed_text': observed_text,
                                     'method': 'Unicode Levenshtein; punctuation/spacing retained; no ASR'}
    if run_speaker:
        if not reference:
            raise ValueError('Speaker similarity requires the actual reference audio')
        try:
            from timbreforge.score import score_one
            value = float(score_one(str(reference), str(p)))
            if not math.isfinite(value) or not -1 <= value <= 1:
                raise ValueError('Invalid CAM++ cosine')
            result['speaker_similarity'].update(status='measured', value=value)
        except Exception as exc:
            result['speaker_similarity'].update(status='unavailable', error=str(exc))
    if run_quality:
        try:
            from timbreforge.mosmodels import score_files
            values = score_files([str(p)])
            if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], dict):
                raise ValueError('Invalid MOS response')
            metrics = {k: v for k, v in values[0].items()
                       if k == 'utmos' or k.startswith('dnsmos')}
            if not metrics or any(isinstance(v, bool) or not isinstance(v, (float, int))
                                  or not math.isfinite(v) for v in metrics.values()):
                raise ValueError('Missing or non-finite MOS metrics')
            result['quality'].update(status='measured', value=metrics)
        except Exception as exc:
            result['quality'].update(status='unavailable', error=str(exc))
    return result


def compare_audio(a_path, b_path, match_volume=False, output_dir=None):
    """Create optional matched-RMS listening copies without touching either raw.

    Common target is quieter RMS, bounded by both peaks and -1 dBFS headroom.
    This is amplitude/RMS matching, NOT ITU integrated loudness measurement.
    """
    a, xa, sa = _audio(a_path)
    b, xb, sb = _audio(b_path)
    result = {'a': str(a), 'b': str(b), 'gain_a_db': 0., 'gain_b_db': 0.,
              'matched': False, 'method': 'original', 'raw_a': str(a), 'raw_b': str(b)}
    if not isinstance(match_volume, bool):
        raise ValueError('match_volume must be boolean')
    if not match_volume:
        return result
    rms = [float(np.sqrt(np.mean(x.astype('float64') ** 2))) for x in (xa, xb)]
    if min(rms) <= 1e-12:
        raise ValueError('Cannot volume-match silent audio')
    peaks = [float(np.max(np.abs(x))) for x in (xa, xb)]
    target = min(*rms, *(r * 10 ** (-1 / 20) / peak for r, peak in zip(rms, peaks)))
    directory = Path(output_dir or get_config().outputs_dir / 'playback').resolve() / uuid.uuid4().hex
    directory.mkdir(parents=True)
    for name, x, sr, level in zip(('a', 'b'), (xa, xb), (sa, sb), rms):
        gain = target / level
        out = directory / f'{name}.wav'
        sf.write(out, x * gain, sr, subtype='FLOAT')
        result[name] = str(out)
        result[f'gain_{name}_db'] = 20 * math.log10(gain)
    result.update(matched=True, method='RMS amplitude match; playback copies only; -1 dBFS peak ceiling')
    return result
