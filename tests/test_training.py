"""Tests for the offline advanced training preparation module.

All WAVs are synthetic and all state lives in temporary directories. Environment
variables are always patched with ``unittest.mock.patch.dict`` so a test can never
leak a trainer configuration into the surrounding process.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from timbreforge import training
from timbreforge.characters import CharacterStore
from timbreforge.model_registry import ModelRegistry, ModelRegistryError
from timbreforge.training import (
    TrainingError,
    TrainingUnavailable,
    create_training_job,
    dataset_summary,
    estimate_vram_gb,
    load_dataset,
    prepare_dataset,
    record_evaluation,
    register_trained_model,
    rollback_model,
    run_training,
)

SR = 24000
TRAINER_ENV = 'TIMBREFORGE_TRAINING_COMMAND'


# ------------------------------------------------------------------- fixtures
def write_voice(path, *, freq=180.0, seconds=2.5, lead=0.35, amp=0.18,
                noise=2e-4):
    """A short voiced tone with low noise and a short leading silence."""
    rng = np.random.default_rng(int(freq * 7) + int(seconds * 1000))
    total = int(round(seconds * SR))
    t = np.arange(total) / SR
    envelope = 0.8 + 0.2 * np.sin(2 * np.pi * 0.9 * t + 0.2)
    data = amp * envelope * np.sin(2 * np.pi * freq * t)
    data += rng.normal(0.0, noise, total)
    lead_frames = int(round(lead * SR))
    data[:lead_frames] = rng.normal(0.0, noise * 0.6, lead_frames)
    sf.write(str(path), data.astype(np.float32), SR, subtype='FLOAT')
    return Path(path)


def write_silence(path):
    sf.write(str(path), np.zeros(SR * 2, dtype=np.float32), SR, subtype='FLOAT')
    return Path(path)


def write_short(path):
    return write_voice(path, seconds=1.0, lead=0.15)


def write_clipped(path):
    total = SR * 2
    data = np.sign(np.sin(2 * np.pi * 150 * np.arange(total) / SR)) * 0.999
    sf.write(str(path), data.astype(np.float32), SR, subtype='FLOAT')
    return Path(path)


def write_low_snr(path):
    rng = np.random.default_rng(9)
    total = SR * 2
    t = np.arange(total) / SR
    data = 0.2 * np.sin(2 * np.pi * 170 * t) + rng.normal(0.0, 0.15, total)
    sf.write(str(path), data.astype(np.float32), SR, subtype='FLOAT')
    return Path(path)


def write_too_much_silence(path):
    total = SR * 2
    data = np.zeros(total)
    voiced = int(0.3 * SR)
    t = np.arange(voiced) / SR
    data[:voiced] = 0.15 * np.sin(2 * np.pi * 200 * t)
    sf.write(str(path), data.astype(np.float32), SR, subtype='FLOAT')
    return Path(path)


@contextmanager
def trainer_env(value=None):
    """Patch the trainer environment variable completely (set or absent)."""
    with patch.dict(os.environ):
        os.environ.pop(TRAINER_ENV, None)
        if value is not None:
            os.environ[TRAINER_ENV] = (
                value if isinstance(value, str) else json.dumps(value))
        yield


def read_jsonl(path):
    text = Path(path).read_text(encoding='utf-8')
    return [json.loads(line) for line in text.splitlines()]


def read_job(job_dir):
    return json.loads((Path(job_dir) / 'job.json').read_text(encoding='utf-8'))


def reopen_job(job_dir):
    """Simulate a deliberate retrain attempt on an existing job directory."""
    data = read_job(job_dir)
    data['status'] = 'prepared'
    data.pop('error', None)
    data.pop('finished_at', None)
    (Path(job_dir) / 'job.json').write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def success_runner(payload=b'v1', *, write_config=True):
    def runner(command, request_path, result_path, log_path, env, timeout):
        request = json.loads(Path(request_path).read_text(encoding='utf-8'))
        assert env.get('HF_HUB_OFFLINE') == '1'
        staging = Path(request['staging_dir'])
        staging.mkdir(parents=True, exist_ok=True)
        if write_config:
            (staging / 'config.json').write_text('{"model_type": "lora"}',
                                                 encoding='utf-8')
        (staging / 'adapter_model.safetensors').write_bytes(payload)
        Path(result_path).write_text(
            json.dumps({'status': 'ok', 'checkpoint_dir': str(staging)}),
            encoding='utf-8')
        Path(log_path).write_text('训练完成\n', encoding='utf-8')
        return {'returncode': 0}
    return runner


class TrainingTestCase(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.root = Path(self._temp.name)

    def make_sources(self, count=2, name='sources'):
        directory = self.root / name
        directory.mkdir(parents=True, exist_ok=True)
        items = []
        paths = []
        for index in range(count):
            path = write_voice(directory / f'voice-{index}.wav',
                               freq=140.0 + index * 17.0)
            paths.append(path)
            items.append({'audio': path, 'transcript': f'第 {index} 条测试台词。'})
        return items, paths

    def make_dataset(self, count=2, name='dataset'):
        items, _paths = self.make_sources(count)
        return prepare_dataset(items, self.root / name, license_confirmed=True)

    def make_base_model(self, size=4096, name='base-model'):
        base = self.root / name
        base.mkdir(parents=True, exist_ok=True)
        (base / 'config.json').write_text('{"model_type": "qwen3"}',
                                          encoding='utf-8')
        (base / 'model.safetensors').write_bytes(b'\0' * size)
        return base

    def create_job(self, *, dataset=None, base=None, params=None,
                   name='测试 LoRA 任务', env=None, output='jobs'):
        with trainer_env(env):
            return create_training_job(
                dataset if dataset is not None else self.dataset['path'],
                base_model=base if base is not None else self.base,
                output_dir=self.root / output,
                name=name, params=params)


# -------------------------------------------------------------------- dataset
class DatasetTests(TrainingTestCase):
    def test_license_must_be_explicitly_confirmed(self):
        items, _paths = self.make_sources(1)
        with trainer_env():
            for value in (False, None, 1, 'yes'):
                with self.assertRaises(TrainingError) as ctx:
                    prepare_dataset(items, self.root / 'dataset',
                                    license_confirmed=value)
                self.assertIn('使用权', str(ctx.exception))
                self.assertFalse((self.root / 'dataset').exists())

    def test_mixed_items_publish_rejected_reasons_and_exact_copies(self):
        good = write_voice(self.root / 'good.wav')
        silence = write_silence(self.root / 'silence.wav')
        short = write_short(self.root / 'short.wav')
        clipped = write_clipped(self.root / 'clipped.wav')
        low_snr = write_low_snr(self.root / 'low-snr.wav')
        too_silent = write_too_much_silence(self.root / 'too-silent.wav')
        sources = {
            'good.wav': good, 'silence.wav': silence, 'short.wav': short,
            'clipped.wav': clipped, 'low-snr.wav': low_snr,
            'too-silent.wav': too_silent,
        }
        before = {name: path.read_bytes() for name, path in sources.items()}
        items = [
            (good, '这是一条正常的测试台词。'),
            {'audio': silence, 'transcript': '静音录音。'},
            {'audio': short, 'transcript': '过短录音。'},
            (clipped, '削波录音。'),
            (low_snr, '低信噪比录音。'),
            (too_silent, '静音过多的录音。'),
        ]
        with trainer_env():
            dataset = prepare_dataset(items, self.root / 'dataset',
                                      license_confirmed=True)
        self.assertEqual(dataset['counts'], {'train': 1, 'valid': 0, 'total': 1})
        self.assertEqual(dataset['manifest']['train_count'], 1)
        self.assertEqual(dataset['manifest']['valid_count'], 0)
        self.assertEqual(dataset['rejected_count'], 5)
        self.assertEqual(len(dataset['rejected']), 5)
        reasons = {Path(entry['audio']).name: entry['reason']
                   for entry in dataset['rejected']}
        self.assertIn('近乎静音', reasons['silence.wav'])
        self.assertIn('时长', reasons['short.wav'])
        self.assertIn('削波', reasons['clipped.wav'])
        self.assertIn('信噪比', reasons['low-snr.wav'])
        self.assertIn('有效语音', reasons['too-silent.wav'])
        self.assertIn('valid', ' '.join(dataset['manifest']['warnings']))

        accepted = dataset['train'][0]
        digest = hashlib.sha256(before['good.wav']).hexdigest()
        self.assertEqual(accepted['sha256'], digest)
        clip = Path(dataset['path']) / accepted['audio']
        self.assertEqual(clip.name, digest[:16] + '.wav')
        self.assertEqual(clip.read_bytes(), before['good.wav'])
        self.assertGreater(accepted['snr_db'], 12.0)
        self.assertGreaterEqual(accepted['voiced_ratio'], 0.25)
        self.assertEqual(accepted['sample_rate'], SR)
        self.assertEqual(accepted['channels'], 1)
        self.assertTrue(dataset['manifest']['license_confirmed'])
        self.assertEqual(dataset['manifest']['schema_version'], training.DATASET_SCHEMA)
        # Byte copies never modify or re-encode a source recording.
        for name, path in sources.items():
            self.assertEqual(path.read_bytes(), before[name], name)
        for source in sources.values():
            self.assertEqual(source.suffix, '.wav')

    def test_input_contract_and_duplicate_bytes_are_rejected(self):
        good = write_voice(self.root / 'good.wav')
        duplicate = self.root / 'duplicate.wav'
        duplicate.write_bytes(good.read_bytes())
        cases = [
            [],
            'not-a-list',
            [123],
            [('only-one',)],
            [{'audio': good}],
            [{'audio': good, 'transcript': '台词', 'extra': 1}],
            [(good, '')],
            [(good, '   ')],
            [(good, 'x' * 5001)],
            [(good, '重复'), (duplicate, '重复')],
        ]
        with trainer_env():
            for index, items in enumerate(cases):
                output = self.root / f'dataset-{index}'
                with self.assertRaises(TrainingError):
                    prepare_dataset(items, output, license_confirmed=True)
                self.assertFalse(output.exists(), index)
        with trainer_env():
            with self.assertRaises(TrainingError) as ctx:
                prepare_dataset([(good, '重复'), (duplicate, '重复')],
                                self.root / 'dataset-dup', license_confirmed=True)
            self.assertIn('重复', str(ctx.exception))

    def test_deterministic_split_and_jsonl_match_manifest(self):
        items, _paths = self.make_sources(6)
        with trainer_env():
            first = prepare_dataset(items, self.root / 'first', license_confirmed=True)
            second = prepare_dataset(items, self.root / 'second', license_confirmed=True)
        self.assertEqual(first['counts'], {'train': 5, 'valid': 1, 'total': 6})
        self.assertGreaterEqual(first['counts']['valid'], 1)
        self.assertEqual(first['manifest']['items'], second['manifest']['items'])
        self.assertEqual(
            {item['sha256']: item['split'] for item in first['manifest']['items']},
            {item['sha256']: item['split'] for item in second['manifest']['items']})

        for path, expected_split in (('train.jsonl', 'train'),
                                     ('valid.jsonl', 'valid')):
            rows = read_jsonl(Path(first['path']) / path)
            expected = [item for item in first['manifest']['items']
                        if item['split'] == expected_split]
            self.assertEqual(rows, expected)
            self.assertEqual(len(rows), first['counts'][expected_split])
            for row in rows:
                self.assertEqual(row['split'], expected_split)

        loaded = load_dataset(first['path'])
        self.assertEqual(loaded['train'], first['train'])
        self.assertEqual(loaded['valid'], first['valid'])
        self.assertEqual(loaded['counts'], first['counts'])
        summary = dataset_summary(first['path'])
        self.assertEqual(summary['counts'], first['counts'])
        self.assertEqual(summary['rejected_count'], 0)
        self.assertEqual(summary['total_duration'], first['total_duration'])

    def test_existing_output_is_refused_and_failed_publish_leaves_no_temp(self):
        items, _paths = self.make_sources(2)
        with trainer_env():
            prepare_dataset(items, self.root / 'existing', license_confirmed=True)
            with self.assertRaises(TrainingError) as ctx:
                prepare_dataset(items, self.root / 'existing', license_confirmed=True)
            self.assertIn('已存在', str(ctx.exception))

            with patch('timbreforge.training.os.rename',
                       side_effect=OSError('发布失败')):
                with self.assertRaises(OSError):
                    prepare_dataset(items, self.root / 'failed',
                                    license_confirmed=True)
        self.assertFalse((self.root / 'failed').exists())
        leftovers = [path.name for path in self.root.iterdir()
                     if path.name.startswith('.dataset-tmp-')]
        self.assertEqual(leftovers, [])

    def test_dataset_settings_are_validated(self):
        items, _paths = self.make_sources(1)
        with trainer_env():
            for kwargs in (
                {'target_sample_rate': 4000},
                {'min_seconds': 2.0, 'max_seconds': 1.0},
                {'max_seconds': 16.0, 'min_seconds': -1.0},
                {'train_ratio': 1.0},
                {'train_ratio': 0.0},
                {'seed': -1},
                {'seed': 1.5},
            ):
                with self.assertRaises(TrainingError):
                    prepare_dataset(items, self.root / 'settings',
                                    license_confirmed=True, **kwargs)

    def test_load_dataset_detects_mutated_clip_and_jsonl_tampering(self):
        items, _paths = self.make_sources(2)
        with trainer_env():
            dataset = prepare_dataset(items, self.root / 'tampered',
                                      license_confirmed=True)
        clip = Path(dataset['path']) / dataset['manifest']['items'][0]['audio']
        data = bytearray(clip.read_bytes())
        data[-1] ^= 0xFF
        clip.write_bytes(bytes(data))
        with self.assertRaises(TrainingError) as ctx:
            load_dataset(dataset['path'])
        self.assertIn('哈希', str(ctx.exception))
        # The summary is deliberately hash-free and still reports the counts.
        summary = dataset_summary(dataset['path'])
        self.assertEqual(summary['counts']['total'], 2)

        with trainer_env():
            other = prepare_dataset(items, self.root / 'jsonl-tampered',
                                    license_confirmed=True)
        train_jsonl = Path(other['path']) / 'train.jsonl'
        with train_jsonl.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'sha256': '0' * 64}) + '\n')
        with self.assertRaises(TrainingError):
            load_dataset(other['path'])

        (Path(other['path']) / other['manifest']['items'][0]['audio']).unlink()
        with self.assertRaises(TrainingError):
            dataset_summary(other['path'])


# ------------------------------------------------------------------------ jobs
class JobTests(TrainingTestCase):
    def setUp(self):
        super().setUp()
        self.dataset = self.make_dataset(2)
        self.base = self.make_base_model()

    def test_defaults_metadata_and_vram_estimate(self):
        with patch('timbreforge.training.subprocess.run',
                   side_effect=AssertionError('创建任务不得执行训练器')):
            job = self.create_job()
        self.assertRegex(job['id'], r'^[0-9a-f]{32}$')
        self.assertEqual(job['schema_version'], training.JOB_SCHEMA)
        self.assertEqual(job['status'], 'prepared')
        self.assertEqual(job['dataset']['counts'], self.dataset['counts'])
        self.assertEqual(job['dataset']['path'], self.dataset['path'])
        self.assertEqual(job['dataset']['clips'], self.dataset['counts']['total'])
        self.assertGreater(job['base_model']['parameter_billions'], 0)
        self.assertEqual(job['params'], training._DEFAULT_PARAMS)
        self.assertAlmostEqual(
            job['vram_estimate_gb'],
            estimate_vram_gb(job['base_model']['parameter_billions']))
        self.assertGreater(job['vram_estimate_gb'], 0)
        self.assertEqual(job['trainer'], {'configured': False, 'command': None})
        self.assertIsNone(job['evaluation'])
        self.assertIsNone(job['registered_model'])
        self.assertIsNone(job['role_id'])
        self.assertIsInstance(job['warnings'], list)
        joined = ' '.join(job['warnings'])
        for theme in ('离线', '显存', 'Base', '授权', '估算'):
            self.assertIn(theme, joined)
        job_dir = Path(job['checkpoint_dir']).parent
        self.assertEqual(job['path'], str(job_dir))
        self.assertEqual(read_job(job_dir), job)
        self.assertEqual(Path(job['log_file']), job_dir / 'train.log')
        self.assertEqual(Path(job['checkpoint_dir']), job_dir / 'checkpoints')
        self.assertFalse((job_dir / 'checkpoints').exists())

    def test_vram_heuristic_is_documented_and_validated(self):
        self.assertEqual(estimate_vram_gb(0.6), 2.8)
        self.assertEqual(estimate_vram_gb(2.0), 5.9)
        self.assertEqual(estimate_vram_gb(0), 1.5)
        self.assertGreater(estimate_vram_gb(1.7), 0)
        doc = estimate_vram_gb.__doc__ or ''
        self.assertIn('2.2', doc)
        self.assertIn('estimate', doc.lower())
        for value in (-0.1, float('nan'), float('inf'), True, '0.6'):
            with self.assertRaises(TrainingError):
                estimate_vram_gb(value)

    def test_base_model_and_parameters_are_strictly_validated(self):
        missing = self.root / 'missing-base'
        no_config = self.root / 'no-config'
        no_config.mkdir()
        (no_config / 'model.safetensors').write_bytes(b'\0' * 32)
        for base in (missing, no_config):
            with self.assertRaises(TrainingError):
                self.create_job(base=base, output='jobs-bad-base')
        self.assertFalse((self.root / 'jobs-bad-base').exists())

        for params in (
            {'unknown': 1},
            {'rank': 3},
            {'rank': 257},
            {'rank': 8.0},
            {'rank': True},
            {'alpha': 513},
            {'epochs': 0},
            {'epochs': 51},
            {'learning_rate': 1e-7},
            {'learning_rate': 1e-2 + 1e-9},
            {'batch_size': 0},
            {'batch_size': 33},
            {'gradient_accumulation': 65},
            {'save_every': 21},
            {'seed': -1},
        ):
            with self.assertRaises(TrainingError):
                self.create_job(params=params, output='jobs-bad-params')

        job = self.create_job(params={
            'rank': 4, 'alpha': 4, 'epochs': 1, 'learning_rate': 1e-6,
            'batch_size': 32, 'gradient_accumulation': 64, 'save_every': 20,
            'seed': 0,
        }, output='jobs-boundary')
        self.assertEqual(job['params']['rank'], 4)
        self.assertEqual(job['params']['seed'], 0)

        for name, role in (('', None), ('x' * 121, None), ('ok', 123)):
            with self.assertRaises(TrainingError):
                if role is None:
                    self.create_job(name=name, output='jobs-bad-name')
                else:
                    with trainer_env():
                        create_training_job(
                            self.dataset['path'], base_model=self.base,
                            output_dir=self.root / 'jobs-bad-name', name=name,
                            role_id=role)

    def test_trainer_command_is_read_from_environment_without_execution(self):
        with patch('timbreforge.training.subprocess.run',
                   side_effect=AssertionError('创建任务不得执行训练器')):
            job = self.create_job(env=['python', 'train.py', '--request',
                                       '{request}', '--result', '{result}'])
        self.assertTrue(job['trainer']['configured'])
        self.assertEqual(job['trainer']['command'],
                         ['python', 'train.py', '--request',
                          '{request}', '--result', '{result}'])
        self.assertFalse(any('占位符' in warning for warning in job['warnings']))

        with self.assertRaises(TrainingError):
            self.create_job(env='{not json')
        with self.assertRaises(TrainingError):
            self.create_job(env='{"request": "x"}')
        empty = self.create_job(env='[]')
        self.assertEqual(empty['trainer'], {'configured': False, 'command': None})
        self.assertTrue(any('空数组' in warning for warning in empty['warnings']))
        no_placeholder = self.create_job(env=['python', 'train.py'])
        self.assertTrue(no_placeholder['trainer']['configured'])
        self.assertTrue(any('占位符' in warning
                            for warning in no_placeholder['warnings']))


# ------------------------------------------------------------------- training
class RunTrainingTests(TrainingTestCase):
    def setUp(self):
        super().setUp()
        self.dataset = self.make_dataset(2)
        self.base = self.make_base_model()
        self.job = self.create_job()
        self.job_dir = Path(self.job['checkpoint_dir']).parent

    def run_success(self, payload=b'v1'):
        job = run_training(
            self.job_dir,
            trainer_command=['fake-trainer', '{request}', '{result}'],
            runner=success_runner(payload), timeout=30)
        self.assertEqual(job['status'], 'completed')
        return job

    def test_missing_trainer_raises_unavailable_and_runs_nothing(self):
        with trainer_env():
            with self.assertRaises(TrainingUnavailable) as ctx:
                run_training(self.job_dir, timeout=30)
        message = str(ctx.exception)
        self.assertIsInstance(ctx.exception, TrainingError)
        self.assertIn(TRAINER_ENV, message)
        self.assertIn('{request}', message)
        self.assertIn('{result}', message)
        self.assertIn('没有运行任何训练', message)
        self.assertEqual(read_job(self.job_dir)['status'], 'prepared')
        self.assertFalse((self.job_dir / 'request.json').exists())
        self.assertFalse((self.job_dir / 'result.json').exists())
        self.assertFalse((self.job_dir / 'checkpoints').exists())
        self.assertEqual(list(self.job_dir.glob('checkpoints.staging-*')), [])

    def test_success_publishes_checkpoint_and_preserves_previous(self):
        with patch('timbreforge.training.subprocess.run',
                   side_effect=AssertionError('注入 runner 时不得启动子进程')):
            job = self.run_success(b'v1')
        checkpoints = self.job_dir / 'checkpoints'
        self.assertTrue((checkpoints / 'config.json').is_file())
        self.assertEqual((checkpoints / 'adapter_model.safetensors').read_bytes(), b'v1')
        self.assertTrue((self.job_dir / 'train.log').is_file())
        self.assertTrue((self.job_dir / 'request.json').is_file())
        request = json.loads((self.job_dir / 'request.json').read_text(encoding='utf-8'))
        self.assertEqual(request['job_id'], self.job['id'])
        self.assertEqual(Path(request['base_model']), self.base.resolve())
        self.assertEqual(Path(request['output_dir']), checkpoints)
        self.assertEqual(Path(request['staging_dir']).parent, self.job_dir)
        self.assertEqual(request['params'], self.job['params'])
        self.assertEqual(request['seed'], self.job['params']['seed'])
        self.assertTrue(Path(request['train_jsonl']).is_file())
        self.assertTrue(Path(request['valid_jsonl']).is_file())

        self.assertEqual(job['status'], 'completed')
        self.assertEqual(job['job']['status'], 'completed')
        self.assertEqual(Path(job['job']['path']), self.job_dir)
        self.assertEqual(job['trainer']['configured'], False)
        self.assertEqual(job['trainer']['command'], None)
        self.assertEqual(job['trainer']['returncode'], 0)
        self.assertIn('command', job['last_run'])
        self.assertEqual(job['last_run']['command'],
                         ['fake-trainer', '{request}', '{result}'])
        self.assertIsNotNone(job['last_run']['finished_at'])
        self.assertEqual(read_job(self.job_dir)['status'], 'completed')

        reopen_job(self.job_dir)
        second = self.run_success(b'v2')
        previous = list(self.job_dir.glob('checkpoints.previous-*'))
        self.assertEqual(len(previous), 1)
        self.assertEqual(
            (previous[0] / 'adapter_model.safetensors').read_bytes(), b'v1')
        self.assertEqual(
            (checkpoints / 'adapter_model.safetensors').read_bytes(), b'v2')
        self.assertEqual(Path(second['last_run']['previous_checkpoint']), previous[0])
        self.assertEqual(list(self.job_dir.glob('checkpoints.staging-*')), [])

    def test_failure_keeps_previous_checkpoint_untouched(self):
        self.run_success(b'keep')
        checkpoints = self.job_dir / 'checkpoints'
        reopen_job(self.job_dir)

        def failing_runner(command, request_path, result_path, log_path, env, timeout):
            Path(log_path).write_text('显存不足\n', encoding='utf-8')
            return 1

        failed = run_training(
            self.job_dir,
            trainer_command=['fake-trainer', '{request}', '{result}'],
            runner=failing_runner, timeout=30)
        self.assertEqual(failed['status'], 'failed')
        self.assertIn('日志', failed['error'])
        self.assertIn('train.log', failed['error'])
        self.assertEqual((checkpoints / 'adapter_model.safetensors').read_bytes(), b'keep')
        self.assertEqual(read_job(self.job_dir)['status'], 'failed')
        self.assertEqual(list(self.job_dir.glob('checkpoints.previous-*')), [])

        # A stale-looking successful result file must not override a non-zero code.
        reopen_job(self.job_dir)

        def nonzero_with_result(command, request_path, result_path, log_path,
                                env, timeout):
            request = json.loads(Path(request_path).read_text(encoding='utf-8'))
            staging = Path(request['staging_dir'])
            (staging / 'adapter_model.safetensors').write_bytes(b'stale')
            Path(result_path).write_text(
                json.dumps({'status': 'ok', 'checkpoint_dir': str(staging)}),
                encoding='utf-8')
            return 2

        still_failed = run_training(
            self.job_dir,
            trainer_command=['fake-trainer', '{request}', '{result}'],
            runner=nonzero_with_result, timeout=30)
        self.assertEqual(still_failed['status'], 'failed')
        self.assertIn('非零退出码', still_failed['error'])
        self.assertEqual((checkpoints / 'adapter_model.safetensors').read_bytes(), b'keep')

    def test_invalid_result_marks_failed_without_publishing(self):
        self.run_success(b'keep')
        checkpoints = self.job_dir / 'checkpoints'
        reopen_job(self.job_dir)

        def bad_status(command, request_path, result_path, log_path, env, timeout):
            Path(result_path).write_text(json.dumps({'status': 'error'}),
                                         encoding='utf-8')
            return 0

        failed = run_training(
            self.job_dir,
            trainer_command=['fake-trainer', '{request}', '{result}'],
            runner=bad_status, timeout=30)
        self.assertEqual(failed['status'], 'failed')
        self.assertIn('status', failed['error'])
        self.assertEqual((checkpoints / 'adapter_model.safetensors').read_bytes(), b'keep')

        reopen_job(self.job_dir)

        def missing_checkpoint(command, request_path, result_path, log_path,
                               env, timeout):
            Path(result_path).write_text(json.dumps({
                'status': 'ok',
                'checkpoint_dir': str(self.job_dir / 'does-not-exist'),
            }), encoding='utf-8')
            return 0

        failed = run_training(
            self.job_dir,
            trainer_command=['fake-trainer', '{request}', '{result}'],
            runner=missing_checkpoint, timeout=30)
        self.assertEqual(failed['status'], 'failed')
        self.assertIn('checkpoint_dir', failed['error'])
        self.assertEqual((checkpoints / 'adapter_model.safetensors').read_bytes(), b'keep')

    def test_timeout_marks_failed_and_keeps_checkpoint(self):
        self.run_success(b'keep')
        checkpoints = self.job_dir / 'checkpoints'
        reopen_job(self.job_dir)

        def timeout_runner(command, request_path, result_path, log_path, env, timeout):
            raise subprocess.TimeoutExpired(command, timeout)

        failed = run_training(
            self.job_dir,
            trainer_command=['fake-trainer', '{request}', '{result}'],
            runner=timeout_runner, timeout=1)
        self.assertEqual(failed['status'], 'failed')
        self.assertIn('超时', failed['error'])
        self.assertEqual((checkpoints / 'adapter_model.safetensors').read_bytes(), b'keep')

    def test_default_runner_executes_command_with_offline_environment(self):
        script = self.root / 'fake_trainer.py'
        script.write_text(
            'import json, os, sys\n'
            'request = json.load(open(sys.argv[1], encoding="utf-8"))\n'
            'staging = request["staging_dir"]\n'
            'open(os.path.join(staging, "config.json"), "w", encoding="utf-8").write("{}")\n'
            'open(os.path.join(staging, "adapter_model.safetensors"), "wb").write(b"default")\n'
            'open(sys.argv[2], "w", encoding="utf-8").write(\n'
            '    json.dumps({"status": "ok", "checkpoint_dir": staging}))\n'
            'print("offline" if os.environ.get("HF_HUB_OFFLINE") == "1" else "online")\n',
            encoding='utf-8')
        command = [sys.executable, str(script), '{request}', '{result}']
        job = run_training(self.job_dir, trainer_command=command, timeout=120)
        self.assertEqual(job['status'], 'completed')
        self.assertEqual(
            (self.job_dir / 'checkpoints' / 'adapter_model.safetensors').read_bytes(),
            b'default')
        log = (self.job_dir / 'train.log').read_text(encoding='utf-8')
        self.assertIn('offline', log)
        self.assertNotIn('online', log)
        request = json.loads((self.job_dir / 'request.json').read_text(encoding='utf-8'))
        self.assertTrue(Path(request['staging_dir']).name.startswith('checkpoints.staging-'))

    def test_completed_job_cannot_be_trained_again(self):
        self.run_success(b'v1')
        with self.assertRaises(TrainingError):
            run_training(
                self.job_dir,
                trainer_command=['fake-trainer', '{request}', '{result}'],
                runner=success_runner(b'v2'), timeout=30)
        self.assertEqual(read_job(self.job_dir)['status'], 'completed')
        self.assertEqual(
            (self.job_dir / 'checkpoints' / 'adapter_model.safetensors').read_bytes(),
            b'v1')

    def test_runner_call_shapes_are_supported(self):
        def four_argument_runner(command, request_path, result_path, log_path):
            request = json.loads(Path(request_path).read_text(encoding='utf-8'))
            staging = Path(request['staging_dir'])
            (staging / 'config.json').write_text('{}', encoding='utf-8')
            (staging / 'adapter_model.safetensors').write_bytes(b'four')
            Path(result_path).write_text(
                json.dumps({'status': 'ok', 'checkpoint_dir': str(staging)}),
                encoding='utf-8')
            return 0

        job = run_training(
            self.job_dir,
            trainer_command=['fake-trainer', '{request}', '{result}'],
            runner=four_argument_runner, timeout=30)
        self.assertEqual(job['status'], 'completed')

        reopen_job(self.job_dir)

        def six_argument_runner(command, request_path, result_path, log_path,
                                env, timeout):
            request = json.loads(Path(request_path).read_text(encoding='utf-8'))
            staging = Path(request['staging_dir'])
            (staging / 'config.json').write_text('{}', encoding='utf-8')
            (staging / 'adapter_model.safetensors').write_bytes(b'six')
            Path(result_path).write_text(
                json.dumps({'status': 'ok', 'checkpoint_dir': str(staging)}),
                encoding='utf-8')
            return subprocess.CompletedProcess(args=command, returncode=0)

        job = run_training(
            self.job_dir,
            trainer_command=['fake-trainer', '{request}', '{result}'],
            runner=six_argument_runner, timeout=30)
        self.assertEqual(job['status'], 'completed')
        self.assertEqual(
            (self.job_dir / 'checkpoints' / 'adapter_model.safetensors').read_bytes(),
            b'six')


# ------------------------------------------------------- evaluation/registry
class EvaluationAndRegistrationTests(TrainingTestCase):
    def setUp(self):
        super().setUp()
        self.dataset = self.make_dataset(2)
        self.base = self.make_base_model()
        self.job = self.create_job()
        self.job_dir = Path(self.job['checkpoint_dir']).parent
        self.voice = write_voice(self.root / 'card-reference.wav', freq=210.0)

    def complete_job(self, *, write_config=True):
        job = run_training(
            self.job_dir,
            trainer_command=['fake-trainer', '{request}', '{result}'],
            runner=success_runner(b'checkpoint', write_config=write_config),
            timeout=30)
        self.assertEqual(job['status'], 'completed')
        return job

    def make_card(self, store):
        card = store.create(
            '训练角色', self.voice, reference_text='这是一段参考台词。',
            backend='qwen3', model_name='BaseX', model_revision='rev-1')
        prompts = Path(card['reference_audio']).parent / 'prompts'
        prompts.mkdir()
        (prompts / 'clone.pt').write_bytes(b'prompt')
        return store.set_clone_prompt(card['id'], 'prompts/clone.pt', 'icl')

    def test_record_evaluation_validates_and_writes_atomically(self):
        # A prepared job cannot carry evaluation evidence.
        with self.assertRaises(TrainingError) as ctx:
            record_evaluation(self.job_dir, {'cer': 0.1})
        self.assertIn('已完成', str(ctx.exception))

        self.complete_job()
        metrics = {'cer': 0.05, 'wer': 0.1, 'speaker_similarity': 0.82,
                   'mos': 3.9, 'notes': '已完成试听。'}
        result = record_evaluation(self.job_dir, metrics)
        self.assertEqual(result['evaluation']['metrics'], metrics)
        self.assertEqual(result['evaluation']['job_id'], self.job['id'])
        evaluation_path = self.job_dir / 'evaluation.json'
        self.assertTrue(evaluation_path.is_file())
        stored = json.loads(evaluation_path.read_text(encoding='utf-8'))
        self.assertEqual(stored['metrics'], metrics)
        self.assertEqual(result['job']['evaluation']['metrics'], metrics)
        self.assertEqual(Path(result['job']['evaluation']['path']), evaluation_path)
        self.assertEqual(read_job(self.job_dir)['evaluation']['metrics'], metrics)
        self.assertEqual(list(self.job_dir.glob('*.tmp*')), [])
        before = evaluation_path.read_bytes()

        invalid = [
            {},
            {'notes': '只有备注'},
            {'cer': -0.1},
            {'wer': -1},
            {'speaker_similarity': 1.5},
            {'speaker_similarity': -1.5},
            {'mos': 5.1},
            {'mos': -0.1},
            {'cer': float('nan')},
            {'cer': True},
            {'unknown': 1},
            {'notes': 123},
            {'cer': 0.1, 'notes': 'x' * 1001},
        ]
        for metrics_value in invalid:
            with self.assertRaises(TrainingError):
                record_evaluation(self.job_dir, metrics_value)
            self.assertEqual(evaluation_path.read_bytes(), before)

    def test_registration_requires_confirmation_and_evaluation(self):
        with self.assertRaises(TrainingError) as ctx:
            register_trained_model(self.job_dir, registry=ModelRegistry(self.root / 'models'),
                                   confirmed=True)
        self.assertIn('已完成', str(ctx.exception))

        self.complete_job()
        with self.assertRaises(TrainingError) as ctx:
            register_trained_model(self.job_dir,
                                   registry=ModelRegistry(self.root / 'models'),
                                   confirmed=False)
        self.assertIn('确认', str(ctx.exception))
        with self.assertRaises(TrainingError) as ctx:
            register_trained_model(self.job_dir,
                                   registry=ModelRegistry(self.root / 'models'),
                                   confirmed=True)
        self.assertIn('评估', str(ctx.exception))
        self.assertIsNone(read_job(self.job_dir)['registered_model'])

        record_evaluation(self.job_dir, {'cer': 0.02})
        with self.assertRaises(TrainingError):
            register_trained_model(self.job_dir,
                                   registry=ModelRegistry(self.root / 'models'),
                                   confirmed=False)

    def test_registration_binding_invalidates_prompt_and_rollback_restores(self):
        self.complete_job()
        record_evaluation(self.job_dir, {
            'cer': 0.03, 'wer': 0.06, 'speaker_similarity': 0.8, 'mos': 4.0})
        base_files = {path.relative_to(self.base): path.read_bytes()
                      for path in self.base.rglob('*') if path.is_file()}

        registry = ModelRegistry(self.root / 'registry')
        store = CharacterStore(self.root / 'characters')
        card = self.make_card(store)
        self.assertIsNotNone(card['clone_prompt'])

        # Registering without a role id must not modify any character card.
        first = register_trained_model(self.job_dir, registry=registry, confirmed=True)
        self.assertIsNone(first['card'])
        self.assertEqual(store.load(card['id']), card)
        entry = first['model']
        self.assertEqual(entry['kind'], 'lora')
        self.assertEqual(entry['name'], self.job['name'])
        self.assertEqual(entry['revision'], f"job-{self.job['id'][:12]}")
        self.assertEqual(Path(entry['base_model']), self.base.resolve())
        self.assertIn('评估', entry['notes'])
        self.assertEqual(read_job(self.job_dir)['registered_model']['id'], entry['id'])

        # Binding activates the model on the card and drops its clone prompt.
        second = register_trained_model(
            self.job_dir, registry=registry, store=store,
            bind_role_id=card['id'], confirmed=True)
        self.assertEqual(second['model']['id'], entry['id'])
        self.assertEqual(second['entry'], second['model'])
        bound = second['card']
        self.assertEqual(bound['model_name'], self.job['name'])
        self.assertEqual(bound['model_revision'], f"job-{self.job['id'][:12]}")
        self.assertIsNone(bound['clone_prompt'])
        self.assertEqual(bound['version'], card['version'] + 1)
        self.assertEqual(store.load(card['id']), bound)

        restored = rollback_model(card['id'], registry=registry, store=store)
        self.assertEqual(restored['restored'], {'name': 'BaseX', 'revision': 'rev-1'})
        self.assertEqual(restored['card']['model_name'], 'BaseX')
        self.assertEqual(restored['card']['model_revision'], 'rev-1')
        self.assertEqual(store.load(card['id'])['model_name'], 'BaseX')

        current_base = {path.relative_to(self.base): path.read_bytes()
                        for path in self.base.rglob('*') if path.is_file()}
        self.assertEqual(current_base, base_files)

    def test_registry_errors_are_propagated_for_invalid_checkpoints(self):
        self.complete_job(write_config=False)
        record_evaluation(self.job_dir, {'cer': 0.1})
        with self.assertRaises(ModelRegistryError):
            register_trained_model(self.job_dir,
                                   registry=ModelRegistry(self.root / 'registry'),
                                   confirmed=True)
        self.assertIsNone(read_job(self.job_dir)['registered_model'])

    def test_rollback_requires_activation_history(self):
        store = CharacterStore(self.root / 'characters')
        card = store.create('无历史角色', self.voice, model_name='BaseX',
                            model_revision='rev-1')
        registry = ModelRegistry(self.root / 'registry')
        with self.assertRaises(ModelRegistryError):
            rollback_model(card['id'], registry=registry, store=store)
        self.assertEqual(store.load(card['id'])['model_name'], 'BaseX')


if __name__ == '__main__':
    unittest.main()

