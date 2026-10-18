"""实验性本机语音转换（Voice Conversion）适配层，与 TTS 主流程相互独立。

本模块不实现 VC 模型，也不导入 torch：它把源音频和角色卡中的目标音色交给
操作者自己配置的外部本地脚本处理。不下载权重、不访问网络，也不会把用户
音频上传或用本项目代码复制到输出目录之外；外部脚本必须把结果写回本次
请求指定的本机输出文件。

必须诚实说明：语音转换可能改变源音频的**情绪、语速和咬字**，不是完美的
音色替换，也不保证跨环境逐样本一致。结果必须逐条试听，并在使用前自行
确认模型许可证、声音授权和显存需求。任何新 VC 后端都应实现
:class:`VoiceConversionBackend`，并报告真实的 ``available()`` 状态。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

import numpy as np
import soundfile as sf

from timbreforge.characters import CharacterStore, audio_path
from timbreforge.config import get_config

SCHEMA_VERSION = 1
TIMEOUT_SECONDS = 1800.0
MAX_AUDIO_SECONDS = 300.0
MIN_OUTPUT_SAMPLE_RATE = 8000
MAX_OUTPUT_SAMPLE_RATE = 192000
_MAX_JSON_BYTES = 1 << 20
_NAME_RE = re.compile(r'[a-z0-9][a-z0-9._-]{0,63}\Z')
_HASH_RE = re.compile(r'[0-9a-f]{64}\Z')
_TARGET_BACKENDS = frozenset({'qwen3', 'indextts', 'cosyvoice3'})


class VoiceConversionError(RuntimeError):
    """语音转换失败；消息包含中文原因，运行期错误还会包含日志路径。"""


class VoiceConversionUnavailable(VoiceConversionError):
    """外部 VC 后端或本地模型文件不可用；不会自动下载或安装。"""


def _require_text(value, name, *, label, limit=2000):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{label}的 {name} 必须是非空文本。')
    if len(value) > limit:
        raise ValueError(f'{label}的 {name} 不得超过 {limit} 个字符。')


@dataclass(frozen=True)
class VoiceConversionSpec:
    """VC 后端的静态身份与诚实的能力边界说明。"""

    name: str
    model_name: str
    model_revision: str
    description: str
    preserves_content: str
    license_note: str
    estimated_vram_gb: float
    experimental: bool = True

    def __post_init__(self):
        _require_text(self.name, 'name', label='VC 后端规格', limit=64)
        if not _NAME_RE.fullmatch(self.name):
            raise ValueError(
                f'VC 后端名称 {self.name!r} 无效；只能使用小写字母、数字、点、'
                '下划线和连字符，并以字母或数字开头。')
        _require_text(self.model_name, 'model_name', label='VC 后端规格', limit=200)
        _require_text(self.model_revision, 'model_revision', label='VC 后端规格', limit=120)
        _require_text(self.description, 'description', label='VC 后端规格', limit=2000)
        _require_text(self.preserves_content, 'preserves_content',
                      label='VC 后端规格', limit=2000)
        _require_text(self.license_note, 'license_note', label='VC 后端规格', limit=2000)
        value = self.estimated_vram_gb
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('estimated_vram_gb 必须是有限正数（单位 GB）。')
        object.__setattr__(self, 'estimated_vram_gb', float(value))
        if type(self.experimental) is not bool:
            raise ValueError('experimental 必须是布尔值。')


class VoiceConversionBackend(ABC):
    """所有语音转换适配器必须实现的接口。"""

    spec: VoiceConversionSpec

    @abstractmethod
    def available(self) -> tuple[bool, str]:
        """只检查本地文件，不加载模型；返回 ``(是否可用, 中文原因)``。"""

    @abstractmethod
    def convert(self, source_audio, target_role_id, *, preserve_prosody=True,
                store=None, output_dir=None, cancel=None) -> dict:
        """把源音频转换为目标角色音色，返回可审计的元数据字典。"""


def _path_argument(value, name):
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f'{name} 必须是本机路径文本或 Path。')
    return Path(value).expanduser().resolve()


def _hash_file(path):
    digest = hashlib.sha256()
    try:
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b''):
                digest.update(chunk)
    except OSError as exc:
        raise VoiceConversionError(f'无法读取音频以计算 SHA-256：{path}；{exc}') from exc
    return digest.hexdigest()


def _validate_audio(path, label, *, max_seconds=None, require_float=False):
    """Decode one local audio file and reject invalid or unsafe content."""
    try:
        info = sf.info(str(path))
    except (OSError, RuntimeError, TypeError) as exc:
        raise VoiceConversionError(f'{label}不是可读取的音频文件：{path}；{exc}') from exc
    if info.frames <= 0 or info.channels <= 0 or info.samplerate <= 0:
        raise VoiceConversionError(f'{label}没有有效采样：{path}。')
    duration = info.frames / info.samplerate
    if duration <= 0:
        raise VoiceConversionError(f'{label}时长为零：{path}。')
    if max_seconds is not None and duration > max_seconds:
        raise VoiceConversionError(
            f'{label}不得超过 {max_seconds:g} 秒；实际约 {duration:.2f} 秒。'
            '请先在本机裁切后重试。')
    if require_float and (info.format != 'WAV' or info.subtype != 'FLOAT'):
        raise VoiceConversionError(
            f'{label}必须是 FLOAT WAV；实际为 {info.format}/{info.subtype}。')
    nonfinite = False
    frames_read = 0
    peak = 0.0
    try:
        with sf.SoundFile(str(path)) as handle:
            for block in handle.blocks(blocksize=65536, dtype='float32', always_2d=True):
                if not np.isfinite(block).all():
                    nonfinite = True
                    break
                frames_read += len(block)
                if block.size:
                    peak = max(peak, float(np.max(np.abs(block))))
    except (OSError, RuntimeError, TypeError) as exc:
        raise VoiceConversionError(f'{label}读取失败：{path}；{exc}') from exc
    if nonfinite:
        raise VoiceConversionError(f'{label}包含非有限采样（NaN/Inf），已拒绝：{path}。')
    if frames_read != info.frames:
        raise VoiceConversionError(f'{label}不完整或被截断：{path}。')
    if peak <= 1e-8:
        raise VoiceConversionError(f'{label}是静音或为空：{path}。')
    return info.samplerate, duration


def _read_json(path, label, *, log_path=None):
    suffix = f'；日志：{log_path}' if log_path else ''
    if path.is_symlink() or not path.is_file():
        raise VoiceConversionError(f'{label}缺失或不是普通文件：{path}{suffix}')
    try:
        if path.stat().st_size > _MAX_JSON_BYTES:
            raise VoiceConversionError(
                f'{label}异常过大（超过 {_MAX_JSON_BYTES // 1024} KiB），已拒绝读取：'
                f'{path}{suffix}')
        return json.loads(path.read_text(encoding='utf-8'))
    except VoiceConversionError:
        raise
    except (OSError, UnicodeError, ValueError) as exc:
        raise VoiceConversionError(f'{label}不是有效 JSON：{exc}{suffix}') from exc


def _result_error(path):
    """Best-effort error text from a failed external result; never raises."""
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_JSON_BYTES:
            return '未生成可读的结果 JSON'
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, ValueError):
        return '未生成可读的结果 JSON'
    if isinstance(data, dict) and isinstance(data.get('error'), str) and data['error'].strip():
        return data['error'].strip()[:1000]
    return '未生成可读的结果 JSON'


def _kill_process(process):
    """Terminate the external converter and any child it started."""
    try:
        if os.name == 'nt':
            subprocess.run(['taskkill', '/F', '/T', '/PID', str(process.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           creationflags=subprocess.CREATE_NO_WINDOW, timeout=15)
        else:
            process.kill()
    except (OSError, subprocess.SubprocessError):
        try:
            process.kill()
        except OSError:
            pass


def _verify_unchanged(path, expected_sha, label, log_path):
    try:
        actual = _hash_file(path)
    except VoiceConversionError as exc:
        raise VoiceConversionError(
            f'{label}在转换期间不可读或已变化：{exc}；日志：{log_path}') from exc
    if actual != expected_sha:
        raise VoiceConversionError(
            f'{label}在转换期间发生变化，已拒绝发布结果；请重新发起转换。'
            f'日志：{log_path}')


def _validate_result_metadata(raw, *, output, model_dir, preserve_prosody,
                              source_sha, target_sha, log_path):
    required = ('file', 'sample_rate', 'duration', 'model_dir',
                'preserve_prosody', 'source_sha256', 'target_sha256')
    missing = [key for key in required if key not in raw]
    if missing:
        raise VoiceConversionError(
            f'外部语音转换结果缺少字段：{"、".join(missing)}；日志：{log_path}')
    file_value = raw['file']
    if not isinstance(file_value, str) or not file_value.strip():
        raise VoiceConversionError(f'外部语音转换结果的 file 必须是非空路径文本；日志：{log_path}')
    try:
        reported_output = Path(file_value).expanduser().resolve()
    except (OSError, ValueError) as exc:
        raise VoiceConversionError(
            f'外部语音转换结果的 file 路径无效：{exc}；日志：{log_path}') from exc
    if reported_output != output.resolve():
        raise VoiceConversionError(
            f'外部语音转换结果写到了未授权路径：{file_value}；本次只允许写入 {output}。'
            f'日志：{log_path}')
    sample_rate = raw['sample_rate']
    if type(sample_rate) is not int or not MIN_OUTPUT_SAMPLE_RATE <= sample_rate <= MAX_OUTPUT_SAMPLE_RATE:
        raise VoiceConversionError(
            f'外部语音转换结果的 sample_rate 必须是 '
            f'{MIN_OUTPUT_SAMPLE_RATE}–{MAX_OUTPUT_SAMPLE_RATE} 的整数；'
            f'实际为 {sample_rate!r}。日志：{log_path}')
    duration = raw['duration']
    if (type(duration) not in (int, float) or not math.isfinite(duration)
            or not 0 < duration <= MAX_AUDIO_SECONDS):
        raise VoiceConversionError(
            f'外部语音转换结果的 duration 必须是 (0, {MAX_AUDIO_SECONDS:g}] 的有限数；'
            f'实际为 {duration!r}。日志：{log_path}')
    reported_model = raw['model_dir']
    if not isinstance(reported_model, str) or not reported_model.strip():
        raise VoiceConversionError(f'外部语音转换结果的 model_dir 必须是非空路径文本；日志：{log_path}')
    try:
        reported_model_path = Path(reported_model).expanduser().resolve()
    except (OSError, ValueError) as exc:
        raise VoiceConversionError(
            f'外部语音转换结果的 model_dir 路径无效：{exc}；日志：{log_path}') from exc
    if reported_model_path != model_dir.resolve():
        raise VoiceConversionError(
            f'外部语音转换结果使用的模型目录与配置不一致：{reported_model}；'
            f'期望 {model_dir}。日志：{log_path}')
    if type(raw['preserve_prosody']) is not bool or raw['preserve_prosody'] != preserve_prosody:
        raise VoiceConversionError(
            '外部语音转换结果的 preserve_prosody 与请求不一致；'
            f'拒绝把未经请求的处理当作成功。日志：{log_path}')
    for key, expected, label in (('source_sha256', source_sha, '源音频'),
                                 ('target_sha256', target_sha, '目标角色音频')):
        value = raw[key]
        if (not isinstance(value, str) or not _HASH_RE.fullmatch(value)
                or value != expected):
            raise VoiceConversionError(
                f'外部语音转换结果记录的{label} SHA-256 与请求不一致；'
                f'拒绝发布结果。日志：{log_path}')


def _validate_output(path, raw, log_path):
    if path.is_symlink():
        raise VoiceConversionError(f'外部语音转换输出不能是符号链接：{path}；日志：{log_path}')
    if not path.is_file():
        raise VoiceConversionError(
            f'外部语音转换没有生成输出文件：{path}；日志：{log_path}')
    try:
        sample_rate, duration = _validate_audio(
            path, '外部语音转换输出', max_seconds=MAX_AUDIO_SECONDS, require_float=True)
    except VoiceConversionError as exc:
        raise VoiceConversionError(f'{exc}；日志：{log_path}') from exc
    if not MIN_OUTPUT_SAMPLE_RATE <= sample_rate <= MAX_OUTPUT_SAMPLE_RATE:
        raise VoiceConversionError(
            f'外部语音转换输出的采样率必须在 {MIN_OUTPUT_SAMPLE_RATE}–'
            f'{MAX_OUTPUT_SAMPLE_RATE} Hz；实际为 {sample_rate} Hz。日志：{log_path}')
    if sample_rate != raw['sample_rate']:
        raise VoiceConversionError(
            f'外部语音转换输出的实际采样率 {sample_rate} Hz 与结果 JSON 记录的 '
            f'{raw["sample_rate"]} Hz 不一致。日志：{log_path}')
    if abs(duration - float(raw['duration'])) > 1 / sample_rate + 1e-6:
        raise VoiceConversionError(
            f'外部语音转换输出的实际时长 {duration:.6f} 秒与结果 JSON 记录的 '
            f'{raw["duration"]} 秒不一致。日志：{log_path}')
    return sample_rate, duration, _hash_file(path)


class ExternalVoiceConversionBackend(VoiceConversionBackend):
    """调用操作者提供的本地 VC 脚本和模型目录。

    脚本属于受信任的本机代码：本项目只传递路径与 SHA-256，不隔离其文件
    访问。结果可能改变情绪、语速和咬字，不能宣称是完美的音色替换。
    """

    def __init__(self, *, python, script, model_dir, name='external-vc',
                 model_name='external-vc', model_revision='local',
                 license_note='使用前请自行确认模型许可证与声音授权。',
                 estimated_vram_gb=4.0):
        self.python = _path_argument(python, 'python')
        self.script = _path_argument(script, 'script')
        self.model_dir = _path_argument(model_dir, 'model_dir')
        self.spec = VoiceConversionSpec(
            name=name,
            model_name=model_name,
            model_revision=model_revision,
            description=(
                '实验性外部语音转换后端：只在本机调用已配置的 Python 脚本，'
                '不下载模型、不访问网络；结果可能改变情绪、语速和咬字，'
                '不是完美的音色替换。'),
            preserves_content='保留源音频的内容与主要韵律，但可能改变情绪、语速和咬字。',
            license_note=license_note,
            estimated_vram_gb=estimated_vram_gb,
        )

    # ------------------------------------------------------------ readiness
    def available(self) -> tuple[bool, str]:
        try:
            missing = []
            if not self.python.is_file():
                missing.append(f'本地 Python 不存在：{self.python}')
            if not self.script.is_file():
                missing.append(f'外部转换脚本不存在：{self.script}')
            if not self.model_dir.is_dir():
                missing.append(f'模型目录不存在：{self.model_dir}')
        except (OSError, ValueError, RuntimeError) as exc:
            return False, f'无法检查外部语音转换后端：{exc}；请检查路径权限。'
        if missing:
            return False, ('外部语音转换后端不可用：' + '；'.join(missing)
                           + '。请检查本机配置；不会自动下载模型或安装依赖。')
        return True, '外部语音转换后端文件已就绪；模型能否加载需实际运行验证。'

    # ----------------------------------------------------------- conversion
    def convert(self, source_audio, target_role_id, *, preserve_prosody=True,
                store=None, output_dir=None, cancel=None) -> dict:
        ok, reason = self.available()
        if not ok:
            raise VoiceConversionUnavailable(reason)
        if type(preserve_prosody) is not bool:
            raise ValueError('preserve_prosody 必须是布尔值。')
        if cancel is not None and not callable(getattr(cancel, 'is_set', None)):
            raise ValueError('cancel 必须是 threading.Event 或提供 is_set() 的对象。')
        if cancel is not None and cancel.is_set():
            raise VoiceConversionError('语音转换已取消；未启动外部进程。')
        if not isinstance(target_role_id, str) or not target_role_id.strip():
            raise VoiceConversionError('target_role_id 必须是角色卡 ID 文本。')

        store = store if store is not None else CharacterStore()
        if not callable(getattr(store, 'load', None)):
            raise VoiceConversionError('store 必须提供 load(role_id) 方法。')
        try:
            card = store.load(target_role_id)
        except (KeyError, OSError, TypeError, ValueError) as exc:
            raise VoiceConversionError(
                f'无法读取目标角色卡 {target_role_id}：{exc}') from exc
        if not isinstance(card, dict):
            raise VoiceConversionError('目标角色卡必须是字典记录。')
        backend = card.get('backend')
        if backend not in _TARGET_BACKENDS:
            raise VoiceConversionError(
                f'目标角色卡的后端不受支持：{backend!r}；'
                f'只接受 {"、".join(sorted(_TARGET_BACKENDS))}。')
        try:
            target = Path(audio_path(card)).expanduser().resolve()
        except (KeyError, TypeError, ValueError) as exc:
            raise VoiceConversionError(
                f'目标角色卡缺少可读的音频路径：{exc}') from exc
        if not target.is_file():
            raise VoiceConversionError(f'目标角色卡音频不存在：{target}。')

        source = self._source_path(source_audio)
        source_sha = _hash_file(source)
        target_sha = _hash_file(target)
        _validate_audio(source, '源音频', max_seconds=MAX_AUDIO_SECONDS)
        _validate_audio(target, '目标角色音频')

        outdir = self._output_directory(output_dir)
        try:
            outdir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise VoiceConversionError(f'无法创建语音转换输出目录：{outdir}；{exc}') from exc
        if not outdir.is_dir():
            raise VoiceConversionError(f'语音转换输出目录不可用：{outdir}。')

        run_id = uuid.uuid4().hex
        request_path = outdir / f'{run_id}.request.json'
        result_path = outdir / f'{run_id}.result.json'
        log_path = outdir / f'{run_id}.log'
        output = outdir / f'{run_id}.wav'
        if output.exists() or output.is_symlink():
            raise VoiceConversionError(f'语音转换输出文件已存在，拒绝覆盖：{output}。')

        request = {
            'mode': 'voice_conversion',
            'source_audio': str(source),
            'target_audio': str(target),
            'target_role_id': target_role_id,
            'preserve_prosody': preserve_prosody,
            'output': str(output),
            'model_dir': str(self.model_dir),
            'source_sha256': source_sha,
            'target_sha256': target_sha,
        }
        try:
            request_path.write_text(
                json.dumps(request, ensure_ascii=False, indent=2, allow_nan=False),
                encoding='utf-8')
        except (OSError, UnicodeError, ValueError) as exc:
            raise VoiceConversionError(
                f'无法写入语音转换请求文件：{request_path}；{exc}') from exc

        start = time.perf_counter()
        try:
            raw = self._run(request_path, result_path, log_path, cancel)
            _validate_result_metadata(
                raw, output=output, model_dir=self.model_dir,
                preserve_prosody=preserve_prosody, source_sha=source_sha,
                target_sha=target_sha, log_path=log_path)
            sample_rate, duration, output_sha = _validate_output(output, raw, log_path)
            _verify_unchanged(source, source_sha, '源音频', log_path)
            _verify_unchanged(target, target_sha, '目标角色音频', log_path)
        except VoiceConversionError:
            output.unlink(missing_ok=True)
            raise
        except (OSError, ValueError, TypeError, subprocess.SubprocessError) as exc:
            output.unlink(missing_ok=True)
            raise VoiceConversionError(f'语音转换失败：{exc}；日志：{log_path}') from exc

        warnings = [
            '实验性语音转换可能改变情绪、语速和咬字，不是完美的音色替换；'
            '每条结果都必须试听确认。',
            '请确认源音频与目标说话者的使用授权；公开发布前核实模型许可证和声音权利。',
        ]
        if not preserve_prosody:
            warnings.append(
                '已关闭 preserve_prosody：外部脚本可能进一步改变韵律和节奏，请重点试听。')
        return {
            'schema_version': SCHEMA_VERSION,
            'backend': self.spec.name,
            'experimental': True,
            'target_role_id': target_role_id,
            'source_audio': str(source),
            'source_sha256': source_sha,
            'target_audio': str(target),
            'target_sha256': target_sha,
            'output_audio': str(output),
            'output_sha256': output_sha,
            'sample_rate': sample_rate,
            'duration': duration,
            'preserve_prosody': preserve_prosody,
            'model_dir': str(self.model_dir),
            'model_name': self.spec.model_name,
            'model_revision': self.spec.model_revision,
            'preserves_content': self.spec.preserves_content,
            'license_note': self.spec.license_note,
            'warnings': warnings,
            'request_file': str(request_path),
            'result_file': str(result_path),
            'log': str(log_path),
            'elapsed': time.perf_counter() - start,
            'raw': raw,
        }

    def _source_path(self, value):
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise VoiceConversionError('source_audio 必须是本机音频路径。')
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise VoiceConversionError(
                f'源音频不存在：{path}。请提供本机可读的音频文件。')
        return path

    def _output_directory(self, value):
        if value is None:
            try:
                return (Path(get_config().outputs_dir) / 'vc').expanduser().resolve()
            except (AttributeError, OSError, TypeError, ValueError) as exc:
                raise VoiceConversionError(
                    f'无法读取输出目录配置：{exc}；请显式传入 output_dir。') from exc
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise ValueError('output_dir 必须是本机目录路径或 None。')
        return Path(value).expanduser().resolve()

    def _run(self, request_path, result_path, log_path, cancel):
        if cancel is not None and cancel.is_set():
            raise VoiceConversionError(
                f'语音转换已取消；未启动外部进程。日志：{log_path}')
        command = [str(self.python), '-B', str(self.script),
                   '--request', str(request_path), '--result', str(result_path)]
        env = dict(os.environ, PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   HF_HUB_DISABLE_TELEMETRY='1')
        flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        process = None
        try:
            with log_path.open('w', encoding='utf-8') as log:
                process = subprocess.Popen(
                    command, cwd=str(self.model_dir), env=env,
                    stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
                deadline = time.monotonic() + TIMEOUT_SECONDS
                while process.poll() is None:
                    if cancel is not None and cancel.is_set():
                        _kill_process(process)
                        raise VoiceConversionError(
                            '语音转换已取消；已停止外部进程并清理本次未完成的输出。'
                            f'日志：{log_path}')
                    if time.monotonic() >= deadline:
                        _kill_process(process)
                        raise VoiceConversionError(
                            f'语音转换超过 {TIMEOUT_SECONDS:g} 秒；'
                            '已停止外部进程并清理本次未完成的输出。'
                            f'日志：{log_path}')
                    time.sleep(.1)
            returncode = process.returncode
        except OSError as exc:
            raise VoiceConversionError(
                f'无法启动外部语音转换进程：{exc}；日志：{log_path}') from exc
        finally:
            if process is not None and process.poll() is None:
                _kill_process(process)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
        if returncode != 0:
            raise VoiceConversionError(
                f'外部语音转换进程失败（退出码 {returncode}）：'
                f'{_result_error(result_path)}；日志：{log_path}')
        raw = _read_json(result_path, '外部语音转换结果', log_path=log_path)
        if not isinstance(raw, dict):
            raise VoiceConversionError(
                f'外部语音转换结果必须是 JSON 对象；日志：{log_path}')
        if raw.get('error'):
            raise VoiceConversionError(
                f'外部语音转换返回错误：{raw["error"]}；日志：{log_path}')
        return raw


#: 进程内 VC 后端登记表；本发行版不预置任何模型，默认是空的。
VC_REGISTRY: dict[str, VoiceConversionBackend] = {}


def register_vc_backend(backend):
    """Register one process-local VC backend; duplicate names are rejected."""
    if not isinstance(backend, VoiceConversionBackend):
        raise VoiceConversionError('VC 后端必须实现 VoiceConversionBackend 接口。')
    name = backend.spec.name
    if name in VC_REGISTRY:
        raise VoiceConversionError(f'VC 后端名称重复：{name}；名称必须唯一。')
    VC_REGISTRY[name] = backend
    return backend


def default_vc_backends():
    """Return registered VC backends; this release ships no VC model."""
    return list(VC_REGISTRY.values())


def vc_status_markdown(backends=None) -> str:
    """Chinese, honest readiness guidance for the experimental VC stage."""
    selected = default_vc_backends() if backends is None else list(backends)
    lines = [
        '### 实验性语音转换（VC）',
        '',
        '- **状态**：实验阶段，独立于 TTS 主流程；本发行版不包含、不下载任何 VC 模型。',
        '- **能力边界**：可能改变情绪、语速和咬字，不是完美的音色替换；每条结果都必须试听。',
        '- **使用前**：自行确认模型许可证、声音授权与显存需求；一次只运行一个重模型。',
        '- **范围**：仅支持第一阶段本机离线实验，不接入正式生成流程，也不保证跨环境一致。',
        '- **数据**：只在本机读写，不上传音频，也不会把音频复制到配置的输出目录之外。',
    ]
    if not selected:
        lines.append(
            '- **当前配置**：未登记外部 VC 后端；请用 ExternalVoiceConversionBackend '
            '指定本机 Python、脚本和模型目录。')
        return '\n'.join(lines)
    lines.extend(['', '| 后端 | 状态 | 说明 |', '| --- | --- | --- |'])
    for backend in selected:
        if not isinstance(backend, VoiceConversionBackend):
            raise ValueError('backends 必须包含 VoiceConversionBackend 实例。')
        try:
            ok, reason = backend.available()
        except Exception as exc:  # status must stay readable even for a broken plugin
            ok, reason = False, f'检查失败：{exc}'
        lines.append(f'| {backend.spec.name} | {"可用" if ok else "不可用"} | {reason} |')
    return '\n'.join(lines)


__all__ = [
    'SCHEMA_VERSION', 'TIMEOUT_SECONDS', 'MAX_AUDIO_SECONDS',
    'MIN_OUTPUT_SAMPLE_RATE', 'MAX_OUTPUT_SAMPLE_RATE',
    'VoiceConversionError', 'VoiceConversionUnavailable',
    'VoiceConversionSpec', 'VoiceConversionBackend',
    'ExternalVoiceConversionBackend',
    'VC_REGISTRY', 'register_vc_backend', 'default_vc_backends',
    'vc_status_markdown',
]

