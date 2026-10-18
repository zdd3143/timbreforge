"""统一的本机模型适配层：新增 TTS/VC 模型只实现这里的接口。

本模块只做请求编排、能力校验和结果归一化，不导入 torch，也不在核心进程
加载模型。每个新模型都必须实现 :class:`TTSBackend`（语音转换则实现
:mod:`timbreforge.vc_backend` 中的 ``VoiceConversionBackend``），用
:class:`BackendSpec` 声明真实能力，并在 ``synthesize``/``convert`` 中委托给
独立 worker 进程。适配器**不得绕过** :func:`timbreforge.resources.model_slot`
这一统一资源管理器，也不得自行下载模型、发起网络请求或把用户音频复制到
配置输出目录之外。

界面与生成代码只依赖 :class:`BackendRequest`、:class:`BackendResult` 和
:class:`BackendRegistry`，因此新增或替换后端不需要改动 UI 或生成流程。
能力声明必须诚实：没有实现的能力不要写进 ``capabilities``，例如 Qwen Base
不支持独立表演指令，就不能声称支持 IndexTTS/CosyVoice 的指令语义。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from pathlib import Path
import re

from timbreforge import backend_qwen
from timbreforge.config import get_config

#: 统一能力词表；:class:`BackendSpec` 只能声明这些值。
CAPABILITIES = ('design', 'clone', 'clone_prompt', 'longform', 'vc')
#: 适配器种类；``vc`` 仅用于未来接入统一请求模型的语音转换适配器。
KINDS = ('tts', 'vc')
#: 默认的单次任务超时（秒）；具体后端可以覆盖。
DEFAULT_TIMEOUT = 1800.0
#: Qwen worker 与适配器使用同一个本地脚本。
QWEN_WORKER = Path(__file__).resolve().parent / 'workers' / 'qwen_worker.py'

_NAME_RE = re.compile(r'[a-z0-9][a-z0-9._-]{0,63}\Z')
_CLONE_MODES = ('icl', 'xvector')


class BackendError(RuntimeError):
    """适配层基类错误。"""


class BackendUnavailable(BackendError):
    """后端或本地模型文件当前不可用；本工具不会自动下载或安装。"""


class CapabilityError(ValueError):
    """请求的 mode/能力与后端声明的能力不匹配。"""


def _require_text(value, name, *, label, limit=5000):
    """Validate one user-facing text field with a Chinese message."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{label}的 {name} 必须是非空文本。')
    if len(value) > limit:
        raise ValueError(f'{label}的 {name} 不得超过 {limit} 个字符。')


@dataclass(frozen=True)
class BackendSpec:
    """一个后端的静态身份与能力声明。

    ``capabilities`` 必须是 :data:`CAPABILITIES` 的子集；声明
    ``clone_prompt`` 时必须同时声明 ``clone``，避免出现只能缓存提示词却
    不能克隆的矛盾组合。
    """

    name: str
    kind: str
    capabilities: tuple[str, ...]
    model_name: str
    model_revision: str
    description: str

    def __post_init__(self):
        _require_text(self.name, 'name', label='后端规格', limit=64)
        if not _NAME_RE.fullmatch(self.name):
            raise ValueError(
                f'后端名称 {self.name!r} 无效；只能使用小写字母、数字、点、下划线和连字符，'
                '并以字母或数字开头。')
        if self.kind not in KINDS:
            raise ValueError(f'后端类型必须是 {" 或 ".join(KINDS)}；收到 {self.kind!r}。')
        if not isinstance(self.capabilities, (tuple, list)):
            raise ValueError('capabilities 必须是 tuple[str, ...]。')
        normalized = []
        for capability in self.capabilities:
            if not isinstance(capability, str) or capability not in CAPABILITIES:
                raise ValueError(
                    f'不支持的能力 {capability!r}；可用能力：{"、".join(CAPABILITIES)}。')
            if capability in normalized:
                raise ValueError(f'能力重复：{capability}。')
            normalized.append(capability)
        if not normalized:
            raise ValueError('后端必须至少声明一个能力。')
        if 'clone_prompt' in normalized and 'clone' not in normalized:
            raise ValueError('声明 clone_prompt 能力时必须同时声明 clone。')
        object.__setattr__(self, 'capabilities', tuple(normalized))
        _require_text(self.model_name, 'model_name', label='后端规格', limit=200)
        _require_text(self.model_revision, 'model_revision', label='后端规格', limit=120)
        _require_text(self.description, 'description', label='后端规格', limit=2000)


@dataclass
class BackendRequest:
    """与具体模型无关的一次生成请求。

    ``mode`` 只使用统一词表 ``design``/``clone``；具体后端负责把它映射成
    自己的协议（例如 Qwen 的 ``voice_design``/``voice_clone``）。
    """

    mode: str
    text: str
    params: object
    instruction: str = ''
    ref_audio: str | None = None
    ref_text: str = ''
    clone_prompt: str | None = None
    clone_mode: str = 'icl'
    reference_sha256: str | None = None
    device: str = 'cuda:0'
    dtype: str = 'float16'
    attention: str = 'sdpa'
    timeout: float = DEFAULT_TIMEOUT
    cancel: object | None = None
    adapter_path: str | None = None
    adapter_scale: float = 1.0
    adapter_sha256: str | None = None


@dataclass
class BackendResult:
    """归一化后的生成结果；``raw`` 保留后端原始字典供审计。"""

    audio_path: str
    mode: str
    seed: int
    params: dict
    model_dir: str
    model_config_sha256: str
    clone_mode: str
    clone_prompt_sha256: str | None
    duration: float
    sample_rate: int
    raw: dict


class TTSBackend(ABC):
    """所有文本转语音适配器必须实现的接口。"""

    spec: BackendSpec

    @abstractmethod
    def available(self) -> tuple[bool, str]:
        """只检查本地文件，不加载模型，也不抛异常；返回 ``(是否可用, 中文原因)``。"""

    @abstractmethod
    def synthesize(self, request: BackendRequest) -> BackendResult:
        """执行一次设计或克隆，返回归一化结果。"""

    @abstractmethod
    def create_clone_prompt(self, ref_audio, ref_text, *, clone_mode='icl',
                            device='cuda:0', dtype='float16', attention='sdpa',
                            timeout=DEFAULT_TIMEOUT, output_dir=None,
                            cancel=None) -> dict:
        """构建可复用的克隆条件缓存，返回后端原始结果字典。"""


def _qwen_params(value):
    if not isinstance(value, backend_qwen.QwenParams):
        raise ValueError('Qwen3 后端的 params 必须是 backend_qwen.QwenParams 实例。')
    return value


def _result(raw, request):
    """Validate the adapter result and normalize it without inventing values."""
    if not isinstance(raw, dict):
        raise RuntimeError('Qwen3 后端返回结果不是字典；请检查 worker 日志。')
    required = ('file', 'seed', 'params', 'model_dir', 'model_config_sha256',
                'clone_mode', 'duration', 'sample_rate')
    missing = [key for key in required if key not in raw]
    if missing:
        raise RuntimeError(
            f'Qwen3 后端结果缺少字段：{"、".join(missing)}；请检查 worker 日志。')
    audio_path = raw['file']
    if not isinstance(audio_path, str) or not audio_path.strip():
        raise RuntimeError('Qwen3 后端结果的 file 必须是非空路径文本。')
    seed = raw['seed']
    if type(seed) is not int or seed < 0:
        raise RuntimeError('Qwen3 后端结果的 seed 必须是非负整数。')
    params = raw['params']
    if not isinstance(params, dict):
        raise RuntimeError('Qwen3 后端结果的 params 必须是字典。')
    model_dir = raw['model_dir']
    if not isinstance(model_dir, str) or not model_dir.strip():
        raise RuntimeError('Qwen3 后端结果的 model_dir 必须是非空路径文本。')
    config_hash = raw['model_config_sha256']
    if not isinstance(config_hash, str) or not config_hash.strip():
        raise RuntimeError('Qwen3 后端结果的 model_config_sha256 必须是非空文本。')
    clone_mode = raw['clone_mode']
    if clone_mode not in _CLONE_MODES + ('none',):
        raise RuntimeError('Qwen3 后端结果的 clone_mode 无效。')
    prompt_hash = raw.get('clone_prompt_sha256')
    if prompt_hash is not None and (
            not isinstance(prompt_hash, str) or not prompt_hash.strip()):
        raise RuntimeError('Qwen3 后端结果的 clone_prompt_sha256 必须是非空文本或 null。')
    duration = raw['duration']
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
        raise RuntimeError('Qwen3 后端结果的 duration 必须是有限正数。')
    sample_rate = raw['sample_rate']
    if type(sample_rate) is not int or sample_rate <= 0:
        raise RuntimeError('Qwen3 后端结果的 sample_rate 必须是正整数。')
    return BackendResult(
        audio_path=audio_path,
        # The unified API keeps the request vocabulary; raw['mode'] stays in raw.
        mode=request.mode,
        seed=seed,
        params=dict(params),
        model_dir=model_dir,
        model_config_sha256=config_hash,
        clone_mode=clone_mode,
        clone_prompt_sha256=prompt_hash,
        duration=float(duration),
        sample_rate=sample_rate,
        raw=dict(raw),
    )


class Qwen3Backend(TTSBackend):
    """Qwen3-TTS 0.1.1 适配器；只委托 :mod:`timbreforge.backend_qwen`。

    VoiceDesign 消费提示词设计音色，Base 消费参考音频或 Clone Prompt。
    Qwen Base **没有**独立的表演指令通道，不能把 IndexTTS/CosyVoice 的指令
    或参数名做语义等价映射；``BackendRequest.instruction`` 只对 design 有效。
    """

    spec = BackendSpec(
        name='qwen3',
        kind='tts',
        capabilities=('design', 'clone', 'clone_prompt'),
        model_name='Qwen3-TTS-12Hz-0.6B-Base',
        model_revision=backend_qwen.QWEN_PACKAGE_VERSION,
        description=(
            'Qwen3-TTS 0.1.1 本地适配器：VoiceDesign 用提示词设计音色，'
            'Base 用参考音频或 Clone Prompt 克隆；Base 没有独立表演指令，'
            '指令与参数语义不能直接照搬 IndexTTS/CosyVoice。'),
    )

    # ------------------------------------------------------------ readiness
    def available(self) -> tuple[bool, str]:
        """Check local files only; never loads a model or touches the network."""
        try:
            cfg = get_config()
            python = Path(cfg.qwen_python).expanduser().resolve()
            model = Path(cfg.qwen_base_model).expanduser().resolve()
        except (AttributeError, OSError, TypeError, ValueError, RuntimeError) as exc:
            return False, (f'无法读取 Qwen 本地配置：{exc}。'
                           '请检查配置文件；不会自动下载模型。')
        worker = QWEN_WORKER
        checks = (
            (python, 'Qwen 本地 Python', '[qwen] python'),
            (worker, 'Qwen worker 脚本', '安装包完整性'),
            (model / 'config.json', 'Qwen Base 模型 config.json', '[qwen] base_model'),
        )
        try:
            for path, label, hint in checks:
                if not path.is_file():
                    return False, (f'缺少{label}：{path}。'
                                   f'请检查 {hint} 配置；不会自动下载模型。')
            has_weights = any(model.glob('*.safetensors'))
        except (OSError, ValueError, RuntimeError) as exc:
            return False, (f'无法读取 Qwen 本地文件：{exc}。'
                           '请检查路径权限；不会自动下载模型。')
        if not has_weights:
            return False, (f'Qwen Base 模型目录缺少 *.safetensors 权重：{model}。'
                           '请准备完整的本地模型；不会自动下载。')
        return True, 'Qwen3-TTS 本地 Python、worker 与 Base 模型文件已就绪。'

    # ---------------------------------------------------------- generation
    def _require_capability(self, capability):
        if capability not in self.spec.capabilities:
            raise CapabilityError(
                f'Qwen3 后端不支持 {capability}；声明能力：'
                f'{"、".join(self.spec.capabilities) or "（无）"}。')

    def synthesize(self, request: BackendRequest) -> BackendResult:
        if not isinstance(request, BackendRequest):
            raise ValueError('request 必须是 BackendRequest 实例。')
        mode = request.mode
        if mode == 'design':
            self._require_capability('design')
            if request.clone_prompt is not None:
                raise CapabilityError(
                    '设计模式不接受 Clone Prompt；请改用 clone 模式，或清空 clone_prompt。')
            if request.ref_audio is not None or request.reference_sha256 is not None:
                raise CapabilityError(
                    '设计模式不接受参考音频或参考哈希；请改用 clone 模式。')
            params = _qwen_params(request.params)
            try:
                raw = backend_qwen.design(
                    request.text, request.instruction, params,
                    device=request.device, dtype=request.dtype,
                    attention=request.attention, timeout=request.timeout,
                    cancel=request.cancel)
            except FileNotFoundError as exc:
                raise BackendUnavailable(
                    f'Qwen3 本地模型文件缺失，无法设计音色：{exc}。'
                    '请检查配置；不会自动下载模型。') from exc
            except ValueError as exc:
                raise ValueError(f'Qwen3 设计请求参数无效：{exc}') from exc
            return _result(raw, request)

        if mode == 'clone':
            self._require_capability('clone')
            params = _qwen_params(request.params)
            prompt = request.clone_prompt
            if prompt is not None:
                self._require_capability('clone_prompt')
                if request.ref_audio is not None:
                    raise CapabilityError(
                        '参考克隆不能同时传入 ref_audio 与 clone_prompt；'
                        '请只保留一个声音来源。')
                if request.reference_sha256 is None:
                    raise ValueError(
                        '使用 Clone Prompt 克隆时必须提供 reference_sha256，'
                        '用于校验角色卡音频来源。')
                try:
                    options = dict(device=request.device, dtype=request.dtype,
                                   attention=request.attention, timeout=request.timeout,
                                   clone_prompt=str(prompt), clone_mode=request.clone_mode,
                                   reference_sha256=request.reference_sha256,
                                   cancel=request.cancel)
                    if request.adapter_path is not None:
                        options.update(adapter_path=request.adapter_path,
                                      adapter_scale=request.adapter_scale,
                                      adapter_sha256=request.adapter_sha256)
                    raw = backend_qwen.clone(None, request.ref_text, request.text, params,
                                             **options)
                except FileNotFoundError as exc:
                    raise BackendUnavailable(
                        f'Qwen3 本地 Clone Prompt 或模型文件缺失：{exc}。'
                        '请检查配置；不会自动下载模型。') from exc
                except ValueError as exc:
                    raise ValueError(f'Qwen3 克隆请求参数无效：{exc}') from exc
            else:
                if request.ref_audio is None:
                    raise ValueError('参考克隆需要 ref_audio 或 clone_prompt。')
                try:
                    options = dict(device=request.device, dtype=request.dtype,
                                   attention=request.attention, timeout=request.timeout,
                                   clone_mode=request.clone_mode, cancel=request.cancel)
                    if request.adapter_path is not None:
                        options.update(adapter_path=request.adapter_path,
                                      adapter_scale=request.adapter_scale,
                                      adapter_sha256=request.adapter_sha256)
                    raw = backend_qwen.clone(request.ref_audio, request.ref_text,
                                             request.text, params, **options)
                except FileNotFoundError as exc:
                    raise BackendUnavailable(
                        f'Qwen3 本地参考音频或模型文件缺失：{exc}。'
                        '请检查配置；不会自动下载模型。') from exc
                except ValueError as exc:
                    raise ValueError(f'Qwen3 克隆请求参数无效：{exc}') from exc
            return _result(raw, request)

        raise CapabilityError(f'未知模式 {mode!r}；只支持 design 或 clone。')

    def create_clone_prompt(self, ref_audio, ref_text, *, clone_mode='icl',
                            device='cuda:0', dtype='float16', attention='sdpa',
                            timeout=DEFAULT_TIMEOUT, output_dir=None,
                            cancel=None) -> dict:
        self._require_capability('clone_prompt')
        ok, reason = self.available()
        if not ok:
            raise BackendUnavailable(reason)
        try:
            result = backend_qwen.create_clone_prompt(
                ref_audio, ref_text, clone_mode=clone_mode, device=device,
                dtype=dtype, attention=attention, timeout=timeout,
                output_dir=output_dir, cancel=cancel)
        except FileNotFoundError as exc:
            raise BackendUnavailable(
                f'Qwen3 本地模型文件缺失，无法创建 Clone Prompt：{exc}。'
                '请检查配置；不会自动下载模型。') from exc
        except ValueError as exc:
            raise ValueError(f'Qwen3 Clone Prompt 请求参数无效：{exc}') from exc
        if not isinstance(result, dict):
            raise RuntimeError('Qwen3 后端没有返回有效的 Clone Prompt 字典；请检查 worker 日志。')
        return result


class BackendRegistry:
    """进程内后端登记表；名称唯一，不做隐式回退。"""

    def __init__(self):
        self._backends: dict[str, TTSBackend] = {}

    def register(self, backend: TTSBackend) -> None:
        # Subclassing TTSBackend is the documented route; a structural
        # implementation with the same three methods is accepted so a VC or
        # third-party adapter can be exposed without inheriting TTS names.
        spec = getattr(backend, 'spec', None)
        if not isinstance(spec, BackendSpec):
            raise BackendError('后端必须实现统一适配接口并提供有效的 BackendSpec。')
        if spec.name in self._backends:
            raise BackendError(f'后端名称重复：{spec.name}；名称必须唯一。')
        for method in ('available', 'synthesize', 'create_clone_prompt'):
            if not callable(getattr(backend, method, None)):
                raise BackendError(f'后端 {spec.name} 缺少 {method} 方法。')
        self._backends[spec.name] = backend

    def get(self, name: str) -> TTSBackend:
        if not isinstance(name, str) or not name.strip():
            raise BackendUnavailable('后端名称必须是非空文本。')
        try:
            return self._backends[name]
        except KeyError:
            known = '、'.join(self._backends) or '（无）'
            raise BackendUnavailable(f'未登记的后端：{name}；已登记：{known}。') from None

    def require(self, name: str, capability: str | None = None) -> TTSBackend:
        backend = self.get(name)
        status = backend.available()
        if (not isinstance(status, tuple) or len(status) != 2
                or type(status[0]) is not bool or not isinstance(status[1], str)):
            raise BackendError(
                f'后端 {name} 的 available() 必须返回 (bool, 中文说明)。')
        ok, reason = status
        if not ok:
            raise BackendUnavailable(
                f'后端 {name} 当前不可用：{reason or "未提供原因"}')
        if capability is not None:
            if not isinstance(capability, str) or capability not in CAPABILITIES:
                raise CapabilityError(
                    f'未知能力 {capability!r}；可用能力：{"、".join(CAPABILITIES)}。')
            if capability not in backend.spec.capabilities:
                declared = '、'.join(backend.spec.capabilities) or '（无）'
                raise CapabilityError(
                    f'后端 {name} 不支持 {capability}；声明能力：{declared}。')
        return backend

    def list(self, *, kind: str | None = None) -> list[BackendSpec]:
        if kind is not None and kind not in KINDS:
            raise ValueError(f'kind 必须是 {" 或 ".join(KINDS)}；收到 {kind!r}。')
        return [backend.spec for backend in self._backends.values()
                if kind is None or backend.spec.kind == kind]

    def names(self) -> list[str]:
        return list(self._backends)


def default_registry() -> BackendRegistry:
    """Return the built-in registry without touching files or loading a model."""
    registry = BackendRegistry()
    registry.register(Qwen3Backend())
    return registry


__all__ = [
    'CAPABILITIES', 'KINDS', 'DEFAULT_TIMEOUT', 'QWEN_WORKER',
    'BackendError', 'BackendUnavailable', 'CapabilityError',
    'BackendSpec', 'BackendRequest', 'BackendResult',
    'TTSBackend', 'Qwen3Backend', 'BackendRegistry', 'default_registry',
]

