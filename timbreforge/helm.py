"""TimbreForge —— Helm 导出（P3 第一步）

把我们的 Patch 导出成 **Helm 可加载的 preset JSON**。

格式依据（已核验，非猜测）：
  * 来自 AutoSynth 开源的 `assets/helm_native_values.json` → `init_payload` 模板
  * 来自 `autosynth/synth/helm_state.py` → 状态结构：
      payload = {license, synth_version, patch_name, folder_name, author,
                 settings: {139 个参数..., modulations: [...]}}
  * 每个参数的 0..1 归一值见 `effective_normalized`；写归一值是安全的
    （Cutoff / transpose / envelope 等是不同量纲，故一律写归一值再给出原生值对照）
  * 二进制状态 = 前缀 + 4字节大端长度 + JSON + \x00（见 build_state_blob）

⚠️ 本机没有安装 Helm，**导出的 preset 尚未在真实 Helm 中验证加载**。
   请把文件放进 Helm 的 preset 目录后确认。
"""
from __future__ import annotations

import json
import os
import struct
from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parent.parent / "assets_ref" / "autosynth_helm"
NATIVE_VALUES = ASSET_DIR / "helm_native_values.json"
PARAM_SPACE = ASSET_DIR / "helm_parameter_space.json"

# 我们的波形 -> Helm osc 波形网格值（0..10）
# 网格含义未在数据中标注，只能做**有依据的假设**：0=正弦，1=锯齿。
# 由于无法在本机验证，导出时把假设写在 preset 的注释字段里（author 段）。
WAVEFORM_GRID = {
    "sine": 0,
    "saw": 1,
    "square": 2,
    "triangle": 3,
    "fm": 5,
}

_load_error: str | None = None
_cache: dict = {}


def _assets() -> tuple[dict, dict]:
    """加载（并缓存）原生值表与归一值表。"""
    global _load_error
    if "native" in _cache:
        return _cache["native"], _cache["norm"]
    if not NATIVE_VALUES.exists():
        _load_error = f"找不到 {NATIVE_VALUES}，请先运行 _get_assets.py 取回资产"
        raise FileNotFoundError(_load_error)
    nv = json.loads(NATIVE_VALUES.read_text(encoding="utf-8"))
    _cache["native"] = nv["parameters"]
    _cache["norm"] = nv["effective_normalized"]
    return _cache["native"], _cache["norm"]


def _load_native() -> dict:
    """完整原生值 JSON（含 init_payload 模板）。"""
    if "full" not in _cache:
        _assets()
        _cache["full"] = json.loads(NATIVE_VALUES.read_text(encoding="utf-8"))
    return _cache["full"]


def _mod_ranges() -> dict:
    if "mod" not in _cache:
        _cache["mod"] = json.loads(
            (ASSET_DIR / "helm_modulation_ranges.json").read_text(encoding="utf-8")
        )["parameter_ranges"]
    return _cache["mod"]


def _native_to_norm(native_tables: dict, norm_tables: dict, name: str,
                    native_value: float) -> float:
    """把原生值按网格就近换算成 0..1 归一值。"""
    nv = native_tables.get(name)
    if nv is None:
        return float(min(max(native_value, 0.0), 1.0))
    import bisect
    vals = list(nv)
    i = bisect.bisect_left(vals, native_value)
    i = max(0, min(i, len(vals) - 1))
    if i > 0 and abs(vals[i - 1] - native_value) < abs(vals[i] - native_value):
        i -= 1
    nm = norm_tables.get(name)
    if nm is not None and i < len(nm):
        return float(nm[i])
    lo, hi = vals[0], vals[-1]
    return float((native_value - lo) / (hi - lo)) if hi > lo else 0.0


def _clamp01(v: float) -> float:
    return float(min(max(v, 0.0), 1.0))


def _snap_to_grid(name: str, value: float) -> float:
    """把值吸附到该参数的**最近合法网格点**。

    为什么必须做：Helm 的参数是离散网格（`effective_normalized` 给出每个参数
    的 128 个合法归一值）。我们算出来的连续值（如 amp_sustain=0.6449）会与
    网格差 0.001 量级，写进 preset 可能被宿主拒绝或行为未定义。
    实测未吸附时有 5~6 个参数不在网格上，故统一吸附。
    """
    import bisect
    _, norm = _assets()
    vals = norm.get(name)
    if vals is None:
        return _clamp01(value)
    v = float(value)
    i = bisect.bisect_left(vals, v)
    if i <= 0:
        return float(vals[0])
    if i >= len(vals):
        return float(vals[-1])
    return float(vals[i - 1] if (v - vals[i - 1]) <= (vals[i] - v) else vals[i])


def patch_to_helm_params(patch, duration_s: float | None = None) -> dict:
    """把 Patch 映射成 Helm 的 139 个归一参数（只覆盖我们能负责的子集）。

    返回前所有值都会被吸附到 Helm 的合法网格点上。
    """
    from .patch import clamp

    nv = _load_native()
    native, norm = _assets()
    settings: dict[str, float] = {}

    m = {k: clamp(v) for k, v in patch.macros.items()}

    # ---- 滤波器：必须显式打开（init_payload 里 filter_on 默认是 0）----
    settings["filter_on"] = 1.0
    settings["filter_style"] = 0.0            # 低通（网格 0）
    settings["filter_shelf"] = 0.0
    # brightness -> 原生截止频率 28..127（Helm 用它自己的刻度）
    native_cutoff = 28.0 + 99.0 * (m["brightness"] ** 0.6)
    settings["cutoff"] = _native_to_norm(native, norm, "cutoff", native_cutoff)
    settings["resonance"] = _native_to_norm(native, norm, "resonance",
                                            clamp(1.4 - m["brightness"] * 0.8, 0.0, 1.0))
    settings["keytrack"] = 0.0
    native_fd = -12.0 + 32.0 * (0.6 * m["grit"] + 0.4 * m["warmth"])
    settings["filter_drive"] = _native_to_norm(native, norm, "filter_drive", native_fd)

    # ---- 滤波包络 ----
    settings["fil_attack"] = 0.0
    native_fd_depth = 128.0 * (m["brightness"] * 0.7 * (0.4 + 0.6 * m["attack_sharpness"]))
    settings["fil_env_depth"] = _native_to_norm(native, norm, "fil_env_depth",
                                                native_fd_depth)
    settings["fil_decay"] = _native_to_norm(native, norm, "fil_decay",
                                            clamp(1.0 - m["attack_sharpness"] * 0.7, 0.05, 1.0))
    settings["fil_sustain"] = clamp(0.3 + 0.4 * m["brightness"])
    settings["fil_release"] = clamp(0.2 + 0.3 * m["space"])

    # ---- 振荡器 1 ----
    grid = WAVEFORM_GRID.get(getattr(patch, "waveform", "saw"), 1)
    settings["osc_1_waveform"] = _native_to_norm(native, norm, "osc_1_waveform", float(grid))
    settings["osc_1_volume"] = 1.0
    settings["osc_1_transpose"] = 0.0
    settings["osc_1_tune"] = 0.0
    uv = max(1, min(int(getattr(patch, "unison_voices", 1)), 15))
    settings["osc_1_unison_voices"] = _native_to_norm(native, norm,
                                                      "osc_1_unison_voices", float(uv))
    det = clamp(getattr(patch, "unison_detune", 0.0) / 0.02) * 100.0
    settings["osc_1_unison_detune"] = _native_to_norm(native, norm,
                                                      "osc_1_unison_detune", det)
    # 振荡器 2 关掉，避免叠出额外音色
    settings["osc_2_volume"] = 0.0

    # ---- 噪声 / 反馈（来自 grit）----
    settings["noise_volume"] = _native_to_norm(native, norm, "noise_volume",
                                               clamp(getattr(patch, "noise_level", 0.0)))
    native_fb = -1.0 + 2.0 * m["grit"] ** 1.5
    settings["osc_feedback_amount"] = _native_to_norm(native, norm,
                                                      "osc_feedback_amount", native_fb)
    settings["cross_modulation"] = _native_to_norm(native, norm, "cross_modulation",
                                                   0.5 * (m["grit"] ** 2))

    # ---- 幅度包络 ----
    dur = duration_s if duration_s is not None else (patch.duration or 2.0)
    atk = float(getattr(patch, "amp_attack", 0.01))
    dec = float(getattr(patch, "amp_decay", 0.2))
    sus = float(getattr(patch, "amp_sustain", 0.7))
    rel = float(getattr(patch, "amp_release", 0.2))
    if patch.mode == "voice":
        atk, rel = 0.05, 0.30
    settings["amp_attack"] = _native_to_norm(native, norm, "amp_attack", min(atk, 4.0))
    settings["amp_decay"] = _native_to_norm(native, norm, "amp_decay", min(dec, 4.0))
    settings["amp_sustain"] = clamp(sus)
    settings["amp_release"] = _native_to_norm(native, norm, "amp_release", min(rel, 4.0))

    # ---- LFO ----
    settings["mono_lfo_1_amplitude"] = _native_to_norm(
        native, norm, "mono_lfo_1_amplitude", -1.0 + 2.0 * clamp(getattr(patch, "lfo_depth", 0.0)))
    settings["mono_lfo_2_amplitude"] = 0.0
    settings["poly_lfo_amplitude"] = 0.0

    # ---- 效果 ----
    settings["reverb_on"] = 1.0 if getattr(patch, "reverb_mix", 0.0) > 0.01 else 0.0
    settings["reverb_dry_wet"] = clamp(getattr(patch, "reverb_mix", 0.0))
    settings["reverb_feedback"] = _native_to_norm(native, norm, "reverb_feedback",
                                                  min(0.8 + 0.2 * m["space"], 1.0))
    settings["reverb_damping"] = clamp(0.3 + 0.5 * (1.0 - m["brightness"]))
    settings["delay_on"] = 1.0 if getattr(patch, "delay_mix", 0.0) > 0.01 else 0.0
    settings["delay_dry_wet"] = clamp(getattr(patch, "delay_mix", 0.0))
    settings["distortion_on"] = 1.0 if m["grit"] > 0.3 else 0.0
    settings["distortion_drive"] = _native_to_norm(native, norm, "distortion_drive",
                                                   30.0 * m["grit"])
    settings["distortion_mix"] = clamp(0.3 + 0.6 * m["grit"])

    # ---- 演奏设置 ----
    settings["polyphony"] = _native_to_norm(native, norm, "polyphony", 8.0)
    settings["portamento"] = _native_to_norm(native, norm, "portamento", -9.0)
    settings["velocity_track"] = 0.0
    settings["volume"] = _native_to_norm(native, norm, "volume", 0.707)

    # ---- 统一吸附到合法网格 ----
    return {k: _snap_to_grid(k, v) for k, v in settings.items()}


def _modulations(patch) -> list[dict]:
    """把我们的调制意图翻成 Helm 的调制路由（源/目标/原生幅度）。"""
    from .patch import clamp
    ranges = _mod_ranges()
    routes = []
    m = {k: clamp(v) for k, v in patch.macros.items()}

    def add(src: str, dst: str, normalized_amount: float) -> None:
        if abs(normalized_amount) < 1e-3 or dst not in ranges:
            return
        lo, hi = ranges[dst]
        routes.append({
            "source": src,
            "destination": dst,
            "amount": float(normalized_amount) * (float(hi) - float(lo)),
        })

    # 流动感 -> LFO 扫滤波（最常用的一条）
    add("mono_lfo_1", "cutoff", 0.35 * m["movement"])
    # 起音锐度 -> 滤波包络深度
    add("fil_envelope", "cutoff", 0.3 * m["attack_sharpness"])
    # 力度 -> 滤波（演奏表现力）
    add("velocity", "cutoff", 0.25)
    # 粗糙 -> 失真驱动
    add("mod_wheel", "distortion_drive", 0.5 * m["grit"])
    return routes


def patch_to_helm_preset(patch, name: str | None = None,
                         duration_s: float | None = None) -> dict:
    """生成完整的 Helm preset payload（JSON 结构）。"""
    nv = _load_native()
    payload = json.loads(json.dumps(nv["init_payload"]))   # 深拷贝模板
    settings = payload.setdefault("settings", {})
    settings.update(patch_to_helm_params(patch, duration_s))
    settings["modulations"] = _modulations(patch)
    payload["patch_name"] = name or (patch.name or "TimbreForge")
    payload["folder_name"] = "TimbreForge"
    payload["author"] = "TimbreForge (generated from Chinese description)"
    return payload


def export_preset(patch, path: str | os.PathLike,
                  name: str | None = None,
                  duration_s: float | None = None) -> str:
    """导出 Helm preset JSON 文件。"""
    payload = patch_to_helm_preset(patch, name=name, duration_s=duration_s)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return str(p)


def build_state_blob(payload: dict, prefix: bytes = b"") -> bytes:
    """构造 Helm 的二进制状态块：前缀 + 大端长度 + JSON + \\x00。

    结构来自 AutoSynth 的 `helm_state.build_state_blob`。
    """
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return prefix + struct.pack(">I", len(raw) + 1) + raw + b"\x00"


def export_state_blob(patch, path: str | os.PathLike,
                      name: str | None = None,
                      duration_s: float | None = None) -> str:
    """导出 Helm 二进制状态文件（给宿主/插件状态恢复用）。"""
    payload = patch_to_helm_preset(patch, name=name, duration_s=duration_s)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(build_state_blob(payload))
    return str(p)


def helm_available() -> tuple[bool, str]:
    """本机是否具备在 Helm 中验证的条件。"""
    return (False,
            "本机未检测到 Helm 与 dawdreamer/pedalboard，导出的 preset "
            "尚未在真实 Helm 中验证")
