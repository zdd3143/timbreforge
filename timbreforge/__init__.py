"""TimbreForge —— 自然语言生成音色（中文）。

最小用法：
    from timbreforge import build_patch, render, extract
    p = build_patch("温暖厚实的复古模拟低音", seed=42)
    audio = render(p)
"""

from .patch import (  # noqa: F401
    MACRO_NAMES,
    DEFAULT_MACROS,
    FAMILIES,
    Patch,
    apply_deltas,
    randomize,
    to_dict,
)
from .lexicon import (  # noqa: F401
    LEXICON,
    build_patch,
    explain,
    parse_description,
)
from .engine import (  # noqa: F401
    SR,
    midi_to_hz,
    render,
    render_patch,
    render_preview,
    render_voice_patch,
    svf_bandpass,
)
from .voice import (  # noqa: F401
    GENDER,
    VOWEL_FORMANTS,
    VOICE_CHARACTER,
    profile_hint,
    render_voice,
)
from .perception import (  # noqa: F401
    FEATURE_NAMES,
    VOICE_FEATURE_NAMES,
    compare,
    distance,
    extract,
    extract_voice,
    profile_text,
    profile_voice_text,
)

__version__ = "0.6.0a1"
__all__ = [
    "Patch", "build_patch", "render", "render_patch", "render_voice_patch",
    "render_preview", "render_voice", "extract", "extract_voice",
    "profile_text", "profile_voice_text", "compare", "distance", "explain",
    "MACRO_NAMES", "FEATURE_NAMES", "VOICE_FEATURE_NAMES",
    "apply_deltas", "randomize", "to_dict",
    "LEXICON", "FAMILIES", "DEFAULT_MACROS", "midi_to_hz", "SR",
    "VOWEL_FORMANTS", "VOICE_CHARACTER", "GENDER", "profile_hint",
    "svf_bandpass",
]

# Helm 导出（延迟导入，避免没取回资产时报错）
try:
    from .helm import (  # noqa: F401
        export_preset,
        export_state_blob,
        helm_available,
        patch_to_helm_params,
        patch_to_helm_preset,
    )
    __all__ += ["export_preset", "export_state_blob", "helm_available",
                "patch_to_helm_params", "patch_to_helm_preset"]
except Exception:  # pragma: no cover - 资产缺失时不影响主链路
    pass

# 角色声线（主线）：数据集检索 + IndexTTS 合成
try:
    from .voicelib import (  # noqa: F401
        LEXICON as VOICE_LEXICON,
        age_to_params,
        describe_params,
        parse_voice_text,
    )
    from .voicebank import (  # noqa: F401
        VoiceBank,
        VoiceEntry,
        find_voice,
        get_bank,
    )
    __all__ += ["VoiceBank", "VoiceEntry", "find_voice", "get_bank",
                "parse_voice_text", "describe_params", "age_to_params",
                "VOICE_LEXICON"]
except Exception:  # pragma: no cover - 数据集未就绪时不影响其它能力
    pass

