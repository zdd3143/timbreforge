"""TimbreForge —— 自动打分模块（S1）

把"像不像参考"变成可复现的分数，供批量抽卡自动挑选。

两级实现：
  1) **CAM++ 说话人嵌入余弦**（主）—— 测参考身份相似度，不是文字-声音 CLIP
     通过 `timbreforge/workers/campplus_sim.py` 在 ComfyUI venv 里跑（需要 torch/indextts）
  2) **MFCC 手工指标**（兜底）—— 无 torch 环境时使用，详见 simmetric.py
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from timbreforge.config import get_config

_CAMPP_SCRIPT = Path(__file__).resolve().parent / "workers" / "campplus_sim.py"


def __getattr__(name: str):
    # Retain the old module attributes for callers that inspect/patch them,
    # while consulting current config on every access (e.g. after --config).
    if name == "VENV_PY":
        return str(get_config().scorer_python)
    if name == "CAMPP_SCRIPT":
        return _CAMPP_SCRIPT
    raise AttributeError(name)


def _configured_path(name: str, fallback: Path | None = None) -> Path | None:
    value = getattr(get_config(), name, None)
    return Path(value).expanduser() if value else fallback


def campplus_available() -> tuple[bool, str]:
    python = Path(globals().get("VENV_PY", str(get_config().scorer_python)))
    script = Path(globals().get("CAMPP_SCRIPT", _CAMPP_SCRIPT))
    if not python.exists():
        return False, f"未找到 ComfyUI venv: {python}"
    if not script.exists():
        return False, f"未找到 {script}"
    return True, "CAM++ 可用（经 ComfyUI venv）"


def score_pairs(pairs: list[dict], device: str = "cpu",
                timeout: int = 1800) -> list[dict]:
    """对 [(label, ref, test)] 批量算相似度，一次调用进程（模型只加载一次）。

    返回 [{"label", "ref", "test", "cosine"}]，cosine ∈ [-1, 1]，同人通常 ≥0.6。
    """
    ok, msg = campplus_available()
    if not ok:
        raise RuntimeError(msg)
    with tempfile.TemporaryDirectory() as td:
        pin = Path(td) / "pairs.json"
        pout = Path(td) / "out.json"
        pin.write_text(json.dumps(pairs, ensure_ascii=False), encoding="utf-8")
        command = [str(globals().get("VENV_PY", get_config().scorer_python)),
                   str(globals().get("CAMPP_SCRIPT", _CAMPP_SCRIPT)),
                   "--pairs", str(pin), "--out", str(pout), "--device", device]
        for flag, field in (("--indextts-repo", "indextts_repo"),
                            ("--checkpoint", "campplus_checkpoint")):
            value = _configured_path(field)
            if value is not None:
                command.extend([flag, str(value)])
        from timbreforge.resources import model_slot
        with model_slot(getattr(get_config(), 'outputs_dir', Path(td))):
            r = subprocess.run(
                command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
        if r.returncode != 0 or not pout.exists():
            raise RuntimeError(
                f"CAM++ 打分失败（code={r.returncode}）：{(r.stderr or '')[-600:]}")
        return json.loads(pout.read_text(encoding="utf-8"))


def score_one(ref: str, test: str, label: str = "pair", device: str = "cpu") -> float:
    """单个对比的相似度。"""
    return score_pairs([{"label": label, "ref": ref, "test": test}],
                       device=device)[0]["cosine"]


def cosine_to_100(c: float) -> float:
    """把 CAM++ 余弦映射到便于阅读的 0~100 分。

    校准依据：实测同人 0.68~0.89、异人 ≈0、自比 1.0。
    映射到 0~100 时以 0.0 -> 0、1.0 -> 100 的线性为主，
    但把 0.6 以下强烈压缩，避免"完全不像"也拿高分。
    """
    c = float(min(max(c, 0.0), 1.0))
    if c <= 0.05:
        return 0.0
    if c < 0.6:
        return (c - 0.05) / 0.55 * 40.0          # 0.05~0.6  -> 0~40
    return 40.0 + (c - 0.6) / 0.4 * 60.0          # 0.6~1.0   -> 40~100


def grade(c: float) -> str:
    """文字档位（便于快速浏览）。"""
    v = cosine_to_100(c)
    if v >= 80:
        return "很像"
    if v >= 65:
        return "接近"
    if v >= 45:
        return "一般"
    if v >= 20:
        return "偏弱"
    return "不像"


def _self_test() -> None:
    import io
    import sys

    sys.stdout = io.TextIOWrapper(open("_score_selftest.txt", "wb"), encoding="utf-8")
    ok, msg = campplus_available()
    print(f"CAM++ 可用性: {ok}  ({msg})")
    print("\n=== 余弦 -> 100 分 映射 ===")
    for c in (-0.1, 0.0, 0.2, 0.4, 0.6, 0.68, 0.75, 0.83, 0.89, 0.95, 1.0):
        print(f"  cosine {c:+.2f} -> {cosine_to_100(c):6.1f} 分  ({grade(c)})")

    if not ok:
        return
    print("\n=== 真实文件打分（沿用已有产物）===")
    out = Path("outputs_voicegen")
    pairs = []
    for stem in ("new_1_冰冷沙哑成熟女声", "new_5_空灵飘渺女声"):
        r = out / f"{stem}_参考.wav"
        s = out / f"{stem}_合成.wav"
        if r.exists() and s.exists():
            pairs.append({"label": stem, "ref": str(r), "test": str(s)})
    if len(pairs) >= 2:
        pairs.append({"label": "交叉对照", "ref": pairs[0]["ref"],
                      "test": pairs[1]["test"]})
    if pairs:
        res = score_pairs(pairs)
        for x in sorted(res, key=lambda z: -z["cosine"]):
            print(f"  {x['label']:26s} cosine={x['cosine']:+.4f}  "
                  f"-> {cosine_to_100(x['cosine']):5.1f} 分 ({grade(x['cosine'])})")
    sys.stdout.flush()


if __name__ == "__main__":
    _self_test()
