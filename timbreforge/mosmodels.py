"""TimbreForge —— 论文级 MOS 指标推理（UTMOS22 / DNSMOS）

用社区维护的**原生 PyTorch 移植版**（只需 torch/torchaudio/transformers），
避免依赖 speechmos / onnxruntime：

  * UTMOS22 strong learner —— 自然度 MOS 预测（MIT）
    `prj-beatrice/utmos22-torch-native`，数值与官方 `tarepan/SpeechMOS` 一致
  * DNSMOS P.835 / P.808 —— 语音质量四指标（推理 MIT，权重 CC-BY-4.0）
    `prj-beatrice/dnsmos-torch-native`，数值与 Microsoft 官方 ONNX 一致

必须在 **ComfyUI venv** 里运行（需要 transformers + torch）。
本模块通过子进程调用 `timbreforge/workers/mos_infer.py`，与 CAM++ 打分同一模式。
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from timbreforge.config import get_config

_MOS_SCRIPT = Path(__file__).resolve().parent / "workers" / "mos_infer.py"


def __getattr__(name: str):
    # Preserve the former constants as lazy compatibility attributes.
    cfg = get_config()
    if name == "VENV_PY":
        return str(cfg.scorer_python)
    if name == "MOS_SCRIPT":
        return _MOS_SCRIPT
    if name == "MODEL_DIR":
        return cfg.mos_models_dir
    raise AttributeError(name)


def mos_available() -> tuple[bool, str]:
    python = Path(globals().get("VENV_PY", str(get_config().scorer_python)))
    script = Path(globals().get("MOS_SCRIPT", _MOS_SCRIPT))
    model_dir = Path(globals().get("MODEL_DIR", get_config().mos_models_dir))
    if not python.exists():
        return False, f"未找到 ComfyUI venv: {python}"
    if not script.exists():
        return False, f"未找到 {script}"
    missing = []
    for name, need in (("utmos22", "model.safetensors"), ("dnsmos", "model.safetensors")):
        if not (model_dir / name / need).exists():
            missing.append(name)
    if missing:
        return False, f"缺少模型权重: {missing}（请运行 _mos_dl.py 下载）"
    return True, "UTMOS22 + DNSMOS 可用"


def score_files(paths: list[str], device: str = "cpu",
                timeout: int = 1800) -> list[dict]:
    """对多个音频文件打分。返回 [{"file", "utmos", "dnsmos_ovrl", ...}]。"""
    ok, msg = mos_available()
    if not ok:
        raise RuntimeError(msg)
    with tempfile.TemporaryDirectory() as td:
        pin = Path(td) / "files.json"
        pout = Path(td) / "out.json"
        pin.write_text(json.dumps(paths, ensure_ascii=False), encoding="utf-8")
        from timbreforge.resources import model_slot
        with model_slot(getattr(get_config(), 'outputs_dir', Path(td))):
            r = subprocess.run(
                [str(globals().get("VENV_PY", get_config().scorer_python)),
                 str(globals().get("MOS_SCRIPT", _MOS_SCRIPT)),
                 "--files", str(pin), "--out", str(pout), "--device", device,
                 "--model-dir", str(globals().get("MODEL_DIR", get_config().mos_models_dir))],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
        if r.returncode != 0 or not pout.exists():
            raise RuntimeError(
                f"MOS 打分失败（code={r.returncode}）：{(r.stderr or '')[-800:]}")
        return json.loads(pout.read_text(encoding="utf-8"))


def score_naturalness(path: str, device: str = "cpu") -> dict:
    """只要自然度（UTMOS）。"""
    rec = score_files([path], device=device)[0]
    return {"utmos": rec.get("utmos")}


def score_quality(path: str, device: str = "cpu") -> dict:
    """只要质量（DNSMOS 四指标）。"""
    rec = score_files([path], device=device)[0]
    return {k: v for k, v in rec.items()
            if k.startswith("dnsmos")}


# ---------------------------------------------------------------- 综合分
def combined_score(sim100: float, utmos: float,
                   dnsmos_ovrl: float | None = None) -> dict:
    """把"相似度 + 自然度 + 质量"合成为可排序的综合分（论文双指标思路）。

    * sim100     : CAM++ 相似度映射的 0~100（像不像参考）
    * utmos      : 自然度 MOS（约 1~5，常见 2~4.5）
    * dnsmos_ovrl: 整体质量（约 1~5）
    """
    u = float(utmos) if utmos is not None else 0.0
    nat = max(0.0, min((u - 1.0) / 3.5, 1.0)) * 100.0     # 1~4.5 -> 0~100
    if dnsmos_ovrl is None:
        total = 0.5 * sim100 + 0.5 * nat
    else:
        q = max(0.0, min((float(dnsmos_ovrl) - 1.0) / 3.5, 1.0)) * 100.0
        total = 0.45 * sim100 + 0.40 * nat + 0.15 * q
    return {"sim100": round(sim100, 1), "naturalness100": round(nat, 1),
            "combined": round(total, 1)}


def grade_combined(total: float) -> str:
    if total >= 80:
        return "优秀"
    if total >= 65:
        return "良好"
    if total >= 50:
        return "可用"
    if total >= 35:
        return "偏弱"
    return "不可用"


def _self_test() -> None:
    import io
    import sys

    sys.stdout = io.TextIOWrapper(open("_mos_selftest.txt", "wb"), encoding="utf-8")
    ok, msg = mos_available()
    print(f"可用性: {ok}  ({msg})")
    print("\n=== 综合分映射自检 ===")
    print(f"{'相似度':>7s} {'UTMOS':>7s} {'DNSMOS':>7s} -> {'综合':>7s}  档位")
    for sim, u, d in ((90, 4.0, 4.0), (70, 3.5, 3.8), (70, 2.5, 3.0),
                      (50, 3.0, 3.2), (30, 2.0, 2.5)):
        c = combined_score(sim, u, d)
        print(f"{sim:7.0f} {u:7.1f} {d:7.1f} -> {c['combined']:7.1f}  "
              f"{grade_combined(c['combined'])}")
    if not ok:
        print("\n模型未就绪，跳过真实推理测试")
        sys.stdout.flush()
        return
    print("\n=== 真实推理测试 ===")
    out = Path("outputs_voicegen")
    files = []
    for stem in ("new_5_空灵飘渺女声", "new_2_睿智沉稳老者"):
        for suf in ("_参考", "_合成"):
            p = out / f"{stem}{suf}.wav"
            if p.exists():
                files.append(str(p))
    if files:
        try:
            recs = score_files(files)
            for r in recs:
                print(f"  {Path(r['file']).name:34s} UTMOS={r.get('utmos')} "
                      f"OVRL={r.get('dnsmos_ovrl')}")
        except Exception as e:
            print(f"  推理失败: {type(e).__name__}: {str(e)[:300]}")
    sys.stdout.flush()


if __name__ == "__main__":
    _self_test()
