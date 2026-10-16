"""TimbreForge —— P0 验收自检（P0-10）

跑法：
    python -m timbreforge.selftest
    python selftest.py

检验 5 条验收标准（见《TimbreForge-阶段任务.md》P0 验收标准）：
  1. 一句中文能产出有效音频（无 NaN、峰值合理）
  2. 单调性：brightness 宏 0→1 时频谱质心单调上升
  3. 可分性：不同描述的距离 > 同描述不同 seed 的距离
  4. 可复现：同 seed 逐样本一致
  5. 无新增第三方依赖（只用到 numpy / soundfile）
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from timbreforge import (  # noqa: E402
    MACRO_NAMES, Patch, build_patch, extract, render,
)
from timbreforge.engine import SR  # noqa: E402

PASS, FAIL = "[通过]", "[失败]"
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"{PASS if ok else FAIL} {name}" + (f"\n        {detail}" if detail else ""))


def main() -> int:
    print("=" * 74)
    print("TimbreForge P0 验收自检")
    print("=" * 74)

    # ---------------------------------------------------------- 1. 有效音频
    print("\n[1] 中文描述 -> 有效音频")
    t0 = time.time()
    p = build_patch("温暖厚实的复古模拟低音", seed=42)
    audio = render(p)
    el = time.time() - t0
    valid = (audio.dtype == np.float32 and audio.size > SR * 0.2
             and np.isfinite(audio).all())
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    check("音频有效（有限值/长度合理/float32）", valid,
          f"{audio.size} 样本 ({audio.size/SR:.2f}s), 峰值 {peak:.3f}, 渲染 {el*1000:.0f}ms")

    # ---------------------------------------------------------- 2. 单调性
    print("\n[2] brightness 宏单调性（0 -> 1 时质心应单调上升）")
    centroids = []
    for b in (0.0, 0.25, 0.5, 0.75, 1.0):
        q = build_patch("铺底", seed=1, jitter=0.0)
        q.macros["brightness"] = b
        q.expand()
        a = render(q, duration=1.0)
        centroids.append(extract(a)[0])
    mono = all(centroids[i] <= centroids[i + 1] + 1e-6 for i in range(len(centroids) - 1))
    strictly_up = centroids[-1] > centroids[0] * 1.5
    check("质心随亮度单调不减", mono, "质心序列(kHz): " + ", ".join(f"{c:.2f}" for c in centroids))
    check("亮度两端差异显著（>1.5x）", strictly_up,
          f"{centroids[0]:.2f} -> {centroids[-1]:.2f} (x{centroids[-1]/max(centroids[0],1e-9):.2f})")

    # ---------------------------------------------------------- 3. 可分性
    print("\n[3] 描述可分性（不同描述的距离 > 同描述不同 seed 的距离）")
    warm = render(build_patch("温暖厚实的复古模拟低音", seed=100, jitter=0.0), duration=1.5)
    cold = render(build_patch("冰冷尖锐的金属拨弦", seed=100, jitter=0.0), duration=1.5)
    warm_b = render(build_patch("温暖厚实的复古模拟低音", seed=200, jitter=0.12), duration=1.5)

    f_warm, f_cold, f_warm_b = extract(warm), extract(cold), extract(warm_b)
    d_between = float(np.linalg.norm(f_warm - f_cold))
    d_within = float(np.linalg.norm(f_warm - f_warm_b))
    check("不同描述差异 > 同描述 seed 差异", d_between > d_within * 1.5,
          f"描述间距离={d_between:.4f}  同描述 seed 距离={d_within:.4f}  "
          f"比值={d_between/max(d_within,1e-9):.2f}x")
    contrast = {
        "亮度": (f_warm[0], f_cold[0]),
        "噪感": (f_warm[2], f_cold[2]),
    }
    detail = "  ".join(f"{k}: 暖{n1:.2f}/冷{n2:.2f}" for k, (n1, n2) in contrast.items())
    check("方向符合直觉（冷调更亮/更噪）",
          f_cold[0] >= f_warm[0] and f_cold[2] >= f_warm[2] * 0.9, detail)

    # ---------------------------------------------------------- 4. 可复现
    print("\n[4] 可复现性（同 seed 应逐样本一致）")
    a1 = render(build_patch("空灵飘渺的女声铺底", seed=7, jitter=0.15), duration=1.2)
    a2 = render(build_patch("空灵飘渺的女声铺底", seed=7, jitter=0.15), duration=1.2)
    a3 = render(build_patch("空灵飘渺的女声铺底", seed=8, jitter=0.15), duration=1.2)
    check("同 seed 完全一致", a1.shape == a2.shape and np.array_equal(a1, a2),
          f"最大样本差 = {float(np.max(np.abs(a1 - a2))) if a1.shape == a2.shape else float('nan'):.2e}")
    check("不同 seed 确实不同", not np.array_equal(a1, a3),
          f"最大样本差 = {float(np.max(np.abs(a1 - a3))):.2e}")

    # ---------------------------------------------------------- 5. 依赖
    print("\n[5] 依赖检查（不应引入 numpy/soundfile 之外的新依赖）")
    base = {"numpy", "soundfile"}
    loaded = {m.split(".")[0] for m in sys.modules
              if not m.startswith("_") and "." not in m}
    heavy = {"torch", "tensorflow", "librosa", "scipy", "transformers", "onnxruntime"}
    found = heavy & loaded
    check("未意外加载重型依赖", not found, f"检测到: {sorted(found) if found else '无'}")

    # ---------------------------------------------------------- 汇总
    print("\n" + "=" * 74)
    ok_n = sum(1 for _, ok, _ in results if ok)
    print(f"结果：{ok_n}/{len(results)} 项通过")
    for name, ok, _ in results:
        if not ok:
            print(f"  [未通过] {name}")
    if ok_n == len(results):
        print("P0 验收标准全部满足 -> P0 可以结项")
    print("=" * 74)
    return 0 if ok_n == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
