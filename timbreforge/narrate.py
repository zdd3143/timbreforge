"""P1-A：自然语言朗读（旁白）模块。

用途：生成音色时，先用中文念出"描述 + 关键参数"，再播放音色，
      让人能明确知道自己在听什么（解决"听不出特点"）。

后端顺序：
  1. edge-tts（音质好，需联网）
  2. Windows SAPI（离线兜底，zh-CN 语音）
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile


def _edge_tts(text: str, out_path: str, voice: str = "zh-CN-XiaoxiaoNeural",
              rate: str = "+0%") -> bool:
    """用 edge-tts 合成语音（异步库，包一层）。"""
    try:
        import edge_tts
    except ImportError:
        return False

    async def _run():
        comm = edge_tts.Communicate(text, voice=voice, rate=rate)
        await comm.save(out_path)

    try:
        asyncio.run(_run())
        return os.path.exists(out_path) and os.path.getsize(out_path) > 1000
    except Exception:
        return False


def _sapi(text: str, out_path: str, rate: int = 0) -> bool:
    """Windows SAPI 离线合成（中文语音 Huihui）。"""
    ps = f'''
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.Rate = {rate}
$zh = $s.GetInstalledVoices() | Where-Object {{ $_.VoiceInfo.Culture.Name -like 'zh*' }} | Select-Object -First 1
if ($zh) {{ $s.SelectVoice($zh.VoiceInfo.Name) }}
$s.SetOutputToWaveFile('{out_path}')
$s.Speak([Console]::In.ReadToEnd())
$s.Dispose()
'''
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                           input=text.encode("utf-8"),
                           capture_output=True, timeout=60)
        return r.returncode == 0 and os.path.exists(out_path) \
            and os.path.getsize(out_path) > 1000
    except Exception:
        return False


def narrate(text: str, out_path: str | None = None,
            prefer: str = "edge") -> str | None:
    """把一句中文念出来，返回 wav 路径；失败返回 None。"""
    out_path = out_path or os.path.join(tempfile.gettempdir(), "tf_narration.wav")
    if prefer == "edge":
        if _edge_tts(text, out_path):
            return out_path
        if _sapi(text, out_path):
            return out_path
    else:
        if _sapi(text, out_path):
            return out_path
        if _edge_tts(text, out_path):
            return out_path
    return None


def narration_script(description: str, profile: str = "",
                     extras: str = "") -> str:
    """生成旁白文案：让听者知道要关注什么。"""
    parts = [f"音色描述：{description}。"]
    if profile:
        parts.append(f"感知特征：{profile}。")
    if extras:
        parts.append(extras)
    return "".join(parts)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    txt = "音色描述：温暖厚实的复古模拟低音。这是测试旁白。"
    for prefer in ("edge", "sapi"):
        p = narrate(txt, f"_narration_{prefer}.wav", prefer=prefer)
        print(f"{prefer:6s} -> {p}  "
              + (f"{os.path.getsize(p)} bytes" if p else "失败"))
