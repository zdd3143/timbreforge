"""Long-form segmentation, analysis, concatenation and orchestration tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest

import numpy as np
import soundfile as sf

from timbreforge import longform

SR = 24000


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tone(freq=220.0, amp=0.2, seconds=1.0, lead=0.0, tail=0.0, sample_rate=SR):
    """Deterministic sine whose whole-cycle endpoints stay silent."""
    time_axis = np.arange(int(round(seconds * sample_rate))) / sample_rate
    signal = amp * np.sin(2 * np.pi * freq * time_axis)
    return np.concatenate([
        np.zeros(int(round(lead * sample_rate))),
        signal,
        np.zeros(int(round(tail * sample_rate))),
    ])


def write_wav(path, samples, sample_rate=SR):
    sf.write(str(path), np.asarray(samples, dtype=np.float64), sample_rate,
             format="WAV", subtype="FLOAT")
    return Path(path)


class FakeGenerator:
    """Injectable generation callable that writes deterministic synthetic WAVs."""

    def __init__(self, directory, durations=None, fail_on=None, on_call=None,
                 sample_rate=SR):
        self.directory = Path(directory)
        self.durations = list(durations or [])
        self.fail_on = fail_on
        self.on_call = on_call
        self.sample_rate = sample_rate
        self.calls = []

    def __call__(self, role_id, text, *, params=None, seed=None,
                 device="cuda:0", dtype="float16", cancel=None):
        self.calls.append({
            "role_id": role_id, "text": text, "params": params, "seed": seed,
            "device": device, "dtype": dtype, "cancel": cancel,
        })
        index = len(self.calls)
        if self.fail_on is not None and index == self.fail_on:
            raise RuntimeError("模拟合成失败")
        seconds = (self.durations[index - 1]
                   if index - 1 < len(self.durations) else 0.8)
        path = self.directory / f"segment-{index}.wav"
        write_wav(path, tone(freq=220.0, amp=0.15, seconds=seconds),
                  self.sample_rate)
        self.calls[-1]["file"] = str(path)
        if self.on_call is not None:
            self.on_call(index)
        return {
            "file": str(path),
            "metadata_file": None,
            "engine": "fake-generator",
            "seed": seed,
            "characters": len(text),
            "note": None,
        }


class SplitTextTests(unittest.TestCase):
    def test_preserves_every_character_and_respects_bound(self):
        text = "第一段第一句。第一段第二句！\n第二段：这是一个问题吗？最后一句没有标点"
        segments = longform.split_text(text, max_chars=20)
        self.assertEqual("".join(segments), text)
        self.assertTrue(all(segment for segment in segments))
        self.assertTrue(all(len(segment) <= 20 for segment in segments))
        self.assertIn("第一段第一句。", segments)

    def test_sentence_terminators_end_segments(self):
        self.assertEqual(
            longform.split_text("第一句。第二句！第三句？"),
            ["第一句。", "第二句！", "第三句？"])

    def test_long_unpunctuated_run_is_hard_split_without_loss(self):
        text = "啊" * 500
        segments = longform.split_text(text, max_chars=20)
        self.assertEqual("".join(segments), text)
        self.assertEqual(len(segments), 25)
        self.assertTrue(all(len(segment) == 20 for segment in segments))

    def test_long_sentence_splits_at_commas_then_hard_slices(self):
        clause = "这是一个很长的分句没有句号，"
        text = clause * 8 + "尾巴" * 30
        segments = longform.split_text(text, max_chars=40)
        self.assertEqual("".join(segments), text)
        self.assertTrue(all(len(segment) <= 40 for segment in segments))
        self.assertTrue(any(segment.endswith("，") for segment in segments))

    def test_whitespace_and_newlines_are_preserved(self):
        text = "  你好。\n\n世界！  "
        segments = longform.split_text(text, max_chars=20)
        self.assertEqual("".join(segments), text)

    def test_trailing_whitespace_does_not_break_the_bound(self):
        text = "啊" * 40 + "  "
        segments = longform.split_text(text, max_chars=20)
        self.assertEqual("".join(segments), text)
        self.assertTrue(all(len(segment) <= 20 for segment in segments))
        self.assertTrue(all(segment.strip() for segment in segments))

    def test_rejections(self):
        for bad_text in ("", "   ", "\n\t "):
            with self.assertRaises(ValueError):
                longform.split_text(bad_text)
        with self.assertRaises(ValueError):
            longform.split_text(None)
        with self.assertRaises(ValueError):
            longform.split_text("哈" * (longform.MAX_TEXT_CHARS + 1))
        for bad_chars in (19, 501, 20.0, True, "120", None):
            with self.assertRaises(ValueError):
                longform.split_text("你好。", max_chars=bad_chars)


class CheckSegmentsTests(unittest.TestCase):
    def setUp(self):
        self.text = "春风吹过江面。秋雨落在窗前。"
        self.segments = longform.split_text(self.text)

    def test_full_split_covers_source(self):
        report = longform.check_segments(self.text, self.segments)
        self.assertTrue(report["ok"])
        self.assertEqual(report["coverage"], 1.0)
        self.assertEqual(report["missing"], [])
        self.assertEqual(report["duplicates"], [])
        self.assertEqual(report["problems"], [])
        self.assertEqual(report["segment_count"], 2)

    def test_missing_fragment_is_reported(self):
        report = longform.check_segments(self.text, self.segments[:1])
        self.assertFalse(report["ok"])
        self.assertEqual(len(report["missing"]), 1)
        self.assertIn("秋雨落在窗前。", report["missing"][0])
        self.assertAlmostEqual(report["coverage"], 0.5)
        self.assertTrue(any("没有" in problem for problem in report["problems"]))

    def test_duplicate_fragment_is_reported(self):
        report = longform.check_segments(
            self.text, [self.segments[0], self.segments[0]])
        self.assertFalse(report["ok"])
        self.assertTrue(report["duplicates"])
        self.assertTrue(any("重复" in problem for problem in report["problems"]))

    def test_hallucinated_fragment_is_reported(self):
        report = longform.check_segments(self.text, ["原文没有这句话。"])
        self.assertFalse(report["ok"])
        self.assertEqual(report["coverage"], 0.0)
        self.assertTrue(any("找不到" in problem for problem in report["problems"]))

    def test_out_of_order_fragment_is_reported(self):
        report = longform.check_segments(
            self.text, [self.segments[1], self.segments[0]])
        self.assertFalse(report["ok"])
        self.assertEqual(report["coverage"], 1.0)
        self.assertTrue(any("顺序" in problem for problem in report["problems"]))

    def test_whitespace_is_ignored_when_matching(self):
        report = longform.check_segments("你好 世界。", ["你好", "世界。"])
        self.assertTrue(report["ok"])
        self.assertEqual(report["coverage"], 1.0)

    def test_rejections(self):
        with self.assertRaises(ValueError):
            longform.check_segments("", self.segments)
        with self.assertRaises(ValueError):
            longform.check_segments(self.text, "不是列表")
        with self.assertRaises(ValueError):
            longform.check_segments(self.text, [1, 2])


class AnalyzeSegmentsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _write(self, name, samples):
        return str(write_wav(self.root / name, samples))

    def test_detects_acceleration_volume_jump_and_timbre_drift(self):
        paths = [
            self._write("a.wav", tone(220.0, 0.2, 1.0)),
            self._write("b.wav", tone(220.0, 0.2, 1.0)),
            self._write("c.wav", tone(220.0, 0.9, 1.0)),
            self._write("d.wav", tone(4000.0, 0.2, 1.0)),
        ]
        texts = ["一二三四", "一二三四五六七八", "一二三四", "一二三四"]
        report = longform.analyze_segments(paths, texts)

        self.assertFalse(report["ok"])
        self.assertTrue(report["problems"])
        for problem in report["problems"]:
            self.assertRegex(problem, r"[\u4e00-\u9fff]")

        segments = report["segments"]
        self.assertEqual(len(segments), 4)
        self.assertTrue(segments[1]["flags"]["sudden_acceleration"])
        self.assertTrue(segments[2]["flags"]["volume_jump"])
        self.assertTrue(segments[3]["flags"]["timbre_drift"])
        self.assertFalse(segments[0]["flags"]["sudden_acceleration"])
        self.assertFalse(segments[0]["flags"]["volume_jump"])
        self.assertFalse(segments[0]["flags"]["timbre_drift"])
        self.assertTrue(all(segment["click_count"] == 0 for segment in segments))
        self.assertGreater(segments[3]["spectral_centroid_hz"], 2000.0)
        self.assertLess(segments[0]["spectral_centroid_hz"], 600.0)
        self.assertAlmostEqual(segments[0]["duration"], 1.0, delta=0.001)
        self.assertEqual(segments[0]["sample_rate"], SR)
        self.assertAlmostEqual(segments[1]["chars_per_second"], 8.0, delta=0.05)

        summary = report["summary"]
        self.assertEqual(summary["segment_count"], 4)
        self.assertAlmostEqual(summary["total_duration"], 4.0, delta=0.01)
        self.assertAlmostEqual(summary["median_chars_per_second"], 4.0, delta=0.05)
        self.assertGreaterEqual(summary["flagged_segments"], 3)
        self.assertEqual(summary["clipped_samples"], 0)

    def test_single_segment_has_no_median_flags(self):
        path = self._write("single.wav", tone(440.0, 0.3, 1.0))
        report = longform.analyze_segments([path], ["你好世界。"])
        self.assertTrue(report["ok"])
        self.assertEqual(report["problems"], [])
        segment = report["segments"][0]
        self.assertFalse(any(segment["flags"].values()))
        self.assertAlmostEqual(segment["chars_per_second"], 5.0, delta=0.05)
        self.assertIsNotNone(segment["loudness_lufs"])

    def test_clean_consistent_segments_are_ok(self):
        paths = [self._write(f"clean-{index}.wav", tone(330.0, 0.25, 0.6))
                 for index in range(3)]
        report = longform.analyze_segments(paths, ["第一句。", "第二句。", "第三句。"])
        self.assertTrue(report["ok"])
        self.assertEqual(report["problems"], [])

    def test_click_is_detected_and_reported_in_chinese(self):
        samples = tone(220.0, 0.2, 1.0)
        samples[len(samples) // 2] = 0.95
        path = self._write("click.wav", samples)
        report = longform.analyze_segments([path], ["一二三四"])
        segment = report["segments"][0]
        self.assertEqual(segment["click_count"], 1)
        self.assertFalse(report["ok"])
        self.assertTrue(any("点击声" in problem for problem in report["problems"]))

    def test_leading_and_trailing_silence_are_measured(self):
        path = self._write("padded.wav", tone(220.0, 0.3, 1.0, lead=0.2, tail=0.3))
        segment = longform.analyze_segments([path], ["你好世界。"])["segments"][0]
        self.assertAlmostEqual(segment["leading_silence"], 0.2, delta=0.021)
        self.assertAlmostEqual(segment["trailing_silence"], 0.3, delta=0.021)
        self.assertAlmostEqual(segment["voiced_seconds"], 1.0, delta=0.021)

    def test_rejections(self):
        good = self._write("good.wav", tone())
        with self.assertRaises(ValueError):
            longform.analyze_segments([good], ["文本", "多余"])
        with self.assertRaises(ValueError):
            longform.analyze_segments([], [])
        with self.assertRaises(ValueError):
            longform.analyze_segments([good], [123])
        with self.assertRaises(ValueError):
            longform.analyze_segments([str(self.root / "missing.wav")], ["文本"])
        nonfinite = self.root / "nonfinite.wav"
        sf.write(str(nonfinite), np.array([np.nan, 0.1, -0.1]), SR, subtype="FLOAT")
        with self.assertRaises(ValueError):
            longform.analyze_segments([str(nonfinite)], ["文本"])


class ConcatenateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.first = write_wav(self.root / "first.wav", tone(220.0, 0.2, 1.0))
        self.second = write_wav(self.root / "second.wav", tone(220.0, 0.2, 1.0))
        self.output = self.root / "joined.wav"

    def tearDown(self):
        self.temp.cleanup()

    def test_duration_rate_gaps_and_fades(self):
        first_hash = sha256(self.first)
        second_hash = sha256(self.second)
        result = longform.concatenate(
            [self.first, self.second], self.output, crossfade_ms=25, gap_ms=80)
        self.assertEqual(result["segments"], 2)
        self.assertEqual(result["sample_rate"], SR)
        self.assertAlmostEqual(result["duration"], 2.08, delta=0.001)
        self.assertTrue(Path(result["path"]).is_file())
        info = sf.info(str(self.output))
        self.assertEqual(info.subtype, "FLOAT")
        self.assertEqual(info.samplerate, SR)
        self.assertAlmostEqual(info.duration, 2.08, delta=0.001)
        # Source files are never touched by concatenation.
        self.assertEqual(sha256(self.first), first_hash)
        self.assertEqual(sha256(self.second), second_hash)

        data, rate = sf.read(str(self.output), dtype="float64", always_2d=True)
        self.assertEqual(rate, SR)
        second_start = int(round(1.08 * rate))
        self.assertLess(abs(data[second_start, 0]), 0.01)
        window = data[second_start:second_start + int(0.1 * rate), 0]
        self.assertGreater(float(np.max(np.abs(window))), 0.1)
        self.assertLess(abs(data[int(round(1.0 * rate)) - 1, 0]), 0.01)

    def test_zero_gap_and_zero_fade_keep_exact_length(self):
        result = longform.concatenate(
            [self.first, self.second], self.output, crossfade_ms=0, gap_ms=0)
        self.assertAlmostEqual(result["duration"], 2.0, delta=0.001)
        with sf.SoundFile(str(self.output)) as recording:
            self.assertEqual(recording.frames, 2 * SR)

    def test_refuses_overwrite_unless_allowed(self):
        longform.concatenate([self.first], self.output)
        with self.assertRaises(FileExistsError):
            longform.concatenate([self.first], self.output)
        self.assertEqual({path.name for path in self.root.iterdir()},
                         {"first.wav", "joined.wav", "second.wav"})
        result = longform.concatenate([self.first, self.second], self.output,
                                      overwrite=True)
        self.assertAlmostEqual(result["duration"], 2.08, delta=0.001)

    def test_refuses_mismatched_sample_rates(self):
        other = write_wav(self.root / "other.wav",
                          tone(220.0, 0.2, 1.0, sample_rate=16000),
                          sample_rate=16000)
        with self.assertRaises(ValueError):
            longform.concatenate([self.first, other], self.output)

    def test_refuses_output_that_is_an_input(self):
        with self.assertRaises(ValueError):
            longform.concatenate([self.first], self.first, overwrite=True)

    def test_rejections(self):
        for kwargs in ({"crossfade_ms": -1}, {"crossfade_ms": 201},
                       {"gap_ms": -1}, {"gap_ms": 2001},
                       {"crossfade_ms": "25"}, {"gap_ms": float("nan")}):
            with self.assertRaises(ValueError):
                longform.concatenate([self.first], self.output, **kwargs)
        with self.assertRaises(ValueError):
            longform.concatenate([], self.output)
        with self.assertRaises(ValueError):
            longform.concatenate([self.first], self.output, overwrite="yes")
        with self.assertRaises(ValueError):
            longform.concatenate([self.first], self.root)


class GenerateLongformTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.audio_dir = self.root / "audio"
        self.audio_dir.mkdir()
        self.text = "春风吹过江面。秋雨落在窗前。远山安静地等待。"

    def tearDown(self):
        self.temp.cleanup()

    def test_generates_segments_sidecar_and_report(self):
        generator = FakeGenerator(self.audio_dir)
        progress = []
        result = longform.generate_longform(
            "role-abc", self.text, output_dir=self.root / "out", max_chars=80,
            generate=generator, params={"speed": 1.0}, seed=10, device="cpu",
            dtype="float32", progress=lambda value, desc=None: progress.append(
                (value, desc)))

        expected_segments = longform.split_text(self.text, max_chars=80)
        self.assertTrue(result["ok"])
        self.assertEqual(len(generator.calls), len(expected_segments))
        self.assertEqual([call["text"] for call in generator.calls],
                         expected_segments)
        self.assertEqual([call["seed"] for call in generator.calls],
                         [10, 11, 12])
        self.assertTrue(all(call["params"] == {"speed": 1.0}
                            for call in generator.calls))
        self.assertTrue(all(call["device"] == "cpu" for call in generator.calls))
        self.assertTrue(all(call["dtype"] == "float32"
                            for call in generator.calls))
        self.assertTrue(all(call["role_id"] == "role-abc"
                            for call in generator.calls))
        self.assertEqual(len(progress), len(expected_segments))
        self.assertAlmostEqual(progress[-1][0], 1.0, delta=1e-9)
        self.assertTrue(all(isinstance(desc, str) for _value, desc in progress))

        output = Path(result["file"])
        report_path = Path(result["report_file"])
        self.assertTrue(output.is_file())
        self.assertTrue(report_path.is_file())
        self.assertEqual(output.parent, (self.root / "out").resolve())
        self.assertEqual(report_path.suffix, ".json")
        self.assertEqual(report_path.stem, output.stem)
        self.assertEqual(result["segments"],
                         [{"index": index, "text": text, "file": call["file"]}
                          for index, (text, call) in enumerate(
                              zip(expected_segments, generator.calls))])

        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["role_id"], "role-abc")
        self.assertEqual(report["text"], self.text)
        self.assertEqual(report["split"],
                         {"max_chars": 80, "segment_count": len(expected_segments)})
        self.assertEqual(len(report["segments"]), len(expected_segments))
        self.assertEqual(report["segments"][0]["text"], expected_segments[0])
        self.assertEqual(report["segments"][0]["metadata"]["engine"],
                         "fake-generator")
        self.assertIsNone(report["segments"][0]["metadata"]["note"])
        self.assertEqual(report["segments"][0]["role_id"], "role-abc")
        self.assertEqual(report["segments"][0]["seed"], 10)
        self.assertEqual(report["segments"][0]["params"], {"speed": 1.0})
        self.assertEqual(len(report["generation"]["files"]), len(expected_segments))
        self.assertEqual(report["generation"]["seed"], 10)
        # Three 0.8 s segments plus two 80 ms gaps.
        self.assertAlmostEqual(report["audio"]["duration"], 2.56, delta=0.01)
        self.assertEqual(report["check"]["ok"], True)
        self.assertIn("analysis", report)
        self.assertEqual(len(report["analysis"]["segments"]), len(expected_segments))
        self.assertTrue(result["analysis"]["ok"])
        self.assertIsNotNone(
            __import__("datetime").datetime.fromisoformat(
                report["created_at"]).tzinfo)
        self.assertTrue(all(isinstance(warning, str)
                            for warning in result["warnings"]))

    def test_default_output_directory_is_longform_sibling(self):
        generator = FakeGenerator(self.audio_dir)
        result = longform.generate_longform(
            "role-abc", "只有一句话。", generate=generator)
        self.assertEqual(Path(result["file"]).parent,
                         (self.audio_dir / "longform").resolve())

    def test_warnings_include_analysis_problems(self):
        generator = FakeGenerator(self.audio_dir, durations=[1.0, 1.0, 0.2])
        result = longform.generate_longform(
            "role-abc", self.text, output_dir=self.root / "out",
            generate=generator)
        self.assertFalse(result["analysis"]["ok"])
        self.assertTrue(result["analysis"]["problems"])
        self.assertTrue(result["warnings"])
        self.assertTrue(any("语速" in warning for warning in result["warnings"]))
        report = json.loads(Path(result["report_file"]).read_text(encoding="utf-8"))
        self.assertFalse(report["analysis"]["ok"])

    def test_cancel_before_first_segment_publishes_nothing(self):
        cancel = threading.Event()
        cancel.set()
        generator = FakeGenerator(self.audio_dir)
        output_dir = self.root / "out"
        with self.assertRaises(RuntimeError) as caught:
            longform.generate_longform(
                "role-abc", self.text, output_dir=output_dir,
                generate=generator, cancel=cancel)
        self.assertEqual(str(caught.exception), "任务已取消")
        self.assertEqual(generator.calls, [])
        self.assertFalse(output_dir.exists())

    def test_cancel_between_segments_publishes_nothing(self):
        cancel = threading.Event()
        generator = FakeGenerator(self.audio_dir,
                                  on_call=lambda index: cancel.set())
        output_dir = self.root / "out"
        with self.assertRaises(RuntimeError) as caught:
            longform.generate_longform(
                "role-abc", self.text, output_dir=output_dir,
                generate=generator, cancel=cancel)
        self.assertEqual(str(caught.exception), "任务已取消")
        self.assertEqual(len(generator.calls), 1)
        self.assertFalse(output_dir.exists())

    def test_generation_failure_does_not_publish_output(self):
        generator = FakeGenerator(self.audio_dir, fail_on=2)
        output_dir = self.root / "out"
        with self.assertRaises(RuntimeError):
            longform.generate_longform(
                "role-abc", self.text, output_dir=output_dir,
                generate=generator)
        self.assertFalse(output_dir.exists())

    def test_invalid_arguments_are_rejected_before_generation(self):
        generator = FakeGenerator(self.audio_dir)
        cases = [
            {"role_id": ""},
            {"role_id": None},
            {"text": "   "},
            {"max_chars": 10},
            {"progress": 42},
            {"cancel": object()},
            {"params": []},
            {"seed": -1},
            {"seed": 1.5},
            {"device": ""},
            {"dtype": None},
        ]
        for case in cases:
            options = dict(case)
            role_id = options.pop("role_id", "role-abc")
            text = options.pop("text", self.text)
            with self.assertRaises(ValueError):
                longform.generate_longform(role_id, text, generate=generator,
                                           **options)
        self.assertEqual(generator.calls, [])

    def test_output_dir_that_is_a_file_is_rejected(self):
        blocker = self.root / "blocker"
        blocker.write_bytes(b"x")
        with self.assertRaises(ValueError):
            longform.generate_longform(
                "role-abc", self.text, output_dir=blocker,
                generate=FakeGenerator(self.audio_dir))

    def test_seed_overflow_is_rejected(self):
        with self.assertRaises(ValueError):
            longform.generate_longform(
                "role-abc", self.text,
                generate=FakeGenerator(self.audio_dir),
                seed=2 ** 63 - 1)


if __name__ == "__main__":
    unittest.main()

