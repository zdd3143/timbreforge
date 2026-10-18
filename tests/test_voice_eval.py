"""Fixed evaluation set, error rates, signal metrics, drift and reporting tests."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from timbreforge import voice_eval

SR = 24000


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tone(freq=440.0, amp=0.3, seconds=1.0, lead=0.0, tail=0.0,
         sample_rate=SR):
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


class FixedTestSetTests(unittest.TestCase):
    def test_categories_are_complete_and_ordered(self):
        expected = (
            "low_emotion_short",
            "question",
            "command",
            "long_sentence",
            "numbers_and_names",
            "high_emotion",
            "cold_low_energy",
            "phonetic_coverage",
        )
        self.assertEqual(voice_eval.TEST_CATEGORIES, expected)
        items = voice_eval.fixed_test_set()
        self.assertEqual(len(items), len(expected))
        self.assertEqual([item["category"] for item in items], list(expected))
        self.assertEqual(len({item["category"] for item in items}), len(items))
        for item in items:
            self.assertEqual(set(item), {"category", "label", "text"})
            self.assertTrue(item["text"].strip())
            self.assertRegex(item["label"], r"[\u4e00-\u9fff]")
            self.assertRegex(item["text"], r"[\u4e00-\u9fff]")

    def test_fixed_set_returns_an_independent_copy(self):
        first = voice_eval.fixed_test_set()
        first[0]["text"] = "被修改"
        first.append({"category": "extra"})
        second = voice_eval.fixed_test_set()
        self.assertNotEqual(second[0]["text"], "被修改")
        self.assertEqual(len(second), len(voice_eval.TEST_CATEGORIES))


class ErrorRateTests(unittest.TestCase):
    def test_cer_exact_math(self):
        result = voice_eval.character_error_rate("你好世界", "你好世")
        self.assertEqual(result["status"], "measured")
        self.assertAlmostEqual(result["cer"], 0.25)
        self.assertEqual(result["distance"], 1)
        self.assertEqual(result["expected_length"], 4)

        self.assertAlmostEqual(
            voice_eval.character_error_rate("kitten", "sitting")["cer"], 0.5)
        self.assertEqual(
            voice_eval.character_error_rate("你好世界", "你好世界")["cer"], 0.0)
        self.assertAlmostEqual(
            voice_eval.character_error_rate("你好世界", "你好世界啊")["cer"], 0.25)

    def test_cer_counts_punctuation(self):
        result = voice_eval.character_error_rate("你好，世界！", "你好世界")
        self.assertEqual(result["distance"], 2)
        self.assertEqual(result["expected_length"], 6)
        self.assertAlmostEqual(result["cer"], 2 / 6)

    def test_cer_rejections(self):
        with self.assertRaises(ValueError):
            voice_eval.character_error_rate("", "你好")
        with self.assertRaises(ValueError):
            voice_eval.character_error_rate(None, "你好")
        with self.assertRaises(ValueError):
            voice_eval.character_error_rate("你好", 123)
        with self.assertRaises(ValueError):
            voice_eval.character_error_rate("啊" * 5001, "啊")

    def test_wer_for_space_separated_text(self):
        result = voice_eval.word_error_rate("hello world again", "hello world")
        self.assertEqual(result["status"], "measured")
        self.assertAlmostEqual(result["wer"], 1 / 3)
        self.assertEqual(result["distance"], 1)
        self.assertEqual(result["expected_length"], 3)
        self.assertAlmostEqual(
            voice_eval.word_error_rate("hello world", "hello word")["wer"], 0.5)

    def test_wer_is_not_faked_for_chinese(self):
        result = voice_eval.word_error_rate("你好世界", "你好世")
        self.assertEqual(result["status"], "not_applicable")
        self.assertIsNone(result["wer"])
        self.assertIn("CER", result["reason"])

    def test_wer_rejections(self):
        with self.assertRaises(ValueError):
            voice_eval.word_error_rate("", "hello")
        with self.assertRaises(ValueError):
            voice_eval.word_error_rate("   ", "hello")
        with self.assertRaises(ValueError):
            voice_eval.word_error_rate(1, "hello")


class SignalMetricTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_known_silence_and_tone_metrics(self):
        path = write_wav(self.root / "known.wav",
                         tone(440.0, 0.5, 2.0, lead=0.5, tail=0.5))
        metrics = voice_eval.measure_signal(path)
        self.assertAlmostEqual(metrics["duration"], 3.0, delta=0.001)
        self.assertEqual(metrics["sample_rate"], SR)
        self.assertEqual(metrics["channels"], 1)
        self.assertAlmostEqual(metrics["leading_silence"], 0.5, delta=0.021)
        self.assertAlmostEqual(metrics["trailing_silence"], 0.5, delta=0.021)
        self.assertAlmostEqual(metrics["voiced_seconds"], 2.0, delta=0.021)
        self.assertEqual(metrics["clipped_samples"], 0)
        self.assertEqual(metrics["near_full_scale_fraction"], 0.0)
        self.assertAlmostEqual(metrics["peak_dbfs"], -6.0206, delta=0.05)
        # RMS covers the whole signal, including both silent halves.
        self.assertAlmostEqual(metrics["rms_dbfs"], -10.7918, delta=0.05)
        self.assertIsNotNone(metrics["loudness_lufs"])
        self.assertTrue(math.isfinite(metrics["loudness_lufs"]))
        self.assertLess(metrics["loudness_lufs"], 0.0)

    def test_clipping_is_counted(self):
        samples = tone(440.0, 0.5, 1.0)
        samples[100] = 1.0
        samples[200] = -1.0
        path = write_wav(self.root / "clipped.wav", samples)
        metrics = voice_eval.measure_signal(path)
        self.assertEqual(metrics["clipped_samples"], 2)
        self.assertAlmostEqual(metrics["near_full_scale_fraction"],
                               2 / len(samples))
        self.assertAlmostEqual(metrics["peak_dbfs"], 0.0, delta=0.01)

    def test_digital_silence_is_measured_without_fake_loudness(self):
        path = write_wav(self.root / "silent.wav", np.zeros(SR))
        metrics = voice_eval.measure_signal(path)
        self.assertIsNone(metrics["loudness_lufs"])
        self.assertAlmostEqual(metrics["leading_silence"], 1.0, delta=0.001)
        self.assertAlmostEqual(metrics["trailing_silence"], 1.0, delta=0.001)
        self.assertEqual(metrics["voiced_seconds"], 0.0)
        self.assertEqual(metrics["clipped_samples"], 0)

    def test_signal_rejections(self):
        with self.assertRaises(ValueError):
            voice_eval.measure_signal(self.root / "missing.wav")
        text_file = self.root / "not-audio.wav"
        text_file.write_text("这不是音频", encoding="utf-8")
        with self.assertRaises(ValueError):
            voice_eval.measure_signal(text_file)
        nonfinite = self.root / "nonfinite.wav"
        sf.write(str(nonfinite), np.array([np.nan, 0.1, -0.1]), SR,
                 subtype="FLOAT")
        with self.assertRaises(ValueError):
            voice_eval.measure_signal(nonfinite)
        empty = self.root / "empty.wav"
        with sf.SoundFile(str(empty), mode="w", samplerate=SR, channels=1,
                          format="WAV"):
            pass
        with self.assertRaises(ValueError):
            voice_eval.measure_signal(empty)


class TimbreDriftTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_low_and_high_tones_drift_far(self):
        low = write_wav(self.root / "low.wav", tone(200.0, 0.3, 1.0))
        high = write_wav(self.root / "high.wav", tone(4000.0, 0.3, 1.0))
        result = voice_eval.timbre_drift(high, low)
        self.assertEqual(result["status"], "measured")
        self.assertGreater(result["centroid_hz"], 2000.0)
        self.assertLess(result["reference_centroid_hz"], 600.0)
        self.assertGreater(result["ratio"], 1.35)
        self.assertGreater(result["drift"], 0.35)

    def test_same_file_has_no_drift(self):
        path = write_wav(self.root / "same.wav", tone(1000.0, 0.3, 1.0))
        result = voice_eval.timbre_drift(path, path)
        self.assertAlmostEqual(result["ratio"], 1.0, delta=0.05)
        self.assertLess(result["drift"], 0.05)

    def test_silence_raises(self):
        silent = write_wav(self.root / "silent.wav", np.zeros(SR))
        audible = write_wav(self.root / "audible.wav", tone())
        with self.assertRaises(ValueError):
            voice_eval.timbre_drift(silent, audible)
        with self.assertRaises(ValueError):
            voice_eval.timbre_drift(audible, silent)
        with self.assertRaises(ValueError):
            voice_eval.timbre_drift(audible, self.root / "missing.wav")


class EvaluateCandidateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.audio = write_wav(self.root / "candidate.wav", tone(440.0, 0.4, 1.0))
        self.reference = write_wav(self.root / "reference.wav",
                                   tone(200.0, 0.3, 1.0))

    def tearDown(self):
        self.temp.cleanup()

    def test_accuracy_is_unverified_without_transcript(self):
        result = voice_eval.evaluate_candidate(self.audio, "你好世界", rating=4)
        self.assertEqual(result["metrics"]["cer"]["status"], "unverified")
        self.assertIsNone(result["metrics"]["cer"]["value"])
        self.assertEqual(result["metrics"]["wer"]["status"], "unverified")
        self.assertIsNone(result["metrics"]["wer"]["value"])
        self.assertEqual(result["metrics"]["speaker_similarity"]["status"],
                         "skipped")
        self.assertEqual(result["metrics"]["timbre_drift"]["status"],
                         "not_requested")
        self.assertEqual(result["file"], str(self.audio.resolve()))
        self.assertEqual(result["sha256"], sha256(self.audio))
        self.assertEqual(result["expected_text"], "你好世界")
        self.assertIsNone(result["observed_text"])
        self.assertEqual(result["rating"], 4)
        self.assertIsInstance(result["checks"], list)
        for check in result["checks"]:
            self.assertRegex(check, r"[\u4e00-\u9fff]")
        self.assertTrue(any("辅助" in warning for warning in result["warnings"]))
        self.assertTrue(any("未验证" in warning for warning in result["warnings"]))

    def test_measured_cer_when_transcript_is_supplied(self):
        result = voice_eval.evaluate_candidate(
            self.audio, "你好世界", observed_text="你好世")
        self.assertEqual(result["metrics"]["cer"]["status"], "measured")
        self.assertAlmostEqual(result["metrics"]["cer"]["value"], 0.25)
        self.assertEqual(result["metrics"]["wer"]["status"], "not_applicable")
        self.assertIsNone(result["metrics"]["wer"]["value"])
        self.assertEqual(result["observed_text"], "你好世")

    def test_timbre_drift_requires_and_uses_reference(self):
        result = voice_eval.evaluate_candidate(
            self.audio, "你好世界", reference_audio=self.reference)
        drift = result["metrics"]["timbre_drift"]
        self.assertEqual(drift["status"], "measured")
        self.assertGreater(drift["value"], 0.35)
        self.assertGreater(drift["ratio"], 1.35)
        self.assertIn("method", drift)

    def test_silent_reference_marks_drift_unavailable(self):
        silent = write_wav(self.root / "silent-reference.wav", np.zeros(SR))
        result = voice_eval.evaluate_candidate(
            self.audio, "你好世界", reference_audio=silent)
        drift = result["metrics"]["timbre_drift"]
        self.assertEqual(drift["status"], "unavailable")
        self.assertIsNone(drift["value"])
        self.assertIn("静音", drift["error"])

    def test_clipping_and_rate_checks(self):
        samples = tone(440.0, 0.4, 1.0)
        samples[50] = 1.0
        clipped = write_wav(self.root / "clipped.wav", samples)
        result = voice_eval.evaluate_candidate(clipped, "你好世界")
        self.assertGreater(result["metrics"]["signal"]["clipped_samples"], 0)
        self.assertTrue(any("削波" in check for check in result["checks"]))

        fast = voice_eval.evaluate_candidate(self.audio, "啊" * 100)
        self.assertTrue(any("偏快" in check for check in fast["checks"]))
        slow = voice_eval.evaluate_candidate(self.audio, "啊")
        self.assertTrue(any("偏慢" in check for check in slow["checks"]))

    def test_rejections(self):
        with self.assertRaises(ValueError):
            voice_eval.evaluate_candidate(self.audio, "")
        with self.assertRaises(ValueError):
            voice_eval.evaluate_candidate(self.audio, None)
        with self.assertRaises(ValueError):
            voice_eval.evaluate_candidate(self.audio, "你好", observed_text=123)
        with self.assertRaises(ValueError):
            voice_eval.evaluate_candidate(self.audio, "你好", run_speaker=True)
        with self.assertRaises(ValueError):
            voice_eval.evaluate_candidate(self.audio, "你好",
                                          reference_audio=self.root / "missing.wav")
        for bad_rating in (0, 6, 1.5, "4", True):
            with self.assertRaises(ValueError):
                voice_eval.evaluate_candidate(self.audio, "你好", rating=bad_rating)


class SummarizeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.first = write_wav(self.root / "first.wav", tone(440.0, 0.4, 1.0))
        samples = tone(440.0, 0.4, 1.0)
        samples[50] = 1.0
        samples[100] = -1.0
        self.second = write_wav(self.root / "second.wav", samples)
        self.reference = write_wav(self.root / "reference.wav",
                                   tone(200.0, 0.3, 1.0))

    def tearDown(self):
        self.temp.cleanup()

    def test_averages_worst_and_unmeasured(self):
        first = voice_eval.evaluate_candidate(
            self.first, "你好世界", observed_text="你好世")
        first["category"] = "甲"
        second = voice_eval.evaluate_candidate(
            self.second, "你好世界", observed_text="你好世界")
        second["category"] = "乙"
        summary = voice_eval.summarize([first, second])

        self.assertEqual(summary["count"], 2)
        self.assertAlmostEqual(summary["averages"]["cer"], 0.125)
        self.assertAlmostEqual(summary["averages"]["clipped_samples"], 1.0)
        self.assertIn("duration", summary["averages"])
        self.assertIn("wer", summary["unmeasured"])
        self.assertIn("speaker_similarity", summary["unmeasured"])
        self.assertIn("drift", summary["unmeasured"])
        self.assertRegex(summary["note"], r"[\u4e00-\u9fff]")

        self.assertEqual(summary["worst"]["cer"]["file"],
                         str(self.first.resolve()))
        self.assertEqual(summary["worst"]["cer"]["category"], "甲")
        self.assertEqual(summary["worst"]["clipped_samples"]["file"],
                         str(self.second.resolve()))
        self.assertIsNone(summary["worst"]["drift"])
        self.assertIsNone(summary["worst"]["speaker_similarity"])

    def test_drift_and_speaker_directions(self):
        first = voice_eval.evaluate_candidate(
            self.first, "你好世界", reference_audio=self.reference)
        second = voice_eval.evaluate_candidate(
            self.second, "你好世界", reference_audio=self.reference)
        summary = voice_eval.summarize([first, second])
        self.assertIn("drift", summary["averages"])
        self.assertIsNotNone(summary["worst"]["drift"])
        self.assertNotIn("drift", summary["unmeasured"])

        synthetic = [
            {"file": "a.wav", "category": "A",
             "metrics": {"signal": {"clipped_samples": 0},
                         "speaker_similarity": {"status": "measured",
                                                "value": 0.8}}},
            {"file": "b.wav", "category": "B",
             "metrics": {"signal": {"clipped_samples": 0},
                         "speaker_similarity": {"status": "measured",
                                                "value": 0.5}}},
        ]
        result = voice_eval.summarize(synthetic)
        self.assertAlmostEqual(result["averages"]["speaker_similarity"], 0.65)
        self.assertEqual(result["worst"]["speaker_similarity"]["file"], "b.wav")

    def test_empty_results(self):
        summary = voice_eval.summarize([])
        self.assertEqual(summary["count"], 0)
        self.assertEqual(summary["averages"], {})
        self.assertIn("cer", summary["unmeasured"])
        self.assertIsNone(summary["worst"]["cer"])

    def test_rejections(self):
        with self.assertRaises(ValueError):
            voice_eval.summarize("不是列表")
        with self.assertRaises(ValueError):
            voice_eval.summarize([{"metrics": {}}, "不是字典"])


class SaveReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_writes_utf8_json_atomically(self):
        report = {"文本": "中文报告", "value": 1.5, "items": [1, 2, 3]}
        path = self.root / "reports" / "report.json"
        returned = voice_eval.save_report(report, path)
        self.assertEqual(returned, str(path.resolve()))
        self.assertTrue(path.is_file())
        raw = path.read_text(encoding="utf-8")
        self.assertIn("中文报告", raw)
        self.assertNotIn("\\u4e2d", raw)
        self.assertEqual(json.loads(raw), report)
        self.assertEqual([item.name for item in path.parent.iterdir()],
                         ["report.json"])

    def test_refuses_overwrite_unless_allowed(self):
        path = self.root / "report.json"
        voice_eval.save_report({"value": 1}, path)
        original = path.read_text(encoding="utf-8")
        with self.assertRaises(FileExistsError):
            voice_eval.save_report({"value": 2}, path)
        self.assertEqual(path.read_text(encoding="utf-8"), original)
        voice_eval.save_report({"value": 2}, path, overwrite=True)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["value"], 2)

    def test_rejects_unserializable_and_directory_paths(self):
        path = self.root / "bad.json"
        with self.assertRaises(ValueError):
            voice_eval.save_report({"value": float("nan")}, path)
        self.assertFalse(path.exists())
        self.assertEqual(list(self.root.glob(".report-*")), [])
        with self.assertRaises(ValueError):
            voice_eval.save_report(["不是字典"], path)
        with self.assertRaises(ValueError):
            voice_eval.save_report({"value": 1}, self.root)
        with self.assertRaises(ValueError):
            voice_eval.save_report({"value": 1}, path, overwrite="yes")


if __name__ == "__main__":
    unittest.main()

