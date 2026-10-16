import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from timbreforge.grid import Variant, default_variants, run_grid
from timbreforge.backend_indextts import SynthParams


class DefaultVariantsTests(unittest.TestCase):
    def test_configurations_are_unique(self):
        variants = default_variants(base_seed=7, n=20)
        configs = {(v.seed, v.ref_seconds, v.num_beams, v.do_sample,
                    v.temperature, v.duration_factor, v.emotion_weight,
                    tuple(sorted(v.extra.items()))) for v in variants}
        self.assertEqual(len(variants), len(configs))

    def test_invalid_n_and_vary(self):
        for n in (0, -1, True, 1.5):
            with self.subTest(n=n):
                with self.assertRaises(ValueError):
                    default_variants(n=n)
        with self.assertRaises(ValueError):
            default_variants(vary=())
        with self.assertRaises(ValueError):
            default_variants(vary=("unknown",))
        with self.assertRaises(ValueError):
            default_variants(n=6, vary=("ref_len",))


class RunGridMockTests(unittest.TestCase):
    def test_base_params_crops_fixed_reference_and_top_n(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            source = td / "source.wav"
            generated = td / "generated.wav"
            sf.write(source, np.ones(1600, dtype=np.float32) * .1, 16000)
            sf.write(generated, np.ones(1600, dtype=np.float32) * .2, 16000)
            prepared = td / "prepared.wav"
            crops = {4.0: td / "crop4.wav", 5.0: td / "crop5.wav"}
            for p in [prepared, *crops.values()]:
                sf.write(p, np.ones(1600, dtype=np.float32) * .1, 16000)
            calls = []

            def prep(path, max_seconds=6.0, cache_dir="data_ref/prepared"):
                calls.append((str(path), max_seconds, cache_dir))
                return prepared if max_seconds is None else crops[max_seconds]

            class Entry:
                cid = "x"; name = "X"; gender = ""; age = ""
            class Bank:
                def get(self, cid): return Entry()

            params = SynthParams(model="custom", precision="fp32", top_p=.33,
                                 top_k=11, emotion={"happy": .8})
            seen = []
            def synth(ref, p, base, verbose=False):
                seen.append((ref, p, base))
                return {"file": str(generated)}
            variants = [Variant("a", ref_seconds=4), Variant("b", ref_seconds=5),
                        Variant("c", ref_seconds=4), Variant("d", ref_seconds=5)]
            def pairs(items):
                self.assertEqual({x["ref"] for x in items}, {str(prepared)})
                return [{"label": x["label"], "cosine": .7} for x in items]
            mos = lambda files, device="cpu": [
                {"file": f, "utmos": 4.0, "dnsmos_ovrl": 4.0} for f in files]
            with patch("timbreforge.backend_indextts.ping", return_value=(True, "ok")), \
                 patch("timbreforge.backend_indextts.synthesize", side_effect=synth), \
                 patch("timbreforge.voicebank.get_bank", return_value=Bank()), \
                 patch("timbreforge.reference.prepare_reference", side_effect=prep), \
                 patch("timbreforge.score.score_pairs", side_effect=pairs), \
                 patch("timbreforge.mosmodels.score_files", side_effect=mos), \
                 patch("timbreforge.mosmodels.combined_score", return_value={"combined": 70, "naturalness100": 80}), \
                 patch("timbreforge.mosmodels.grade_combined", return_value="良好"):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    result = run_grid("p", "hello", variants, outdir=td / "out",
                                      base_url="http://custom", base_params=params,
                                      top_n=3, verbose=True, ref_wav_override=str(source))
            self.assertEqual(len(result["rows"]), 4)
            rendered = output.getvalue()
            table_rows = [line for line in rendered.splitlines()
                          if any(f" {v.label:<12s} " in line for v in variants)]
            self.assertEqual(len(table_rows), 3)
            self.assertEqual({x[0] for x in seen}, {str(crops[4.0]), str(crops[5.0])})
            self.assertTrue(all(x[2] == "http://custom" for x in seen))
            self.assertTrue(all(x[1].model == "custom" and x[1].top_p == .33 for x in seen))
            saved = json.loads((td / "out" / "grid.json").read_text(encoding="utf-8"))
            self.assertEqual(len(saved["rows"]), 4)
            self.assertTrue(all("params" in row for row in saved["rows"]))

    def test_missing_mos_is_error(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td); source = td / "r.wav"; generated = td / "g.wav"
            sf.write(source, np.ones(800, dtype=np.float32) * .1, 8000)
            sf.write(generated, np.ones(800, dtype=np.float32) * .1, 8000)
            class Bank:
                def get(self, cid): return None
            with patch("timbreforge.backend_indextts.ping", return_value=(True, "ok")), \
                 patch("timbreforge.backend_indextts.synthesize", return_value={"file": str(generated)}), \
                 patch("timbreforge.voicebank.get_bank", return_value=Bank()), \
                 patch("timbreforge.reference.prepare_reference", return_value=source), \
                 patch("timbreforge.score.score_pairs", return_value=[{"label": "a", "cosine": .7}]), \
                 patch("timbreforge.mosmodels.score_files", return_value=[{"file": "wrong", "utmos": 4, "dnsmos_ovrl": 4}]):
                with self.assertRaises(RuntimeError):
                    run_grid("", "x", [Variant("a")], outdir=td / "out",
                              ref_wav_override=str(source), verbose=False)

    def test_missing_or_nonfinite_speaker_score_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / 'ref.wav'
            sf.write(source, np.ones(1600, dtype=np.float32) * .1, 16000)
            for records in ([], [{'label': 'a', 'cosine': float('nan')}],
                            [{'label': 'wrong', 'cosine': .7}]):
                with self.subTest(records=records), \
                     patch('timbreforge.backend_indextts.ping', return_value=(True, 'ok')), \
                     patch('timbreforge.backend_indextts.synthesize', return_value={'file': str(source)}), \
                     patch('timbreforge.voicebank.get_bank', return_value=[]), \
                     patch('timbreforge.reference.prepare_reference', return_value=source), \
                     patch('timbreforge.score.score_pairs', return_value=records):
                    with self.assertRaisesRegex(RuntimeError, 'CAM'):
                        run_grid('', 'x', [Variant('a')], outdir=root / 'out',
                                 ref_wav_override=str(source), verbose=False, use_mos=False)


if __name__ == "__main__":
    unittest.main()
