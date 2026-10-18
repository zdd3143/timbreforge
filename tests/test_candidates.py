import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import soundfile as sf
from timbreforge.candidates import compare_audio, measure_candidate


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.a = self.root / 'a.wav'
        self.b = self.root / 'b.wav'
        x = np.sin(2*np.pi*160*np.arange(16000)/16000)
        sf.write(self.a, .1*x, 16000, subtype='FLOAT')
        sf.write(self.b, .5*x, 16000, subtype='FLOAT')

    def test_matching_copies_preserve_raw_and_equal_rms(self):
        hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (self.a,self.b)]
        result = compare_audio(self.a,self.b,True,self.root/'playback')
        x = sf.read(result['a'])[0]; y = sf.read(result['b'])[0]
        self.assertAlmostEqual(float(np.sqrt(np.mean(x*x))),float(np.sqrt(np.mean(y*y))),places=7)
        self.assertLessEqual(max(np.max(np.abs(x)),np.max(np.abs(y))),10**(-1/20)+1e-6)
        self.assertEqual(hashes,[hashlib.sha256(p.read_bytes()).hexdigest() for p in (self.a,self.b)])
        self.assertNotEqual(result['b'],str(self.b))
        self.assertAlmostEqual(result['gain_b_db'],-13.9794,places=3)

    def test_without_matching_no_writes(self):
        result=compare_audio(self.a,self.b)
        self.assertEqual(result['a'],str(self.a))
        self.assertFalse(result['matched'])
        self.assertFalse((self.root/'playback').exists())

    def test_unknown_scores_are_not_fabricated(self):
        result=measure_candidate(self.a,expected_text='你好')
        for axis in ('speaker_similarity','content_accuracy','quality'):
            self.assertEqual(result[axis]['status'],'unverified')
            self.assertIsNone(result[axis]['value'])
        self.assertNotIn('winner',result)

    def test_user_transcript_cer_independent_of_similarity(self):
        result=measure_candidate(self.a,expected_text='你好世界',observed_text='你好世间')
        self.assertEqual(result['content_accuracy']['value'],.25)
        self.assertEqual(result['speaker_similarity']['status'],'unverified')
        self.assertIn('no ASR',result['content_accuracy']['method'])

    def test_optional_missing_scorers_report_unavailable(self):
        with patch('timbreforge.score.score_one',side_effect=RuntimeError('no model')), \
             patch('timbreforge.mosmodels.score_files',side_effect=RuntimeError('no model')):
            result=measure_candidate(self.a,reference=self.b,run_speaker=True,run_quality=True)
        self.assertEqual(result['speaker_similarity']['status'],'unavailable')
        self.assertEqual(result['quality']['status'],'unavailable')

    def test_quality_keeps_metrics_separate_and_rejects_nan(self):
        with patch('timbreforge.mosmodels.score_files', return_value=[{'file': 'x', 'utmos': 3.2, 'dnsmos_ovrl': 2.8}]):
            result=measure_candidate(self.a,run_quality=True)
        self.assertEqual(result['quality']['value'],{'utmos':3.2,'dnsmos_ovrl':2.8})
        with patch('timbreforge.mosmodels.score_files', return_value=[{'utmos':float('nan')}]):
            result=measure_candidate(self.a,run_quality=True)
        self.assertEqual(result['quality']['status'],'unavailable')

    def test_silent_match_and_invalid_audio_rejected(self):
        sf.write(self.b,np.zeros(16000),16000)
        with self.assertRaisesRegex(ValueError,'silent'):
            compare_audio(self.a,self.b,True,self.root/'listen')
        sf.write(self.b,np.array([np.nan]),16000,subtype='FLOAT')
        with self.assertRaises(ValueError): measure_candidate(self.b)


if __name__=='__main__': unittest.main()
