import shutil
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from timbreforge.audio_composer import compose_audio


class AudioComposerTests(unittest.TestCase):
    def setUp(self):
        self.root = Path('tmp_audio_composer_test')
        self.root.mkdir(exist_ok=True)
        for index, value in enumerate((0.1, 0.2)):
            sf.write(self.root / f'{index}.wav', np.full(800, value, dtype=np.float32),
                     8000, subtype='FLOAT')

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_join_is_new_file_with_gap_and_fade(self):
        result = compose_audio([str(self.root / '0.wav'), str(self.root / '1.wav')],
                               gap_ms=100, crossfade_ms=10, output_dir=self.root)
        self.assertTrue(Path(result['path']).is_file())
        self.assertEqual(result['segments'], 2)
        self.assertEqual(result['sample_rate'], 8000)
        with sf.SoundFile(result['path']) as stream:
            self.assertEqual(stream.frames, 2400)

    def test_requires_two_inputs(self):
        with self.assertRaises(ValueError):
            compose_audio([str(self.root / '0.wav')], output_dir=self.root)


if __name__ == '__main__':
    unittest.main()

