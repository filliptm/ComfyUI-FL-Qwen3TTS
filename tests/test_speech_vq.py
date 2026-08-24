import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qwen_tts.core.tokenizer_25hz.vq.speech_vq import peak_normalize


class PeakNormalizeTests(unittest.TestCase):
    def test_normalizes_to_minus_six_db_without_mutating_input(self):
        waveform = torch.tensor([-0.25, 0.5], dtype=torch.float32)
        original = waveform.clone()

        normalized = peak_normalize(waveform)

        self.assertTrue(torch.equal(waveform, original))
        self.assertAlmostEqual(normalized.abs().max().item(), 10 ** (-6 / 20), places=6)
        self.assertEqual(normalized.dtype, waveform.dtype)

    def test_silence_remains_finite_silence(self):
        normalized = peak_normalize(torch.zeros(16000, dtype=torch.float32))

        self.assertTrue(torch.isfinite(normalized).all())
        self.assertEqual(torch.count_nonzero(normalized).item(), 0)


if __name__ == "__main__":
    unittest.main()
