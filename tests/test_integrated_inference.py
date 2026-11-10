"""Unit test verifying authentic end-to-end inference and token logits with AirColibri."""

import os
import unittest
import torch
from src.aircolibri_integrated import AirColibriModel

MODEL_DIR = r"C:\air-colibri\models\deepseek-v4-9b-colibri"


class TestIntegratedInference(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = AirColibriModel(
            model_dir=MODEL_DIR,
            vram_resident_layers=4,
            dtype=torch.bfloat16
        )

    def test_first_token_logits_and_capital_paris(self):
        prompt = "The capital of France is"
        out_text, tokens, step_times = self.engine.generate(
            prompt=prompt,
            max_new_tokens=2,
            temperature=0.0
        )
        # Token 1 must be ' Paris' (11751), Token 2 must be '.' (13)
        self.assertIn("Paris", out_text)
        self.assertEqual(tokens[-2], 11751)  # ' Paris'
        self.assertEqual(tokens[-1], 13)     # '.'
        self.assertGreater(len(step_times), 0)
        print(f"\n[Test Verified Text]: '{out_text}' in {sum(step_times):.1f} ms")


if __name__ == "__main__":
    unittest.main()
