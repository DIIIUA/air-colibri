"""Unit test verifying authentic end-to-end inference and token logits with AirColibri."""

import gc
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
            max_ram_layers=2,
            max_ram_bytes=1024 * 2**20,
            dtype=torch.bfloat16
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "engine", None) is not None:
            cls.engine.close()
            cls.engine = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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


class TestIntegratedInferenceQ4(unittest.TestCase):
    """End-to-end интеграция 0c: W4A16 (g64_asym) на 118 MiB арене с lm_head=bf16 по умолчанию."""

    def test_q4_g64_asym_default_lm_head_bf16_and_paris(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        with AirColibriModel(
            model_dir=MODEL_DIR,
            vram_resident_layers=4,
            max_ram_layers=2,
            max_ram_bytes=256 * 2**20,
            dtype=torch.bfloat16,
            weight_format="q4_g64_asym",
            q4_cache_dir=r"C:\air-colibri\data\q4-g64-cache",
            checkpoint_manifest=r"C:\air-colibri\data\integration-20261112\trusted-checkpoint-manifest.json",
        ) as q4_engine:
            self.assertEqual(q4_engine.weight_format, "q4_g64_asym")
            self.assertFalse(q4_engine.quantize_lm_head)
            self.assertTrue(q4_engine.lazy_embed_lookup)

            mib = 1024 * 1024
            self.assertEqual(q4_engine.pipeline.slot_stride, 118 * mib)
            self.assertEqual(q4_engine.pipeline.arena.nbytes, 3 * 118 * mib)
            self.assertTrue(
                all("lut" not in k.lower() for k in q4_engine.weight_manager.pinned_pool.layout.specs)
            )

            out_text, tokens, step_times = q4_engine.generate(
                prompt="The capital of France is",
                max_new_tokens=2,
                temperature=0.0,
            )
            peak_vram_mb = torch.cuda.max_memory_allocated() / mib

            self.assertIn("Paris", out_text)
            self.assertEqual(tokens[-2], 11751)
            self.assertEqual(tokens[-1], 13)
            self.assertLess(peak_vram_mb, 4096.0)
            print(
                f"\n[Test Verified Q4 Text]: '{out_text}' in {sum(step_times):.1f} ms "
                f"| PeakVRAM={peak_vram_mb:.1f} MB | arena={q4_engine.pipeline.arena.nbytes / mib:.1f} MB"
            )


if __name__ == "__main__":
    unittest.main()
