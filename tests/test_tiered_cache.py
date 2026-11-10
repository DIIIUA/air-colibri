"""Unit tests for DynamicTieredCache, StaticWeightMap, and Zero Disk Re-read."""

import os
import unittest
import torch
from src.aircolibri_engine import (
    AirColibriEngine,
    DynamicTieredCache,
    StaticWeightMap
)

MODEL_DIR = r"C:\air-colibri\models\deepseek-v4-9b-colibri"


class TestTieredCacheAndReUse(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.weight_map = StaticWeightMap(MODEL_DIR)

    def test_static_weight_map_indexing(self):
        self.assertEqual(len(self.weight_map.layers), 32)
        # Check Layer 0
        layer0 = self.weight_map.layers[0]
        self.assertEqual(layer0.layer_idx, 0)
        self.assertEqual(layer0.layer_type, "linear_attention")
        self.assertGreater(len(layer0.active_keys), 0)

        # Check Layer 3 (Full Attention)
        layer3 = self.weight_map.layers[3]
        self.assertEqual(layer3.layer_type, "full_attention")

    def test_tiered_cache_vram_budgets(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Test 2 GB VRAM budget
        cache_2gb = DynamicTieredCache(self.weight_map, vram_budget_gb=2.0, ram_budget_gb=5.0, device=device)
        layers_in_vram_2gb = len(cache_2gb.vram_resident_layers)

        # 2. Test 4 GB VRAM budget
        cache_4gb = DynamicTieredCache(self.weight_map, vram_budget_gb=4.0, ram_budget_gb=5.0, device=device)
        layers_in_vram_4gb = len(cache_4gb.vram_resident_layers)

        # 3. Test 6 GB VRAM budget
        cache_6gb = DynamicTieredCache(self.weight_map, vram_budget_gb=6.0, ram_budget_gb=5.0, device=device)
        layers_in_vram_6gb = len(cache_6gb.vram_resident_layers)

        self.assertLess(layers_in_vram_2gb, layers_in_vram_4gb)
        self.assertLess(layers_in_vram_4gb, layers_in_vram_6gb)
        print(f"\n[Test Budgets] 2GB VRAM: {layers_in_vram_2gb} layers | 4GB VRAM: {layers_in_vram_4gb} layers | 6GB VRAM: {layers_in_vram_6gb} layers")

    def test_zero_disk_reread_guarantee(self):
        """Crucial test: when model fits in VRAM + RAM (6GB VRAM + 5GB RAM), disk read MUST be 0.0 MB!"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        engine = AirColibriEngine(
            MODEL_DIR,
            vram_budget_gb=6.0,
            ram_budget_gb=5.0,
            preload_at_init=True
        )

        x = torch.randn(1, 1, 4096, dtype=torch.float16, device=device)

        # Run 3 consecutive tokens
        for tok in range(3):
            x, step_ms, disk_read_mb, pcie_mb = engine.execute_token_step(x, token_idx=tok)
            # ZERO DISK READ VERIFICATION:
            self.assertEqual(disk_read_mb, 0.0, f"Token {tok} must not read any bytes from disk when fully resident!")
            self.assertFalse(torch.isnan(x).any(), "Output tensor must not contain NaNs")
            print(f"[Full Resident Token {tok}] Step: {step_ms:.1f} ms | Disk Read: {disk_read_mb:.1f} MB | PCIe Transfer: {pcie_mb:.1f} MB")

    def test_partial_vram_budget_streaming(self):
        """Test with constrained 2GB VRAM + 5GB RAM: 19 layers cached in memory, 13 layers streamed."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        engine = AirColibriEngine(
            MODEL_DIR,
            vram_budget_gb=2.0,
            ram_budget_gb=5.0,
            preload_at_init=True
        )

        x = torch.randn(1, 1, 4096, dtype=torch.float16, device=device)
        x, step_ms, disk_read_mb, pcie_mb = engine.execute_token_step(x, token_idx=0)
        # Exactly 13 layers * ~320 MB = ~4160 MB read from SSD, rest from memory
        self.assertGreater(disk_read_mb, 0.0)
        self.assertLess(disk_read_mb, 10240.0)
        print(f"[Partial Stream Token 0] Step: {step_ms:.1f} ms | Disk Read: {disk_read_mb:.1f} MB | PCIe Transfer: {pcie_mb:.1f} MB")


if __name__ == "__main__":
    unittest.main()
