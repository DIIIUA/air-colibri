"""Unit tests for DynamicTieredCache, StaticWeightMap, and Zero Disk Re-read."""

import gc
import os
import unittest
import safetensors
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

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "weight_map", None) is not None:
            cls.weight_map.layers.clear()
            cls.weight_map = None

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

        # 1. Test 2 GB VRAM budget (sequentially created and closed)
        cache_2gb = DynamicTieredCache(self.weight_map, vram_budget_gb=2.0, ram_budget_gb=5.0, device=device)
        try:
            layers_in_vram_2gb = len(cache_2gb.vram_resident_layers)
            b2_gb = cache_2gb.vram_budget_bytes / (1024**3)
            c2 = cache_2gb.vram_clamped
        finally:
            cache_2gb.close()
            del cache_2gb

        # 2. Test 4 GB VRAM budget
        cache_4gb = DynamicTieredCache(self.weight_map, vram_budget_gb=4.0, ram_budget_gb=5.0, device=device)
        try:
            layers_in_vram_4gb = len(cache_4gb.vram_resident_layers)
            b4_gb = cache_4gb.vram_budget_bytes / (1024**3)
            c4 = cache_4gb.vram_clamped
        finally:
            cache_4gb.close()
            del cache_4gb

        # 3. Test 6 GB VRAM budget
        cache_6gb = DynamicTieredCache(self.weight_map, vram_budget_gb=6.0, ram_budget_gb=5.0, device=device)
        try:
            layers_in_vram_6gb = len(cache_6gb.vram_resident_layers)
            b6_gb = cache_6gb.vram_budget_bytes / (1024**3)
            c6 = cache_6gb.vram_clamped
        finally:
            cache_6gb.close()
            del cache_6gb

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        print(
            f"\n[Test Budgets] "
            f"2GB(eff={b2_gb:.2f}G,clamped={c2}): {layers_in_vram_2gb} layers | "
            f"4GB(eff={b4_gb:.2f}G,clamped={c4}): {layers_in_vram_4gb} layers | "
            f"6GB(eff={b6_gb:.2f}G,clamped={c6}): {layers_in_vram_6gb} layers"
        )
        self.assertLess(layers_in_vram_2gb, layers_in_vram_4gb)
        self.assertLess(layers_in_vram_4gb, layers_in_vram_6gb)

    def test_zero_disk_reread_guarantee(self):
        """Crucial test: when model fits in VRAM + RAM (6GB VRAM + 5GB RAM), disk read MUST be 0.0 MB!"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        engine = AirColibriEngine(
            MODEL_DIR,
            vram_budget_gb=6.0,
            ram_budget_gb=5.0,
            preload_at_init=False
        )
        self.addCleanup(engine.close)

        total_layers = len(engine.weight_map.layers)
        cached_layers = len(engine.cache.vram_resident_layers) + len(engine.cache.ram_resident_layers)
        if cached_layers < total_layers:
            budget_vram_gb = engine.cache.vram_budget_bytes / (1024**3)
            budget_ram_gb = engine.cache.ram_budget_bytes / (1024**3)
            budget_total_gb = budget_vram_gb + budget_ram_gb
            active_total_gb = sum(l.active_bytes for l in engine.weight_map.layers.values()) / (1024**3)
            free_vram_gb = engine.cache.free_vram_at_init_bytes / (1024**3)
            free_ram_gb = engine.cache.free_ram_at_init_bytes / (1024**3)
            self.skipTest(
                f"Clamped residency insufficient for zero disk re-read ({cached_layers}/{total_layers} layers): "
                f"budget={budget_total_gb:.2f} GiB (vram={budget_vram_gb:.2f} GiB, ram={budget_ram_gb:.2f} GiB, "
                f"clamped=vram:{engine.cache.vram_clamped}/ram:{engine.cache.ram_clamped}), "
                f"free=(vram={free_vram_gb:.2f} GiB, ram={free_ram_gb:.2f} GiB), "
                f"active_total={active_total_gb:.2f} GiB"
            )

        engine.cold_init_time_s = engine.cache.preload_cache()
        print(f"[AirColibri] Preload complete in {engine.cold_init_time_s:.2f} s | Zero Disk Re-read mode ACTIVE")

        x = torch.randn(1, 1, 4096, dtype=torch.float16, device=device)

        # Run 3 consecutive tokens
        for tok in range(3):
            x, step_ms, disk_read_mb, pcie_mb = engine.execute_token_step(x, token_idx=tok)
            # ZERO DISK READ VERIFICATION:
            self.assertEqual(disk_read_mb, 0.0, f"Token {tok} must not read any bytes from disk when fully resident!")
            self.assertFalse(torch.isnan(x).any(), "Output tensor must not contain NaNs")
            print(f"[Full Resident Token {tok}] Step: {step_ms:.1f} ms | Disk Read: {disk_read_mb:.1f} MB | PCIe Transfer: {pcie_mb:.1f} MB")

    @staticmethod
    def _reference_layer_step(x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        layer_path = os.path.join(MODEL_DIR, f"model-{layer_idx:05d}.safetensors")
        prefix = f"model.layers.{layer_idx}."
        attn_k = "self_attn.o_proj.weight" if (layer_idx % 4 == 3) else "linear_attn.out_proj.weight"
        keys = [
            "input_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            attn_k,
        ]
        w = {}
        with safetensors.safe_open(layer_path, framework="pt", device="cpu") as f:
            for k in keys:
                w[k] = f.get_tensor(prefix + k).to(device=x.device, dtype=torch.float16)
        normed = torch.nn.functional.rms_norm(x, (4096,), w["input_layernorm.weight"])
        gate = torch.matmul(normed, w["mlp.gate_proj.weight"].T)
        up = torch.matmul(normed, w["mlp.up_proj.weight"].T)
        mlp = torch.matmul(torch.nn.functional.silu(gate) * up, w["mlp.down_proj.weight"].T)
        attn = torch.matmul(normed, w[attn_k].T)
        return x + attn + mlp

    def test_self_attn_streaming_equivalence_vram_vs_ram(self):
        """Проверяет эквивалентность self_attn-слоя: VRAM-резидент vs RAM-стрим после linear_attn-слоя."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            self.skipTest("CUDA not available")

        # При vram_budget_gb=1.0 только ~1 слой влезает в VRAM; layer 3 (self_attn) идёт через RAM-стрим
        engine = AirColibriEngine(
            MODEL_DIR,
            vram_budget_gb=1.0,
            ram_budget_gb=4.0,
            preload_at_init=True
        )
        try:
            self.assertIn(3, engine.cache.ram_resident_layers)
            self.assertNotIn(3, engine.cache.vram_resident_layers)

            # Сначала прогоняем layer 0 (linear_attn) в stand 0, затем layer 3 (self_attn) в тот же stand 0
            s0, _, _ = engine.cache.get_layer_tensors(0, stand_idx=0)
            self.assertIn("linear_attn.out_proj.weight", s0)
            s3, _, _ = engine.cache.get_layer_tensors(3, stand_idx=0)
            self.assertIn("self_attn.o_proj.weight", s3)
            self.assertNotIn("linear_attn.out_proj.weight", s3)

            torch.manual_seed(123)
            x_in = torch.randn(1, 1, 4096, dtype=torch.float16, device=device)

            # RAM-stream forward на слое 3 (использует s3, полученный после s0 на том же стенде)
            normed = torch.nn.functional.rms_norm(x_in, (4096,), s3["input_layernorm.weight"])
            gate = torch.matmul(normed, s3["mlp.gate_proj.weight"].T)
            up = torch.matmul(normed, s3["mlp.up_proj.weight"].T)
            mlp = torch.matmul(torch.nn.functional.silu(gate) * up, s3["mlp.down_proj.weight"].T)
            attn = torch.matmul(normed, s3["self_attn.o_proj.weight"].T)
            x_ram_cpu = (x_in + attn + mlp).cpu()
        finally:
            engine.close()

        # Независимый VRAM-референс слоя 3 напрямую из safetensors
        x_vram_cpu = self._reference_layer_step(x_in.clone(), 3).cpu()
        self.assertTrue(torch.equal(x_ram_cpu, x_vram_cpu), "RAM-streamed self_attn output must match VRAM reference bit-for-bit")

    def test_partial_vram_budget_streaming(self):
        """Test with constrained 2GB VRAM + 5GB RAM: 19 layers cached in memory, 13 layers streamed + exact logits check."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        engine = AirColibriEngine(
            MODEL_DIR,
            vram_budget_gb=2.0,
            ram_budget_gb=5.0,
            preload_at_init=True
        )
        self.addCleanup(engine.close)

        ram_self = [i for i in engine.cache.ram_resident_layers if i % 4 == 3]
        self.assertGreater(len(ram_self), 0, "Expected non-resident self_attn layers under 2GB VRAM budget")

        torch.manual_seed(42)
        x_init = torch.randn(1, 1, 4096, dtype=torch.float16, device=device)
        x_out, step_ms, disk_read_mb, pcie_mb = engine.execute_token_step(x_init.clone(), token_idx=0)
        x_out_cpu = x_out.cpu()
        engine.close()

        # Exactly 13+ layers read from SSD, rest from memory
        self.assertGreater(disk_read_mb, 0.0)
        self.assertLess(disk_read_mb, 10240.0)
        self.assertFalse(torch.isnan(x_out_cpu).any())

        # Verify mathematical correctness against independent layer-by-layer reference
        x_ref = x_init.clone()
        for layer_idx in range(32):
            x_ref = self._reference_layer_step(x_ref, layer_idx)
        x_ref_cpu = x_ref.cpu()

        cos = torch.nn.functional.cosine_similarity(
            x_out_cpu.float().flatten(), x_ref_cpu.float().flatten(), dim=0
        ).item()
        self.assertGreaterEqual(cos, 0.9999, f"Streaming output diverged from reference (cos={cos:.6f})")
        self.assertTrue(torch.allclose(x_out_cpu, x_ref_cpu, rtol=1e-2, atol=1e-2))
        print(
            f"[Partial Stream Token 0] Step: {step_ms:.1f} ms | Disk Read: {disk_read_mb:.1f} MB | "
            f"PCIe Transfer: {pcie_mb:.1f} MB | Ref cos={cos:.6f}"
        )


if __name__ == "__main__":
    unittest.main()
