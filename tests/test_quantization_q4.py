"""Юнит- и контрактные тесты квантизации W4A16 (g64_asym) и Q4-раскладки 0c."""

import os
import tempfile
import unittest
import safetensors
from safetensors.torch import save_file
import torch
import torch.nn.functional as F

import src.quantization_q4 as q4_mod
from src.bounded_memory_manager import (
    ARENA_ALIGN_BYTES,
    LazyGlobalsMap,
    StableWeightManager,
    align_up,
)
from src.quantization_q4 import (
    Q4_GROUP_SIZE,
    Q4_SCRATCH_ELEMS,
    ColibriQ4Linear,
    _compute_tile_rows,
    dequantize_q4_tensor,
    is_q4_quantizable,
    q4_subkey_names,
    q4_tiled_linear,
    quantize_tensor_q4,
)

MODEL_DIR = r"C:\air-colibri\models\deepseek-v4-9b-colibri"


class TestQuantizationQ4(unittest.TestCase):

    def test_bitwise_unpack_no_lut_and_exact_nibbles(self):
        # Ни LUT-кэша, ни вспомогательных LUT-функций больше не существует
        self.assertFalse(hasattr(q4_mod, "get_nibble_lut"))
        self.assertFalse(hasattr(q4_mod, "clear_nibble_lut_cache"))
        self.assertFalse(hasattr(q4_mod, "_NIBBLE_LUT_CACHE"))

        # Проверяем распаковку байта b -> (b & 0x0F, b >> 4) на граничных значениях
        edge_bytes = [0x00, 0x0F, 0x10, 0xA5, 0xFF]
        pattern = torch.tensor(edge_bytes, dtype=torch.uint8)
        qw = pattern.repeat((2, Q4_GROUP_SIZE // 2))[:, : Q4_GROUP_SIZE // 2].contiguous()
        # Повторим 7 раз, чтобы покрыть ровно 32 байта = 64 колонки (1 группа)
        qw = pattern.repeat(1, 7)[:, : Q4_GROUP_SIZE // 2].repeat(2, 1).contiguous()
        scales = torch.tensor([[0.5], [1.25]], dtype=torch.bfloat16)
        zeros = torch.tensor([[-2.0], [3.0]], dtype=torch.bfloat16)

        w_rec = dequantize_q4_tensor(qw, scales, zeros, group_size=Q4_GROUP_SIZE, dtype=torch.bfloat16)
        self.assertEqual(tuple(w_rec.shape), (2, Q4_GROUP_SIZE))
        lo = (qw.to(torch.int64) & 0x0F).to(torch.bfloat16)
        hi = (qw.to(torch.int64) >> 4).to(torch.bfloat16)
        ref = torch.stack([lo, hi], dim=-1).reshape(2, Q4_GROUP_SIZE) * scales + zeros
        self.assertTrue(torch.equal(w_rec, ref))

    def test_q4_layout_composition_no_lut_and_exact_stride(self):
        """Условие D17 #1 + Z5: в layout.specs нет LUT-тензоров, только qweight/scales/zeros и bf16-остатки."""
        wm = StableWeightManager(
            model_dir=MODEL_DIR,
            max_ram_layers=1,
            num_pinned_slots=3,
            dtype=torch.bfloat16,
            weight_format="q4_g64_asym",
        )
        self.addCleanup(wm.close)

        layout = wm.pinned_pool.layout
        specs = layout.specs

        expected_q4_subkeys = set()
        expected_bf16_remainders = set()
        for l_idx in (0, 3):
            path = os.path.join(MODEL_DIR, f"model-{l_idx:05d}.safetensors")
            with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                for full_k in f.keys():
                    short_k = full_k.replace(f"model.layers.{l_idx}.", "")
                    shape = tuple(f.get_slice(full_k).get_shape())
                    if is_q4_quantizable(short_k, shape, group_size=Q4_GROUP_SIZE):
                        self.assertEqual(shape[1] % Q4_GROUP_SIZE, 0, f"{short_k} shape={shape} not divisible by 64")
                        expected_q4_subkeys.update(q4_subkey_names(short_k))
                    else:
                        expected_bf16_remainders.add(short_k)

        self.assertEqual(set(specs.keys()), expected_q4_subkeys | expected_bf16_remainders)
        for name, spec in specs.items():
            low_name = name.lower()
            self.assertNotIn("lut", low_name)
            self.assertNotIn("table", low_name)
            self.assertNotIn("nibble", low_name)
            self.assertNotIn("codebook", low_name)
            if name.endswith(".qweight"):
                self.assertEqual(spec.dtype, torch.uint8)
            else:
                self.assertEqual(spec.dtype, torch.bfloat16)

        self.assertEqual(
            layout.shared_bytes,
            sum(s.nbytes for s in specs.values() if s.region == "shared"),
        )
        self.assertEqual(
            layout.variant_bytes["linear_attn"],
            sum(s.nbytes for s in specs.values() if s.region == "linear_attn"),
        )
        self.assertEqual(
            layout.variant_bytes["self_attn"],
            sum(s.nbytes for s in specs.values() if s.region == "self_attn"),
        )
        self.assertEqual(layout.payload_bytes, max(s.offset_bytes + s.nbytes for s in specs.values()))
        self.assertEqual(layout.slot_stride, align_up(layout.payload_bytes, ARENA_ALIGN_BYTES))
        self.assertEqual(layout.slot_stride, 118 * 1024 * 1024)
        self.assertEqual(wm.pinned_pool.arena.nbytes, 3 * 118 * 1024 * 1024)

    def test_q4_tiled_linear_exactness_and_bounded_scratch(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        device = torch.device("cuda")
        torch.manual_seed(7)
        w = torch.randn((12288, 4096), dtype=torch.bfloat16, device=device) * 0.05
        x = torch.randn((2, 4, 4096), dtype=torch.bfloat16, device=device)
        qw, sc, zr = quantize_tensor_q4(w, group_size=Q4_GROUP_SIZE, quant_device=device, target_device=device)

        w_deq = dequantize_q4_tensor(qw, sc, zr, group_size=Q4_GROUP_SIZE, dtype=torch.bfloat16)
        y_ref = F.linear(x, w_deq)
        del w_deq, w

        # Прогрев cuBLAS workspace под геометрию тайла до снятия base_alloc
        tile_rows = _compute_tile_rows(12288, 4096, Q4_SCRATCH_ELEMS)
        _warm_scratch = torch.empty(Q4_SCRATCH_ELEMS, dtype=torch.bfloat16, device=device)
        _ = F.linear(x.reshape(-1, 4096), _warm_scratch[: tile_rows * 4096].view(tile_rows, 4096))
        del _warm_scratch, _

        # Базу VRAM снимаем ДО выделения scratch (12 MiB) и ColibriQ4Linear (условие D20 #1)
        torch.cuda.synchronize()
        base_alloc = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        scratch = torch.empty(Q4_SCRATCH_ELEMS, dtype=torch.bfloat16, device=device)
        mod = ColibriQ4Linear(
            out_features=12288,
            in_features=4096,
            scratch_provider=lambda _d: scratch,
            device=torch.device("meta"),
        )
        mod.bind_q4_tensors(qw, sc, zr)
        torch.cuda.synchronize()
        pre_fwd_alloc = torch.cuda.memory_allocated()
        y_tiled = mod(x)
        torch.cuda.synchronize()
        peak_from_pre_scratch_mib = (torch.cuda.max_memory_allocated() - base_alloc) / (1024 ** 2)
        peak_over_scratch_mib = (torch.cuda.max_memory_allocated() - pre_fwd_alloc) / (1024 ** 2)
        scratch_mib = scratch.nbytes / (1024 ** 2)
        tile_tmp_budget_mib = (tile_rows * 4096) / (1024 ** 2)  # 1024 * 4096 B = 4.0 MiB
        print(
            f"\n[Q4 Tiled Linear] tile_rows={tile_rows} | scratch={scratch_mib:.2f} MiB | "
            f"PeakVRAM_from_pre_scratch={peak_from_pre_scratch_mib:.2f} MiB | "
            f"over_scratch={peak_over_scratch_mib:.2f} MiB (budget={tile_tmp_budget_mib:.2f} MiB)"
        )

        w_small = torch.randn((512, 4096), dtype=torch.bfloat16, device=device) * 0.05
        qw_s, sc_s, zr_s = quantize_tensor_q4(w_small, group_size=Q4_GROUP_SIZE, quant_device=device, target_device=device)
        y_small_ref = F.linear(x, dequantize_q4_tensor(qw_s, sc_s, zr_s, group_size=Q4_GROUP_SIZE, dtype=torch.bfloat16))
        y_small_tiled = q4_tiled_linear(x, qw_s, sc_s, zr_s, group_size=Q4_GROUP_SIZE, scratch=scratch)
        self.assertTrue(torch.equal(y_small_ref, y_small_tiled))

        self.assertTrue(
            torch.allclose(y_ref, y_tiled, rtol=2e-2, atol=0.1),
            f"max_diff={(y_ref - y_tiled).abs().max().item()}",
        )
        # Порог из _compute_tile_rows (D23 #1): тёплый over_scratch <= tile*K байт и pre-scratch <= scratch + бюджет + 8 MiB
        self.assertLessEqual(
            peak_over_scratch_mib,
            tile_tmp_budget_mib,
            f"Warm over_scratch PeakVRAM too high: {peak_over_scratch_mib:.2f} > {tile_tmp_budget_mib:.2f} MiB",
        )
        self.assertLessEqual(
            peak_from_pre_scratch_mib,
            scratch_mib + tile_tmp_budget_mib + 8.0,
            f"Pre-scratch PeakVRAM too high: {peak_from_pre_scratch_mib:.2f} MiB",
        )

    def test_q4_real_layer_weight_cosine_f64(self):
        min_cos = 1.0
        for l_idx in (0, 3):
            path = os.path.join(MODEL_DIR, f"model-{l_idx:05d}.safetensors")
            with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                for full_k in f.keys():
                    short_k = full_k.replace(f"model.layers.{l_idx}.", "")
                    w = f.get_tensor(full_k)
                    if not is_q4_quantizable(short_k, tuple(w.shape), group_size=Q4_GROUP_SIZE):
                        continue
                    qw, sc, zr = quantize_tensor_q4(w, group_size=Q4_GROUP_SIZE, target_device=torch.device("cpu"))
                    w_rec = dequantize_q4_tensor(qw, sc, zr, group_size=Q4_GROUP_SIZE, dtype=torch.bfloat16)
                    wd = w.double().flatten()
                    wrd = w_rec.double().flatten()
                    cos = float((torch.dot(wd, wrd) / (wd.norm() * wrd.norm())).item())
                    self.assertLessEqual(cos, 1.0 + 1e-9)
                    self.assertGreaterEqual(cos, 0.995, f"L{l_idx} {short_k} cos={cos:.5f} < 0.995")
                    min_cos = min(min_cos, cos)
        self.assertGreaterEqual(min_cos, 0.995)

    def test_lazy_embed_lookup_equivalence(self):
        globals_path = os.path.join(MODEL_DIR, "model-globals.safetensors")
        gmap = LazyGlobalsMap(globals_path, dtype=torch.bfloat16)
        self.addCleanup(gmap.close)
        token_ids = torch.tensor([[11751, 13, 0, 220]], dtype=torch.long)
        got = gmap.lookup_rows("model.embed_tokens.weight", token_ids, dtype=torch.bfloat16)
        self.assertNotIn("model.embed_tokens.weight", gmap._materialized)

        sl = gmap.get_tensor_slice("model.embed_tokens.weight")
        ref = torch.stack([
            torch.stack([sl[int(idx)].to(torch.bfloat16) for idx in row], dim=0)
            for row in token_ids
        ], dim=0)
        self.assertTrue(torch.equal(got, ref))

    def test_lazy_globals_map_chunked_read_equivalence(self):
        torch.manual_seed(11)
        big_f16 = (torch.randn((9000, 64)) * 0.1).to(torch.float16)
        small_f16 = (torch.randn((256,)) * 0.1).to(torch.float16)
        with tempfile.TemporaryDirectory() as tmpdir:
            sf_path = os.path.join(tmpdir, "globals_tmp.safetensors")
            save_file({"big.weight": big_f16, "small.weight": small_f16}, sf_path)
            gmap = LazyGlobalsMap(sf_path, dtype=torch.bfloat16)
            try:
                got_big = gmap["big.weight"]
                got_small = gmap["small.weight"]
                self.assertEqual(got_big.dtype, torch.bfloat16)
                self.assertEqual(got_small.dtype, torch.bfloat16)
                self.assertTrue(torch.equal(got_big, big_f16.to(torch.bfloat16)))
                self.assertTrue(torch.equal(got_small, small_f16.to(torch.bfloat16)))
            finally:
                gmap.close()


if __name__ == "__main__":
    unittest.main()
