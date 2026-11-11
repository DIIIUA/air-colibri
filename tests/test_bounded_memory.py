"""Unit tests for Bounded Memory Manager (Stage 5 / Refactored)."""

import os
import unittest
import torch
from src.bounded_memory_manager import (
    ARENA_ALIGN_BYTES,
    BoundedRAMCache,
    BoundedPinnedBufferPool,
    SlotArenaLayout,
    StableWeightManager,
)

MODEL_DIR = r"C:\air-colibri\models\deepseek-v4-9b-colibri"


class TestBoundedMemory(unittest.TestCase):

    def test_bounded_ram_cache_lru_eviction(self):
        cache = BoundedRAMCache(max_cached_layers=3, policy="lru")
        # Add 3 layers
        cache.put(0, {"w": torch.zeros(10)})
        cache.put(1, {"w": torch.zeros(10)})
        cache.put(2, {"w": torch.zeros(10)})
        self.assertEqual(cache.current_layers_count, 3)

        # Access layer 0 to make it recently used
        cache.get(0)

        # Add 4th layer: layer 1 should be evicted (as least recently used)
        cache.put(3, {"w": torch.zeros(10)})
        self.assertEqual(cache.current_layers_count, 3)
        self.assertEqual(cache.evictions, 1)

        self.assertIsNotNone(cache.get(0))
        self.assertIsNone(cache.get(1))  # evicted!
        self.assertIsNotNone(cache.get(2))
        self.assertIsNotNone(cache.get(3))

    def test_bounded_ram_cache_static_pinned(self):
        cache = BoundedRAMCache(max_cached_layers=2, policy="static_pinned")
        cache.put(0, {"w": torch.zeros(10)})
        cache.put(1, {"w": torch.zeros(10)})
        # Trying to add 3rd layer in static pinned mode: should NOT evict layers 0 and 1!
        cache.put(2, {"w": torch.zeros(10)})
        self.assertEqual(cache.current_layers_count, 2)
        self.assertEqual(cache.evictions, 0)
        self.assertIsNotNone(cache.get(0))
        self.assertIsNotNone(cache.get(1))
        self.assertIsNone(cache.get(2))

    def test_pinned_buffer_pool(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        template = {"matrix": torch.zeros(100, 100, dtype=torch.bfloat16)}
        pool = BoundedPinnedBufferPool(template_tensors=template, num_slots=2, dtype=torch.bfloat16)
        self.addCleanup(pool.close)
        self.assertEqual(len(pool.slots), 2)
        self.assertTrue(pool.slots[0]["matrix"].is_pinned())
        self.assertTrue(pool.slots[1]["matrix"].is_pinned())

        # Test copying into slot
        src = {"matrix": torch.ones(100, 100, dtype=torch.bfloat16)}
        slot0 = pool.copy_into_slot(0, src)
        self.assertEqual(slot0["matrix"][0, 0].item(), 1.0)

    def test_stable_weight_manager_stage_a(self):
        manager = StableWeightManager(
            model_dir=MODEL_DIR,
            max_ram_layers=4,
            num_pinned_slots=2,
            cache_policy="static_pinned",
            dtype=torch.bfloat16
        )
        self.addCleanup(manager.close)
        # Load layer 0 and layer 1
        s0 = manager.load_layer_to_pinned_slot(layer_idx=0, target_slot_idx=0)
        self.assertIn("input_layernorm.weight", s0)
        self.assertEqual(s0["input_layernorm.weight"].dtype, torch.bfloat16)

        self.assertEqual(manager.pinned_pool.layout.slot_stride, 418 * 1024 * 1024)
        self.assertEqual(manager.pinned_pool.arena.nbytes, 2 * 418 * 1024 * 1024)
        self.assertEqual(len(s0), 14)

        s1 = manager.load_layer_to_pinned_slot(layer_idx=1, target_slot_idx=1)
        self.assertIn("input_layernorm.weight", s1)
        manager.close()

    def test_slot_arena_layout_non_union_and_disjoint_slots(self):
        # 2 MiB shared + overlapping 4 MiB linear_attn / 2 MiB self_attn => payload ~6 MiB (not 8 MiB union!)
        template = {
            "mlp.weight": (1024, 1024),                 # 2.0 MiB in bf16
            "linear_attn.proj.weight": (2048, 1024),    # 4.0 MiB in bf16
            "self_attn.q_proj.weight": (1024, 1024),    # 2.0 MiB in bf16
        }
        pool = BoundedPinnedBufferPool(template_tensors=template, num_slots=2, dtype=torch.bfloat16)
        self.addCleanup(pool.close)

        layout = pool.layout
        mib = 1024 * 1024
        self.assertEqual(layout.shared_bytes, 2 * mib)
        self.assertEqual(layout.attn_offset, 2 * mib)
        self.assertEqual(layout.variant_bytes["linear_attn"], 4 * mib)
        self.assertEqual(layout.variant_bytes["self_attn"], 2 * mib)
        self.assertEqual(layout.payload_bytes, 6 * mib)
        self.assertEqual(layout.slot_stride, 6 * mib)
        self.assertEqual(layout.slot_stride % ARENA_ALIGN_BYTES, 0)
        self.assertEqual(pool.arena.nbytes, 2 * 6 * mib)

        # Соседние слоты не перекрываются в виртуальных адресах
        s0_ptrs = [(t.data_ptr(), t.data_ptr() + t.nbytes) for t in pool.slots[0].values()]
        s1_ptrs = [(t.data_ptr(), t.data_ptr() + t.nbytes) for t in pool.slots[1].values()]
        self.assertLessEqual(max(hi for _, hi in s0_ptrs), min(lo for lo, _ in s1_ptrs))

        # copy_into_slot возвращает только ключи переданной ветки без stale-алиаса другой ветки
        ret_self = pool.copy_into_slot(
            0,
            {
                "mlp.weight": torch.full((1024, 1024), 2.0, dtype=torch.bfloat16),
                "self_attn.q_proj.weight": torch.full((1024, 1024), 5.0, dtype=torch.bfloat16),
            },
        )
        self.assertEqual(set(ret_self.keys()), {"mlp.weight", "self_attn.q_proj.weight"})
        self.assertEqual(ret_self["self_attn.q_proj.weight"][0, 0].item(), 5.0)


if __name__ == "__main__":
    unittest.main()
