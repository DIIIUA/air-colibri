"""Unit tests for Bounded Memory Manager (Stage A)."""

import os
import unittest
import torch
from src.bounded_memory_manager import (
    BoundedRAMCache,
    BoundedPinnedBufferPool,
    StableWeightManager
)

MODEL_DIR = r"C:\air-colibri\models\deepseek-v4-9b-colibri"


class TestBoundedMemory(unittest.TestCase):

    def test_bounded_ram_cache_lru_eviction(self):
        cache = BoundedRAMCache(max_cached_layers=3)
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

    def test_pinned_buffer_pool(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        template = {"matrix": torch.zeros(100, 100, dtype=torch.float16)}
        pool = BoundedPinnedBufferPool(template_tensors=template, num_slots=2, dtype=torch.float16)
        self.assertEqual(len(pool.slots), 2)
        self.assertTrue(pool.slots[0]["matrix"].is_pinned())
        self.assertTrue(pool.slots[1]["matrix"].is_pinned())

        # Test copying into slot
        src = {"matrix": torch.ones(100, 100, dtype=torch.float16)}
        slot0 = pool.copy_into_slot(0, src)
        self.assertEqual(slot0["matrix"][0, 0].item(), 1.0)

    def test_stable_weight_manager_stage_a(self):
        manager = StableWeightManager(
            model_dir=MODEL_DIR,
            max_ram_layers=4,
            num_pinned_slots=2,
            dtype=torch.float16
        )
        # Load layer 0 and layer 1
        s0 = manager.load_layer_to_pinned_slot(layer_idx=0, target_slot_idx=0)
        self.assertIn("input_layernorm.weight", s0)
        self.assertEqual(s0["input_layernorm.weight"].dtype, torch.float16)

        s1 = manager.load_layer_to_pinned_slot(layer_idx=1, target_slot_idx=1)
        self.assertIn("input_layernorm.weight", s1)


if __name__ == "__main__":
    unittest.main()
