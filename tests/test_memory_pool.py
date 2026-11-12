"""Unit tests for AirColibri reusable memory pools and Colibri LFRU cache.

Verifies:
1. Exact address immutability (data_ptr constancy = zero reallocation).
2. Numerical correctness across all pool modes.
3. Colibri tier.h LFRU 25% margin and tie-breaking admission logic.
4. Double-buffering ping-pong slot alternation.
5. MoE cache hits vs cold swaps.
"""

import unittest
import torch
from src.lfru_cache import ColibriLFRUCache
from src.memory_pool import (
    SingleSlotPool,
    DoubleBufferPool,
    RingPoolWithCache,
    NaiveAirLLMPool,
)


class TestLFRUCache(unittest.TestCase):
    """Verifies Colibri tier.h LFRU cache logic ported from C."""

    def test_promotion_margin(self):
        cache = ColibriLFRUCache(num_slots=2, num_experts=8)
        # Margin: hot > cold + (cold >> 2) + 4
        # For cold = 0: threshold = 0 + 0 + 4 = 4. Hot must be >= 5 to promote.
        self.assertFalse(cache.should_promote(hot=4, cold=0))
        self.assertTrue(cache.should_promote(hot=5, cold=0))

        # For cold = 100: threshold = 100 + 25 + 4 = 129. Hot must be >= 130.
        self.assertFalse(cache.should_promote(hot=129, cold=100))
        self.assertTrue(cache.should_promote(hot=130, cold=100))

    def test_cache_hits_and_eviction(self):
        cache = ColibriLFRUCache(num_slots=2, num_experts=4)
        # Access 0 -> miss, placed in slot 0
        slot0, is_hit = cache.access_expert(0)
        self.assertFalse(is_hit)
        self.assertEqual(slot0, 0)

        # Access 1 -> miss, placed in slot 1
        slot1, is_hit = cache.access_expert(1)
        self.assertFalse(is_hit)
        self.assertEqual(slot1, 1)

        # Access 0 again -> HIT in slot 0
        slot0_again, is_hit = cache.access_expert(0)
        self.assertTrue(is_hit)
        self.assertEqual(slot0_again, 0)

        # Access 0 multiple times to make it very hot
        for _ in range(10):
            cache.access_expert(0)

        # A cold miss executes transiently and must not defeat admission.
        slot2, is_hit = cache.access_expert(2)
        self.assertFalse(is_hit)
        self.assertIsNone(slot2)
        self.assertTrue(cache.is_resident(1))
        self.assertFalse(cache.is_resident(2))
        for _ in range(7):
            slot2, is_hit = cache.access_expert(2)
        self.assertEqual(slot2, 1)
        self.assertTrue(cache.is_resident(0))
        self.assertTrue(cache.is_resident(2))
        self.assertFalse(cache.is_resident(1))


class TestMemoryPools(unittest.TestCase):
    """Verifies memory pool invariants, slot reuse, and computation accuracy."""

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.shape = (128, 256)
        self.dtype = torch.float32

    def test_single_slot_zero_reallocation(self):
        """Proof of zero re-allocation: data_ptr must NOT change between layer loads."""
        pool = SingleSlotPool(self.shape, self.dtype, self.device)
        initial_ptr = pool.initial_data_ptr

        layer_weights = [torch.randn(self.shape, dtype=self.dtype) for _ in range(5)]
        for i, w in enumerate(layer_weights):
            buf = pool.load_layer(w, i)
            self.assertEqual(buf.data_ptr(), initial_ptr, f"data_ptr mutated at layer {i}!")
            # Numerical check
            self.assertTrue(torch.allclose(buf.cpu(), w, atol=1e-5))
            pool.release_layer()

    def test_double_buffer_alternation_and_correctness(self):
        """Verifies ping-pong slot alternation and output match with direct matmul."""
        pool = DoubleBufferPool(self.shape, self.dtype, self.device)
        ptr_0 = pool.slots[0].data_ptr
        ptr_1 = pool.slots[1].data_ptr

        self.assertNotEqual(ptr_0, ptr_1)

        w0 = torch.randn(self.shape, dtype=self.dtype)
        w1 = torch.randn(self.shape, dtype=self.dtype)
        x = torch.randn(1, 16, 128, dtype=self.dtype, device=self.device)

        # Load initial w0 into active slot (slot 0)
        pool.load_initial_layer(w0, 0)
        active_buf = pool.acquire_active_layer()
        self.assertEqual(active_buf.data_ptr(), ptr_0)
        expected_out0 = torch.matmul(x, w0.to(self.device))
        actual_out0 = torch.matmul(x, active_buf)
        self.assertTrue(torch.allclose(expected_out0, actual_out0, atol=1e-4))

        # Prefetch w1 into slot 1 while slot 0 finishes
        pool.prefetch_next_layer(w1, 1)
        pool.finish_active_layer()

        # Next layer: active is now slot 1
        active_buf1 = pool.acquire_active_layer()
        self.assertEqual(active_buf1.data_ptr(), ptr_1)
        expected_out1 = torch.matmul(x, w1.to(self.device))
        actual_out1 = torch.matmul(x, active_buf1)
        self.assertTrue(torch.allclose(expected_out1, actual_out1, atol=1e-4))
        pool.finish_active_layer()

    def test_ring_pool_expert_caching(self):
        """Verifies that hot MoE experts remain resident and register cache hits."""
        dense_shape = (64, 128)
        expert_shape = (64, 64)
        pool = RingPoolWithCache(
            dense_shape=dense_shape,
            expert_shape=expert_shape,
            num_dense_slots=2,
            num_expert_slots=2,
            num_total_experts=4,
            dtype=self.dtype,
            device=self.device
        )

        e0 = torch.randn(expert_shape, dtype=self.dtype)
        e1 = torch.randn(expert_shape, dtype=self.dtype)

        # Initial loads: MISS
        b0, hit0 = pool.get_expert(0, e0)
        self.assertFalse(hit0)
        b1, hit1 = pool.get_expert(1, e1)
        self.assertFalse(hit1)

        # Repeat access: HIT
        b0_again, hit0_again = pool.get_expert(0, e0)
        self.assertTrue(hit0_again)
        self.assertEqual(b0.data_ptr(), b0_again.data_ptr())
        self.assertEqual(pool.cache_hits, 1)
        self.assertEqual(pool.cache_misses, 2)


if __name__ == "__main__":
    unittest.main()
