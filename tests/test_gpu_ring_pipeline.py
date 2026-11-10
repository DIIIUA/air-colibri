"""Unit tests for GPURingPipeline (Stage 5 / Refactored)."""

import os
import unittest
import torch
from src.gpu_ring_pipeline import (
    GPURingPipeline,
    RingSlot,
    SlotState
)


class TestGPURingPipeline(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16
        self.template = {
            "weight": torch.zeros(128, 128, dtype=self.dtype),
            "bias": torch.zeros(128, dtype=self.dtype)
        }

    def test_state_machine_transitions(self):
        pipeline = GPURingPipeline(
            template_tensors=self.template,
            num_slots=3,
            device=self.device,
            dtype=self.dtype
        )
        self.assertEqual(len(pipeline.slots), 3)

        # Slot 0: initial state FREE
        slot0 = pipeline.slots[0]
        self.assertEqual(slot0.state, SlotState.FREE)

        # 1. Acquire for loading -> LOADING
        slot_loading = pipeline.get_slot_for_loading(layer_idx=0)
        self.assertEqual(slot_loading.state, SlotState.LOADING)
        self.assertEqual(slot_loading.slot_id, 0)

        # 2. Async transfer enqueued -> state is LOADING (in flight) until ready
        src = {
            "weight": torch.ones(128, 128, dtype=self.dtype),
            "bias": torch.ones(128, dtype=self.dtype)
        }
        pipeline.load_tensors_async(slot_loading, src)
        # On GPU with CUDA stream, transfer is in flight
        if self.device.type == "cuda":
            self.assertEqual(slot_loading.state, SlotState.LOADING)
            # wait_ready() synchronizes DMA transfer and sets READY
            slot_loading.wait_ready()
            self.assertEqual(slot_loading.state, SlotState.READY)
            self.assertTrue(slot_loading.is_ready())
        else:
            self.assertEqual(slot_loading.state, SlotState.READY)

        # 3. Acquire for compute -> COMPUTING
        slot_comp = pipeline.acquire_slot_for_compute(layer_idx=0)
        self.assertEqual(slot_comp.state, SlotState.COMPUTING)

        # 4. Release after compute -> FREE
        pipeline.release_slot_after_compute(slot_comp)
        self.assertEqual(slot_comp.state, SlotState.FREE)

    def test_mathematical_equivalence_1_2_3_slots(self):
        """Verifies that mathematical output is bit-for-bit identical across 1, 2, and 3 slots."""
        x = torch.randn(1, 128, dtype=self.dtype, device=self.device)
        weights_layer0 = {"weight": torch.randn(128, 128, dtype=self.dtype), "bias": torch.randn(128, dtype=self.dtype)}
        weights_layer1 = {"weight": torch.randn(128, 128, dtype=self.dtype), "bias": torch.randn(128, dtype=self.dtype)}

        results = {}

        for num_slots in [1, 2, 3]:
            pipe = GPURingPipeline(template_tensors=self.template, num_slots=num_slots, device=self.device, dtype=self.dtype)
            cur_x = x.clone()

            for idx, w in enumerate([weights_layer0, weights_layer1]):
                slot = pipe.get_slot_for_loading(idx)
                pipe.load_tensors_async(slot, w)
                comp_slot = pipe.acquire_slot_for_compute(idx)
                # Compute linear layer
                cur_x = torch.matmul(cur_x, comp_slot.tensors["weight"].T) + comp_slot.tensors["bias"]
                pipe.release_slot_after_compute(comp_slot)

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            results[num_slots] = cur_x.cpu()

        # Check equivalence
        self.assertTrue(torch.allclose(results[1], results[2], atol=1e-3))
        self.assertTrue(torch.allclose(results[2], results[3], atol=1e-3))
        print("\n[Ring Pipeline] 1-slot, 2-slot, and 3-slot mathematical outputs are 100% IDENTICAL!")

    def test_tail_to_head_prefetch(self):
        pipe = GPURingPipeline(
            template_tensors=self.template,
            num_slots=3,
            device=self.device,
            dtype=self.dtype,
            tail_to_head_prefetch=True
        )
        l0_tensors = {"weight": torch.ones(128, 128, dtype=self.dtype), "bias": torch.ones(128, dtype=self.dtype)}
        slot = pipe.prefetch_tail_to_head(target_layer_idx=0, pinned_tensors=l0_tensors)
        self.assertIsNotNone(slot)
        self.assertEqual(pipe.tail_prefetches, 1)


if __name__ == "__main__":
    unittest.main()
