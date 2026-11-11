"""Stress test for Pinned Memory Buffer recycling and DMA Race Condition prevention."""

import unittest
import torch
from src.bounded_memory_manager import BoundedPinnedBufferPool
from src.gpu_ring_pipeline import GPURingPipeline, SlotState


class TestPinnedBufferRaceCondition(unittest.TestCase):

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = torch.device("cuda:0")
        self.dtype = torch.bfloat16
        # 16 MB matrix per slot
        self.template = {"matrix": torch.zeros(2048, 4096, dtype=self.dtype)}

    def test_pinned_buffer_dma_race_prevention(self):
        """Verifies that CPU cannot overwrite a pinned buffer while GPU DMA is in flight."""
        num_slots = 2
        pool = BoundedPinnedBufferPool(
            template_tensors=self.template,
            num_slots=num_slots,
            dtype=self.dtype
        )
        self.addCleanup(pool.close)
        pipe = GPURingPipeline(
            template_tensors=self.template,
            num_slots=num_slots,
            device=self.device,
            dtype=self.dtype
        )
        self.addCleanup(pipe.close)

        # Run 30 rapid buffer recycling cycles
        for cycle in range(30):
            slot_idx = cycle % num_slots
            val = float(cycle + 1)

            # 1. CPU prepares unique data
            src_data = {"matrix": torch.full((2048, 4096), val, dtype=self.dtype)}

            # 2. Copy into pinned slot
            pinned_slot = pool.copy_into_slot(slot_idx, src_data)

            # 3. Launch async DMA transfer to GPU
            gpu_slot = pipe.get_slot_for_loading(cycle)
            pipe.load_tensors_async(gpu_slot, pinned_slot, pinned_pool=pool)

            # 4. CPU immediately attempts to overwrite the SAME pinned slot with garbage
            # Our BoundedPinnedBufferPool must block/synchronize until the DMA is done!
            garbage_data = {"matrix": torch.full((2048, 4096), -999.0, dtype=self.dtype)}
            pool.copy_into_slot(slot_idx, garbage_data)

            # 5. Acquire GPU slot for compute and verify that GPU received the TRUE value (val), NOT garbage (-999.0)!
            comp_slot = pipe.acquire_slot_for_compute(cycle)
            gpu_mat = comp_slot.tensors["matrix"]

            # Compute stream synchronizes with transfer_done_event
            torch.cuda.synchronize()

            # Verify that NO tearing or corruption occurred on the GPU
            is_valid = torch.all(gpu_mat == val).item()
            self.assertTrue(is_valid, f"Race condition detected at cycle {cycle}: expected {val}, found corrupted data!")

            pipe.release_slot_after_compute(comp_slot)

        print(f"\n[Stress Test]: Successfully executed 30 DMA recycling cycles with ZERO races and zero memory corruption!")


if __name__ == "__main__":
    unittest.main()
