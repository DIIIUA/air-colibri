"""Unit tests for Adaptive Cross-Layer Selective Streaming Engine."""

import os
import unittest
import torch
from src.adaptive_streaming import (
    AdaptiveCrossLayerEngine,
    CancellablePrefetcher,
    ComponentType,
    CrossLayerPlanMap,
    ResidencyTier
)

MODEL_DIR = r"C:\air-colibri\models\deepseek-v4-9b-colibri"


class TestAdaptiveStreaming(unittest.TestCase):

    def test_cross_layer_plan_map(self):
        plan_map = CrossLayerPlanMap(num_layers=4)
        plan_map.register_component(
            layer_idx=0,
            comp_id="expert_1",
            comp_type=ComponentType.ROUTED_EXPERT,
            keys=["k1"],
            nbytes=1000,
            initial_tier=ResidencyTier.DISK_STREAM
        )

        # Record multiple hits on expert_1
        for step in range(5):
            plan_map.record_access(0, "expert_1", step)

        meta = plan_map.matrix[0]["expert_1"]
        self.assertEqual(meta.access_count, 5)

        # Trigger replan: expert_1 should be promoted due to high frequency
        result = plan_map.trigger_replan(trigger_reason="test_frequency", vram_slot_budget=2, ram_slot_budget=2)
        self.assertGreater(result["promotions"], 0)
        self.assertIn(meta.current_tier, (ResidencyTier.VRAM_STAND, ResidencyTier.RAM_PINNED))

    def test_cancellable_prefetcher(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        prefetcher = CancellablePrefetcher(device=device)

        # Submit 3 requests
        r1 = prefetcher.submit(layer_idx=1, component_id="exp_1", target_slot=0, priority=1, token_step=0, nbytes_est=50000)
        r2 = prefetcher.submit(layer_idx=2, component_id="exp_2", target_slot=1, priority=2, token_step=0, nbytes_est=50000)
        r3 = prefetcher.submit(layer_idx=0, component_id="exp_old", target_slot=0, priority=1, token_step=0, nbytes_est=50000)

        # Cancel stale requests for layer <= 1 at step 0
        cancelled = prefetcher.cancel_stale_requests(current_step=0, current_layer=1)
        self.assertGreater(cancelled, 0)
        self.assertTrue(r1.is_cancelled() or r3.is_cancelled())
        self.assertGreater(prefetcher.bytes_saved_by_cancel, 0)

        prefetcher.shutdown()

    def test_adaptive_cross_layer_engine_execution(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        engine = AdaptiveCrossLayerEngine(
            model_dir=MODEL_DIR,
            num_layers=4,
            device=device,
            replan_interval=2
        )

        # Execute 3 steps across 4 layers
        for step in range(3):
            for l in range(4):
                slot_idx = l % 2
                active_expert = (step + l) % 8
                ms, was_cancelled = engine.execute_layer_with_safeguards(
                    layer_idx=l,
                    active_expert_idx=active_expert,
                    slot_idx=slot_idx,
                    step=step
                )
                self.assertGreater(ms, 0.0)

        # Check that replan occurred
        self.assertGreater(engine.plan_map.replan_count, 0)
        engine.shutdown()


if __name__ == "__main__":
    unittest.main()
