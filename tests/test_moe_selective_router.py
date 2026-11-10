"""Unit tests for MoE Selective Router and Tracker (Stage F)."""

import unittest
from src.moe_selective_router import OLMoERoutingTracker, MoEExpertLFRUCache


class TestMoESelectiveRouter(unittest.TestCase):

    def test_moe_expert_cache_and_hit_rate(self):
        cache = MoEExpertLFRUCache(layer_idx=0, num_experts=64, capacity=8)

        # Token 0: routes experts [0, 1, 2, 3, 4, 5, 6, 7] -> all 8 are misses initially
        hits, misses = cache.query_experts([0, 1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(len(misses), 8)
        self.assertEqual(len(hits), 0)

        # Token 1: routes experts [0, 1, 2, 3, 10, 11, 12, 13] -> [0, 1, 2, 3] are HITS!
        hits, misses = cache.query_experts([0, 1, 2, 3, 10, 11, 12, 13])
        self.assertEqual(len(hits), 4)
        self.assertEqual(hits, [0, 1, 2, 3])
        self.assertEqual(len(misses), 4)
        self.assertGreater(cache.hit_rate, 20.0)
        print(f"\n[MoE Cache Test]: Hit rate after locality run = {cache.hit_rate:.1f}%")

    def test_olmoe_routing_tracker_co_activation(self):
        tracker = OLMoERoutingTracker(num_layers=4, num_experts=64, vram_slots_per_layer=8)

        # Simulate co-activation: experts 5 and 9 always co-occur
        for _ in range(5):
            tracker.on_router_step(layer_idx=0, routed_experts=[5, 9, 12, 14])

        # Predict candidates when expert 5 is active
        candidates = tracker.predict_next_candidates(current_active=[5], top_n=2)
        self.assertIn(9, candidates)
        print(f"[MoE Predictor Test]: Predicted co-activated experts: {candidates}")


if __name__ == "__main__":
    unittest.main()
