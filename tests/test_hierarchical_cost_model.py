"""Unit tests for Hierarchical Cost Model (Stage E)."""

import unittest
from src.hierarchical_cost_model import HierarchicalCostModel, HardwareProfile


class TestHierarchicalCostModel(unittest.TestCase):

    def setUp(self):
        self.model = HierarchicalCostModel()

    def test_mlp_decision_favors_gpu(self):
        # MLP block: ~400 MB, ~200 GFLOPs
        decision = self.model.decide_placement(
            component_name="layer_mlp",
            nbytes=400 * 1024 * 1024,
            flops=200 * 10**9,
            current_tier="RAM_PINNED"
        )
        self.assertEqual(decision["decision"], "GPU")
        self.assertGreater(decision["speedup"], 1.0)
        print(f"\n[CostModel MLP]: Decision = {decision['decision']} | Speedup = {decision['speedup']}x")

    def test_tiny_embedding_favors_cpu(self):
        # 1-token embedding lookup: 8 KB, virtually 0 FLOPs
        decision = self.model.decide_placement(
            component_name="token_embed_lookup",
            nbytes=8192,
            flops=8192,
            current_tier="RAM_PINNED"
        )
        # Due to GPU kernel launch and PCIe overhead, small lookups are faster on CPU
        self.assertEqual(decision["decision"], "CPU")
        print(f"[CostModel Embed Lookup]: Decision = {decision['decision']} (avoids 2GB VRAM waste and PCIe latency)")


if __name__ == "__main__":
    unittest.main()
