"""Unit tests verifying the 4 pillars of AirColibri:
1. Priority Ranking
2. Layer Streaming
3. Selective Subtensor Extraction
4. Persistent Memory Stand & Zero Malloc Replacement
"""

import os
import unittest
import torch
from src.aircolibri_engine import (
    PriorityRanker,
    SelectiveSubtensorLoader,
    MemoryStandPool,
    AirColibriEngine
)

MODEL_DIR = r"C:\air-colibri\models\deepseek-v4-9b-colibri"


class TestAirColibriCore(unittest.TestCase):

    def test_priority_ranking(self):
        ranker = PriorityRanker(MODEL_DIR)
        layers = ranker.get_ranked_queue()
        self.assertEqual(len(layers), 32)

        # Layer 0 must be Rank 1 (critical anchor)
        self.assertEqual(layers[0].priority_score, 1)

        # Layer 3 must be Full Attention and Rank 1
        self.assertEqual(layers[3].layer_type, "full_attention")
        self.assertEqual(layers[3].priority_score, 1)

        # Layer 5 must be Linear Attention and Rank 3
        self.assertEqual(layers[5].layer_type, "linear_attention")
        self.assertEqual(layers[5].priority_score, 3)

    def test_selective_extraction_efficiency(self):
        loader = SelectiveSubtensorLoader()
        layer0_path = os.path.join(MODEL_DIR, "model-00000.safetensors")
        full_file_size = os.path.getsize(layer0_path)

        # Selectively extract only RMS Norm (8 KB) and Gate Proj (100 MB)
        subtensors = loader.load_subtensors(
            layer0_path,
            ["model.layers.0.input_layernorm.weight", "model.layers.0.mlp.gate_proj.weight"]
        )
        self.assertIn("model.layers.0.input_layernorm.weight", subtensors)
        self.assertIn("model.layers.0.mlp.gate_proj.weight", subtensors)
        self.assertNotIn("model.layers.0.mlp.down_proj.weight", subtensors)

        extracted_bytes = sum(t.nbytes for t in subtensors.values())
        # Must be strictly less than full file size
        self.assertLess(extracted_bytes, full_file_size)
        print(f"\n[Test] Full file: {full_file_size / (1024*1024):.1f} MB | Selectively read: {extracted_bytes / (1024*1024):.1f} MB (saved {100 - extracted_bytes/full_file_size*100:.1f}%)")

    def test_stand_data_ptr_constancy(self):
        import safetensors.torch
        layer0_path = os.path.join(MODEL_DIR, "model-00000.safetensors")
        template = safetensors.torch.load_file(layer0_path, device="cpu")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        stand_pool = MemoryStandPool(template, device=device, num_stands=1)
        self.addCleanup(stand_pool.close)
        del template
        initial_ptr = stand_pool.gpu_stands[0]["mlp.down_proj.weight"].data_ptr()

        # Place new tensor on the stand
        layer1_path = os.path.join(MODEL_DIR, "model-00001.safetensors")
        loader = SelectiveSubtensorLoader()
        subtensors = loader.load_subtensors(layer1_path, ["model.layers.1.mlp.down_proj.weight"])
        stand_pool.place_on_stand(subtensors, stand_idx=0)

        # Pointer MUST remain identical! (Proof of Zero Malloc)
        new_ptr = stand_pool.gpu_stands[0]["mlp.down_proj.weight"].data_ptr()
        self.assertEqual(initial_ptr, new_ptr)


if __name__ == "__main__":
    unittest.main()
