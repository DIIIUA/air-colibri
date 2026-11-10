"""Unit tests for Progressive Live Acceleration (Stage D)."""

import os
import time
import unittest
import torch
from src.progressive_acceleration import ProgressiveLiveAttacher


class TestProgressiveLiveAcceleration(unittest.TestCase):

    def test_progressive_layer_attachment(self):
        attached_layers = []

        def on_attach(layer_idx):
            attached_layers.append(layer_idx)

        attacher = ProgressiveLiveAttacher(
            num_layers=8,
            initial_gpu_layers=2,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            attach_callback=on_attach
        )

        # Initially, layers 0 and 1 are on GPU, others are not
        self.assertTrue(attacher.is_layer_on_gpu(0))
        self.assertTrue(attacher.is_layer_on_gpu(1))
        self.assertFalse(attacher.is_layer_on_gpu(2))
        self.assertFalse(attacher.is_layer_on_gpu(7))

        # Simulate ongoing token generation loop while background thread works
        token_step = 0
        while attacher.attached_count < 8 and token_step < 50:
            time.sleep(0.01)
            token_step += 1

        attacher.stop()

        # Check that all layers became attached progressively
        self.assertEqual(attacher.attached_count, 8)
        self.assertTrue(attacher.is_layer_on_gpu(7))
        self.assertEqual(len(attached_layers), 6)  # layers 2..7 attached in background
        print(f"\n[Progressive Acceleration] Successfully attached 6 layers live in {token_step} token cycles!")


if __name__ == "__main__":
    unittest.main()
