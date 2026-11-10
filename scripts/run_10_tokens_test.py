"""Run verified 10-token genuine inference test with AirColibri Integrated Engine."""

import os
import sys
import time
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.aircolibri_integrated import AirColibriModel

MODEL_DIR = r"C:\air-colibri\models\deepseek-v4-9b-colibri"

def main():
    print("=" * 80)
    print("AirColibri: Real 10-Token Generation Test on DeepSeek-V4-9B")
    print(f"Device: {torch.cuda.get_device_name(0)} | VRAM: 8 GB")
    print("=" * 80)

    t0_init = time.perf_counter()
    # 4 layers resident in VRAM (~1.6 GB), remaining 28 streamed from RAM
    engine = AirColibriModel(
        model_dir=MODEL_DIR,
        vram_resident_layers=4,
        dtype=torch.bfloat16
    )
    init_time_s = time.perf_counter() - t0_init
    print(f"[Init] Model initialized in {init_time_s:.2f} s")

    prompt = "The capital of France is"
    out_text, tokens, step_times = engine.generate(prompt=prompt, max_new_tokens=10, temperature=0.0)

    print("\n" + "=" * 80)
    print("RESULTS:")
    print(f"Prompt: '{prompt}'")
    print(f"Full Generated Text: '{out_text}'")
    print(f"Step Times: {[round(t, 1) for t in step_times]} ms")
    print(f"Average Step Time: {sum(step_times) / len(step_times):.1f} ms")
    print(f"Steady State (tokens 2..10): {sum(step_times[1:]) / len(step_times[1:]):.1f} ms")
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / (1024*1024):.1f} MB")
    print("=" * 80)

if __name__ == "__main__":
    main()
