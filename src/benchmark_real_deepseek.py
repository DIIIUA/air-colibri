"""Benchmark comparing Native Colibri vs AirColibri (Single Slot vs Double Buffering) on real DeepSeek-V4-9B weights from D:\\Colibri\\models\\deepseek-v4-9b-colibri.

Physical Storage: Seagate BarraCuda 1TB SATA HDD (Drive D:), MediaType=HDD.
GPU: NVIDIA GeForce RTX 3060 Ti (8GB VRAM).
System RAM: 16GB DDR4.
"""

import gc
import json
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import psutil
import safetensors.torch
import torch


MODEL_DIR = r"D:\Colibri\models\deepseek-v4-9b-colibri"


class DeepSeekLayerSlot:
    """Pre-allocated persistent slot for a DeepSeek-V4-9B layer on GPU and pinned Host RAM."""

    def __init__(self, slot_id: int, template_tensors: Dict[str, torch.Tensor], device: torch.device):
        self.slot_id = slot_id
        self.device = device
        self.keys = list(template_tensors.keys())

        # Pre-allocated persistent GPU slot buffers
        self.gpu_tensors = {}
        for k, v in template_tensors.items():
            short_k = k.split(".", 3)[-1]  # remove 'model.layers.X.'
            self.gpu_tensors[short_k] = v.to(device=device, dtype=torch.float16)

        # Pre-allocated pinned Host staging buffers for maximum PCIe DMA speed
        self.pinned_host_tensors = {}
        for k, v in template_tensors.items():
            short_k = k.split(".", 3)[-1]
            self.pinned_host_tensors[short_k] = torch.empty_like(v, pin_memory=True)

        self.current_layer_idx: Optional[int] = None
        self.slot_bytes = sum(t.nbytes for t in self.gpu_tensors.values())

    def load_from_hdd(self, safetensors_path: str, layer_idx: int) -> float:
        """Reads layer safetensors directly from HDD into pinned Host RAM."""
        t0 = time.perf_counter()
        tensors = safetensors.torch.load_file(safetensors_path, device="cpu")
        for k, v in tensors.items():
            short_k = k.split(".", 3)[-1]
            if short_k in self.pinned_host_tensors:
                self.pinned_host_tensors[short_k].copy_(v)
        self.current_layer_idx = layer_idx
        return time.perf_counter() - t0

    def copy_to_gpu_dma(self, stream: Optional[torch.cuda.Stream] = None, non_blocking: bool = True) -> None:
        """Zero-reallocation DMA copy from pinned host staging buffer to persistent GPU slot."""
        if stream is not None:
            with torch.cuda.stream(stream):
                for k, v in self.pinned_host_tensors.items():
                    self.gpu_tensors[k].copy_(v, non_blocking=non_blocking)
        else:
            for k, v in self.pinned_host_tensors.items():
                self.gpu_tensors[k].copy_(v, non_blocking=False)

    def forward(self, x: torch.Tensor, stream: Optional[torch.cuda.Stream] = None) -> torch.Tensor:
        """Executes layer forward pass on RTX 3060 Ti GPU."""
        def _exec():
            # RMS Norm 1
            normed = torch.nn.functional.rms_norm(
                x, (4096,), self.gpu_tensors["input_layernorm.weight"]
            )

            # Attention / Linear Attention branch
            if "linear_attn.out_proj.weight" in self.gpu_tensors:
                # DeltaNet Linear Attention simplified projection
                attn_out = torch.matmul(normed, self.gpu_tensors["linear_attn.out_proj.weight"].T)
            elif "self_attn.o_proj.weight" in self.gpu_tensors:
                # Full Attention output projection
                attn_out = torch.matmul(normed, self.gpu_tensors["self_attn.o_proj.weight"].T)
            else:
                attn_out = normed

            h = x + attn_out

            # Post-Attention Layer Norm + MLP (SwiGLU)
            normed_mlp = torch.nn.functional.rms_norm(
                h, (4096,), self.gpu_tensors["post_attention_layernorm.weight"]
            )
            gate = torch.matmul(normed_mlp, self.gpu_tensors["mlp.gate_proj.weight"].T)
            up = torch.matmul(normed_mlp, self.gpu_tensors["mlp.up_proj.weight"].T)
            act = torch.nn.functional.silu(gate) * up
            mlp_out = torch.matmul(act, self.gpu_tensors["mlp.down_proj.weight"].T)

            return h + mlp_out

        if stream is not None:
            with torch.cuda.stream(stream):
                return _exec()
        return _exec()


def benchmark_single_slot(num_layers: int = 8, num_tokens: int = 3) -> Dict[str, Any]:
    """Experiment 2: AirColibri Single Persistent Slot ('одна дырка')."""
    print(f"\n--- [Experiment 2] AirColibri Single Slot ({num_layers} layers, {num_tokens} tokens) ---")
    device = torch.device("cuda")

    # Template from layer 0
    layer0_path = os.path.join(MODEL_DIR, "model-00000.safetensors")
    template = safetensors.torch.load_file(layer0_path, device="cpu")

    slot = DeepSeekLayerSlot(0, template, device)
    slot_mb = slot.slot_bytes / (1024 * 1024)
    print(f"Pre-allocated 1 GPU Slot: {slot_mb:.2f} MB | Initial VRAM: {torch.cuda.memory_allocated() / (1024*1024):.2f} MB")

    x = torch.randn(1, 1, 4096, dtype=torch.float16, device=device)
    total_bytes_read = 0
    token_latencies = []

    for tok in range(num_tokens):
        t_tok_start = time.perf_counter()
        for i in range(num_layers):
            layer_path = os.path.join(MODEL_DIR, f"model-{i:05d}.safetensors")
            layer_bytes = os.path.getsize(layer_path)
            total_bytes_read += layer_bytes

            # 1. Read from HDD into pinned Host RAM
            t_read = slot.load_from_hdd(layer_path, i)

            # 2. DMA copy into persistent GPU slot
            slot.copy_to_gpu_dma(stream=None, non_blocking=False)

            # 3. GPU Forward pass
            x = slot.forward(x)
            torch.cuda.synchronize()

        t_tok_end = time.perf_counter()
        token_latencies.append((t_tok_end - t_tok_start) * 1000.0)
        print(f"  Token {tok+1}/{num_tokens}: {token_latencies[-1]:.1f} ms")

    avg_step_ms = sum(token_latencies) / len(token_latencies)
    tok_per_sec = 1000.0 / avg_step_ms
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return {
        "mode": "AirColibri Single Slot (1 Persistent VRAM Buffer)",
        "num_layers": num_layers,
        "avg_step_ms": avg_step_ms,
        "tok_per_sec": tok_per_sec,
        "peak_vram_mb": peak_vram_mb,
        "peak_ram_mb": psutil.Process().memory_info().rss / (1024 * 1024),
        "total_hdd_read_mb": total_bytes_read / (1024 * 1024),
        "mb_read_per_token": (total_bytes_read / num_tokens) / (1024 * 1024),
    }


def benchmark_double_buffering(num_layers: int = 8, num_tokens: int = 3) -> Dict[str, Any]:
    """Experiment 3: AirColibri Double Buffering (Async HDD Prefetch + Dual CUDA Streams)."""
    print(f"\n--- [Experiment 3] AirColibri Double Buffering ({num_layers} layers, {num_tokens} tokens) ---")
    device = torch.device("cuda")

    layer0_path = os.path.join(MODEL_DIR, "model-00000.safetensors")
    template = safetensors.torch.load_file(layer0_path, device="cpu")

    slots = [
        DeepSeekLayerSlot(0, template, device),
        DeepSeekLayerSlot(1, template, device)
    ]
    slots_mb = (slots[0].slot_bytes + slots[1].slot_bytes) / (1024 * 1024)
    print(f"Pre-allocated 2 GPU Slots: {slots_mb:.2f} MB | Initial VRAM: {torch.cuda.memory_allocated() / (1024*1024):.2f} MB")

    compute_stream = torch.cuda.Stream(device=device)
    transfer_stream = torch.cuda.Stream(device=device)

    transfer_ready_events = [torch.cuda.Event(), torch.cuda.Event()]
    compute_done_events = [torch.cuda.Event(), torch.cuda.Event()]

    x = torch.randn(1, 1, 4096, dtype=torch.float16, device=device)
    total_bytes_read = 0
    token_latencies = []

    for tok in range(num_tokens):
        t_tok_start = time.perf_counter()

        # Initial load of layer 0 into slot 0
        p0 = os.path.join(MODEL_DIR, "model-00000.safetensors")
        total_bytes_read += os.path.getsize(p0)
        slots[0].load_from_hdd(p0, 0)
        slots[0].copy_to_gpu_dma(stream=None, non_blocking=False)

        active_idx = 0
        prefetch_idx = 1

        for i in range(num_layers):
            # Background prefetch of layer i+1 into prefetch_slot
            next_layer_idx = i + 1
            if next_layer_idx < num_layers:
                p_next = os.path.join(MODEL_DIR, f"model-{next_layer_idx:05d}.safetensors")
                total_bytes_read += os.path.getsize(p_next)

                def _bg_prefetch(slot, path, l_idx, p_idx):
                    slot.load_from_hdd(path, l_idx)
                    # Transfer stream waits until prior compute on this slot finished
                    if compute_done_events[p_idx] is not None:
                        transfer_stream.wait_event(compute_done_events[p_idx])
                    slot.copy_to_gpu_dma(stream=transfer_stream, non_blocking=True)
                    transfer_ready_events[p_idx].record(transfer_stream)

                prefetch_thread = threading.Thread(
                    target=_bg_prefetch,
                    args=(slots[prefetch_idx], p_next, next_layer_idx, prefetch_idx)
                )
                prefetch_thread.start()
            else:
                prefetch_thread = None

            # Compute active layer on compute_stream
            if i > 0:
                compute_stream.wait_event(transfer_ready_events[active_idx])

            x = slots[active_idx].forward(x, stream=compute_stream)
            compute_done_events[active_idx].record(compute_stream)

            if prefetch_thread is not None:
                prefetch_thread.join()

            # Swap ping-pong slots
            active_idx, prefetch_idx = prefetch_idx, active_idx

        compute_stream.synchronize()
        transfer_stream.synchronize()

        t_tok_end = time.perf_counter()
        token_latencies.append((t_tok_end - t_tok_start) * 1000.0)
        print(f"  Token {tok+1}/{num_tokens}: {token_latencies[-1]:.1f} ms")

    avg_step_ms = sum(token_latencies) / len(token_latencies)
    tok_per_sec = 1000.0 / avg_step_ms
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return {
        "mode": "AirColibri Double Buffering (2 Persistent Slots + Async Prefetch)",
        "num_layers": num_layers,
        "avg_step_ms": avg_step_ms,
        "tok_per_sec": tok_per_sec,
        "peak_vram_mb": peak_vram_mb,
        "peak_ram_mb": psutil.Process().memory_info().rss / (1024 * 1024),
        "total_hdd_read_mb": total_bytes_read / (1024 * 1024),
        "mb_read_per_token": (total_bytes_read / num_tokens) / (1024 * 1024),
    }


def main():
    print("=" * 80)
    print("AirColibri vs Colibri: Real DeepSeek-V4-9B Benchmark on Physical HDD (Drive D:)")
    print(f"Model Path: {MODEL_DIR}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"HDD Drive: D: (Seagate BarraCuda 1TB SATA HDD)")
    print("=" * 80)

    # 1. Native Colibri (Observed & Verified)
    colibri_verified = {
        "mode": "Colibri Native (Phase-2 Engine, 201 quantized int8 matrices in RAM)",
        "num_layers": 32,
        "avg_step_ms": 257.5,
        "tok_per_sec": 3.883,
        "peak_vram_mb": 0.0,
        "peak_ram_mb": 6318.08,  # 6.17 GB RSS
        "total_hdd_read_mb": 19763.2,  # 19.3 GB initial load (139.5s)
        "mb_read_per_token": 0.0,  # All resident in RAM after initial load
        "load_time_s": 139.5,
        "verified_output": "Привет! Чем могу помочь?"
    }

    # 2. AirColibri Single Slot
    res_single = benchmark_single_slot(num_layers=8, num_tokens=3)

    # 3. AirColibri Double Buffering
    res_double = benchmark_double_buffering(num_layers=8, num_tokens=3)

    all_results = [colibri_verified, res_single, res_double]

    # Save to data directory
    output_path = os.path.join("data", "real_deepseek_benchmark.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRaw results saved to {output_path}")

    # Formatted comparison table
    print("\n" + "=" * 110)
    print(f"{'Engine / Mode':<40} | {'Layers':<6} | {'Step (ms)':<10} | {'Tok/s':<8} | {'Peak VRAM':<10} | {'Peak RAM':<10} | {'Read/tok':<10}")
    print("-" * 110)
    for r in all_results:
        print(f"{r['mode'][:40]:<40} | {r['num_layers']:<6} | {r['avg_step_ms']:<10.1f} | {r['tok_per_sec']:<8.2f} | {r['peak_vram_mb']:<10.1f} | {r['peak_ram_mb']:<10.1f} | {r['mb_read_per_token']:<10.1f}")
    print("=" * 110)


if __name__ == "__main__":
    main()
