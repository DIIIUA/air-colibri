"""Hybrid Layer Runner: Executes transformer layers using various memory pool strategies.

Supports:
- Sequential Dense transformer layers
- Mixture of Experts (MoE) layers with router-directed execution
- Profiling of Peak VRAM, Peak RAM, Transfer Volume, and Step Latency
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import torch
import psutil
from src.memory_pool import (
    NaiveAirLLMPool,
    SingleSlotPool,
    DoubleBufferPool,
    RingPoolWithCache,
)


class HybridLayerRunner:
    """Executes multi-layer transformer forward passes across different memory pool configurations."""

    def __init__(
        self,
        num_layers: int = 16,
        hidden_dim: int = 2048,
        inter_dim: int = 4096,
        num_experts: int = 16,
        top_k: int = 2,
        dtype: torch.dtype = torch.float16,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.inter_dim = inter_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.dtype = dtype
        self.device = device

        # Pre-generate synthetic layer weights on Host CPU (pinned if CUDA)
        self.dense_shape = (hidden_dim, inter_dim)
        self.expert_shape = (hidden_dim, inter_dim // 2)

        self.dense_weights_host = [
            torch.randn(self.dense_shape, dtype=dtype) for _ in range(num_layers)
        ]
        self.expert_weights_host = [
            torch.randn(self.expert_shape, dtype=dtype) for _ in range(num_experts)
        ]

    def _sync(self) -> None:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

    def run_naive_airllm(self, batch_size: int = 1, seq_len: int = 1, num_tokens: int = 5) -> Dict[str, Any]:
        """Runs baseline Naive AirLLM execution (dynamic allocation + empty_cache per layer)."""
        pool = NaiveAirLLMPool(self.dense_shape, self.dtype, self.device)
        x = torch.randn(batch_size, seq_len, self.hidden_dim, dtype=self.dtype, device=self.device)

        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        start_time = time.perf_counter()
        token_latencies = []

        for tok in range(num_tokens):
            t0 = time.perf_counter()
            for i in range(self.num_layers):
                weight = pool.prepare_layer(self.dense_weights_host[i], i)
                # Compute forward pass
                out = torch.matmul(x, weight)
                x = out[:, :, :self.hidden_dim]
                self._sync()
                pool.release_layer()
            t1 = time.perf_counter()
            token_latencies.append((t1 - t0) * 1000.0)

        total_time = time.perf_counter() - start_time
        peak_vram = torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" and torch.cuda.is_available() else 0

        return {
            "mode": "Naive AirLLM (Dynamic Alloc + empty_cache)",
            "total_time_s": total_time,
            "avg_step_ms": sum(token_latencies) / len(token_latencies),
            "tok_per_sec": num_tokens / total_time,
            "peak_vram_bytes": peak_vram,
            "peak_vram_mb": peak_vram / (1024 * 1024),
            "bytes_transferred": pool.total_bytes_transferred,
            "mb_transferred": pool.total_bytes_transferred / (1024 * 1024),
        }

    def run_single_slot(self, batch_size: int = 1, seq_len: int = 1, num_tokens: int = 5) -> Dict[str, Any]:
        """Runs Single Persistent Slot execution (zero reallocation, slot reuse)."""
        pool = SingleSlotPool(self.dense_shape, self.dtype, self.device)
        x = torch.randn(batch_size, seq_len, self.hidden_dim, dtype=self.dtype, device=self.device)

        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        start_time = time.perf_counter()
        token_latencies = []

        for tok in range(num_tokens):
            t0 = time.perf_counter()
            for i in range(self.num_layers):
                weight = pool.load_layer(self.dense_weights_host[i], i)
                out = torch.matmul(x, weight)
                x = out[:, :, :self.hidden_dim]
                self._sync()
                pool.release_layer()
            t1 = time.perf_counter()
            token_latencies.append((t1 - t0) * 1000.0)

        total_time = time.perf_counter() - start_time
        peak_vram = torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" and torch.cuda.is_available() else 0

        return {
            "mode": "Single Slot Pool (Persistent Buffer)",
            "total_time_s": total_time,
            "avg_step_ms": sum(token_latencies) / len(token_latencies),
            "tok_per_sec": num_tokens / total_time,
            "peak_vram_bytes": peak_vram,
            "peak_vram_mb": peak_vram / (1024 * 1024),
            "bytes_transferred": pool.total_bytes_transferred,
            "mb_transferred": pool.total_bytes_transferred / (1024 * 1024),
        }

    def run_double_buffer(self, batch_size: int = 1, seq_len: int = 1, num_tokens: int = 5) -> Dict[str, Any]:
        """Runs Double Buffering execution (async prefetch on transfer stream)."""
        pool = DoubleBufferPool(self.dense_shape, self.dtype, self.device)
        x = torch.randn(batch_size, seq_len, self.hidden_dim, dtype=self.dtype, device=self.device)

        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        start_time = time.perf_counter()
        token_latencies = []

        for tok in range(num_tokens):
            t0 = time.perf_counter()

            # Load initial layer 0 into active slot
            pool.load_initial_layer(self.dense_weights_host[0], 0)

            for i in range(self.num_layers):
                # Prefetch layer i+1 into prefetch slot while layer i is computing
                if i + 1 < self.num_layers:
                    pool.prefetch_next_layer(self.dense_weights_host[i + 1], i + 1)

                weight = pool.acquire_active_layer()

                # Execution on compute stream (if CUDA)
                if pool.is_cuda:
                    with torch.cuda.stream(pool.compute_stream):
                        out = torch.matmul(x, weight)
                        x = out[:, :, :self.hidden_dim]
                else:
                    out = torch.matmul(x, weight)
                    x = out[:, :, :self.hidden_dim]

                pool.finish_active_layer()

            pool.synchronize()
            t1 = time.perf_counter()
            token_latencies.append((t1 - t0) * 1000.0)

        total_time = time.perf_counter() - start_time
        peak_vram = torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" and torch.cuda.is_available() else 0

        return {
            "mode": "Double Buffer Pool (Compute/Transfer Overlap)",
            "total_time_s": total_time,
            "avg_step_ms": sum(token_latencies) / len(token_latencies),
            "tok_per_sec": num_tokens / total_time,
            "peak_vram_bytes": peak_vram,
            "peak_vram_mb": peak_vram / (1024 * 1024),
            "bytes_transferred": pool.total_bytes_transferred,
            "mb_transferred": pool.total_bytes_transferred / (1024 * 1024),
        }

    def run_ring_pool_with_cache(
        self,
        batch_size: int = 1,
        seq_len: int = 1,
        num_tokens: int = 10,
        hot_expert_bias: float = 0.75
    ) -> Dict[str, Any]:
        """Runs Ring Pool with Colibri LFRU Expert Cache for hybrid Dense + MoE routing."""
        num_expert_slots = 4
        pool = RingPoolWithCache(
            dense_shape=self.dense_shape,
            expert_shape=self.expert_shape,
            num_dense_slots=2,
            num_expert_slots=num_expert_slots,
            num_total_experts=self.num_experts,
            dtype=self.dtype,
            device=self.device
        )
        x = torch.randn(batch_size, seq_len, self.hidden_dim, dtype=self.dtype, device=self.device)

        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        start_time = time.perf_counter()
        token_latencies = []

        # Synthetic router distribution: simulate zipfian / skewed routing where 4 hot experts get 75% of hits
        import random
        random.seed(42)

        for tok in range(num_tokens):
            t0 = time.perf_counter()

            # 1. Dense backbone prefetch and execution
            pool.dense_pool.load_initial_layer(self.dense_weights_host[0], 0)
            for i in range(min(4, self.num_layers)):  # Dense blocks
                if i + 1 < min(4, self.num_layers):
                    pool.dense_pool.prefetch_next_layer(self.dense_weights_host[i + 1], i + 1)
                weight = pool.dense_pool.acquire_active_layer()
                out = torch.matmul(x, weight)
                x = out[:, :, :self.hidden_dim]
                pool.dense_pool.finish_active_layer()
            pool.dense_pool.synchronize()

            # 2. MoE Layer: route top-k experts per token
            # Skewed routing: hot experts are 0, 1, 2
            if random.random() < hot_expert_bias:
                chosen_experts = random.sample([0, 1, 2, 3], self.top_k)
            else:
                chosen_experts = random.sample(range(4, self.num_experts), self.top_k)

            for exp_id in chosen_experts:
                exp_weight, is_hit = pool.get_expert(exp_id, self.expert_weights_host[exp_id])
                # MoE computation
                exp_out = torch.matmul(x, exp_weight)
                x = x + exp_out[:, :, :self.hidden_dim] * 0.5

            self._sync()
            t1 = time.perf_counter()
            token_latencies.append((t1 - t0) * 1000.0)

        total_time = time.perf_counter() - start_time
        peak_vram = torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" and torch.cuda.is_available() else 0
        total_accesses = pool.cache_hits + pool.cache_misses
        hit_rate = (pool.cache_hits / total_accesses * 100.0) if total_accesses > 0 else 0.0

        return {
            "mode": "Ring Pool + Colibri LFRU Cache (Dense Ring + MoE Resident)",
            "total_time_s": total_time,
            "avg_step_ms": sum(token_latencies) / len(token_latencies),
            "tok_per_sec": num_tokens / total_time,
            "peak_vram_bytes": peak_vram,
            "peak_vram_mb": peak_vram / (1024 * 1024),
            "bytes_transferred": pool.total_bytes_transferred,
            "mb_transferred": pool.total_bytes_transferred / (1024 * 1024),
            "cache_hits": pool.cache_hits,
            "cache_misses": pool.cache_misses,
            "hit_rate_pct": hit_rate,
        }
