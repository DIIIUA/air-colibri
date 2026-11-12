"""AirColibri Memory Pools: Reusable RAM/VRAM slot architectures for sequential & MoE LLMs.

Eliminates PyTorch allocator churn, empty_cache() stalls, and dynamic cudaMalloc/cudaFree overhead.
Supports:
1. NaiveAirLLMPool: baseline mimicking AirLLM's dynamic allocation + empty_cache() per layer.
2. SingleSlotPool: 1 persistent pre-allocated slot; in-place zero-malloc weight overwriting.
3. DoubleBufferPool: 2 persistent slots + 2 CUDA streams (compute & transfer) with Event synchronization.
4. RingPoolWithCache: Circular prefetch ring for dense layers + Colibri LFRU cache for MoE experts.
"""

import gc
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
from src.lfru_cache import ColibriLFRUCache


class DeviceSlot:
    """A persistent pre-allocated memory slot on GPU and pinned Host RAM.

    Guarantees that memory is allocated once during pool initialization.
    No cudaMalloc/cudaFree or GC is triggered during forward passes.
    """

    def __init__(self, slot_id: int, shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device):
        self.slot_id = slot_id
        self.shape = shape
        self.dtype = dtype
        self.device = device

        # Persistent GPU buffer
        self.buffer = torch.zeros(shape, dtype=dtype, device=device)
        self.data_ptr = self.buffer.data_ptr()

        # Pinned host staging buffer for zero-copy PCIe DMA transfer
        if device.type == "cuda" and torch.cuda.is_available():
            self.host_pinned_staging = torch.empty(shape, dtype=dtype, pin_memory=True)
        else:
            self.host_pinned_staging = torch.empty(shape, dtype=dtype)

        self.current_layer_id: Optional[int] = None

    def copy_from_host(self, host_tensor: torch.Tensor, stream: Optional[torch.cuda.Stream] = None, non_blocking: bool = True) -> None:
        """Zero-reallocation in-place copy from CPU memory into persistent slot buffer."""
        if stream is not None and self.device.type == "cuda":
            with torch.cuda.stream(stream):
                self.buffer.copy_(host_tensor, non_blocking=non_blocking)
        else:
            self.buffer.copy_(host_tensor, non_blocking=False)


# ==============================================================================
# 1. Baseline: Naive AirLLM Pool (dynamic malloc/free + empty_cache)
# ==============================================================================
class NaiveAirLLMPool:
    """Emulates original AirLLM layer offloading: allocates on-the-fly and purges cache."""

    def __init__(self, shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device):
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.active_tensor: Optional[torch.Tensor] = None
        self.total_bytes_transferred: int = 0
        self.layer_nbytes = torch.empty(shape, dtype=dtype).element_size() * torch.empty(shape, dtype=dtype).numel()

    def prepare_layer(self, host_tensor: torch.Tensor, layer_id: int) -> torch.Tensor:
        # Dynamic GPU allocation (AirLLM move_layer_to_device)
        self.active_tensor = host_tensor.to(self.device, non_blocking=False)
        self.total_bytes_transferred += self.layer_nbytes
        return self.active_tensor

    def release_layer(self) -> None:
        # AirLLM post_hook: delete tensor + gc.collect() + empty_cache()
        self.active_tensor = None
        gc.collect()
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


# ==============================================================================
# 2. Single Slot Pool (1 Persistent Pre-allocated Buffer)
# ==============================================================================
class SingleSlotPool:
    """Single persistent pre-allocated slot.

    Overwrites existing GPU buffer in-place.
    Zero allocations, zero deallocations, no empty_cache() calls.
    """

    def __init__(self, shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device):
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.slot = DeviceSlot(0, shape, dtype, device)
        self.initial_data_ptr = self.slot.data_ptr
        self.total_bytes_transferred: int = 0
        self.layer_nbytes = self.slot.buffer.nbytes

    def load_layer(self, host_tensor: torch.Tensor, layer_id: int) -> torch.Tensor:
        self.slot.copy_from_host(host_tensor, stream=None, non_blocking=False)
        self.slot.current_layer_id = layer_id
        self.total_bytes_transferred += self.layer_nbytes
        return self.slot.buffer

    def release_layer(self) -> None:
        # No cudaFree, no gc.collect, no empty_cache! Slot remains resident.
        pass


# ==============================================================================
# 3. Double Buffering Pool (2 Slots + Dual CUDA Streams + Events)
# ==============================================================================
class DoubleBufferPool:
    """Ping-pong double buffering.

    Overlaps execution of layer i on compute_stream with PCIe transfer of layer i+1 on transfer_stream.
    Synchronized via torch.cuda.Event without blocking the CPU thread.
    """

    def __init__(self, shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device):
        self.shape = shape
        self.dtype = dtype
        self.device = device

        self.slots = [
            DeviceSlot(0, shape, dtype, device),
            DeviceSlot(1, shape, dtype, device)
        ]
        self.active_idx: int = 0
        self.prefetch_idx: int = 1
        self.total_bytes_transferred: int = 0
        self.layer_nbytes = self.slots[0].buffer.nbytes

        self.is_cuda = (device.type == "cuda" and torch.cuda.is_available())
        if self.is_cuda:
            self.compute_stream = torch.cuda.Stream(device=device)
            self.transfer_stream = torch.cuda.Stream(device=device)
            # Events for slot readiness and computation finish
            self.transfer_ready_events = [torch.cuda.Event(), torch.cuda.Event()]
            self.compute_done_events = [torch.cuda.Event(), torch.cuda.Event()]
        else:
            self.compute_stream = None
            self.transfer_stream = None
            self.transfer_ready_events = [None, None]
            self.compute_done_events = [None, None]

    def load_initial_layer(self, host_tensor: torch.Tensor, layer_id: int) -> torch.Tensor:
        """Loads layer 0 directly into active slot at pipeline start."""
        slot = self.slots[self.active_idx]
        slot.current_layer_id = layer_id
        self.total_bytes_transferred += self.layer_nbytes
        slot.copy_from_host(host_tensor, stream=None, non_blocking=False)
        return slot.buffer

    def prefetch_next_layer(self, host_tensor: torch.Tensor, layer_id: int) -> None:
        """Asynchronously transfers layer i+1 into prefetch slot on transfer_stream."""
        slot = self.slots[self.prefetch_idx]
        slot.current_layer_id = layer_id
        self.total_bytes_transferred += self.layer_nbytes

        if self.is_cuda:
            # Transfer stream must wait until compute finished with this prefetch slot (from round - 2)
            if self.compute_done_events[self.prefetch_idx] is not None:
                self.transfer_stream.wait_event(self.compute_done_events[self.prefetch_idx])
            with torch.cuda.stream(self.transfer_stream):
                slot.buffer.copy_(host_tensor, non_blocking=True)
                self.transfer_ready_events[self.prefetch_idx].record(self.transfer_stream)
        else:
            slot.buffer.copy_(host_tensor, non_blocking=False)

    def acquire_active_layer(self) -> torch.Tensor:
        """Ensures active slot transfer is complete before compute starts."""
        if self.is_cuda and self.transfer_ready_events[self.active_idx] is not None:
            self.compute_stream.wait_event(self.transfer_ready_events[self.active_idx])
        return self.slots[self.active_idx].buffer

    def finish_active_layer(self) -> None:
        """Records compute completion and swaps active/prefetch slots."""
        if self.is_cuda:
            self.compute_done_events[self.active_idx].record(self.compute_stream)

        # Swap ping-pong slots
        self.active_idx, self.prefetch_idx = self.prefetch_idx, self.active_idx

    def synchronize(self) -> None:
        if self.is_cuda:
            self.compute_stream.synchronize()
            self.transfer_stream.synchronize()


# ==============================================================================
# 4. Ring Pool with Colibri LFRU Expert Cache
# ==============================================================================
class RingPoolWithCache:
    """Hybrid Pool:

    - Dense Ring: Double-buffered prefetch slots for sequential backbone layers.
    - MoE Cache: M persistent slots for MoE experts governed by Colibri's LFRU algorithm.
      Hot experts remain resident across tokens; cold experts are swapped into evicted slots.
    """

    def __init__(
        self,
        dense_shape: Tuple[int, ...],
        expert_shape: Tuple[int, ...],
        num_dense_slots: int = 2,
        num_expert_slots: int = 4,
        num_total_experts: int = 16,
        dtype: torch.dtype = torch.float16,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        self.dense_shape = dense_shape
        self.expert_shape = expert_shape
        self.num_dense_slots = num_dense_slots
        self.num_expert_slots = num_expert_slots
        self.num_total_experts = num_total_experts
        self.dtype = dtype
        self.device = device

        # Dense prefetch pool (using double buffering logic)
        self.dense_pool = DoubleBufferPool(dense_shape, dtype, device)

        # Expert persistent slots in VRAM
        self.expert_slots = [
            DeviceSlot(i, expert_shape, dtype, device)
            for i in range(num_expert_slots)
        ]
        self.transient_expert = DeviceSlot(num_expert_slots, expert_shape, dtype, device)
        self.expert_nbytes = self.transient_expert.buffer.nbytes

        # Colibri LFRU Cache Manager
        self.lfru = ColibriLFRUCache(num_slots=num_expert_slots, num_experts=num_total_experts)

        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.total_expert_bytes_transferred: int = 0

        self.is_cuda = (device.type == "cuda" and torch.cuda.is_available())
        if self.is_cuda:
            self.expert_stream = torch.cuda.Stream(device=device)
            self.expert_event = torch.cuda.Event()
        else:
            self.expert_stream = None
            self.expert_event = None

    def get_expert(self, expert_id: int, host_expert_tensor: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """Routes to an expert using the LFRU resident cache.

        Returns:
            (tensor, is_hit): VRAM buffer for the expert, and whether it was a cache hit.
        """
        slot_idx, is_hit = self.lfru.access_expert(expert_id)
        slot = self.transient_expert if slot_idx is None else self.expert_slots[slot_idx]

        if is_hit:
            self.cache_hits += 1
            return slot.buffer, True

        # Cache MISS: transfer weights into the chosen slot
        self.cache_misses += 1
        self.total_expert_bytes_transferred += self.expert_nbytes

        if self.is_cuda:
            consumer = torch.cuda.current_stream(self.device)
            self.expert_stream.wait_stream(consumer)
            with torch.cuda.stream(self.expert_stream):
                slot.buffer.copy_(host_expert_tensor, non_blocking=True)
                self.expert_event.record(self.expert_stream)
            # Wait for transfer before returning buffer for computation
            consumer.wait_event(self.expert_event)
        else:
            slot.buffer.copy_(host_expert_tensor, non_blocking=False)

        slot.current_layer_id = expert_id
        return slot.buffer, False

    @property
    def total_bytes_transferred(self) -> int:
        return self.dense_pool.total_bytes_transferred + self.total_expert_bytes_transferred
