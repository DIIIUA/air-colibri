"""GPU Ring Pipeline for AirColibri (Stage 5 / Refactored):
Кольцевой конвейер VRAM-буферов с управлением состоянием, строгой семантикой READY/LOADING
и межпоточной синхронизацией через CUDA Streams и Events.

Состояния слота:
FREE -> LOADING (DMA в полете) -> READY (передача завершена) -> COMPUTING -> FREE

Поддерживает режимы:
1. Single Slot (1 слот, последовательный)
2. Double Buffer (2 слота, базовое перекрытие)
3. Triple Ring Pipeline (3 слота, трехфазная «лесенка»)
4. Triple Ring + Tail-to-Head Prefetch (упреждающая загрузка первого потокового слоя следующего токена)
"""

import enum
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

try:
    from src.bounded_memory_manager import SlotArenaLayout, slot_arena_geometry
except ImportError:
    from bounded_memory_manager import SlotArenaLayout, slot_arena_geometry


class SlotState(enum.Enum):
    FREE = "FREE"
    LOADING = "LOADING"
    READY = "READY"
    COMPUTING = "COMPUTING"


class RingSlot:
    """Один слот кольцевого VRAM-буфера с флагами владения и CUDA-событиями."""

    def __init__(
        self,
        slot_id: int,
        template_tensors: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
        arena: Optional[torch.Tensor] = None,
        arena_layout: Optional[SlotArenaLayout] = None,
        arena_offset_bytes: Optional[int] = None,
    ):
        self.slot_id = slot_id
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.state = SlotState.FREE
        self.current_layer_idx: Optional[int] = None
        self.consumer_stream: Optional[torch.cuda.Stream] = None
        self.loaded_keys = set()
        self._owned_arena: Optional[torch.Tensor] = None

        if arena is not None and arena_layout is not None:
            self.tensors = arena_layout.bind_slot_views(arena, slot_id, dtype=dtype,base_offset_bytes=arena_offset_bytes)
            self.slot_bytes = arena_layout.slot_stride
            self.payload_bytes = arena_layout.payload_bytes
        else:
            layout = arena_layout or SlotArenaLayout.from_templates(template_tensors or {}, dtype=dtype)
            self._owned_arena = (
                torch.zeros(layout.slot_stride, dtype=torch.uint8, device=self.device)
                if layout.slot_stride > 0
                else torch.empty(0, dtype=torch.uint8, device=self.device)
            )
            self.tensors = layout.bind_slot_views(self._owned_arena, 0, dtype=dtype)
            self.slot_bytes = layout.slot_stride
            self.payload_bytes = layout.payload_bytes

        # CUDA события для строгой синхронизации без блокировки CPU
        if self.device.type == "cuda":
            self.compute_done_event = torch.cuda.Event()
            self.transfer_done_event = torch.cuda.Event()
        else:
            self.compute_done_event = None
            self.transfer_done_event = None

    def is_ready(self) -> bool:
        """Проверяет фактическую готовность слота на GPU без блокировки."""
        if self.state == SlotState.READY:
            return True
        if self.state == SlotState.LOADING and self.transfer_done_event is not None:
            if self.transfer_done_event.query():
                self.state = SlotState.READY
                return True
        return False

    def wait_ready(self) -> None:
        """Ожидает фактического завершения DMA-передачи слота на CPU."""
        if self.state == SlotState.LOADING and self.transfer_done_event is not None:
            self.transfer_done_event.synchronize()
            self.state = SlotState.READY


class GPURingPipeline:
    """Кольцевой менеджер VRAM с поддержкой 1, 2 или 3 слотов («лесенка»)."""

    def __init__(
        self,
        template_tensors: Optional[Dict[str, Any]] = None,
        num_slots: int = 3,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
        tail_to_head_prefetch: bool = False,
        arena_layout: Optional[SlotArenaLayout] = None,
        slot_layouts: Optional[List[SlotArenaLayout]] = None,
    ):
        self.num_slots = max(1, min(num_slots, 1024))
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.tail_to_head_prefetch = tail_to_head_prefetch

        if arena_layout is not None:
            if arena_layout.dtype != dtype:
                raise ValueError(f"arena_layout.dtype ({arena_layout.dtype}) != dtype ({dtype})")
            self.layout = arena_layout
        elif template_tensors is not None:
            self.layout = SlotArenaLayout.from_templates(template_tensors, dtype=dtype)
        else:
            raise ValueError("Either template_tensors or arena_layout must be provided")

        self.slot_layouts, self.slot_offsets, total_bytes = slot_arena_geometry(self.layout,self.num_slots,slot_layouts)
        self.slot_strides = tuple(item.slot_stride for item in self.slot_layouts)
        self.slot_stride = self.slot_strides[0] if len(set(self.slot_strides))==1 else None
        # Единая неразрывная VRAM-арена на все слоты кольца без фрагментации
        self.arena: Optional[torch.Tensor] = (
            torch.zeros(total_bytes, dtype=torch.uint8, device=self.device)
            if total_bytes > 0
            else torch.empty(0, dtype=torch.uint8, device=self.device)
        )

        # Bind disjoint slot views into the same allocation, including compact layouts.
        self.slots: List[RingSlot] = [
            RingSlot(
                slot_id=i,
                device=self.device,
                dtype=dtype,
                arena=self.arena,
                arena_layout=self.slot_layouts[i],
                arena_offset_bytes=self.slot_offsets[i],
            )
            for i in range(self.num_slots)
        ]

        # Раздельные потоки CUDA для исполнения и передачи
        if self.device.type == "cuda":
            self.compute_stream = torch.cuda.current_stream(self.device)
            self.transfer_stream = torch.cuda.Stream(device=self.device)
            # Arena zero-initialization was enqueued on the construction stream.
            # The first H2D must not race with that write on another stream.
            self.transfer_stream.wait_stream(self.compute_stream)
        else:
            self.compute_stream = None
            self.transfer_stream = None

        # Статистика конвейера
        self.total_loads = 0
        self.total_computes = 0
        self.total_resident_acquires = 0
        self.tail_prefetches = 0
        self.h2d_overlap_time_ms = 0.0

        total_mb = self.arena.nbytes / (1024**2)
        strides_mb = [round(size / (1024**2),2) for size in self.slot_strides]
        strides_str = str(strides_mb) if len(strides_mb) <= 8 else f"{strides_mb[:4]}...({len(strides_mb)} slots)"
        print(
            f"[GPURingPipeline] Initialized {self.num_slots}-slot ring "
            f"({total_mb:.1f} MB VRAM arena | slot MiB {strides_str}) | "
            f"Mode: {'Triple-Ring + Tail-Prefetch' if (self.num_slots >= 3 and tail_to_head_prefetch) else f'{self.num_slots}-Slot Ring'}"
        )

    def get_slot_for_loading(self, layer_idx: int) -> RingSlot:
        """Выбирает слот для загрузки следующего слоя весов (LOADING)."""
        slot_idx = layer_idx % self.num_slots
        slot = self.slots[slot_idx]

        if slot.state != SlotState.FREE:
            raise RuntimeError(f"Cannot reserve active GPU slot: {slot.state.value}")

        # Если на слоте еще висит вычисление — ожидаем завершения на transfer_stream
        if self.device.type == "cuda" and slot.compute_done_event is not None:
            self.transfer_stream.wait_event(slot.compute_done_event)

        slot.state = SlotState.LOADING
        slot.current_layer_idx = layer_idx
        slot.consumer_stream = None
        slot.loaded_keys.clear()
        return slot

    def load_tensors_async(
        self,
        slot: RingSlot,
        pinned_src_tensors: Dict[str, torch.Tensor],
        pinned_pool: Optional[Any] = None,
        pinned_slot_id: Optional[int] = None,
    ) -> None:
        """Асинхронно копирует веса из закрепленной памяти в GPU-слот через transfer_stream."""
        if slot not in self.slots or slot.state != SlotState.LOADING:
            raise RuntimeError("DMA requires a reserved local slot")
        has_linear = any(k.startswith("linear_attn.") for k in pinned_src_tensors)
        has_self = any(k.startswith("self_attn.") for k in pinned_src_tensors)
        if has_linear and has_self:
            raise ValueError(
                "Cannot load both 'linear_attn.' and 'self_attn.' branches into the same overlay slot."
            )
        if not pinned_src_tensors:
            raise ValueError("Empty GPU payload")
        for name, src in pinned_src_tensors.items():
            target = slot.tensors.get(name)
            if target is None or src.shape != target.shape or src.dtype != target.dtype:
                raise ValueError(f"GPU payload shape/dtype/key mismatch: {name}")

        if self.device.type == "cuda" and self.transfer_stream is not None:
            with torch.cuda.stream(self.transfer_stream):
                for name, src_t in pinned_src_tensors.items():
                    if name in slot.tensors:
                        slot.tensors[name].copy_(src_t, non_blocking=True)
                slot.transfer_done_event.record(self.transfer_stream)

            # Регистрируем в pinned-пуле активную DMA-передачу для предотвращения гонок
            if pinned_pool is not None:
                psid = slot.slot_id if pinned_slot_id is None else pinned_slot_id
                pinned_pool.record_dma_in_flight(psid, self.transfer_stream)

            # На CPU семантически состояние остается LOADING до завершения DMA
            slot.state = SlotState.LOADING
        else:
            for name, src_t in pinned_src_tensors.items():
                if name in slot.tensors:
                    slot.tensors[name].copy_(src_t, non_blocking=False)
            slot.state = SlotState.READY

        self.total_loads += 1
        slot.loaded_keys = set(pinned_src_tensors)

    def acquire_slot_for_compute(self, layer_idx: int) -> RingSlot:
        """Ожидает завершения передачи и переводит слот в состояние COMPUTING."""
        slot_idx = layer_idx % self.num_slots
        slot = self.slots[slot_idx]

        if slot.current_layer_idx != layer_idx or slot.state not in (SlotState.LOADING, SlotState.READY) or not slot.loaded_keys:
            raise RuntimeError("Compute requires a matching submitted layer")

        # Вычисления в compute_stream строго ждут завершения DMA-передачи этого слота
        if self.device.type == "cuda" and slot.transfer_done_event is not None:
            slot.consumer_stream = torch.cuda.current_stream(self.device)
            self.compute_stream = slot.consumer_stream
            slot.consumer_stream.wait_event(slot.transfer_done_event)

        slot.state = SlotState.COMPUTING
        return slot

    def release_slot_after_compute(self, slot: RingSlot) -> None:
        """Фиксирует завершение вычислений в слоте и переводит его в FREE."""
        if slot not in self.slots or slot.state != SlotState.COMPUTING:
            raise RuntimeError("Release requires an active local compute slot")
        if self.device.type == "cuda" and slot.compute_done_event is not None:
            # Another slot can be acquired by a different stream before this
            # one is released. Fence this slot's actual consumer, not the most
            # recently acquired stream on the pipeline.
            if slot.consumer_stream is None:
                raise RuntimeError("CUDA release requires the slot's consumer stream")
            slot.compute_done_event.record(slot.consumer_stream)

        slot.state = SlotState.FREE
        self.total_computes += 1

    def acquire_resident_for_compute(self, layer_idx: int, previous_layer_idx: int) -> RingSlot:
        """Reuse retained bytes with a fresh consumer ticket and no H2D.

        Serializing successive readers through their completion events makes
        the last consumer's fence sufficient for a subsequent overwrite.
        """
        slot = self.slots[layer_idx % self.num_slots]
        if (slot.state != SlotState.FREE or slot.current_layer_idx != previous_layer_idx
                or not slot.loaded_keys):
            raise RuntimeError("Resident acquire requires matching released content")
        if self.device.type == "cuda":
            consumer = torch.cuda.current_stream(self.device)
            consumer.wait_event(slot.transfer_done_event)
            consumer.wait_event(slot.compute_done_event)
            slot.consumer_stream = consumer
            self.compute_stream = consumer
        slot.current_layer_idx = layer_idx
        slot.state = SlotState.COMPUTING
        self.total_resident_acquires += 1
        return slot

    def prefetch_tail_to_head(
        self,
        target_layer_idx: int,
        pinned_tensors: Dict[str, torch.Tensor],
        pinned_pool: Optional[Any] = None
    ) -> Optional[RingSlot]:
        """Упреждающая предзагрузка первого стримингового слоя следующего токена на свободный слот кольца.

        Вызывается во время вычисления хвоста токена T.
        """
        if not self.tail_to_head_prefetch or self.num_slots < 2:
            return None

        slot_idx = target_layer_idx % self.num_slots
        slot = self.slots[slot_idx]

        # Запускаем упреждающую загрузку в transfer_stream
        self.get_slot_for_loading(target_layer_idx)
        self.load_tensors_async(slot, pinned_tensors, pinned_pool=pinned_pool)
        self.tail_prefetches += 1
        return slot

    def close(self) -> None:
        """Освобождает GPU-слоты и сбрасывает состояние конвейера."""
        if self.device.type == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        for slot in self.slots:
            slot.tensors.clear()
            slot._owned_arena = None
            slot.compute_done_event = None
            slot.transfer_done_event = None
            slot.current_layer_idx = None
            slot.consumer_stream = None
            slot.loaded_keys.clear()
            slot.state = SlotState.FREE
        self.slots.clear()
        self.arena = None
        self.compute_stream = None
        self.transfer_stream = None
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> "GPURingPipeline":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
