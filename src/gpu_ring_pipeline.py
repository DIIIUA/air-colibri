"""GPU Ring Pipeline for AirColibri (Stage B & Stage C):
Кольцевой конвейер VRAM-буферов с управлением состоянием и перекрытием передачи и вычислений.

Состояния слота:
FREE -> LOADING -> READY -> COMPUTING -> FREE

Поддерживает режимы:
1. Single Slot (1 слот, последовательный)
2. Double Buffer (2 слота, базовое перекрытие)
3. Triple Ring Pipeline (3 слота, трехфазная «лесенка»)
4. Triple Ring + Tail-to-Head Prefetch (упреждающая загрузка слоя 0 для следующего токена)
"""

import enum
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch


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
        template_tensors: Dict[str, torch.Tensor],
        device: torch.device,
        dtype: torch.dtype = torch.float16
    ):
        self.slot_id = slot_id
        self.device = device
        self.dtype = dtype
        self.state = SlotState.FREE
        self.current_layer_idx: Optional[int] = None

        # Выделяем постоянные GPU-тензоры один раз при инициализации
        self.tensors: Dict[str, torch.Tensor] = {}
        for name, tensor in template_tensors.items():
            self.tensors[name] = torch.zeros(tensor.shape, dtype=dtype, device=device)

        # CUDA события для строгой синхронизации без блокировки CPU
        if device.type == "cuda":
            self.compute_done_event = torch.cuda.Event()
            self.transfer_done_event = torch.cuda.Event()
        else:
            self.compute_done_event = None
            self.transfer_done_event = None

        self.slot_bytes = sum(t.nbytes for t in self.tensors.values())


class GPURingPipeline:
    """Кольцевой менеджер VRAM с поддержкой 1, 2 или 3 слотов («лесенка»)."""

    def __init__(
        self,
        template_tensors: Dict[str, torch.Tensor],
        num_slots: int = 3,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
        tail_to_head_prefetch: bool = False
    ):
        self.num_slots = max(1, min(num_slots, 4))
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.tail_to_head_prefetch = tail_to_head_prefetch

        # Создаем кольцевой пул слотов
        self.slots: List[RingSlot] = [
            RingSlot(slot_id=i, template_tensors=template_tensors, device=self.device, dtype=dtype)
            for i in range(self.num_slots)
        ]

        # Раздельные потоки CUDA для исполнения и передачи
        if self.device.type == "cuda":
            self.compute_stream = torch.cuda.current_stream()
            self.transfer_stream = torch.cuda.Stream(device=self.device)
        else:
            self.compute_stream = None
            self.transfer_stream = None

        # Статистика конвейера
        self.total_loads = 0
        self.total_computes = 0
        self.tail_prefetches = 0
        self.h2d_overlap_time_ms = 0.0

        total_mb = sum(s.slot_bytes for s in self.slots) / (1024**2)
        print(f"[GPURingPipeline] Initialized {self.num_slots}-slot ring ({total_mb:.1f} MB VRAM) | Mode: {'Triple-Ring + Tail-Prefetch' if (self.num_slots >= 3 and tail_to_head_prefetch) else f'{self.num_slots}-Slot Ring'}")

    def get_slot_for_loading(self, layer_idx: int) -> RingSlot:
        """Выбирает слот для загрузки следующего слоя весов (LOADING)."""
        slot_idx = layer_idx % self.num_slots
        slot = self.slots[slot_idx]

        # Если на слоте еще висит вычисление — ожидаем завершения на transfer_stream
        if self.device.type == "cuda" and slot.compute_done_event is not None:
            self.transfer_stream.wait_event(slot.compute_done_event)

        slot.state = SlotState.LOADING
        slot.current_layer_idx = layer_idx
        return slot

    def load_tensors_async(
        self,
        slot: RingSlot,
        pinned_src_tensors: Dict[str, torch.Tensor]
    ) -> None:
        """Асинхронно копирует веса из закрепленной памяти в GPU-слот через transfer_stream."""
        if self.device.type == "cuda" and self.transfer_stream is not None:
            with torch.cuda.stream(self.transfer_stream):
                for name, src_t in pinned_src_tensors.items():
                    if name in slot.tensors:
                        slot.tensors[name].copy_(src_t, non_blocking=True)
                slot.transfer_done_event.record(self.transfer_stream)
        else:
            for name, src_t in pinned_src_tensors.items():
                if name in slot.tensors:
                    slot.tensors[name].copy_(src_t, non_blocking=False)

        slot.state = SlotState.READY
        self.total_loads += 1

    def acquire_slot_for_compute(self, layer_idx: int) -> RingSlot:
        """Ожидает завершения передачи и переводит слот в состояние COMPUTING."""
        slot_idx = layer_idx % self.num_slots
        slot = self.slots[slot_idx]

        # Вычисления ждут завершения DMA-передачи этого слота
        if self.device.type == "cuda" and slot.transfer_done_event is not None:
            self.compute_stream.wait_event(slot.transfer_done_event)

        slot.state = SlotState.COMPUTING
        return slot

    def release_slot_after_compute(self, slot: RingSlot) -> None:
        """Фиксирует завершение вычислений в слоте и переводит его в FREE."""
        if self.device.type == "cuda" and slot.compute_done_event is not None:
            slot.compute_done_event.record(self.compute_stream)

        slot.state = SlotState.FREE
        self.total_computes += 1

    def prefetch_tail_to_head(
        self,
        layer_0_pinned_tensors: Dict[str, torch.Tensor]
    ) -> Optional[RingSlot]:
        """Упреждающая предзагрузка слоя 0 токена T+1 на свободный слот кольца.

        Вызывается во время вычисления хвоста (слоя 31) токена T.
        """
        if not self.tail_to_head_prefetch or self.num_slots < 2:
            return None

        # Выбираем слот для слоя 0 токена T+1
        slot_idx = 0 % self.num_slots
        slot = self.slots[slot_idx]

        # Запускаем упреждающую загрузку в transfer_stream
        self.get_slot_for_loading(0)
        self.load_tensors_async(slot, layer_0_pinned_tensors)
        self.tail_prefetches += 1
        return slot
