"""Bounded Memory Manager for AirColibri (Stage A):
1. Строгий лимит RAM-кэша с LRU/LFRU вытеснением (защита от page faults и swap).
2. Ограниченный пул закрепленных буферов (Bounded Pinned Pool) — ровно 2-3 слота (~800-1200 МБ).
3. Нативный Float16 без паразитной конвертации dtypes.
4. Устранение постоянных аллокаций.
"""

import collections
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import psutil
import torch
import safetensors
import safetensors.torch


class BoundedRAMCache:
    """Кэш в оперативной памяти с жестким ограничением объема (LRU вытеснение)."""

    def __init__(self, max_cached_layers: int = 8):
        self.max_cached_layers = max_cached_layers
        # OrderedDict для строгого LRU вытеснения
        self.cache: collections.OrderedDict[int, Dict[str, torch.Tensor]] = collections.OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, layer_idx: int) -> Optional[Dict[str, torch.Tensor]]:
        if layer_idx in self.cache:
            self.hits += 1
            # Перемещаем в конец как недавно использованный
            self.cache.move_to_end(layer_idx)
            return self.cache[layer_idx]
        self.misses += 1
        return None

    def put(self, layer_idx: int, tensors: Dict[str, torch.Tensor]) -> None:
        if layer_idx in self.cache:
            self.cache.move_to_end(layer_idx)
            self.cache[layer_idx] = tensors
            return

        # Если лимит исчерпан — вытесняем самый старый (холодный) слой
        if len(self.cache) >= self.max_cached_layers:
            evicted_idx, evicted_tensors = self.cache.popitem(last=False)
            del evicted_tensors
            self.evictions += 1

        self.cache[layer_idx] = tensors

    def clear(self) -> None:
        self.cache.clear()

    @property
    def current_layers_count(self) -> int:
        return len(self.cache)


class BoundedPinnedBufferPool:
    """Пул закрепленных (pinned) буферов для быстрой передачи по PCIe DMA.

    Выделяет фиксированное число слотов (например, 2 или 3) один раз при старте.
    Исключает исчерпание пула драйвера NVIDIA (cudaHostAlloc) и paging в Windows.
    """

    def __init__(
        self,
        template_tensors: Dict[str, torch.Tensor],
        num_slots: int = 2,
        dtype: torch.dtype = torch.float16
    ):
        self.num_slots = num_slots
        self.dtype = dtype
        self.slots: List[Dict[str, torch.Tensor]] = []

        # Выделяем ровно num_slots закрепленных буферов
        for s in range(num_slots):
            slot = {}
            for name, tensor in template_tensors.items():
                # Выделяем в pinned memory
                pinned_t = torch.empty(tensor.shape, dtype=dtype, pin_memory=True)
                slot[name] = pinned_t
            self.slots.append(slot)

        total_pinned_bytes = sum(t.nbytes for s in self.slots for t in s.values())
        print(f"[BoundedPinnedPool] Allocated {num_slots} pinned DMA slots: {total_pinned_bytes / (1024**2):.1f} MB total")

    def copy_into_slot(self, slot_idx: int, src_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Копирует тензоры в закрепленный слот перед передачей по DMA."""
        target_slot = self.slots[slot_idx % self.num_slots]
        for name, src_t in src_tensors.items():
            if name in target_slot:
                target_slot[name].copy_(src_t, non_blocking=False)
        return target_slot


class StableWeightManager:
    """Стабильный менеджер весов, объединяющий строгий RAM-кэш и закрепленный пул."""

    def __init__(
        self,
        model_dir: str,
        max_ram_layers: int = 8,
        num_pinned_slots: int = 2,
        dtype: torch.dtype = torch.float16
    ):
        self.model_dir = model_dir
        self.dtype = dtype
        self.ram_cache = BoundedRAMCache(max_cached_layers=max_ram_layers)

        # Читаем слои как шаблон для выделения закрепленного пула (учитывая linear_attention и full_attention)
        template = {}
        for l_idx in [0, 3]:
            l_file = os.path.join(model_dir, f"model-{l_idx:05d}.safetensors")
            if os.path.exists(l_file):
                with safetensors.safe_open(l_file, framework="pt", device="cpu") as f:
                    for k in f.keys():
                        short_k = k.replace(f"model.layers.{l_idx}.", "")
                        template[short_k] = f.get_slice(k)[:]

        self.pinned_pool = BoundedPinnedBufferPool(
            template_tensors=template,
            num_slots=num_pinned_slots,
            dtype=dtype
        )

        # Глобальные веса (embed, norm, lm_head)
        self.globals: Dict[str, torch.Tensor] = {}
        self._load_globals()

    def _load_globals(self) -> None:
        globals_path = os.path.join(self.model_dir, "model-globals.safetensors")
        with safetensors.safe_open(globals_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                self.globals[k] = f.get_slice(k)[:].to(dtype=self.dtype)

    def load_layer_to_pinned_slot(self, layer_idx: int, target_slot_idx: int) -> Dict[str, torch.Tensor]:
        """Загружает веса слоя в закрепленный слот для отправки на GPU по DMA."""
        # 1. Проверяем RAM-кэш
        cached = self.ram_cache.get(layer_idx)
        if cached is None:
            # 2. Читаем с SSD
            layer_file = os.path.join(self.model_dir, f"model-{layer_idx:05d}.safetensors")
            cached = {}
            prefix = f"model.layers.{layer_idx}."
            with safetensors.safe_open(layer_file, framework="pt", device="cpu") as f:
                for k in f.keys():
                    short_k = k.replace(prefix, "")
                    cached[short_k] = f.get_slice(k)[:].to(dtype=self.dtype)

            # Сохраняем в ограниченный RAM-кэш (с авто-вытеснением)
            self.ram_cache.put(layer_idx, cached)

        # 3. Копируем в закрепленный буфер пула
        pinned_slot = self.pinned_pool.copy_into_slot(target_slot_idx, cached)
        return pinned_slot
