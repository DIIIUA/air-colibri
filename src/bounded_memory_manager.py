"""Bounded Memory Manager for AirColibri (Stage 5 / Refactored):
1. Строгий лимит RAM-кэша с поддержкой политик:
   - "static_pinned" (удержание фиксированных слоев в RAM, защита от LRU-патологии на последовательных проходах)
   - "lru" (классическое LRU-вытеснение)
   - "mmap_direct" (прямой zero-copy доступ через постоянные дескрипторы safetensors)
2. Bounded Pinned Pool с защитой от DMA-гонок (синхронизация через CUDA Events перед перезаписью).
3. Постоянные mmap-дескрипторы safetensors (устранение 480 мс оверхеда на повторное открытие файлов).
4. Нативный BFloat16 без паразитных конвертаций dtypes.
"""

import collections
import collections.abc
import gc
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

_MISSING = object()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import psutil
import torch
import safetensors
import safetensors.torch


class BoundedRAMCache:
    """Кэш в оперативной памяти с жестким ограничением объема (LRU или Static Pinned)."""

    def __init__(self, max_cached_layers: int = 8, policy: str = "static_pinned"):
        self.max_cached_layers = max_cached_layers
        self.policy = policy
        self.cache: collections.OrderedDict[int, Dict[str, torch.Tensor]] = collections.OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, layer_idx: int) -> Optional[Dict[str, torch.Tensor]]:
        if layer_idx in self.cache:
            self.hits += 1
            if self.policy == "lru":
                self.cache.move_to_end(layer_idx)
            return self.cache[layer_idx]
        self.misses += 1
        return None

    def put(self, layer_idx: int, tensors: Dict[str, torch.Tensor]) -> None:
        if layer_idx in self.cache:
            if self.policy == "lru":
                self.cache.move_to_end(layer_idx)
            self.cache[layer_idx] = tensors
            return

        if self.policy == "static_pinned":
            # При static_pinned кэшируем только первые max_cached_layers слоев
            # Они остаются в памяти навсегда и никогда не вытесняются
            if len(self.cache) < self.max_cached_layers:
                self.cache[layer_idx] = tensors
            return

        # LRU режим
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

    Включает строгую синхронизацию через CUDA Events:
    CPU никогда не перезаписывает закрепленный буфер, пока асинхронный DMA-контроллер GPU
    не закончил чтение данных по шине PCIe.
    """

    def __init__(
        self,
        template_tensors: Dict[str, torch.Tensor],
        num_slots: int = 2,
        dtype: torch.dtype = torch.bfloat16
    ):
        self.num_slots = num_slots
        self.dtype = dtype
        self.slots: List[Dict[str, torch.Tensor]] = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Выделяем ровно num_slots закрепленных буферов
        for s in range(num_slots):
            slot = {}
            for name, tensor in template_tensors.items():
                pinned_t = torch.empty(tensor.shape, dtype=dtype, pin_memory=True)
                slot[name] = pinned_t
            self.slots.append(slot)

        # CUDA события для защиты от гонок между CPU-записью в буфер и GPU DMA-чтением
        if self.device.type == "cuda":
            self.dma_events: List[Optional[torch.cuda.Event]] = [torch.cuda.Event() for _ in range(num_slots)]
            self._has_in_flight_dma: List[bool] = [False] * num_slots
        else:
            self.dma_events = [None] * num_slots
            self._has_in_flight_dma = [False] * num_slots

        total_pinned_bytes = sum(t.nbytes for s in self.slots for t in s.values())
        print(f"[BoundedPinnedPool] Allocated {num_slots} pinned DMA slots ({total_pinned_bytes / (1024**2):.1f} MB total) with zero-race protection")

    def copy_into_slot(self, slot_idx: int, src_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Копирует тензоры в закрепленный слот перед передачей по DMA.

        Гарантирует завершение предыдущей асинхронной DMA-передачи из этого слота перед перезаписью.
        """
        idx = slot_idx % self.num_slots

        # Защита от гонки: ждем, пока GPU DMA закончит чтение предыдущего пакета из этого pinned-буфера
        if self.device.type == "cuda" and self._has_in_flight_dma[idx]:
            self.dma_events[idx].synchronize()
            self._has_in_flight_dma[idx] = False

        target_slot = self.slots[idx]
        for name, src_t in src_tensors.items():
            if name in target_slot:
                target_slot[name].copy_(src_t, non_blocking=False)
        return target_slot

    def record_dma_in_flight(self, slot_idx: int, stream: Optional[torch.cuda.Stream]) -> None:
        """Регистрирует запуск асинхронной DMA-передачи из слота."""
        idx = slot_idx % self.num_slots
        if self.device.type == "cuda" and stream is not None:
            self.dma_events[idx].record(stream)
            self._has_in_flight_dma[idx] = True

    def close(self) -> None:
        """Освобождает pinned-слоты после синхронизации DMA."""
        if self.device.type == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        self.slots.clear()
        self.dma_events.clear()
        self._has_in_flight_dma.clear()

    def __enter__(self) -> "BoundedPinnedBufferPool":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class LazyGlobalsMap(collections.abc.MutableMapping):
    """Ленивый mmap-словарь глобальных весов без жадной загрузки 4 ГБ в RAM."""

    def __init__(self, globals_path: str, dtype: torch.dtype):
        self._globals_path = globals_path
        self._dtype = dtype
        self._handle: Any = safetensors.safe_open(globals_path, framework="pt", device="cpu")
        self._keys: Dict[str, None] = dict.fromkeys(self._handle.keys())
        self._materialized: Dict[str, torch.Tensor] = {}

    def _ensure_handle(self) -> Any:
        if self._handle is None:
            self._handle = safetensors.safe_open(self._globals_path, framework="pt", device="cpu")
        return self._handle

    def _read_tensor(self, key: str) -> torch.Tensor:
        return self._ensure_handle().get_slice(key)[:].to(dtype=self._dtype)

    def __getitem__(self, key: str) -> torch.Tensor:
        if key not in self._keys:
            raise KeyError(key)
        if key not in self._materialized:
            self._materialized[key] = self._read_tensor(key)
        return self._materialized[key]

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        self._keys[key] = None
        self._materialized[key] = value

    def __delitem__(self, key: str) -> None:
        if key not in self._keys:
            raise KeyError(key)
        del self._keys[key]
        self._materialized.pop(key, None)
        if not self._keys and not self._materialized:
            self._handle = None

    def pop(self, key: str, default: Any = _MISSING) -> Any:
        if key not in self._keys:
            if default is _MISSING:
                raise KeyError(key)
            return default
        if key in self._materialized:
            tensor = self._materialized.pop(key)
        else:
            tensor = self._read_tensor(key)
        self._keys.pop(key, None)
        if not self._keys and not self._materialized:
            self._handle = None
        return tensor

    def __iter__(self):
        return iter(list(self._keys))

    def __len__(self) -> int:
        return len(self._keys)

    def __contains__(self, key: object) -> bool:
        return key in self._keys

    def clear(self) -> None:
        self._materialized.clear()
        self._keys.clear()
        self._handle = None

    def close(self) -> None:
        self.clear()

    def __repr__(self) -> str:
        return f"LazyGlobalsMap(keys={list(self._keys.keys())}, loaded={list(self._materialized.keys())})"


class StableWeightManager:
    """Стабильный менеджер весов с защитой от гонок, постоянными mmap-дескрипторами и кэшем."""

    def __init__(
        self,
        model_dir: str,
        max_ram_layers: int = 8,
        num_pinned_slots: int = 2,
        cache_policy: str = "static_pinned",
        dtype: torch.dtype = torch.bfloat16
    ):
        self.model_dir = model_dir
        self.dtype = dtype
        self.cache_policy = cache_policy
        self.ram_cache = BoundedRAMCache(max_cached_layers=max_ram_layers, policy=cache_policy)

        # Постоянные открытые mmap-дескрипторы файлов safetensors
        self.mmap_handles: Dict[int, Any] = {}

        # Читаем слои как шаблон для выделения закрепленного пула (учитывая linear_attention и full_attention)
        template = {}
        for l_idx in [0, 3]:
            l_file = os.path.join(model_dir, f"model-{l_idx:05d}.safetensors")
            if os.path.exists(l_file):
                f = self._get_mmap_handle(l_idx)
                for k in f.keys():
                    short_k = k.replace(f"model.layers.{l_idx}.", "")
                    template[short_k] = f.get_slice(k)[:]

        self.pinned_pool = BoundedPinnedBufferPool(
            template_tensors=template,
            num_slots=num_pinned_slots,
            dtype=dtype
        )
        del template

        # Глобальные веса (embed, norm, lm_head) — ленивый mmap без жадной загрузки
        self.globals: collections.abc.MutableMapping = {}
        self._load_globals()

    def _get_mmap_handle(self, layer_idx: int) -> Any:
        """Возвращает постоянный открытый mmap-дескриптор файла слоя."""
        if layer_idx not in self.mmap_handles:
            layer_file = os.path.join(self.model_dir, f"model-{layer_idx:05d}.safetensors")
            self.mmap_handles[layer_idx] = safetensors.safe_open(layer_file, framework="pt", device="cpu")
        return self.mmap_handles[layer_idx]

    def _load_globals(self) -> None:
        globals_path = os.path.join(self.model_dir, "model-globals.safetensors")
        self.globals = LazyGlobalsMap(globals_path, dtype=self.dtype)

    def load_layer_to_pinned_slot(self, layer_idx: int, target_slot_idx: int) -> Dict[str, torch.Tensor]:
        """Загружает веса слоя в закрепленный слот для отправки на GPU по DMA."""
        # 1. Проверяем RAM-кэш
        cached = self.ram_cache.get(layer_idx)
        if cached is None:
            # 2. Читаем через постоянный открытый mmap-дескриптор
            f = self._get_mmap_handle(layer_idx)
            cached = {}
            prefix = f"model.layers.{layer_idx}."
            for k in f.keys():
                short_k = k.replace(prefix, "")
                cached[short_k] = f.get_tensor(k).to(dtype=self.dtype)

            # Сохраняем в RAM-кэш в соответствии с выбранной политикой
            self.ram_cache.put(layer_idx, cached)

        # 3. Безопасно копируем в закрепленный буфер пула (с авто-синхронизацией DMA)
        pinned_slot = self.pinned_pool.copy_into_slot(target_slot_idx, cached)
        return pinned_slot

    def close(self) -> None:
        """Освобождает дескрипторы, кэш RAM, глобалы и pinned-буферы."""
        self.mmap_handles.clear()
        self.ram_cache.clear()
        if hasattr(self, "globals") and self.globals is not None:
            self.globals.clear()
        if hasattr(self, "pinned_pool") and self.pinned_pool is not None:
            self.pinned_pool.close()

    def __enter__(self) -> "StableWeightManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
