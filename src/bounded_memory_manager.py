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
from dataclasses import dataclass
import gc
import math
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


ARENA_ALIGN_BYTES = 2 * 1024 * 1024  # 2 MiB выравнивание шага слота в единой арене
TENSOR_ALIGN_BYTES = 256             # 256-байтное выравнивание тензоров внутри слота


def align_up(value: int, alignment: int) -> int:
    """Выравнивает размер вверх до ближайшей кратной границы alignment."""
    if alignment <= 1 or value <= 0:
        return max(0, value)
    return ((value + alignment - 1) // alignment) * alignment


@dataclass(frozen=True)
class TensorArenaSpec:
    """Смещение и геометрия одного тензора внутри слота арены."""
    name: str
    shape: Tuple[int, ...]
    offset_bytes: int
    nbytes: int
    region: str  # "shared" | "linear_attn" | "self_attn"


class SlotArenaLayout:
    """Единый шаблон слота из двух зон (shared + перекрытый overlay attention) без union-раздувания."""

    ATTENTION_PREFIXES: Tuple[str, ...] = ("linear_attn.", "self_attn.")

    def __init__(
        self,
        specs: Dict[str, TensorArenaSpec],
        dtype: torch.dtype,
        shared_bytes: int,
        attn_offset: int,
        variant_bytes: Dict[str, int],
        payload_bytes: int,
        slot_stride: int,
    ):
        self.specs = specs
        self.dtype = dtype
        self.shared_bytes = shared_bytes
        self.attn_offset = attn_offset
        self.variant_bytes = variant_bytes
        self.payload_bytes = payload_bytes
        self.slot_stride = slot_stride

    @classmethod
    def _classify_region(cls, name: str) -> str:
        for prefix in cls.ATTENTION_PREFIXES:
            if name.startswith(prefix):
                return prefix[:-1]
        return "shared"

    @classmethod
    def from_templates(
        cls,
        template_tensors: Dict[str, Any],
        dtype: torch.dtype = torch.bfloat16,
        slot_alignment: int = ARENA_ALIGN_BYTES,
        tensor_alignment: int = TENSOR_ALIGN_BYTES,
    ) -> "SlotArenaLayout":
        elem_size = torch.empty((), dtype=dtype).element_size()
        grouped: Dict[str, List[Tuple[str, Tuple[int, ...]]]] = {
            "shared": [],
            "linear_attn": [],
            "self_attn": [],
        }

        for name, spec_val in template_tensors.items():
            shape = tuple(spec_val.shape) if isinstance(spec_val, torch.Tensor) else tuple(spec_val)
            grouped[cls._classify_region(name)].append((name, shape))

        specs: Dict[str, TensorArenaSpec] = {}
        cursor = 0
        shared_logical = 0
        max_end = 0

        for name, shape in grouped["shared"]:
            numel = math.prod(shape) if shape else 1
            nbytes = numel * elem_size
            specs[name] = TensorArenaSpec(
                name=name,
                shape=shape,
                offset_bytes=cursor,
                nbytes=nbytes,
                region="shared",
            )
            shared_logical += nbytes
            end = cursor + nbytes
            if end > max_end:
                max_end = end
            cursor = align_up(end, tensor_alignment)

        attn_offset = cursor
        variant_bytes: Dict[str, int] = {}

        # Ветки attention стартуют от одного attn_offset и разделяют общую память без union-суммирования
        for region_name in ("linear_attn", "self_attn"):
            items = grouped[region_name]
            if not items:
                continue
            v_cursor = attn_offset
            v_logical = 0
            for name, shape in items:
                numel = math.prod(shape) if shape else 1
                nbytes = numel * elem_size
                specs[name] = TensorArenaSpec(
                    name=name,
                    shape=shape,
                    offset_bytes=v_cursor,
                    nbytes=nbytes,
                    region=region_name,
                )
                v_logical += nbytes
                v_end = v_cursor + nbytes
                if v_end > max_end:
                    max_end = v_end
                v_cursor = align_up(v_end, tensor_alignment)
            variant_bytes[region_name] = v_logical

        payload_bytes = max_end
        slot_stride = align_up(payload_bytes, slot_alignment) if payload_bytes > 0 else 0
        return cls(
            specs=specs,
            dtype=dtype,
            shared_bytes=shared_logical,
            attn_offset=attn_offset,
            variant_bytes=variant_bytes,
            payload_bytes=payload_bytes,
            slot_stride=slot_stride,
        )

    def bind_slot_views(
        self,
        arena: torch.Tensor,
        slot_idx: int,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, torch.Tensor]:
        """Создает zero-copy тензорные view слота внутри единой арены uint8."""
        view_dtype = dtype or self.dtype
        base = slot_idx * self.slot_stride
        views: Dict[str, torch.Tensor] = {}
        for name, spec in self.specs.items():
            if spec.nbytes == 0:
                views[name] = torch.empty(spec.shape, dtype=view_dtype, device=arena.device)
            else:
                start = base + spec.offset_bytes
                views[name] = arena[start : start + spec.nbytes].view(view_dtype).view(spec.shape)
        return views


class BoundedPinnedBufferPool:
    """Пул закрепленных (pinned) буферов в единой неразрывной арене для PCIe DMA.

    Включает строгую синхронизацию через CUDA Events:
    CPU никогда не перезаписывает закрепленный буфер, пока асинхронный DMA-контроллер GPU
    не закончил чтение данных по шине PCIe.
    """

    def __init__(
        self,
        template_tensors: Optional[Dict[str, Any]] = None,
        num_slots: int = 2,
        dtype: torch.dtype = torch.bfloat16,
        arena_layout: Optional[SlotArenaLayout] = None,
    ):
        self.num_slots = num_slots
        self.dtype = dtype
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if arena_layout is not None:
            if arena_layout.dtype != dtype:
                raise ValueError(f"arena_layout.dtype ({arena_layout.dtype}) != dtype ({dtype})")
            self.layout = arena_layout
        elif template_tensors is not None:
            self.layout = SlotArenaLayout.from_templates(template_tensors, dtype=dtype)
        else:
            raise ValueError("Either template_tensors or arena_layout must be provided")

        self.slot_stride = self.layout.slot_stride
        total_bytes = self.num_slots * self.slot_stride
        use_pinned = self.device.type == "cuda"
        self.arena: Optional[torch.Tensor] = (
            torch.empty(total_bytes, dtype=torch.uint8, pin_memory=use_pinned)
            if total_bytes > 0
            else torch.empty(0, dtype=torch.uint8)
        )
        self.total_pinned_bytes = self.arena.nbytes

        # Единый пул на все слоты со смещением s * slot_stride без перекрытия и фрагментации
        self.slots: List[Dict[str, torch.Tensor]] = [
            self.layout.bind_slot_views(self.arena, s, dtype=dtype)
            for s in range(self.num_slots)
        ]

        # CUDA события для защиты от гонок между CPU-записью в буфер и GPU DMA-чтением
        if self.device.type == "cuda":
            self.dma_events: List[Optional[torch.cuda.Event]] = [torch.cuda.Event() for _ in range(num_slots)]
            self._has_in_flight_dma: List[bool] = [False] * num_slots
        else:
            self.dma_events = [None] * num_slots
            self._has_in_flight_dma = [False] * num_slots

        stride_mb = self.slot_stride / (1024**2)
        payload_mb = self.layout.payload_bytes / (1024**2)
        print(
            f"[BoundedPinnedPool] Allocated {num_slots} pinned DMA slots "
            f"({self.total_pinned_bytes / (1024**2):.1f} MB total arena | "
            f"stride {stride_mb:.1f} MB | payload {payload_mb:.2f} MB) with zero-race protection"
        )

    def copy_into_slot(self, slot_idx: int, src_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Копирует активные тензоры слоя в закрепленный слот перед передачей по DMA.

        Возвращает только view, соответствующие ключам src_tensors (без неактивного overlay-алиаса),
        чтобы асинхронный DMA не затирал активную ветку attention.
        """
        idx = slot_idx % self.num_slots

        # Защита от гонки: ждем, пока GPU DMA закончит чтение предыдущего пакета из этого pinned-буфера
        if self.device.type == "cuda" and self._has_in_flight_dma[idx]:
            self.dma_events[idx].synchronize()
            self._has_in_flight_dma[idx] = False

        target_slot = self.slots[idx]
        active_views: Dict[str, torch.Tensor] = {}
        for name, src_t in src_tensors.items():
            if name in target_slot:
                target_t = target_slot[name]
                target_t.copy_(src_t, non_blocking=False)
                active_views[name] = target_t
        return active_views

    def record_dma_in_flight(self, slot_idx: int, stream: Optional[torch.cuda.Stream]) -> None:
        """Регистрирует запуск асинхронной DMA-передачи из слота."""
        idx = slot_idx % self.num_slots
        if self.device.type == "cuda" and stream is not None:
            self.dma_events[idx].record(stream)
            self._has_in_flight_dma[idx] = True

    def close(self) -> None:
        """Освобождает единую pinned-арену и слоты после синхронизации DMA."""
        if self.device.type == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        self.slots.clear()
        self.arena = None
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

        # Читаем только метаданные форм весов (без чтения тензоров с диска) для linear_attn и self_attn
        template: Dict[str, Tuple[int, ...]] = {}
        for l_idx in [0, 3]:
            l_file = os.path.join(model_dir, f"model-{l_idx:05d}.safetensors")
            if os.path.exists(l_file):
                f = self._get_mmap_handle(l_idx)
                for k in f.keys():
                    short_k = k.replace(f"model.layers.{l_idx}.", "")
                    template[short_k] = tuple(f.get_slice(k).get_shape())

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
