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

from src.quantization_q4 import (
    Q4_GROUP_SIZE,
    is_q4_quantizable,
    q4_subkey_names,
    q4_tensor_specs,
    quantize_tensor_q4,
)


class BoundedRAMCache:
    """Кэш в оперативной памяти с жестким ограничением объема (LRU или Static Pinned)."""

    def __init__(self, max_cached_layers: int = 8, policy: str = "static_pinned", max_bytes: Optional[int] = None):
        if max_cached_layers < 0 or (max_bytes is not None and max_bytes < 0):
            raise ValueError("Cache budgets must be nonnegative")
        if policy not in ("static_pinned", "lru", "mmap_direct"):
            raise ValueError("Unsupported RAM cache policy")
        self.max_cached_layers = max_cached_layers
        self.max_bytes = max_bytes
        self.policy = policy
        self.cache: collections.OrderedDict[int, Dict[str, torch.Tensor]] = collections.OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.rejected_admissions = 0
        self.admission = None
        self.on_access = None
        self.on_placement = None
        self.generation = 0
        self.peak_bytes = 0
        self.placement_events = collections.deque(maxlen=1024)
        self.placement_counts = collections.Counter()

    @property
    def current_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for ts in self.cache.values() for t in ts.values())

    def get(self, layer_idx: int) -> Optional[Dict[str, torch.Tensor]]:
        if self.on_access is not None:
            self.on_access(layer_idx)
        if layer_idx in self.cache:
            self.hits += 1
            if self.policy == "lru":
                self.cache.move_to_end(layer_idx)
            return self.cache[layer_idx]
        self.misses += 1
        return None

    def put(self, layer_idx: int, tensors: Dict[str, torch.Tensor]) -> None:
        size = sum(t.numel() * t.element_size() for t in tensors.values())
        before = self.current_bytes
        victims = []
        def reject(reason):
            self.rejected_admissions += 1
            self._placement_ack(layer_idx, "rejected", [], before, reason)
        if not self.max_cached_layers or self.policy == "mmap_direct":
            reject("residency_disabled")
            return
        if self.max_bytes is not None and size > self.max_bytes:
            reject("object_exceeds_byte_budget")
            return
        if self.admission is not None:
            accepted, victim = self.admission(layer_idx, list(self.cache))
            if not accepted:
                reject("native_admission_margin")
                return
            if victim is not None:
                if victim not in self.cache or victim == layer_idx:
                    raise ValueError("Invalid admission victim")
                victims.append(victim)
        old_size = sum(t.numel() * t.element_size() for t in self.cache.get(layer_idx, {}).values())
        projected = before - old_size + size
        for victim in victims:
            projected -= sum(t.numel() * t.element_size() for t in self.cache[victim].values())
        remaining = [key for key in self.cache if key != layer_idx and key not in victims]
        def over_budget():
            return len(remaining) + 1 > self.max_cached_layers or (self.max_bytes is not None and projected > self.max_bytes)
        if self.policy == "static_pinned" and over_budget():
            reject("static_residents_reserved")
            return
        # Native admission authorizes exactly its victim. A differently-sized
        # newcomer must not silently trigger extra, unplanned evictions.
        if victims and over_budget():
            reject("native_replacement_exceeds_budget")
            return
        while over_budget() and remaining:
            victim = remaining.pop(0)
            victims.append(victim)
            projected -= sum(t.numel() * t.element_size() for t in self.cache[victim].values())
        if over_budget():
            reject("byte_budget")
            return
        # All checks precede mutation; rejected replacements retain old data.
        for victim in victims:
            del self.cache[victim]
            self.evictions += 1
        self.cache.pop(layer_idx, None)
        self.cache[layer_idx] = tensors
        self.peak_bytes = max(self.peak_bytes, self.current_bytes)
        self._placement_ack(layer_idx, "applied", victims, before, "admitted")

    def _placement_ack(self, layer, status, victims, before, reason):
        self.generation += 1
        event = dict(generation=self.generation, tier="ram", layer=layer,
                     status=status, victims=list(victims), reason=reason,
                     bytes_before=before, bytes_after=self.current_bytes,
                     budget_bytes=self.max_bytes)
        self.placement_counts[status] += 1
        self.placement_events.append(event)
        if self.on_placement is not None:
            self.on_placement(dict(event))

    def clear(self) -> None:
        before = self.current_bytes
        victims = list(self.cache)
        self.cache.clear()
        if victims:
            self._placement_ack(None, "released", victims, before, "close_or_clear")

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
    dtype: torch.dtype = torch.bfloat16


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
    def _parse_template_entry(cls, spec_val: Any, default_dtype: torch.dtype) -> Tuple[Tuple[int, ...], torch.dtype]:
        if isinstance(spec_val, torch.Tensor):
            return tuple(spec_val.shape), spec_val.dtype
        if isinstance(spec_val, tuple) and len(spec_val) == 2 and isinstance(spec_val[1], torch.dtype):
            return tuple(spec_val[0]), spec_val[1]
        return tuple(spec_val), default_dtype

    @classmethod
    def from_templates(
        cls,
        template_tensors: Dict[str, Any],
        dtype: torch.dtype = torch.bfloat16,
        slot_alignment: int = ARENA_ALIGN_BYTES,
        tensor_alignment: int = TENSOR_ALIGN_BYTES,
    ) -> "SlotArenaLayout":
        elem_sizes: Dict[torch.dtype, int] = {}
        grouped: Dict[str, List[Tuple[str, Tuple[int, ...], torch.dtype]]] = {
            "shared": [],
            "linear_attn": [],
            "self_attn": [],
        }

        for name, spec_val in template_tensors.items():
            shape, t_dtype = cls._parse_template_entry(spec_val, dtype)
            if t_dtype not in elem_sizes:
                elem_sizes[t_dtype] = torch.empty((), dtype=t_dtype).element_size()
            grouped[cls._classify_region(name)].append((name, shape, t_dtype))

        specs: Dict[str, TensorArenaSpec] = {}
        cursor = 0
        shared_logical = 0
        max_end = 0

        for name, shape, t_dtype in grouped["shared"]:
            numel = math.prod(shape) if shape else 1
            nbytes = numel * elem_sizes[t_dtype]
            specs[name] = TensorArenaSpec(
                name=name,
                shape=shape,
                offset_bytes=cursor,
                nbytes=nbytes,
                region="shared",
                dtype=t_dtype,
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
            for name, shape, t_dtype in items:
                numel = math.prod(shape) if shape else 1
                nbytes = numel * elem_sizes[t_dtype]
                specs[name] = TensorArenaSpec(
                    name=name,
                    shape=shape,
                    offset_bytes=v_cursor,
                    nbytes=nbytes,
                    region=region_name,
                    dtype=t_dtype,
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
        base_offset_bytes: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Создает zero-copy тензорные view слота внутри единой арены uint8."""
        fallback_dtype = dtype or self.dtype
        base = slot_idx * self.slot_stride if base_offset_bytes is None else base_offset_bytes
        if arena.dtype != torch.uint8 or arena.ndim != 1 or base < 0 or base+self.slot_stride > arena.numel():
            raise ValueError("Slot lies outside its byte arena")
        views: Dict[str, torch.Tensor] = {}
        for name, spec in self.specs.items():
            t_dtype = spec.dtype if spec.dtype is not None else fallback_dtype
            if spec.nbytes == 0:
                views[name] = torch.empty(spec.shape, dtype=t_dtype, device=arena.device)
            else:
                start = base + spec.offset_bytes
                views[name] = arena[start : start + spec.nbytes].view(t_dtype).view(spec.shape)
        return views


def slot_arena_geometry(layout, num_slots, slot_layouts=None):
    """One contiguous arena, with explicit disjoint offsets for each slot."""
    layouts = tuple(slot_layouts) if slot_layouts is not None else (layout,)*num_slots
    if num_slots < 1 or len(layouts) != num_slots:
        raise ValueError("Slot layout count must match the arena")
    offsets, total = [], 0
    for item in layouts:
        if not isinstance(item, SlotArenaLayout) or item.dtype != layout.dtype:
            raise ValueError("Slot layout dtype mismatch")
        if not 0 <= item.payload_bytes <= item.slot_stride:
            raise ValueError("Slot payload exceeds its stride")
        offsets.append(total)
        total += item.slot_stride
    return layouts, tuple(offsets), total


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
        slot_layouts: Optional[List[SlotArenaLayout]] = None,
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

        self.slot_layouts, self.slot_offsets, total_bytes = slot_arena_geometry(self.layout,num_slots,slot_layouts)
        self.slot_strides = tuple(item.slot_stride for item in self.slot_layouts)
        self.slot_stride = self.slot_strides[0] if len(set(self.slot_strides))==1 else None
        use_pinned = self.device.type == "cuda"
        self.arena: Optional[torch.Tensor] = (
            torch.empty(total_bytes, dtype=torch.uint8, pin_memory=use_pinned)
            if total_bytes > 0
            else torch.empty(0, dtype=torch.uint8)
        )
        self.total_pinned_bytes = self.arena.nbytes

        # Slots may have different templates, but still share one allocation.
        self.slots: List[Dict[str, torch.Tensor]] = [
            self.slot_layouts[s].bind_slot_views(self.arena, s, dtype=dtype,base_offset_bytes=self.slot_offsets[s])
            for s in range(self.num_slots)
        ]

        # CUDA события для защиты от гонок между CPU-записью в буфер и GPU DMA-чтением
        if self.device.type == "cuda":
            self.dma_events: List[Optional[torch.cuda.Event]] = [torch.cuda.Event() for _ in range(num_slots)]
            self._has_in_flight_dma: List[bool] = [False] * num_slots
        else:
            self.dma_events = [None] * num_slots
            self._has_in_flight_dma = [False] * num_slots

        strides_mb = [round(size / (1024**2),2) for size in self.slot_strides]
        print(
            f"[BoundedPinnedPool] Allocated {num_slots} pinned DMA slots "
            f"({self.total_pinned_bytes / (1024**2):.1f} MB total arena | "
            f"slot MiB {strides_mb}) with zero-race protection"
        )

    def copy_into_slot(self, slot_idx: int, src_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Копирует активные тензоры слоя в закрепленный слот перед передачей по DMA.

        Возвращает только view, соответствующие ключам src_tensors (без неактивного overlay-алиаса),
        чтобы асинхронный DMA не затирал активную ветку attention.
        """
        if self.arena is None or not self.slots:
            raise RuntimeError("BoundedPinnedBufferPool is closed; cannot copy_into_slot.")

        if any(k.startswith("linear_attn.") for k in src_tensors) and any(k.startswith("self_attn.") for k in src_tensors):
            raise ValueError("Cannot write both attention variants into an overlay slot")
        if not src_tensors:
            raise ValueError("Empty pinned payload")
        idx = slot_idx % self.num_slots
        # Validate the entire payload before the first write (copy_ can broadcast).
        for name, src_t in src_tensors.items():
            target = self.slots[idx].get(name)
            if target is None or not isinstance(src_t, torch.Tensor):
                raise ValueError(f"Unknown pinned tensor: {name}")
            if src_t.shape != target.shape or src_t.dtype != target.dtype or src_t.device.type != "cpu":
                raise ValueError(f"Pinned tensor shape/dtype/device mismatch: {name}")

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

    def __init__(self, globals_path: Optional[str], dtype: torch.dtype,
                 key_map: Optional[Dict[str, Tuple[str, str]]] = None):
        """``key_map`` (canonical name -> (file, stored key)) serves sharded Hugging Face checkpoints."""
        self._globals_path = globals_path
        self._dtype = dtype
        self._key_map = key_map
        self._handles: Dict[str, Any] = {}
        if key_map is None:
            self._handle: Any = safetensors.safe_open(globals_path, framework="pt", device="cpu")
            self._keys: Dict[str, None] = dict.fromkeys(self._handle.keys())
        else:
            self._handle = None
            self._keys = dict.fromkeys(key_map)
        self._materialized: Dict[str, torch.Tensor] = {}

    def _ensure_handle(self) -> Any:
        if self._handle is None:
            self._handle = safetensors.safe_open(self._globals_path, framework="pt", device="cpu")
        return self._handle

    def _slice(self, key: str) -> Any:
        if self._key_map is None:
            return self._ensure_handle().get_slice(key)
        path, stored = self._key_map[key]
        handle = self._handles.get(path)
        if handle is None:
            handle = safetensors.safe_open(path, framework="pt", device="cpu")
            self._handles[path] = handle
        return handle.get_slice(stored)

    def release_handle(self) -> None:
        """Drop the globals mapping after staged copies, retain lazy key metadata.

        Reopening for embedding rows prevents already-copied lm_head pages from
        staying mapped for the whole inference lifetime. Materialized tensors,
        when explicitly requested by callers, retain their own storage owner.
        """
        self._handle = None
        self._handles.clear()

    def _read_tensor(self, key: str) -> torch.Tensor:
        sl = self._slice(key)
        shape = sl.get_shape()
        if len(shape) >= 2 and shape[0] > 4096:
            out = torch.empty(shape, dtype=self._dtype, device="cpu")
            for r0 in range(0, shape[0], 4096):
                r1 = min(r0 + 4096, shape[0])
                out[r0:r1].copy_(sl[r0:r1])
            return out
        return sl[:].to(dtype=self._dtype)

    def get_tensor_slice(self, key: str) -> Any:
        """Возвращает ленивый safetensors slice без материализации всего тензора в RAM."""
        if key not in self._keys:
            raise KeyError(key)
        return self._slice(key)

    def lookup_rows(self, key: str, indices: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Читает только нужные строки embedding-таблицы без загрузки 1.94 ГБ в RAM."""
        if key not in self._keys:
            raise KeyError(key)
        target_dtype = dtype or self._dtype
        if key in self._materialized:
            return self._materialized[key][indices].to(dtype=target_dtype)
        sl = self._slice(key)
        flat_ids = indices.reshape(-1).tolist()
        if not flat_ids:
            cols = sl.get_shape()[1]
            return torch.empty((*indices.shape, cols), dtype=target_dtype)
        rows = torch.cat([sl[int(i) : int(i) + 1] for i in flat_ids], dim=0).to(dtype=target_dtype)
        return rows.view(*indices.shape, -1)

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
        self._handles.clear()

    def close(self) -> None:
        self.clear()

    def __repr__(self) -> str:
        return f"LazyGlobalsMap(keys={list(self._keys.keys())}, loaded={list(self._materialized.keys())})"


class ShardedLayerView:
    """``safe_open``-like view of one decoder layer spread over Hugging Face shards.

    ``entries`` maps canonical names (``model.layers.{i}.<param>``) to ``(file, stored key)``.
    """

    def __init__(self, entries: Dict[str, Tuple[str, str]]):
        self._entries = dict(entries)
        self._handles: Dict[str, Any] = {}

    def _handle(self, path: str) -> Any:
        h = self._handles.get(path)
        if h is None:
            h = safetensors.safe_open(path, framework="pt", device="cpu")
            self._handles[path] = h
        return h

    def keys(self) -> List[str]:
        return list(self._entries)

    def entries(self) -> Dict[str, Tuple[str, str]]:
        return dict(self._entries)

    def get_slice(self, key: str) -> Any:
        path, stored = self._entries[key]
        return self._handle(path).get_slice(stored)

    def get_tensor(self, key: str) -> torch.Tensor:
        path, stored = self._entries[key]
        return self._handle(path).get_tensor(stored)


class StableWeightManager:
    """Стабильный менеджер весов с защитой от гонок, постоянными mmap-дескрипторами и кэшем."""

    def __init__(
        self,
        model_dir: str,
        max_ram_layers: int = 8,
        num_pinned_slots: int = 2,
        cache_policy: str = "static_pinned",
        dtype: torch.dtype = torch.bfloat16,
        weight_format: str = "bf16",
        q4_group_size: int = Q4_GROUP_SIZE,
        q4_cache_dir: Optional[str] = None,
        checkpoint_manifest: Optional[str] = None,
        max_ram_bytes: Optional[int] = None,
        max_mmap_handles: int = 2,
        weight_reader=None,
        model_source=None,
    ):
        self.model_dir = model_dir
        self.model_source = model_source
        self.dtype = dtype
        self.cache_policy = cache_policy
        if weight_format not in ("bf16", "q4", "q4_g16_asym", "q4_g32_asym", "q4_g64_asym"):
            raise ValueError("Unsupported weight format")
        self.is_q4 = weight_format.startswith("q4")
        self.weight_format = f"q4_g{q4_group_size}_asym" if self.is_q4 else "bf16"
        self.q4_group_size = q4_group_size
        self.ram_cache = BoundedRAMCache(max_cached_layers=max_ram_layers, policy=cache_policy, max_bytes=max_ram_bytes)
        if max_mmap_handles < 1:
            raise ValueError("Need at least one mmap handle")
        self.max_mmap_handles = max_mmap_handles
        self.weight_reader = weight_reader
        self.prepacked_cache = None
        self.quantization_calls = 0
        self._closed = False
        if q4_cache_dir is not None:
            if not self.is_q4 or (checkpoint_manifest is None and model_source is None):
                raise ValueError("Q4 cache requires Q4 format and a checkpoint manifest or model source")
            from src.prepacked_q4_cache import PrepackedLayerCache
            self.prepacked_cache = PrepackedLayerCache(model_dir, q4_cache_dir, checkpoint_manifest,
                dtype=dtype, group_size=q4_group_size, reader=weight_reader, source=model_source)

        # Постоянные открытые mmap-дескрипторы файлов safetensors
        self.mmap_handles: Dict[int, Any] = collections.OrderedDict()

        # Читаем только метаданные форм весов (без чтения тензоров с диска) для linear_attn и self_attn
        template: Dict[str, Any] = {}
        for l_idx in [0, 3]:
            l_file = os.path.join(model_dir, f"model-{l_idx:05d}.safetensors")
            if model_source is not None or os.path.exists(l_file):
                f = self._get_mmap_handle(l_idx)
                for k in f.keys():
                    short_k = k.replace(f"model.layers.{l_idx}.", "")
                    shape = tuple(f.get_slice(k).get_shape())
                    if self.is_q4 and is_q4_quantizable(short_k, shape, group_size=self.q4_group_size):
                        template.update(
                            q4_tensor_specs(
                                short_k,
                                shape,
                                group_size=self.q4_group_size,
                                compute_dtype=self.dtype,
                            )
                        )
                    else:
                        template[short_k] = (shape, self.dtype)

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
            if self.model_source is not None:
                self.mmap_handles[layer_idx] = ShardedLayerView(self.model_source.layer_entries(layer_idx))
            else:
                layer_file = os.path.join(self.model_dir, f"model-{layer_idx:05d}.safetensors")
                self.mmap_handles[layer_idx] = safetensors.safe_open(layer_file, framework="pt", device="cpu")
        self.mmap_handles.move_to_end(layer_idx)
        while len(self.mmap_handles) > self.max_mmap_handles:
            self.mmap_handles.popitem(last=False)
        return self.mmap_handles[layer_idx]

    def _load_globals(self) -> None:
        if self.model_source is not None:
            self.globals = LazyGlobalsMap(None, dtype=self.dtype, key_map=self.model_source.globals_entries())
            return
        globals_path = os.path.join(self.model_dir, "model-globals.safetensors")
        self.globals = LazyGlobalsMap(globals_path, dtype=self.dtype)

    def read_layer_tensors(
        self,
        layer_idx: int,
        target_device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """Читает веса слоя из mmap и при weight_format='q4_g64_asym' квантует 2D-проекции в W4A16."""
        if self._closed:
            raise RuntimeError("Weight manager is closed")
        if self.prepacked_cache is not None:
            tensors = self.prepacked_cache.get_or_build(layer_idx,
                lambda: self._read_uncached(layer_idx, torch.device("cpu")))
            dst = target_device or torch.device("cpu")
            return {k: t.to(dst) for k, t in tensors.items()}
        return self._read_uncached(layer_idx, target_device)

    def _read_uncached(self, layer_idx, target_device=None):
        dst = target_device or torch.device("cpu")
        f = self._get_mmap_handle(layer_idx)
        names = list(f.keys())
        if self.weight_reader is None:
            loaded = None
        elif isinstance(f, ShardedLayerView):
            loaded = (self.weight_reader.read_entries(f.entries())
                      if hasattr(self.weight_reader, "read_entries") else None)
        else:
            loaded = self.weight_reader(self.model_dir, f"model-{layer_idx:05d}", names)
        tensors: Dict[str, torch.Tensor] = {}
        prefix = f"model.layers.{layer_idx}."
        for k in names:
            if not k.startswith(prefix):
                raise ValueError(f"Foreign tensor in layer shard: {k}")
            short_k = k[len(prefix):]
            t = loaded[k] if loaded is not None else f.get_tensor(k)
            if self.is_q4 and is_q4_quantizable(short_k, tuple(t.shape), group_size=self.q4_group_size):
                self.quantization_calls += 1
                q_k, s_k, z_k = q4_subkey_names(short_k)
                qweight, scales, zeros = quantize_tensor_q4(
                    t,
                    group_size=self.q4_group_size,
                    compute_dtype=self.dtype,
                    target_device=dst,
                )
                tensors[q_k] = qweight
                tensors[s_k] = scales
                tensors[z_k] = zeros
            else:
                tensors[short_k] = t.to(device=dst, dtype=self.dtype)
        return tensors

    def load_layer_to_pinned_slot(self, layer_idx: int, target_slot_idx: int) -> Dict[str, torch.Tensor]:
        """Загружает веса слоя в закрепленный слот для отправки на GPU по DMA."""
        # 1. Проверяем RAM-кэш
        cached = self.ram_cache.get(layer_idx)
        if cached is None:
            # 2. Читаем через постоянный открытый mmap-дескриптор (и квантуем в Q4 при необходимости)
            cached = self.read_layer_tensors(layer_idx, target_device=torch.device("cpu"))

            # Сохраняем в RAM-кэш в соответствии с выбранной политикой
            self.ram_cache.put(layer_idx, cached)

        # 3. Безопасно копируем в закрепленный буфер пула (с авто-синхронизацией DMA)
        pinned_slot = self.pinned_pool.copy_into_slot(target_slot_idx, cached)
        return pinned_slot

    def close(self) -> None:
        """Освобождает дескрипторы, кэш RAM, глобалы и pinned-буферы."""
        self._closed = True
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
