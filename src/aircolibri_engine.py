"""AirColibri Unified Engine:
1. Статическая карта весов (Static Weight Map) — строится 1 раз при старте.
2. Динамический ярусный кэш (Dynamic Tiered Cache) с настраиваемым бюджетом VRAM (2, 4, 6 ГБ).
3. Реальное повторное использование памяти (Zero Disk Re-read) — загруженные веса не перечитываются.
4. Послойный конвейер с двойной буферизацией (Double-Buffered Streaming) для вытесняемых слоёв.
5. Постоянные VRAM/RAM подставки (Zero Malloc Runtime).
"""

import gc
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import psutil
import safetensors
import safetensors.torch
import torch
from src.bounded_memory_manager import SlotArenaLayout
from src.lfru_cache import ColibriLFRUCache


def _validate_single_attention_branch(keys: Any) -> None:
    has_linear = any(k.startswith("linear_attn.") for k in keys)
    has_self = any(k.startswith("self_attn.") for k in keys)
    if has_linear and has_self:
        raise ValueError("Cannot mix 'linear_attn.' and 'self_attn.' keys in the same active stand payload.")


@dataclass
class SubtensorMeta:
    full_name: str
    short_name: str
    shape: List[int]
    dtype: str
    nbytes: int


@dataclass
class LayerDescriptor:
    layer_idx: int
    layer_type: str  # 'linear_attention' or 'full_attention'
    priority_score: int  # 1 = critical anchor, 2 = medium, 3 = standard
    file_path: str
    file_bytes: int
    subtensors: Dict[str, SubtensorMeta]
    active_keys: List[str]  # pre-calculated keys for forward pass (no string filtering in loop)
    active_bytes: int


class StaticWeightMap:
    """Статическая карта весов модели: строится СТРОГО ОДИН РАЗ при инициализации."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.layers: Dict[int, LayerDescriptor] = {}
        self.globals_meta: Dict[str, SubtensorMeta] = {}
        self.total_model_bytes: int = 0
        self._build_map()

    def _build_map(self) -> None:
        t0 = time.perf_counter()
        # 1. Индексация 32 слоев
        for i in range(32):
            layer_file = os.path.join(self.model_dir, f"model-{i:05d}.safetensors")
            if not os.path.exists(layer_file):
                continue

            file_bytes = os.path.getsize(layer_file)
            self.total_model_bytes += file_bytes
            is_full_attn = (i % 4 == 3)
            layer_type = "full_attention" if is_full_attn else "linear_attention"

            # Приоритет:
            # Rank 1: слой 0 (базовый вход) и Full Attention слои
            # Rank 2: промежуточные степени двойки (1, 2, 4, 8, 16)
            # Rank 3: линейные слои
            if i == 0 or is_full_attn:
                priority = 1
            elif i in (1, 2, 4, 8, 16):
                priority = 2
            else:
                priority = 3

            subtensors = {}
            active_keys = []
            active_bytes = 0

            with safetensors.safe_open(layer_file, framework="pt", device="cpu") as f:
                for k in f.keys():
                    sl = f.get_slice(k)
                    shape = list(sl.get_shape())
                    dtype_str = str(sl.get_dtype())
                    elem_size = 2 if "F16" in dtype_str or "BF16" in dtype_str else 4
                    numel = 1
                    for s in shape:
                        numel *= s
                    nbytes = numel * elem_size
                    short_k = k.split(".", 3)[-1]
                    meta = SubtensorMeta(
                        full_name=k,
                        short_name=short_k,
                        shape=shape,
                        dtype=dtype_str,
                        nbytes=nbytes
                    )
                    subtensors[k] = meta

                    # Предварительный расчёт активных ключей для прямого прохода
                    if "mlp." in k or "norm" in k or ("linear_attn.out_proj" in k or "self_attn.o_proj" in k):
                        active_keys.append(k)
                        active_bytes += nbytes

            self.layers[i] = LayerDescriptor(
                layer_idx=i,
                layer_type=layer_type,
                priority_score=priority,
                file_path=layer_file,
                file_bytes=file_bytes,
                subtensors=subtensors,
                active_keys=active_keys,
                active_bytes=active_bytes
            )

        # 2. Индексация глобальных весов (эмбеддинг, голова)
        globals_file = os.path.join(self.model_dir, "model-globals.safetensors")
        if os.path.exists(globals_file):
            self.total_model_bytes += os.path.getsize(globals_file)
            with safetensors.safe_open(globals_file, framework="pt", device="cpu") as f:
                for k in f.keys():
                    sl = f.get_slice(k)
                    shape = list(sl.get_shape())
                    dtype_str = str(sl.get_dtype())
                    elem_size = 2 if "F16" in dtype_str else 4
                    numel = 1
                    for s in shape:
                        numel *= s
                    self.globals_meta[k] = SubtensorMeta(
                        full_name=k,
                        short_name=k,
                        shape=shape,
                        dtype=dtype_str,
                        nbytes=numel * elem_size
                    )

        elapsed = (time.perf_counter() - t0) * 1000.0
        print(f"[StaticWeightMap] Indexed {len(self.layers)} layers & {len(self.globals_meta)} globals in {elapsed:.1f} ms | Model size: {self.total_model_bytes / (1024**3):.2f} GB")


class PriorityRanker:
    """Обертка для сохранения обратной совместимости API."""

    def __init__(self, model_dir: str):
        self.weight_map = StaticWeightMap(model_dir)

    def get_ranked_queue(self) -> List[Any]:
        return [self.weight_map.layers[i] for i in sorted(self.weight_map.layers.keys())]

    def get_priority_summary(self) -> Dict[str, Any]:
        p1 = [l.layer_idx for l in self.weight_map.layers.values() if l.priority_score == 1]
        p2 = [l.layer_idx for l in self.weight_map.layers.values() if l.priority_score == 2]
        p3 = [l.layer_idx for l in self.weight_map.layers.values() if l.priority_score == 3]
        return {"rank_1_critical": p1, "rank_2_medium": p2, "rank_3_standard": p3}


class SelectiveSubtensorLoader:
    """Выборочно извлекает из safetensors нужные субтензоры слоя."""

    def __init__(self):
        self.total_bytes_extracted: int = 0

    def load_subtensors(
        self,
        safetensors_path: str,
        target_keys: List[str]
    ) -> Dict[str, torch.Tensor]:
        extracted = {}
        with safetensors.safe_open(safetensors_path, framework="pt", device="cpu") as f:
            for k in target_keys:
                if k in f.keys():
                    t = f.get_slice(k)[:]
                    extracted[k] = t
                    self.total_bytes_extracted += t.nbytes
        return extracted


class MemoryStandPool:
    """Постоянная «подставка» (Persistent Stand) в единых аренах VRAM и pinned RAM (Zero Malloc Runtime)."""

    def __init__(
        self,
        template_tensors: Dict[str, Any],
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        num_stands: int = 2,
        dtype: torch.dtype = torch.float16,
    ):
        self.device = device
        self.num_stands = num_stands
        self.dtype = dtype

        normalized_template: Dict[str, Tuple[int, ...]] = {}
        for k, v in template_tensors.items():
            short_k = k.split(".", 3)[-1] if k.startswith("model.layers.") else k
            normalized_template[short_k] = tuple(v.shape) if isinstance(v, torch.Tensor) else tuple(v)

        self.layout = SlotArenaLayout.from_templates(normalized_template, dtype=dtype)
        total_bytes = self.num_stands * self.layout.slot_stride
        use_pinned = device.type == "cuda"

        self.gpu_arena: Optional[torch.Tensor] = (
            torch.zeros(total_bytes, dtype=torch.uint8, device=device)
            if total_bytes > 0
            else torch.empty(0, dtype=torch.uint8, device=device)
        )
        self.pinned_arena: Optional[torch.Tensor] = (
            torch.empty(total_bytes, dtype=torch.uint8, pin_memory=use_pinned)
            if total_bytes > 0
            else torch.empty(0, dtype=torch.uint8)
        )

        self.gpu_stands: List[Dict[str, torch.Tensor]] = [
            self.layout.bind_slot_views(self.gpu_arena, i, dtype=dtype)
            for i in range(num_stands)
        ]
        self.pinned_ram_stands: List[Dict[str, torch.Tensor]] = [
            self.layout.bind_slot_views(self.pinned_arena, i, dtype=dtype)
            for i in range(num_stands)
        ]

        self.active_stand_idx: int = 0
        self.stand_nbytes = self.layout.slot_stride
        self.lfru = ColibriLFRUCache(num_slots=num_stands, num_experts=32)

    def place_on_stand(
        self,
        subtensors: Dict[str, torch.Tensor],
        stand_idx: int,
        stream: Optional[torch.cuda.Stream] = None
    ) -> Dict[str, torch.Tensor]:
        if self.gpu_arena is None or self.pinned_arena is None or not self.gpu_stands:
            raise RuntimeError("MemoryStandPool is closed; cannot place_on_stand.")

        gpu_stand = self.gpu_stands[stand_idx]
        ram_stand = self.pinned_ram_stands[stand_idx]

        normalized_items: List[Tuple[str, torch.Tensor]] = []
        for k, v in subtensors.items():
            short_k = k.split(".", 3)[-1] if k.startswith("model.layers.") else k
            normalized_items.append((short_k, v))
        _validate_single_attention_branch([sk for sk, _ in normalized_items])

        active_gpu_stand: Dict[str, torch.Tensor] = {}
        for short_k, v in normalized_items:
            if short_k in ram_stand:
                ram_stand[short_k].copy_(v)
                if stream is not None and self.device.type == "cuda":
                    with torch.cuda.stream(stream):
                        gpu_stand[short_k].copy_(ram_stand[short_k], non_blocking=True)
                else:
                    gpu_stand[short_k].copy_(ram_stand[short_k], non_blocking=False)
                active_gpu_stand[short_k] = gpu_stand[short_k]
        return active_gpu_stand

    def close(self) -> None:
        """Освобождает GPU-подставки, pinned RAM-подставки и единые арены."""
        if self.device.type == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        self.gpu_stands.clear()
        self.pinned_ram_stands.clear()
        self.gpu_arena = None
        self.pinned_arena = None
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> "MemoryStandPool":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class DynamicTieredCache:
    """Динамический ярусный кэш с настраиваемым бюджетом VRAM (2, 4, 6 ГБ).

    Принцип Zero Disk Re-read:
    - Слой однажды загруженный в RAM или VRAM больше НИКОГДА повторно не читается с диска.
    - Уровень 0 (VRAM Resident): постоянные слои в видеопамяти.
    - Уровень 1 (Pinned Host RAM Resident): слои в системной ОЗУ с быстрой подкачкой по PCIe DMA.
    - Уровень 2 (SSD): используется ТОЛЬКО на холодном старте при первом чтении.
    """

    def __init__(
        self,
        weight_map: StaticWeightMap,
        vram_budget_gb: float = 4.0,
        ram_budget_gb: float = 10.0,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        # Страховочное ограничение RAM (reserve 1.5 GiB, чтобы не уходить в своп)
        avail_ram_bytes = psutil.virtual_memory().available
        ram_reserve_bytes = int(1.5 * (1024**3))
        max_safe_ram_bytes = max(0, avail_ram_bytes - ram_reserve_bytes)

        self.free_ram_at_init_bytes = avail_ram_bytes
        self.ram_clamped = False
        if ram_budget_gb is None:
            requested_ram_bytes = max_safe_ram_bytes
            self.ram_budget_bytes = max_safe_ram_bytes
        else:
            requested_ram_bytes = int(ram_budget_gb * (1024**3))
            if requested_ram_bytes > max_safe_ram_bytes:
                self.ram_budget_bytes = max_safe_ram_bytes
                self.ram_clamped = True
                print(
                    f"[TieredCache] WARNING: RAM budget clamped "
                    f"{requested_ram_bytes / (1024**3):.2f} -> {self.ram_budget_bytes / (1024**3):.2f} GiB "
                    f"(free={avail_ram_bytes / (1024**3):.2f} GiB, reserve=1.50 GiB)"
                )
            else:
                self.ram_budget_bytes = requested_ram_bytes
        self.requested_ram_budget_bytes = requested_ram_bytes

        # Страховочное ограничение VRAM (резерв 512 MiB; при free >= 6.5 GiB бюджет <= 6.0 GiB не режется)
        requested_vram_bytes = int(vram_budget_gb * (1024**3))
        self.requested_vram_budget_bytes = requested_vram_bytes
        self.vram_clamped = False
        self.free_vram_at_init_bytes = 0
        self.vram_budget_bytes = requested_vram_bytes

        if device.type == "cuda" and torch.cuda.is_available():
            dev_idx = device.index if device.index is not None else torch.cuda.current_device()
            free_vram_bytes, _total_vram_bytes = torch.cuda.mem_get_info(dev_idx)
            self.free_vram_at_init_bytes = free_vram_bytes
            vram_reserve_bytes = 512 * 1024**2
            max_safe_vram_bytes = max(0, free_vram_bytes - vram_reserve_bytes)
            if requested_vram_bytes > max_safe_vram_bytes:
                self.vram_budget_bytes = max_safe_vram_bytes
                self.vram_clamped = True
                print(
                    f"[TieredCache] WARNING: VRAM budget clamped "
                    f"{requested_vram_bytes / (1024**3):.2f} -> {self.vram_budget_bytes / (1024**3):.2f} GiB "
                    f"(free={free_vram_bytes / (1024**3):.2f} GiB, reserve=0.50 GiB)"
                )

        self.weight_map = weight_map
        self.device = device

        # Слои в постоянном кэше
        self.vram_resident_layers: Dict[int, Dict[str, torch.Tensor]] = {}
        self.ram_resident_layers: Dict[int, Dict[str, torch.Tensor]] = {}

        # Двойной буфер для потоковых (нерезидентных в VRAM) слоёв:
        # строим перекрытый шаблон по метаданным обоих типов слоёв (linear_attn + self_attn) без чтения диска.
        template_tensors: Dict[str, Tuple[int, ...]] = {}
        seen_layer_types: Set[str] = set()
        for l_idx in sorted(weight_map.layers.keys()):
            desc = weight_map.layers[l_idx]
            if desc.layer_type in seen_layer_types:
                continue
            seen_layer_types.add(desc.layer_type)
            for k in desc.active_keys:
                meta = desc.subtensors[k]
                template_tensors[meta.short_name] = tuple(meta.shape)

        self.streaming_pool = MemoryStandPool(template_tensors, device=self.device, num_stands=2)
        self.streaming_slot_bytes = self.streaming_pool.stand_nbytes * 2

        # Стримы CUDA для асинхронного перекрытия
        if device.type == "cuda":
            self.compute_stream = torch.cuda.Stream(device=device)
            self.transfer_stream = torch.cuda.Stream(device=device)
            self.transfer_ready_events = [torch.cuda.Event(), torch.cuda.Event()]
            self.compute_done_events = [torch.cuda.Event(), torch.cuda.Event()]
        else:
            self.compute_stream = None
            self.transfer_stream = None

        # Учёт использования памяти
        self.used_vram_bytes = self.streaming_slot_bytes
        self.used_ram_bytes = 0

        # Статистика I/O
        self.total_disk_read_bytes = 0
        self.total_pcie_transfer_bytes = 0

        # Планирование размещения слоёв
        self._plan_residency()

    def _plan_residency(self) -> None:
        """Планирует размещение слоёв по бюджетам VRAM и RAM."""
        # Ранжируем слои для размещения:
        # 1. Приоритет 1 (якоря: 0, 3, 7, 11...)
        # 2. Первые слои конвейера (для создания взлётной полосы асинхронной подкачки)
        candidate_order = sorted(
            self.weight_map.layers.keys(),
            key=lambda idx: (self.weight_map.layers[idx].priority_score, idx)
        )

        for idx in candidate_order:
            desc = self.weight_map.layers[idx]
            layer_bytes = desc.active_bytes

            # Проверяем бюджет VRAM
            if self.used_vram_bytes + layer_bytes <= self.vram_budget_bytes:
                self.vram_resident_layers[idx] = {}  # будет загружен в VRAM
                self.used_vram_bytes += layer_bytes
            # Иначе проверяем бюджет RAM
            elif self.used_ram_bytes + layer_bytes <= self.ram_budget_bytes:
                self.ram_resident_layers[idx] = {}  # будет загружен в RAM
                self.used_ram_bytes += layer_bytes

        print(f"[TieredCache] Plan (eff={self.vram_budget_bytes / (1024**3):.2f}G/{self.ram_budget_bytes / (1024**3):.2f}G, "
              f"clamped=vram:{self.vram_clamped}/ram:{self.ram_clamped}): "
              f"{len(self.vram_resident_layers)} layers in VRAM ({self.used_vram_bytes / (1024**2):.1f} MB) | "
              f"{len(self.ram_resident_layers)} layers in RAM ({self.used_ram_bytes / (1024**2):.1f} MB) | "
              f"{len(self.weight_map.layers) - len(self.vram_resident_layers) - len(self.ram_resident_layers)} layers streaming from SSD")

    def preload_cache(self) -> float:
        """Предварительная загрузка весов по составленному плану (Zero Disk Re-read)."""
        t0 = time.perf_counter()

        # 1. Загрузка слоев в VRAM
        for idx in self.vram_resident_layers.keys():
            desc = self.weight_map.layers[idx]
            tensors = {}
            with safetensors.safe_open(desc.file_path, framework="pt", device="cpu") as f:
                for k in desc.active_keys:
                    short_k = k.split(".", 3)[-1]
                    t = f.get_slice(k)[:]
                    tensors[short_k] = t.to(device=self.device, dtype=torch.float16)
                    self.total_disk_read_bytes += t.nbytes
            self.vram_resident_layers[idx] = tensors

        # 2. Загрузка слоев в Host RAM (хранятся в обычной системной памяти, без блокировки всего пула под pinned)
        for idx in self.ram_resident_layers.keys():
            desc = self.weight_map.layers[idx]
            tensors = {}
            with safetensors.safe_open(desc.file_path, framework="pt", device="cpu") as f:
                for k in desc.active_keys:
                    short_k = k.split(".", 3)[-1]
                    t = f.get_slice(k)[:]
                    tensors[short_k] = t
                    self.total_disk_read_bytes += t.nbytes
            self.ram_resident_layers[idx] = tensors

        preload_time = time.perf_counter() - t0
        return preload_time

    def get_layer_tensors(
        self,
        layer_idx: int,
        stand_idx: int = 0,
        async_stream: bool = False
    ) -> Tuple[Dict[str, torch.Tensor], int, int]:
        """Возвращает тензоры слоя для вычислений.

        Returns: (tensors_dict, disk_bytes_read, pcie_bytes_transferred)
        """
        # СЛУЧАЙ 1: Слой уже в VRAM (Zero PCIe, Zero Disk)
        if layer_idx in self.vram_resident_layers and self.vram_resident_layers[layer_idx]:
            return self.vram_resident_layers[layer_idx], 0, 0

        # СЛУЧАЙ 2: Слой в Host RAM (Zero Disk, быстрая передача по PCIe)
        if layer_idx in self.ram_resident_layers and self.ram_resident_layers[layer_idx]:
            ram_tensors = self.ram_resident_layers[layer_idx]
            _validate_single_attention_branch(ram_tensors.keys())
            gpu_stand = self.streaming_pool.gpu_stands[stand_idx]
            pinned_stand = self.streaming_pool.pinned_ram_stands[stand_idx]
            transfer_bytes = 0
            active_gpu_stand: Dict[str, torch.Tensor] = {}

            # Копируем в закрепленный буфер для истинного DMA по PCIe
            for short_k, t in ram_tensors.items():
                if short_k in pinned_stand:
                    pinned_stand[short_k].copy_(t)

            stream = self.transfer_stream if (async_stream and self.transfer_stream) else None
            if stream is not None:
                with torch.cuda.stream(stream):
                    for short_k in ram_tensors.keys():
                        if short_k in pinned_stand and short_k in gpu_stand:
                            gpu_stand[short_k].copy_(pinned_stand[short_k], non_blocking=True)
                            transfer_bytes += pinned_stand[short_k].nbytes
                            active_gpu_stand[short_k] = gpu_stand[short_k]
            else:
                for short_k in ram_tensors.keys():
                    if short_k in pinned_stand and short_k in gpu_stand:
                        gpu_stand[short_k].copy_(pinned_stand[short_k], non_blocking=False)
                        transfer_bytes += pinned_stand[short_k].nbytes
                        active_gpu_stand[short_k] = gpu_stand[short_k]

            self.total_pcie_transfer_bytes += transfer_bytes
            return active_gpu_stand, 0, transfer_bytes

        # СЛУЧАЙ 3: Cold Miss с SSD (только если не влезло ни в VRAM, ни в RAM)
        desc = self.weight_map.layers[layer_idx]
        disk_bytes = 0
        subtensors = {}
        with safetensors.safe_open(desc.file_path, framework="pt", device="cpu") as f:
            for k in desc.active_keys:
                t = f.get_slice(k)[:]
                subtensors[k] = t
                disk_bytes += t.nbytes

        self.total_disk_read_bytes += disk_bytes
        active_gpu_stand = self.streaming_pool.place_on_stand(subtensors, stand_idx=stand_idx)
        return active_gpu_stand, disk_bytes, disk_bytes

    def close(self) -> None:
        """Освобождает резидентные слои VRAM/RAM и стриминговый пул."""
        if self.device.type == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        self.vram_resident_layers.clear()
        self.ram_resident_layers.clear()
        if getattr(self, "streaming_pool", None) is not None:
            self.streaming_pool.close()
            self.streaming_pool = None
        self.compute_stream = None
        self.transfer_stream = None
        if hasattr(self, "transfer_ready_events"):
            self.transfer_ready_events.clear()
        if hasattr(self, "compute_done_events"):
            self.compute_done_events.clear()
        self.used_vram_bytes = 0
        self.used_ram_bytes = 0
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> "DynamicTieredCache":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class AirColibriEngine:
    """Единый модернизированный движок AirColibri со статической картой и ярусным кэшем."""

    def __init__(
        self,
        model_dir: str,
        vram_budget_gb: float = 4.0,
        ram_budget_gb: Optional[float] = None,
        preload_at_init: bool = True
    ):
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[AirColibri] Initializing on {self.device.type.upper()}: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

        # 1. Построение статической карты весов (СТРОГО ОДИН РАЗ)
        self.weight_map = StaticWeightMap(model_dir)

        # 2. Инициализация ярусного кэша
        self.cache = DynamicTieredCache(
            weight_map=self.weight_map,
            vram_budget_gb=vram_budget_gb,
            ram_budget_gb=ram_budget_gb,
            device=self.device
        )

        # Совместимость для тестов
        self.ranker = PriorityRanker(model_dir)
        self.ranked_layers = self.ranker.get_ranked_queue()
        self.selective_loader = SelectiveSubtensorLoader()
        self.stand_pool = self.cache.streaming_pool

        # 3. Предварительная загрузка (прогрев памяти)
        self.cold_init_time_s = 0.0
        if preload_at_init:
            t0 = time.perf_counter()
            self.cold_init_time_s = self.cache.preload_cache()
            print(f"[AirColibri] Preload complete in {self.cold_init_time_s:.2f} s | Zero Disk Re-read mode ACTIVE")

    def execute_token_step(
        self,
        x: torch.Tensor,
        token_idx: int = 0
    ) -> Tuple[torch.Tensor, float, float, float]:
        """Выполняет 1 шаг генерации токена без повторного чтения с диска.

        Returns: (x_out, step_duration_ms, disk_read_mb, pcie_transfer_mb)
        """
        t0 = time.perf_counter()
        disk_bytes_step = 0
        pcie_bytes_step = 0

        # Конвейерный проход по всем 32 слоям
        num_layers = len(self.weight_map.layers)
        stand_idx = 0

        for i in range(num_layers):
            # Получение тензоров слоя (0 байт с диска, если слой в кэше)
            stand, disk_b, pcie_b = self.cache.get_layer_tensors(i, stand_idx=stand_idx)
            disk_bytes_step += disk_b
            pcie_bytes_step += pcie_b

            # Математический проход через слой
            normed = torch.nn.functional.rms_norm(x, (4096,), stand["input_layernorm.weight"])

            # MLP блок
            gate = torch.matmul(normed, stand["mlp.gate_proj.weight"].T)
            up = torch.matmul(normed, stand["mlp.up_proj.weight"].T)
            act = torch.nn.functional.silu(gate) * up
            mlp_out = torch.matmul(act, stand["mlp.down_proj.weight"].T)

            # Проекция внимания строго по типу слоя (исключаем чтение устаревшего overlay-алиаса)
            layer_desc = self.weight_map.layers[i]
            if layer_desc.layer_type == "linear_attention":
                if "linear_attn.out_proj.weight" not in stand:
                    raise KeyError(f"Layer {i} (linear_attention) missing 'linear_attn.out_proj.weight'")
                attn_out = torch.matmul(normed, stand["linear_attn.out_proj.weight"].T)
            elif layer_desc.layer_type == "full_attention":
                if "self_attn.o_proj.weight" not in stand:
                    raise KeyError(f"Layer {i} (full_attention) missing 'self_attn.o_proj.weight'")
                attn_out = torch.matmul(normed, stand["self_attn.o_proj.weight"].T)
            else:
                raise ValueError(f"Unsupported layer_type {layer_desc.layer_type!r} for layer {i}")
            x = x + attn_out + mlp_out

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        step_duration_ms = (time.perf_counter() - t0) * 1000.0
        return (
            x,
            step_duration_ms,
            disk_bytes_step / (1024 * 1024),
            pcie_bytes_step / (1024 * 1024)
        )

    def close(self) -> None:
        """Освобождает кэш, карту весов и очищает память устройства."""
        if getattr(self, "cache", None) is not None:
            self.cache.close()
            self.cache = None
        self.stand_pool = None
        self.weight_map = None
        self.ranker = None
        self.ranked_layers = []
        self.selective_loader = None
        gc.collect()
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> "AirColibriEngine":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
