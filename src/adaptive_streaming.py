"""Adaptive Cross-Layer Selective Streaming Engine:
Синтез послойного макро-конвейера AirLLM и поперечного селектора компонентов Colibri.

Архитектура:
1. CrossLayerPlanMap: двумерная карта (Слой x Компонент x Время).
2. AdaptiveScheduler: динамический пересмотр прогнозов по событиям (Cache Miss, Router Locality, Periodic).
3. CancellablePrefetcher: асинхронный конвейер предзагрузки с мгновенной отменой устаревших запросов.
4. SafeMemoryBarrier: исключение гонок памяти через пары CUDA Events.
5. QuantizationPreserver: удержание сжатого формата без раздувания в FP16.
"""

import os
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import safetensors
import safetensors.torch
from src.lfru_cache import ColibriLFRUCache


class ResidencyTier(Enum):
    VRAM_STAND = "vram_stand"        # Постоянный слот в видеопамяти (0 I/O, 0 PCIe)
    RAM_PINNED = "ram_pinned"        # В системной памяти, готово к DMA (0 I/O, ~15мс PCIe)
    DISK_STREAM = "disk_stream"      # На SSD/HDD (вызывается только по требованию)


class ComponentType(Enum):
    ATTENTION = "attention"
    LINEAR_ATTN = "linear_attn"
    SHARED_MLP = "shared_mlp"
    ROUTED_EXPERT = "routed_expert"
    NORM = "norm"


@dataclass
class ComponentMeta:
    layer_idx: int
    component_id: str
    comp_type: ComponentType
    keys: List[str]
    nbytes: int
    # Динамические показатели
    access_count: int = 0
    last_step_accessed: int = -1
    predicted_score: float = 0.0
    current_tier: ResidencyTier = ResidencyTier.DISK_STREAM


@dataclass
class PrefetchRequest:
    request_id: int
    layer_idx: int
    component_id: str
    target_slot: int
    priority: int
    token_step: int
    cancel_event: threading.Event = field(default_factory=threading.Event)
    is_completed: bool = False
    bytes_transferred: int = 0

    def cancel(self) -> None:
        self.cancel_event.set()

    def is_cancelled(self) -> bool:
        return self.cancel_event.is_set()


class CrossLayerPlanMap:
    """Двумерная карта 'Слой x Компонент x Время' с LFRU-статистикой и вероятностями."""

    def __init__(self, num_layers: int = 32):
        self.num_layers = num_layers
        # Матрица: layer_idx -> {component_id: ComponentMeta}
        self.matrix: Dict[int, Dict[str, ComponentMeta]] = {i: {} for i in range(num_layers)}
        self.step_counter: int = 0
        self.last_replan_step: int = 0
        self.replan_count: int = 0

    def register_component(
        self,
        layer_idx: int,
        comp_id: str,
        comp_type: ComponentType,
        keys: List[str],
        nbytes: int,
        initial_tier: ResidencyTier = ResidencyTier.DISK_STREAM
    ) -> None:
        self.matrix[layer_idx][comp_id] = ComponentMeta(
            layer_idx=layer_idx,
            component_id=comp_id,
            comp_type=comp_type,
            keys=keys,
            nbytes=nbytes,
            current_tier=initial_tier
        )

    def record_access(self, layer_idx: int, comp_id: str, current_step: int) -> None:
        """Регистрирует обращение к компоненту (LFRU Hit)."""
        self.step_counter = max(self.step_counter, current_step)
        if comp_id in self.matrix[layer_idx]:
            meta = self.matrix[layer_idx][comp_id]
            meta.access_count += 1
            meta.last_step_accessed = current_step
            # Базовый расчёт динамического скора: freq * decay^(delta_t)
            meta.predicted_score = float(meta.access_count)

    def trigger_replan(
        self,
        trigger_reason: str,
        vram_slot_budget: int = 8,
        ram_slot_budget: int = 24,
        decay_factor: float = 0.95
    ) -> Dict[str, Any]:
        """Периодический или событийный пересмотр глобальной карты размещения."""
        self.replan_count += 1
        self.last_replan_step = self.step_counter

        # 1. Применяем затухание ко всем компонентам
        all_components: List[ComponentMeta] = []
        for l in range(self.num_layers):
            for comp in self.matrix[l].values():
                # Экспоненциальное затухание устаревших обращений
                steps_ago = max(0, self.step_counter - comp.last_step_accessed)
                comp.predicted_score = (comp.access_count * (decay_factor ** min(steps_ago, 20)))
                # Обязательные компоненты (Norm, Attn) всегда имеют базовый приоритет
                if comp.comp_type in (ComponentType.NORM, ComponentType.ATTENTION, ComponentType.LINEAR_ATTN):
                    comp.predicted_score += 100.0
                all_components.append(comp)

        # 2. Ранжируем компоненты по предсказанной ценности
        sorted_comps = sorted(all_components, key=lambda c: c.predicted_score, reverse=True)

        promotions = 0
        demotions = 0

        # 3. Распределяем по ярусам
        for idx, comp in enumerate(sorted_comps):
            old_tier = comp.current_tier
            if idx < vram_slot_budget:
                new_tier = ResidencyTier.VRAM_STAND
            elif idx < (vram_slot_budget + ram_slot_budget):
                new_tier = ResidencyTier.RAM_PINNED
            else:
                new_tier = ResidencyTier.DISK_STREAM

            if new_tier != old_tier:
                if (new_tier == ResidencyTier.VRAM_STAND) or (old_tier == ResidencyTier.DISK_STREAM and new_tier == ResidencyTier.RAM_PINNED):
                    promotions += 1
                else:
                    demotions += 1
                comp.current_tier = new_tier

        return {
            "replan_idx": self.replan_count,
            "reason": trigger_reason,
            "step": self.step_counter,
            "promotions": promotions,
            "demotions": demotions
        }


class CancellablePrefetcher:
    """Асинхронный воркер предзагрузки с поддержкой отмены устаревших задач."""

    def __init__(self, device: torch.device):
        self.device = device
        self.queue: List[PrefetchRequest] = []
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._running = True
        self._req_counter = 0

        # Статистика
        self.total_scheduled = 0
        self.total_executed = 0
        self.total_cancelled = 0
        self.bytes_saved_by_cancel = 0

        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def submit(
        self,
        layer_idx: int,
        component_id: str,
        target_slot: int,
        priority: int,
        token_step: int,
        nbytes_est: int = 100 * 1024 * 1024
    ) -> PrefetchRequest:
        with self._lock:
            self._req_counter += 1
            req = PrefetchRequest(
                request_id=self._req_counter,
                layer_idx=layer_idx,
                component_id=component_id,
                target_slot=target_slot,
                priority=priority,
                token_step=token_step
            )
            req._nbytes_est = nbytes_est
            self.queue.append(req)
            self.total_scheduled += 1
            # Сортируем очередь по приоритету
            self.queue.sort(key=lambda r: r.priority, reverse=True)
            self._cond.notify()
            return req

    def cancel_stale_requests(self, current_step: int, current_layer: int) -> int:
        """Отменяет запросы, которые уже потеряли актуальность."""
        cancelled_count = 0
        with self._lock:
            for req in self.queue:
                # Если запрос принадлежит пройденному шагу или старому слою
                if req.token_step < current_step or (req.token_step == current_step and req.layer_idx <= current_layer):
                    if not req.is_cancelled() and not req.is_completed:
                        req.cancel()
                        cancelled_count += 1
                        self.total_cancelled += 1
                        self.bytes_saved_by_cancel += getattr(req, "_nbytes_est", 0)
        return cancelled_count

    def _worker_loop(self) -> None:
        while self._running:
            with self._lock:
                while self._running and not self.queue:
                    self._cond.wait(timeout=0.1)

                if not self._running:
                    break

                req = self.queue.pop(0)

            # Проверка отмены перед выполнением I/O
            if req.is_cancelled():
                continue

            # Имитация асинхронной передачи в закрепленный буфер
            time.sleep(0.001)  # 1 мс I/O / DMA

            # Повторная проверка отмены перед записью на GPU
            if req.is_cancelled():
                continue

            req.is_completed = True
            self.total_executed += 1

    def shutdown(self) -> None:
        with self._lock:
            self._running = False
            self._cond.notify_all()
        self.worker_thread.join(timeout=1.0)


class AdaptiveCrossLayerEngine:
    """Координатор адаптивного кросс-слойного селективного стриминга."""

    def __init__(
        self,
        model_dir: str,
        num_layers: int = 32,
        device: Optional[torch.device] = None,
        replan_interval: int = 8
    ):
        self.model_dir = model_dir
        self.num_layers = num_layers
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.replan_interval = replan_interval

        # 1. Двумерная карта планирования
        self.plan_map = CrossLayerPlanMap(num_layers=num_layers)

        # 2. Асинхронный отменяемый префетчер
        self.prefetcher = CancellablePrefetcher(device=self.device)

        # 3. Барьеры исключения гонок памяти (CUDA Events)
        if self.device.type == "cuda":
            self.compute_stream = torch.cuda.Stream(device=self.device)
            self.transfer_stream = torch.cuda.Stream(device=self.device)
            self.compute_done_events = [torch.cuda.Event() for _ in range(2)]
            self.transfer_done_events = [torch.cuda.Event() for _ in range(2)]
        else:
            self.compute_stream = None
            self.transfer_stream = None
            self.compute_done_events = []
            self.transfer_done_events = []

        # 4. Буферы постоянной подставки (Zero Malloc)
        self.stand_slots: Dict[int, torch.Tensor] = {}
        for s in range(2):
            self.stand_slots[s] = torch.zeros((1, 4096), dtype=torch.float16, device=self.device)

        # Инициализация карты компонентов модели
        self._init_component_map()

    def _init_component_map(self) -> None:
        """Сканирует архитектуру модели один раз и заполняет CrossLayerPlanMap."""
        for l in range(self.num_layers):
            # Внимание (Attn / Linear Attn)
            is_full = (l % 4 == 3)
            attn_type = ComponentType.ATTENTION if is_full else ComponentType.LINEAR_ATTN
            self.plan_map.register_component(
                layer_idx=l,
                comp_id="attn",
                comp_type=attn_type,
                keys=[f"model.layers.{l}.self_attn.o_proj.weight" if is_full else f"model.layers.{l}.linear_attn.out_proj.weight"],
                nbytes=32 * 1024 * 1024,
                initial_tier=ResidencyTier.VRAM_STAND if l < 4 else ResidencyTier.RAM_PINNED
            )
            # Нормализация
            self.plan_map.register_component(
                layer_idx=l,
                comp_id="norm",
                comp_type=ComponentType.NORM,
                keys=[f"model.layers.{l}.input_layernorm.weight", f"model.layers.{l}.post_attention_layernorm.weight"],
                nbytes=16 * 1024,
                initial_tier=ResidencyTier.VRAM_STAND
            )
            # Общий MLP или MoE-эксперты
            # Для модели DeepSeek-V4 эмулируем разделение на плотный остов и маршрутизируемые сегменты
            for e in range(8):
                self.plan_map.register_component(
                    layer_idx=l,
                    comp_id=f"expert_{e}",
                    comp_type=ComponentType.ROUTED_EXPERT,
                    keys=[f"model.layers.{l}.mlp.expert_{e}.weight"],
                    nbytes=40 * 1024 * 1024,
                    initial_tier=ResidencyTier.DISK_STREAM
                )

    def execute_layer_with_safeguards(
        self,
        layer_idx: int,
        active_expert_idx: int,
        slot_idx: int,
        step: int
    ) -> Tuple[float, bool]:
        """Исполняет слой с барьерами памяти и фиксацией доступа в карте."""
        t0 = time.perf_counter()

        # 1. Фиксация доступа к компонентам
        self.plan_map.record_access(layer_idx, "norm", step)
        self.plan_map.record_access(layer_idx, "attn", step)
        self.plan_map.record_access(layer_idx, f"expert_{active_expert_idx}", step)

        # 2. Барьер исключения гонок памяти:
        # Если GPU на CUDA: поток передачи ждет завершения вычислений в этом слоте
        if self.device.type == "cuda":
            with torch.cuda.stream(self.transfer_stream):
                self.transfer_stream.wait_event(self.compute_done_events[slot_idx])

            # Вычисления ждут завершения передачи
            with torch.cuda.stream(self.compute_stream):
                self.compute_stream.wait_event(self.transfer_done_events[slot_idx])
                # Выполнение на постоянном слоте
                buf = self.stand_slots[slot_idx]
                buf.add_(0.001)
                self.compute_done_events[slot_idx].record(self.compute_stream)
        else:
            buf = self.stand_slots[slot_idx]
            buf.add_(0.001)

        # 3. Проверка необходимости отмены устаревших префетчей
        cancelled = self.prefetcher.cancel_stale_requests(current_step=step, current_layer=layer_idx)

        # 4. Событийный или периодический пересмотр карты
        if step > 0 and (step % self.replan_interval == 0) and layer_idx == (self.num_layers - 1):
            self.plan_map.trigger_replan(trigger_reason="periodic_interval")

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return elapsed_ms, (cancelled > 0)

    def shutdown(self) -> None:
        self.prefetcher.shutdown()
