"""Progressive Live Acceleration Engine for AirColibri (Stage D):
Постепенное подключение GPU-ускорения во время активной генерации без прерывания потока токенов.

Архитектура:
1. Токен 0 стартует на доступных слоях немедленно (минимальный TTFT).
2. Фоновый поток (GPUProgressiveAttacher) асинхронно готовит и переносит следующие слои на GPU.
3. Атомарное переключение слоев: как только слой подготовлен, флаг is_gpu_active[layer_idx] становится True.
4. Миграция состояния KV-кэша: при переключении слоя его KV-состояние переносится на GPU без перезапуска генерации.
"""

import os
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch


class ProgressiveLayerStatus:
    def __init__(self, layer_idx: int, initially_on_gpu: bool = False):
        self.layer_idx = layer_idx
        self.is_gpu_active = initially_on_gpu
        self.is_loading = False
        self.gpu_ready_time_ms: Optional[float] = None


class ProgressiveLiveAttacher:
    """Управляет динамическим фоновым подключением слоев к GPU во время генерации."""

    def __init__(
        self,
        num_layers: int,
        initial_gpu_layers: int = 4,
        device: Optional[torch.device] = None,
        attach_callback: Optional[Callable[[int], None]] = None
    ):
        self.num_layers = num_layers
        self.initial_gpu_layers = initial_gpu_layers
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attach_callback = attach_callback

        # Статус каждого слоя
        self.layer_statuses: List[ProgressiveLayerStatus] = [
            ProgressiveLayerStatus(i, initially_on_gpu=(i < initial_gpu_layers))
            for i in range(num_layers)
        ]

        self._lock = threading.Lock()
        self._running = True
        self.attached_count = initial_gpu_layers
        self.t0_start = time.perf_counter()

        # Фоновый воркер асинхронного подключения
        self.worker = threading.Thread(target=self._background_attachment_loop, daemon=True)
        self.worker.start()

    def is_layer_on_gpu(self, layer_idx: int) -> bool:
        return self.layer_statuses[layer_idx].is_gpu_active

    def _background_attachment_loop(self) -> None:
        """Постепенно загружает слои на GPU в фоновом режиме."""
        for layer_idx in range(self.initial_gpu_layers, self.num_layers):
            if not self._running:
                break

            # Имитируем асинхронную подготовку весов слоя (DMA пересылка)
            time.sleep(0.02)  # 20 мс на подготовку слоя

            # Вызываем callback для реальной фиксации весов на GPU
            if self.attach_callback is not None:
                self.attach_callback(layer_idx)

            with self._lock:
                status = self.layer_statuses[layer_idx]
                status.is_gpu_active = True
                status.gpu_ready_time_ms = (time.perf_counter() - self.t0_start) * 1000.0
                self.attached_count += 1

    def stop(self) -> None:
        self._running = False
        self.worker.join(timeout=1.0)
