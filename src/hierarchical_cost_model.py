"""Hierarchical Cost Model for AirColibri (Stage E):
Многоуровневая оценка затрат вычислений, передачи и синхронизации для принятия решений:
CPU vs GPU, VRAM vs RAM vs SSD.

Правило:
GPU выгодно, если T_gpu_compute + T_transfer + T_sync < T_cpu_compute.
"""

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@dataclass
class HardwareProfile:
    # Пропускная способность (ГБ/с)
    vram_bandwidth_gbps: float = 448.0   # RTX 3060 Ti GDDR6
    pcie_bandwidth_gbps: float = 14.0    # PCIe Gen4 x8/x16 pinned
    pageable_bandwidth_gbps: float = 4.5 # Pageable RAM transfer
    ssd_bandwidth_gbps: float = 2.5      # NVMe SSD seq read
    # Вычислительная производительность (TFLOPS)
    gpu_tflops_fp16: float = 16.2
    cpu_tflops_fp16: float = 0.5         # OpenMP SIMD CPU
    # Задержки запуска ядра / DMA (мс)
    gpu_kernel_launch_ms: float = 0.015
    pcie_dma_latency_ms: float = 0.020
    ssd_seek_latency_ms: float = 0.200


class HierarchicalCostModel:
    """Оценивает оптимальное размещение и вычислитель для каждого компонента модели."""

    def __init__(self, hw_profile: Optional[HardwareProfile] = None):
        self.hw = hw_profile or HardwareProfile()

    def estimate_cpu_time_ms(self, nbytes: int, flops: int) -> float:
        """Оценивает время вычисления на CPU (мс)."""
        compute_ms = (flops / (self.hw.cpu_tflops_fp16 * 1e12)) * 1000.0
        return compute_ms

    def estimate_gpu_time_ms(
        self,
        nbytes: int,
        flops: int,
        source_tier: str = "RAM_PINNED"
    ) -> Tuple[float, float, float]:
        """Оценивает общее время выполнения на GPU с учетом передачи и синхронизации.

        Returns: (total_ms, transfer_ms, compute_ms)
        """
        # 1. Время передачи весов
        if source_tier == "VRAM":
            transfer_ms = 0.0
        elif source_tier == "RAM_PINNED":
            transfer_ms = self.hw.pcie_dma_latency_ms + (nbytes / (self.hw.pcie_bandwidth_gbps * 1e9)) * 1000.0
        elif source_tier == "RAM_PAGEABLE":
            transfer_ms = self.hw.pcie_dma_latency_ms + (nbytes / (self.hw.pageable_bandwidth_gbps * 1e9)) * 1000.0
        elif source_tier == "SSD":
            transfer_ms = self.hw.ssd_seek_latency_ms + (nbytes / (self.hw.ssd_bandwidth_gbps * 1e9)) * 1000.0
        else:
            transfer_ms = 100.0

        # 2. Время вычисления на GPU
        compute_ms = self.hw.gpu_kernel_launch_ms + (flops / (self.hw.gpu_tflops_fp16 * 1e12)) * 1000.0
        total_ms = transfer_ms + compute_ms
        return total_ms, transfer_ms, compute_ms

    def decide_placement(
        self,
        component_name: str,
        nbytes: int,
        flops: int,
        current_tier: str
    ) -> Dict[str, Any]:
        """Принимает строгое математическое решение: вычислять на GPU или на CPU."""
        t_cpu = self.estimate_cpu_time_ms(nbytes, flops)
        t_gpu_total, t_trans, t_gpu_comp = self.estimate_gpu_time_ms(nbytes, flops, current_tier)

        use_gpu = (t_gpu_total < t_cpu)

        return {
            "component": component_name,
            "decision": "GPU" if use_gpu else "CPU",
            "t_cpu_ms": round(t_cpu, 3),
            "t_gpu_total_ms": round(t_gpu_total, 3),
            "t_gpu_transfer_ms": round(t_trans, 3),
            "t_gpu_compute_ms": round(t_gpu_comp, 3),
            "speedup": round(t_cpu / max(1e-5, t_gpu_total), 2)
        }
