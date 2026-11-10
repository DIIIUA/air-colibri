"""AirColibri Unified Engine:
1. Ранжирование слоев (Priority Ranking)
2. Послойная потоковая подача (Layer Streaming)
3. Выборочное выдергивание весов (Selective Sub-layer Extraction)
4. Замещение памяти на постоянной подставке (Persistent Memory Stand + LFRU)
"""

import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import psutil
import safetensors
import safetensors.torch
import torch
from src.lfru_cache import ColibriLFRUCache


@dataclass
class LayerMetadata:
    layer_idx: int
    layer_type: str  # 'linear_attention' or 'full_attention'
    priority_score: int  # 1 = highest, 3 = standard
    file_path: str
    file_bytes: int
    subtensor_names: List[str]


class PriorityRanker:
    """Анализирует архитектуру модели и ранжирует слои по приоритетности."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.layers: List[LayerMetadata] = []
        self._analyze_model()

    def _analyze_model(self) -> None:
        # 32 layers in DeepSeek-V4-9B
        # Every 4th layer (3, 7, 11...) is Full Attention (high priority routing)
        # Layer 0 is foundation embedding receiver (high priority)
        for i in range(32):
            layer_file = os.path.join(self.model_dir, f"model-{i:05d}.safetensors")
            if not os.path.exists(layer_file):
                continue

            file_bytes = os.path.getsize(layer_file)
            is_full_attn = (i % 4 == 3)
            layer_type = "full_attention" if is_full_attn else "linear_attention"

            # Assign priority:
            # Rank 1: First layer (0) + Full Attention anchors (3, 7, 11, 15, 19, 23, 27, 31)
            # Rank 2: High intermediate layers
            # Rank 3: Standard linear attention
            if i == 0 or is_full_attn:
                priority = 1
            elif i in (1, 2, 4, 8, 16):
                priority = 2
            else:
                priority = 3

            # Inspect subtensor keys via header
            with safetensors.safe_open(layer_file, framework="pt", device="cpu") as f:
                subtensors = list(f.keys())

            self.layers.append(
                LayerMetadata(
                    layer_idx=i,
                    layer_type=layer_type,
                    priority_score=priority,
                    file_path=layer_file,
                    file_bytes=file_bytes,
                    subtensor_names=subtensors
                )
            )

    def get_ranked_queue(self) -> List[LayerMetadata]:
        """Возвращает слои для конвейерного прохода с учётом приоритетов."""
        # Layers execute sequentially 0..31, but priority metadata governs residency
        return sorted(self.layers, key=lambda l: l.layer_idx)

    def get_priority_summary(self) -> Dict[str, Any]:
        p1 = [l.layer_idx for l in self.layers if l.priority_score == 1]
        p2 = [l.layer_idx for l in self.layers if l.priority_score == 2]
        p3 = [l.layer_idx for l in self.layers if l.priority_score == 3]
        return {"rank_1_critical": p1, "rank_2_medium": p2, "rank_3_standard": p3}


class SelectiveSubtensorLoader:
    """Выборочно извлекает из safetensors ТОЛЬКО запрошенные субтензоры слоя.

    Не десериализует весь монолитный 416 МБ файл!
    """

    def __init__(self):
        self.total_bytes_extracted: int = 0

    def load_subtensors(
        self,
        safetensors_path: str,
        target_keys: List[str]
    ) -> Dict[str, torch.Tensor]:
        t0 = time.perf_counter()
        extracted = {}
        with safetensors.safe_open(safetensors_path, framework="pt", device="cpu") as f:
            for k in target_keys:
                if k in f.keys():
                    t = f.get_slice(k)[:]
                    extracted[k] = t
                    self.total_bytes_extracted += t.nbytes
        return extracted


class MemoryStandPool:
    """Постоянная «подставка» (Persistent Stand) в VRAM и pinned RAM.

    Память выделяется один раз при старте.
    Выбранные субтензоры замещают отработавшие прямо на подставке (zero malloc).
    """

    def __init__(
        self,
        template_tensors: Dict[str, torch.Tensor],
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        num_stands: int = 2
    ):
        self.device = device
        self.num_stands = num_stands

        # Выделение постоянных VRAM-подставок
        self.gpu_stands = []
        self.pinned_ram_stands = []

        for i in range(num_stands):
            gpu_stand = {}
            ram_stand = {}
            for k, v in template_tensors.items():
                short_k = k.split(".", 3)[-1]
                gpu_stand[short_k] = v.to(device=device, dtype=torch.float16)
                if device.type == "cuda":
                    ram_stand[short_k] = torch.empty_like(v, pin_memory=True)
                else:
                    ram_stand[short_k] = torch.empty_like(v)
            self.gpu_stands.append(gpu_stand)
            self.pinned_ram_stands.append(ram_stand)

        self.active_stand_idx: int = 0
        self.stand_nbytes = sum(t.nbytes for t in self.gpu_stands[0].values())

        # LFRU кэш для горячих субтензоров
        self.lfru = ColibriLFRUCache(num_slots=num_stands, num_experts=32)

    def place_on_stand(
        self,
        subtensors: Dict[str, torch.Tensor],
        stand_idx: int,
        stream: Optional[torch.cuda.Stream] = None
    ) -> None:
        """Помещает выбранные субтензоры прямо на подставку (in-place copy)."""
        gpu_stand = self.gpu_stands[stand_idx]
        ram_stand = self.pinned_ram_stands[stand_idx]

        for k, v in subtensors.items():
            short_k = k.split(".", 3)[-1]
            if short_k in ram_stand:
                ram_stand[short_k].copy_(v)
                if stream is not None and self.device.type == "cuda":
                    with torch.cuda.stream(stream):
                        gpu_stand[short_k].copy_(ram_stand[short_k], non_blocking=True)
                else:
                    gpu_stand[short_k].copy_(ram_stand[short_k], non_blocking=False)


class AirColibriEngine:
    """Единый исполнительный движок AirColibri."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[AirColibri] Initializing on {self.device.type.upper()}: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

        # 1. Ранжирование слоев
        self.ranker = PriorityRanker(model_dir)
        self.ranked_layers = self.ranker.get_ranked_queue()
        summary = self.ranker.get_priority_summary()
        print(f"[AirColibri] Priority Ranking: {len(summary['rank_1_critical'])} Critical anchors, {len(summary['rank_2_medium'])} High, {len(summary['rank_3_standard'])} Standard")

        # 2. Селективный загрузчик
        self.selective_loader = SelectiveSubtensorLoader()

        # 3. Выделение постоянной «подставки» под слой
        template_file = self.ranked_layers[0].file_path
        template_tensors = safetensors.torch.load_file(template_file, device="cpu")
        self.stand_pool = MemoryStandPool(template_tensors, device=self.device, num_stands=2)
        stand_mb = self.stand_pool.stand_nbytes / (1024 * 1024)
        print(f"[AirColibri] Allocated 2 Persistent Stands: {stand_mb * 2:.1f} MB VRAM (Zero Malloc Runtime)")

    def execute_token_step(
        self,
        x: torch.Tensor,
        selective_mlp_only: bool = False
    ) -> Tuple[torch.Tensor, float, float]:
        """Выполняет 1 шаг генерации токена через конвейер AirColibri."""
        t0 = time.perf_counter()
        bytes_read_start = self.selective_loader.total_bytes_extracted

        # Конвейерный проход по ранжированным слоям
        for layer_meta in self.ranked_layers:
            # Выборочное выдергивание весов:
            # Если активен selective_mlp_only, читаем только MLP (gate/up/down)
            # В стандартном режиме читаем активные проекции слоя
            keys_to_fetch = [
                k for k in layer_meta.subtensor_names
                if "mlp." in k or "norm" in k or ("linear_attn.out_proj" in k or "self_attn.o_proj" in k)
            ]

            # 3. Выборка на лету из safetensors на SSD
            subtensors = self.selective_loader.load_subtensors(layer_meta.file_path, keys_to_fetch)

            # 4. Замещение на постоянной подставке (slot 0)
            self.stand_pool.place_on_stand(subtensors, stand_idx=0)

            # 5. Вычисление GEMM на GPU
            stand = self.stand_pool.gpu_stands[0]
            normed = torch.nn.functional.rms_norm(x, (4096,), stand["input_layernorm.weight"])
            gate = torch.matmul(normed, stand["mlp.gate_proj.weight"].T)
            up = torch.matmul(normed, stand["mlp.up_proj.weight"].T)
            act = torch.nn.functional.silu(gate) * up
            mlp_out = torch.matmul(act, stand["mlp.down_proj.weight"].T)
            x = x + mlp_out

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        step_duration = time.perf_counter() - t0
        bytes_extracted = self.selective_loader.total_bytes_extracted - bytes_read_start
        return x, step_duration, bytes_extracted / (1024 * 1024)
