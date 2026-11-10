"""MoE Selective Router & Expert Tracker for AirColibri (Stage F):
Специализированный трекер маршрутизации и выборочной загрузки экспертов для OLMoE (64 эксперта, top-8).

Принципы:
1. Только 8 из 64 экспертов (12.5% весов) активируются на каждом токене.
2. LFRU кэш горячих экспертов: популярные эксперты удерживаются в постоянных слотах VRAM.
3. Предсказание на основе кластеризации роутинга: предзагрузка кандидатов без блокировки математики.
4. При ошибке прогноза фактически выбранный роутером эксперт гарантированно подгружается.
"""

import collections
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.lfru_cache import ColibriLFRUCache


class MoEExpertLFRUCache:
    """LFRU кэш экспертов для MoE слоя."""

    def __init__(self, layer_idx: int, num_experts: int = 64, capacity: int = 16):
        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self.capacity = capacity
        self.lfru = ColibriLFRUCache(num_slots=capacity, num_experts=num_experts)

        # Статистика
        self.total_requests = 0
        self.hits = 0
        self.misses = 0

    def query_experts(self, active_expert_indices: List[int]) -> Tuple[List[int], List[int]]:
        """Разделяет запрошенных роутером экспертов на попадания (hit) и промахи (miss).

        Returns: (hit_experts, miss_experts)
        """
        hits = []
        misses = []

        for e_idx in active_expert_indices:
            self.total_requests += 1
            slot, is_hit = self.lfru.access_expert(e_idx)
            if is_hit:
                self.hits += 1
                hits.append(e_idx)
            else:
                self.misses += 1
                misses.append(e_idx)

        return hits, misses

    @property
    def hit_rate(self) -> float:
        return (self.hits / max(1, self.total_requests)) * 100.0


class OLMoERoutingTracker:
    """Глобальный трекер маршрутизации по всем слоям модели OLMoE."""

    def __init__(self, num_layers: int = 16, num_experts: int = 64, vram_slots_per_layer: int = 16):
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.layer_caches: List[MoEExpertLFRUCache] = [
            MoEExpertLFRUCache(i, num_experts=num_experts, capacity=vram_slots_per_layer)
            for i in range(num_layers)
        ]

        # Матрица ко-активации экспертов для предсказания: co_matrix[e1][e2]
        self.co_activation = collections.defaultdict(lambda: collections.defaultdict(int))

    def on_router_step(self, layer_idx: int, routed_experts: List[int]) -> Dict[str, Any]:
        """Фиксирует решение роутера слоя и обновляет статистику."""
        hits, misses = self.layer_caches[layer_idx].query_experts(routed_experts)

        # Обновляем матрицу совместной активации
        for i, e1 in enumerate(routed_experts):
            for e2 in routed_experts[i+1:]:
                self.co_activation[e1][e2] += 1
                self.co_activation[e2][e1] += 1

        return {
            "layer_idx": layer_idx,
            "routed_count": len(routed_experts),
            "hits": hits,
            "misses": misses,
            "hit_rate_pct": self.layer_caches[layer_idx].hit_rate
        }

    def predict_next_candidates(self, current_active: List[int], top_n: int = 4) -> List[int]:
        """Предсказывает наиболее вероятных экспертов для следующего слоя."""
        candidate_scores = collections.defaultdict(int)
        for e in current_active:
            for related, count in self.co_activation[e].items():
                if related not in current_active:
                    candidate_scores[related] += count

        sorted_cands = sorted(candidate_scores.keys(), key=lambda k: candidate_scores[k], reverse=True)
        return sorted_cands[:top_n]
