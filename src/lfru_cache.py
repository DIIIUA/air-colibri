"""Colibri LFRU (Least Frequently / Recently Used) Expert Cache Algorithm.

Direct port of tier.h admission logic from Colibri:
- tier_should_promote(hot, cold): 25% margin + 4 threshold prevents ping-pong thrashing.
- tier_lfru_score(heat, last, clock): frequency is primary signal (8-bit shifted),
  recency (0-255) breaks ties without displacing genuinely hotter experts.
- tier_decay_value(heat): exponential decay (heat >> 1).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class ExpertSlot:
    slot_id: int
    expert_id: Optional[int] = None
    is_dirty: bool = False


class ColibriLFRUCache:
    """Manages expert slot allocation in VRAM using Colibri's tier.h LFRU strategy."""

    def __init__(self, num_slots: int, num_experts: int, decay_interval: int = 100):
        if num_slots < 0 or num_experts < 1 or decay_interval < 1:
            raise ValueError("Invalid expert cache dimensions")
        self.num_slots = num_slots
        self.num_experts = num_experts
        self.decay_interval = decay_interval
        self.clock: int = 0

        # Expert tracking
        self.heat = [0] * num_experts
        self.last_access = [0] * num_experts

        # Slot state
        self.slots: List[ExpertSlot] = [ExpertSlot(slot_id=i) for i in range(num_slots)]
        # Map expert_id -> slot_id (if currently resident)
        self.resident_map: Dict[int, int] = {}

    def should_promote(self, hot: int, cold: int) -> bool:
        """25% margin + 4 prevents ping-pong thrashing between closely routed experts."""
        threshold = cold + (cold >> 2) + 4
        return hot > threshold

    def decay_heat(self) -> None:
        """Decay all expert heat counters (heat >> 1)."""
        for i in range(self.num_experts):
            self.heat[i] >>= 1

    def compute_lfru_score(self, expert_id: int) -> int:
        """Score: (frequency << 8) | recent (0..255)."""
        heat = self.heat[expert_id]
        last = self.last_access[expert_id]
        age = (self.clock - last) & 0xFFFFFFFF
        recent = 255 - age if age < 255 else 0
        return (heat << 8) | recent

    def access_expert(self, expert_id: int) -> Tuple[Optional[int], bool]:
        """Accesses an expert.

        Returns:
            (slot_id, is_cache_hit); slot_id=None means execute transiently
            without evicting a resident. Execution and admission are separate.
        """
        if not 0 <= expert_id < self.num_experts:
            raise ValueError("Expert index out of range")
        self.clock = (self.clock + 1) & 0xFFFFFFFF
        if self.clock % self.decay_interval == 0:
            self.decay_heat()

        self.heat[expert_id] = min(0xFFFFFFFF, self.heat[expert_id] + 1)
        self.last_access[expert_id] = self.clock
        if self.num_slots == 0:
            return None, False

        # Cache HIT: expert is already resident in a slot
        if expert_id in self.resident_map:
            return self.resident_map[expert_id], True

        # Cache MISS: check for an empty slot first
        for slot in self.slots:
            if slot.expert_id is None:
                slot.expert_id = expert_id
                self.resident_map[expert_id] = slot.slot_id
                return slot.slot_id, False

        # All slots full: find coldest resident slot by LFRU score
        cold_slot_idx = 0
        min_resident_score = float("inf")
        for i, slot in enumerate(self.slots):
            score = self.compute_lfru_score(slot.expert_id)
            if score < min_resident_score:
                min_resident_score = score
                cold_slot_idx = i

        coldest_resident_expert = self.slots[cold_slot_idx].expert_id
        hot_score = self.compute_lfru_score(expert_id)
        cold_score = self.compute_lfru_score(coldest_resident_expert)
        if hot_score <= cold_score + (cold_score >> 2) + (4 << 8):
            return None, False
        old_expert = self.slots[cold_slot_idx].expert_id
        if old_expert in self.resident_map:
            del self.resident_map[old_expert]

        self.slots[cold_slot_idx].expert_id = expert_id
        self.resident_map[expert_id] = cold_slot_idx
        return cold_slot_idx, False

    def is_resident(self, expert_id: int) -> bool:
        return expert_id in self.resident_map

    def get_slot(self, expert_id: int) -> Optional[int]:
        return self.resident_map.get(expert_id, None)
