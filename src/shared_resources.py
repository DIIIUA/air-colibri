"""Bounded Dense/expert ownership on the existing Ring, pinned pool and ACKs.

Two simultaneous consumers are supported: a decoder's Dense trunk and its
current expert. This is deterministic streaming, not a dynamic VRAM cache.
"""
from collections import Counter, deque
from dataclasses import dataclass

import torch
from src.bounded_memory_manager import BoundedPinnedBufferPool, SlotArenaLayout
from src.gpu_ring_pipeline import GPURingPipeline
from src.lease_ack_bridge import ExistingRingAcknowledgementAdapter


@dataclass(frozen=True)
class ResourceLease:
    slot_id: int
    ticket: int
    key: tuple


class SharedWeightResources:
    def __init__(self, templates, device, dtype, num_slots=2):
        if not isinstance(num_slots, int) or not 1 <= num_slots <= 4:
            raise ValueError("Resource slot count must be between 1 and 4")
        self.active = {}
        self.sequence = 0
        self.closed = False
        self.failed = False
        self.ring = self.pool = self.ack = None
        self.counts = Counter()
        self.logical_h2d_bytes = Counter()
        self.events = deque(maxlen=4096)
        self.peak_active_slots = 0
        try:
            self.layout = SlotArenaLayout.from_templates(templates, dtype=dtype)
            self.pool = BoundedPinnedBufferPool(num_slots=num_slots, dtype=dtype, arena_layout=self.layout)
            self.ring = GPURingPipeline(num_slots=num_slots, device=torch.device(device), dtype=dtype, arena_layout=self.layout)
            self.ack = ExistingRingAcknowledgementAdapter(self.ring).install()
        except BaseException:
            self.close()
            raise

    def acquire(self, slot_id, key, payload):
        if self.closed or self.failed:
            raise RuntimeError("Resource backend is closed or failed")
        if not isinstance(slot_id, int) or not 0 <= slot_id < self.ring.num_slots:
            raise ValueError("Invalid resource slot")
        if not isinstance(key, tuple) or not key or not isinstance(key[0], str):
            raise ValueError("Resource identity must be a nonempty tuple with a kind")
        if slot_id in self.active:
            raise RuntimeError("Cannot overwrite a live resource lease")
        # Pool validates the whole payload, then waits for its previous DMA
        # before CPU staging. Ring reuse independently waits for GPU consumers.
        pinned = self.pool.copy_into_slot(slot_id, payload)
        ticket = self.sequence * self.ring.num_slots + slot_id
        self.sequence += 1
        try:
            slot = self.ring.get_slot_for_loading(ticket)
            self.ring.load_tensors_async(slot, pinned, pinned_pool=self.pool)
            slot = self.ring.acquire_slot_for_compute(ticket)
        except BaseException:
            self.failed = True
            raise
        lease = ResourceLease(slot_id, ticket, key)
        self.active[slot_id] = lease
        self.peak_active_slots = max(self.peak_active_slots, len(self.active))
        self.counts[key[0]] += 1
        self.logical_h2d_bytes[key[0]] += sum(t.nbytes for t in payload.values())
        self.events.append(dict(action="acquire", ticket=ticket, slot=slot_id, resource=key))
        return lease, {name:slot.tensors[name] for name in payload}

    def release(self, lease):
        if not isinstance(lease, ResourceLease) or self.closed or self.active.get(lease.slot_id) is not lease:
            raise RuntimeError("Foreign, stale or already released resource lease")
        self.ring.release_slot_after_compute(self.ring.slots[lease.slot_id])
        del self.active[lease.slot_id]
        self.events.append(dict(action="release_enqueued", ticket=lease.ticket, slot=lease.slot_id, resource=lease.key))

    def snapshot(self):
        ack = self.ack.snapshot() if self.ack else {}
        return dict(kind="shared-dense-expert-ring-v1", counts=dict(self.counts),
            logical_h2d_bytes=dict(self.logical_h2d_bytes), active_slots=len(self.active),
            peak_active_slots=self.peak_active_slots, slot_stride=self.layout.slot_stride,
            gpu_arena_bytes=self.ring.num_slots*self.layout.slot_stride,
            pinned_arena_bytes=self.pool.total_pinned_bytes, loads=self.ring.total_loads,
            computes=self.ring.total_computes, lease_event_counts=ack.get("event_counts", {}),
            violations=ack.get("violations_rejected", 0), leases=ack.get("leases", []),
            events=list(self.events), dynamic_vram_cache=False)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.ring:
            # Failure can leave a partially submitted lease: drain before
            # destroying storage; do not pretend it completed normally.
            self.ring.close()
        if self.ack:
            self.ack.close()
        if self.pool:
            self.pool.close()
        self.active.clear()
