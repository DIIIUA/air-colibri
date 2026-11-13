"""Bounded Dense/expert ownership on the existing Ring, pinned pool and ACKs.

Two simultaneous consumers are supported: a decoder's Dense trunk and its
current expert. Optional bounded expert residency reuses the same Ring/ACKs.
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
        self.contents = {}
        self.residency = None
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
        # A reused pinned source has now finished its earlier DMA. Confirm
        # any previous cache admission before its event can be overwritten.
        if self.residency:
            self.residency.confirm()
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
        self.contents[slot_id] = (key, ticket, tuple(payload), sum(t.nbytes for t in payload.values()))
        self.peak_active_slots = max(self.peak_active_slots, len(self.active))
        self.counts[key[0]] += 1
        self.logical_h2d_bytes[key[0]] += sum(t.nbytes for t in payload.values())
        self.events.append(dict(action="acquire", ticket=ticket, slot=slot_id, resource=key))
        return lease, {name:slot.tensors[name] for name in payload}

    def reuse(self, slot_id, key):
        if self.closed or self.failed:
            raise RuntimeError("Resource backend is closed or failed")
        if slot_id in self.active:
            raise RuntimeError("Cannot reacquire a live resource lease")
        content = self.contents.get(slot_id)
        if content is None or content[0] != key:
            raise RuntimeError("Resident resource identity mismatch")
        _, previous, names, size = content
        ticket = self.sequence*self.ring.num_slots+slot_id
        self.sequence += 1
        slot = self.ring.acquire_resident_for_compute(ticket, previous)
        lease = ResourceLease(slot_id,ticket,key)
        self.active[slot_id] = lease
        self.contents[slot_id] = (key,ticket,names,size)
        self.counts[key[0]] += 1
        self.peak_active_slots = max(self.peak_active_slots,len(self.active))
        self.events.append(dict(action="resident_acquire",ticket=ticket,slot=slot_id,resource=key,new_dma=False))
        return lease,{name:slot.tensors[name] for name in names}

    def enable_expert_residency(self, policy, mode="apply"):
        if self.residency is not None or self.active or self.contents:
            raise RuntimeError("Configure residency once before first use")
        self.residency = ExpertResidency(self,policy,mode)

    def release(self, lease):
        if not isinstance(lease, ResourceLease) or self.closed or self.active.get(lease.slot_id) is not lease:
            raise RuntimeError("Foreign, stale or already released resource lease")
        self.ring.release_slot_after_compute(self.ring.slots[lease.slot_id])
        del self.active[lease.slot_id]
        self.events.append(dict(action="release_enqueued", ticket=lease.ticket, slot=lease.slot_id, resource=lease.key))

    def snapshot(self):
        ack = self.ack.snapshot() if self.ack else {}
        return dict(kind="shared-dense-expert-ring-v2", counts=dict(self.counts),
            logical_h2d_bytes=dict(self.logical_h2d_bytes), active_slots=len(self.active),
            peak_active_slots=self.peak_active_slots, slot_stride=self.layout.slot_stride,
            gpu_arena_bytes=self.ring.num_slots*self.layout.slot_stride,
            pinned_arena_bytes=self.pool.total_pinned_bytes, loads=self.ring.total_loads,
            computes=self.ring.total_computes, lease_event_counts=ack.get("event_counts", {}),
            violations=ack.get("violations_rejected", 0), leases=ack.get("leases", []),
            events=list(self.events), resident_acquires=self.ring.total_resident_acquires,
            expert_residency=self.residency.snapshot() if self.residency else None,
            dynamic_vram_cache=self.residency is not None and self.residency.mode=="apply")

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
        self.contents.clear()
        if self.residency:
            self.residency.clear()


class ExpertResidency:
    """Native admission proposals + physical Ring leases, with distinct ACKs.

    Shadow tracks a virtual resident set but always uses the streaming slot.
    Apply invokes the payload supplier only on misses. One host forward owns
    this object; concurrent host mutation is not a supported scheduling mode.
    """
    def __init__(self, resources, policy, mode):
        slots = tuple(range(2,resources.ring.num_slots))
        if mode not in ("shadow","apply") or not slots or policy.capacity != len(slots):
            raise ValueError("Expert residency needs matching reserved slots and shadow/apply mode")
        if policy.mode != "apply":
            raise ValueError("Policy must return real proposals; backend controls shadow/apply")
        self.resources, self.policy, self.mode = resources, policy, mode
        self.slots = slots
        self.entries = {}
        self.stats = Counter()
        self.events = deque(maxlen=4096)

    def _ack(self, status, expert, **extra):
        event = dict(tier="vram",status=status,expert=expert,mode=self.mode,**extra)
        self.events.append(event)
        self.policy.acknowledge(event)

    def confirm(self):
        if self.mode != "apply":
            return
        for expert, entry in self.entries.items():
            if entry["confirmed"]:
                continue
            sid = entry["slot"]
            if self.resources.contents[sid][0] != entry["key"]:
                raise RuntimeError("Unconfirmed cache content was replaced outside its owner")
            if self.resources.ack.poll(sid)["dma_done_confirmed"]:
                entry["confirmed"] = True
                self.stats["admissions_confirmed"] += 1
                self._ack("applied",expert,slot=sid,submission_ticket=entry["ticket"],dma_physically_complete=True)

    def acquire(self, expert, key, payload_supplier):
        if self.resources.closed or self.resources.failed:
            raise RuntimeError("Resource backend is closed or failed")
        self.confirm()
        self.policy.record_access(expert)
        self.stats["requests"] += 1
        entry = self.entries.get(expert)
        if entry is not None:
            if entry["key"] != key:
                raise ValueError("Expert identity alias")
            if self.mode == "apply":
                lease,views = self.resources.reuse(entry["slot"],key)
                self.stats["hits"] += 1
                self.stats["avoided_h2d_bytes"] += self.resources.contents[entry["slot"]][3]
                return lease,views,0
            self.stats["shadow_hits"] += 1
        else:
            accepted,victim = self.policy.admit(expert,list(self.entries))
            if victim is not None and (victim not in self.entries or victim==expert):
                raise ValueError("Invalid GPU admission victim")
            sid = None
            if accepted:
                occupied = {e["slot"] for e in self.entries.values()}
                free = [s for s in self.slots if s not in occupied]
                if victim is not None:
                    sid = self.entries[victim]["slot"]
                elif free:
                    sid = free[0]
                else:
                    raise ValueError("GPU admission omitted required eviction")
                if sid in self.resources.active:
                    raise RuntimeError("GPU admission targets an active consumer")
            if self.mode == "apply" and accepted:
                payload = payload_supplier()
                lease,views = self.resources.acquire(sid,key,payload)
                if victim is not None:
                    del self.entries[victim]
                    self.stats["evictions"] += 1
                self.entries[expert] = dict(slot=sid,key=key,ticket=lease.ticket,confirmed=False)
                self.stats["admissions_submitted"] += 1
                self._ack("submitted",expert,slot=sid,submission_ticket=lease.ticket,victim=victim,dma_physically_complete=False)
                self.confirm()
                return lease,views,sum(t.nbytes for t in payload.values())
            if self.mode == "shadow" and accepted:
                if victim is not None:
                    del self.entries[victim]
                    self.stats["shadow_evictions"] += 1
                self.entries[expert] = dict(slot=sid,key=key,confirmed=False)
                self.stats["shadow_admissions"] += 1
            self._ack("shadow" if self.mode=="shadow" else "rejected",expert,proposed_admission=bool(accepted),victim=victim)
        payload = payload_supplier()
        lease,views = self.resources.acquire(1,key,payload)
        size = sum(t.nbytes for t in payload.values())
        if entry is not None and self.mode == "shadow":
            self.stats["shadow_avoidable_h2d_bytes"] += size
        return lease,views,size

    def snapshot(self):
        self.confirm()
        return dict(mode=self.mode,capacity=len(self.slots),stats=dict(self.stats),
            resident_experts=list(self.entries) if self.mode=="apply" else [],
            virtual_experts=list(self.entries) if self.mode=="shadow" else [],
            reserved_gpu_bytes=len(self.slots)*self.resources.layout.slot_stride,
            reserved_pinned_bytes=len(self.slots)*self.resources.layout.slot_stride,
            native_policy=dict(self.policy.stats),events=list(self.events))

    def clear(self):
        # Called only after the backend has drained consumers and freed arenas.
        if self.entries:
            self._ack("released",None,experts=list(self.entries))
        self.entries.clear()
