"""Lease / completion acknowledgement for the EXISTING AirColibri GPURingPipeline.

Intercepts its public methods in place without replacing their implementations.
Acknowledges GPU readiness only after CUDA Event.query() confirms the event.
A recorded release is NOT a completed GPU computation.
No loader, SSD reader, quantizer or new compute kernel is implemented.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from collections import Counter, deque
import threading
import time
from typing import Any, Callable

import torch
from src.gpu_ring_pipeline import GPURingPipeline, RingSlot, SlotState


class BackendLeaseViolation(RuntimeError):
    pass


@dataclass
class SlotLease:
    slot_id: int
    generation: int = 0
    layer_idx: int | None = None
    phase: str = "FREE"
    dma_event_issued: bool = False
    dma_done_confirmed: bool = False
    compute_wait_issued: bool = False
    compute_event_issued: bool = False
    compute_done_confirmed: bool = False


class ExistingRingAcknowledgementAdapter:
    """Validates the original GPURingPipeline's real slot and CUDA events.

    Method replacements delegate to captured ORIGINAL methods with no altered
    data path. A new load may be queued after a released compute event, even
    if unfinished, because the ORIGINAL transfer stream waits on that event.
    """
    METHOD_NAMES = ("get_slot_for_loading", "load_tensors_async",
                    "acquire_slot_for_compute", "release_slot_after_compute")

    def __init__(self, backend: GPURingPipeline,
                 on_event: Callable[[dict], None] | None = None):
        if not isinstance(backend, GPURingPipeline):
            raise TypeError("Only the existing GPURingPipeline may be wrapped")
        self.backend = backend
        self.on_event = on_event
        self._lock = threading.RLock()
        self.originals: dict[str, Any] = {}
        self._attached = False
        self.leases = [SlotLease(i) for i in range(backend.num_slots)]
        self.events = deque(maxlen=4096)
        self.event_counts = Counter()
        self.violations = 0

    def _write_event(self, kind: str, lease: SlotLease, **extra):
        event = {
            "kind": kind, "slot_id": lease.slot_id,
            "generation": lease.generation,
            "layer": lease.layer_idx,
            "phase": lease.phase,
            "host_ns": time.perf_counter_ns(),
            **extra
        }
        self.events.append(event)
        self.event_counts[kind] += 1
        if self.on_event:
            self.on_event(dict(event))

    def _reject(self, reason: str):
        self.violations += 1
        raise BackendLeaseViolation(reason)

    def _resolve(self, layer: int):
        if not isinstance(layer, int) or layer < 0:
            self._reject("Invalid layer index")
        sid = layer % self.backend.num_slots
        return self.leases[sid], self.backend.slots[sid]

    def _verify_slot(self, slot):
        if not isinstance(slot, RingSlot):
            self._reject("GPU slot is not a RingSlot instance")
        sid = slot.slot_id
        if sid < 0 or sid >= self.backend.num_slots:
            self._reject("GPU slot ID out of bounds")
        if self.backend.slots[sid] is not slot:
            self._reject("Foreign GPU slot")
        return self.leases[sid]

    def poll(self, slot_id: int):
        """Nonblocking physical event acknowledgements, never metadata guesses."""
        with self._lock:
            if not 0 <= slot_id < self.backend.num_slots:
                self._reject("Invalid slot id")
            lease = self.leases[slot_id]
            slot = self.backend.slots[slot_id]
            if self.backend.device.type != "cuda":
                return asdict(lease)
            if lease.dma_event_issued and not lease.dma_done_confirmed:
                if slot.transfer_done_event is not None and slot.transfer_done_event.query():
                    lease.dma_done_confirmed = True
                    self._write_event("dma_completed_on_gpu", lease)
            if lease.compute_event_issued and not lease.compute_done_confirmed:
                if slot.compute_done_event is not None and slot.compute_done_event.query():
                    lease.compute_done_confirmed = True
                    lease.phase = "FREE"
                    self._write_event("compute_completed_on_gpu", lease)
            return asdict(lease)

    def wait_for_free(self, slot_id: int):
        """Explicit sync for callers that truly need a completed, reusable slot."""
        with self._lock:
            self.poll(slot_id)
            lease = self.leases[slot_id]
            if lease.phase == "FREE":
                return
            if lease.phase != "RELEASE_QUEUED":
                self._reject("Cannot wait/free while lease has active consumer")
            slot = self.backend.slots[slot_id]
            if self.backend.device.type == "cuda":
                if not lease.compute_event_issued or slot.compute_done_event is None:
                    self._reject("Cannot safely recycle without recorded compute event")
                slot.compute_done_event.synchronize()
            lease.compute_done_confirmed = True
            lease.phase = "FREE"
            self._write_event("compute_completed_on_gpu", lease, synchronized=True)

    def install(self):
        with self._lock:
            if self._attached:
                raise RuntimeError("Already installed")
            for name in self.METHOD_NAMES:
                self.originals[name] = getattr(self.backend, name)
            self.backend.get_slot_for_loading = self.get_slot_for_loading
            self.backend.load_tensors_async = self.load_tensors_async
            self.backend.acquire_slot_for_compute = self.acquire_slot_for_compute
            self.backend.release_slot_after_compute = self.release_slot_after_compute
            self._attached = True
            return self

    def close(self):
        with self._lock:
            for name, original in self.originals.items():
                setattr(self.backend, name, original)
            self.originals.clear()
            self._attached = False

    def __enter__(self):
        return self.install()

    def __exit__(self, typ, value, tb):
        self.close()
        return False

    def get_slot_for_loading(self, layer_idx: int):
        with self._lock:
            lease, slot = self._resolve(layer_idx)
            self.poll(lease.slot_id)
            if lease.phase not in ("FREE", "RELEASE_QUEUED"):
                self._reject(
                    f"Attempt to overwrite active lease phase={lease.phase}")
            if lease.phase == "RELEASE_QUEUED":
                # GPU compute may still be running. Original GPU transfer
                # stream MUST wait on a real, already-recorded compute event.
                if self.backend.device.type == "cuda":
                    if not lease.compute_event_issued or slot.compute_done_event is None:
                        self._reject("Release fence absent; unsafe slot reuse")
                self._write_event("reuse_scheduled_after_compute_fence", lease,
                                  physical_done=lease.compute_done_confirmed)
            original = self.originals["get_slot_for_loading"]
            result = original(layer_idx)
            if result is not slot or slot.current_layer_idx != layer_idx:
                self._reject("Original pipeline returned an unexpected slot")
            lease.generation += 1
            lease.layer_idx = layer_idx
            lease.phase = "LOADING_RESERVED"
            lease.dma_event_issued = False
            lease.dma_done_confirmed = False
            lease.compute_wait_issued = False
            lease.compute_event_issued = False
            lease.compute_done_confirmed = False
            self._write_event("slot_reserved", lease)
            return result

    def load_tensors_async(self, slot: RingSlot, pinned_src_tensors: dict,
                           pinned_pool=None):
        with self._lock:
            lease = self._verify_slot(slot)
            if lease.phase != "LOADING_RESERVED":
                self._reject("DMA requires exactly one prior slot reservation")
            if slot.current_layer_idx != lease.layer_idx:
                self._reject("Stale layer ID in GPU slot")
            if not isinstance(pinned_src_tensors, dict) or not pinned_src_tensors:
                self._reject("Empty or invalid transfer payload")
            if (any(k.startswith("linear_attn.") for k in pinned_src_tensors)
                    and any(k.startswith("self_attn.") for k in pinned_src_tensors)):
                self._reject("Overlapping attention variants cannot coexist")
            for name, source in pinned_src_tensors.items():
                if name not in slot.tensors:
                    self._reject("Payload key absent from slot: "+name)
                target = slot.tensors[name]
                if not isinstance(source, torch.Tensor) or source.device.type != "cpu":
                    self._reject("DMA source must be host tensor")
                if source.shape != target.shape or source.dtype != target.dtype:
                    self._reject("DMA shape/dtype mismatch for "+name)
            if pinned_pool is not None:
                if slot.slot_id >= pinned_pool.num_slots:
                    self._reject("Pinned-pool slot index mismatch")
            original = self.originals["load_tensors_async"]
            original(slot, pinned_src_tensors, pinned_pool=pinned_pool)
            if self.backend.device.type == "cuda":
                if slot.transfer_done_event is None:
                    self._reject("Original loader did not provide a CUDA transfer event")
                lease.dma_event_issued = True
                lease.phase = "DMA_QUEUED"
                self._write_event("dma_event_recorded", lease,
                                  physically_finished=slot.transfer_done_event.query())
            else:
                lease.phase = "DMA_QUEUED"
                lease.dma_done_confirmed = True
                self._write_event("cpu_copy_complete", lease)

    def acquire_slot_for_compute(self, layer_idx: int):
        with self._lock:
            lease, slot = self._resolve(layer_idx)
            if lease.phase != "DMA_QUEUED" or lease.layer_idx != layer_idx:
                self._reject("Compute acquire without matching completed DMA submission")
            if self.backend.device.type == "cuda" and not lease.dma_event_issued:
                self._reject("Missing transfer event for GPU compute")
            original = self.originals["acquire_slot_for_compute"]
            actual = original(layer_idx)
            if actual is not slot:
                self._reject("Original pipeline returned foreign compute slot")
            # The ORIGINAL waits on transfer_done_event in compute_stream.
            # This proves only that dependency was ENQUEUED, not completed.
            lease.compute_wait_issued = True
            lease.phase = "COMPUTING"
            self._write_event("compute_dependency_enqueued", lease,
                              dma_physically_done=(
                                  slot.transfer_done_event.query()
                                  if slot.transfer_done_event is not None else True))
            return actual

    def release_slot_after_compute(self, slot: RingSlot):
        with self._lock:
            lease = self._verify_slot(slot)
            if lease.phase != "COMPUTING":
                self._reject("Release without the active compute lease")
            if slot.current_layer_idx != lease.layer_idx:
                self._reject("Release of a mismatched layer")
            original = self.originals["release_slot_after_compute"]
            original(slot)
            if self.backend.device.type == "cuda":
                if slot.compute_done_event is None:
                    self._reject("No compute event available after release")
                lease.compute_event_issued = True
                lease.phase = "RELEASE_QUEUED"
                self._write_event("compute_event_recorded", lease,
                                  physically_finished=slot.compute_done_event.query())
            else:
                lease.compute_done_confirmed = True
                lease.phase = "FREE"
                self._write_event("cpu_compute_released", lease)

    def snapshot(self):
        with self._lock:
            for i in range(len(self.leases)):
                self.poll(i)
            return {"status": "PASS",
                    "original_pipeline_methods_restorable": self._attached,
                    "num_slots": len(self.leases),
                    "leases": [asdict(lease) for lease in self.leases],
                    "events": list(self.events),
                    "event_counts": dict(self.event_counts),
                    "violations_rejected": self.violations,
                    "uses_original_GPU_pipeline": True,
                    "new_loader_or_CUDA_kernel": False,
                    "native_Colibri_C_lease_not_implemented": True}
