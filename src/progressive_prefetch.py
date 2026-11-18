"""Bounded, progressively refined expert regions; no duration estimates or clock."""
from collections import OrderedDict
import threading

import torch

from src.quantized_projection import pack_weight, packed_specs


class ThreeBankExpertPrefetch:
    """Publish experts individually while the producer region is still changing.

    A region identifies the next piece of work, not a completed token. Revisions
    keep its prepared intersection and replace only obsolete entries. A producer
    re-reads the newest revision between experts, including after CPU packing.
    Bank reuse follows actual DMA/reader fences; model execution never waits for
    the CPU producer. Three banks retain consecutive regions concurrently.
    """

    def __init__(self, hidden, intermediate, bits, capacity, load_cpu, close_cpu,
                 device, host_cache_bytes=64*1024**2, group_size=64):
        self.device = torch.device(device)
        self.capacity, self.bits = capacity, bits
        self.load_cpu, self.close_cpu = load_cpu, close_cpu
        self.group_size = group_size
        self.specs = (packed_specs("weight", (2*intermediate, hidden), bits, group_size),
                      packed_specs("weight", (hidden, intermediate), bits, group_size))
        self.expert_bytes = sum(torch.empty((), dtype=dt).element_size()*sh[0]*sh[1]
                                for spec in self.specs for sh, dt in spec.values())
        if capacity <= 0 or host_cache_bytes < self.expert_bytes:
            raise ValueError("Warmup budget must fit at least one expert")
        self.host_cache_bytes = int(host_cache_bytes)
        self.cv = threading.Condition()
        self.epoch = self.serial = self.cursor = 0
        self.requests = OrderedDict()
        self.settled = {}
        self.queue_mode = None
        self.active = self.closed = False
        self.fatal = None
        self.cpu_cache = OrderedDict()
        self.cpu_cache_size = 0
        self.origin = torch.cuda.Event(enable_timing=True)
        self.origin.record(torch.cuda.current_stream(self.device))
        self.stream = torch.cuda.Stream(device=self.device)
        self.stream.wait_event(self.origin)
        self.banks = []
        for i in range(3):
            def alloc(device, pinned=False):
                return [{k: torch.empty((capacity, *sh), dtype=dt, device=device, pin_memory=pinned)
                         for k, (sh, dt) in spec.items()} for spec in self.specs]
            self.banks.append(dict(id=i, host=alloc("cpu", True), gpu=alloc(self.device),
                                   keys={}, epoch=-1, region=None, ready=None, used=None))
        self.counters = dict(submitted=0, superseded=0, published=0, stale=0,
                             bank_busy=0, cpu_conversions=0, host_cache_hits=0,
                             copied_experts=0, ready_at_demand=0, late_at_demand=0,
                             unavailable_at_demand=0, errors=0, refined=0,
                             retained_on_refinement=0, obsolete_after_pack=0,
                             regions_completed=0, bank_reuses=0, trace_dropped=0, updates_dropped=0)
        self.rotations = [0, 0, 0]
        self.consumptions = [0, 0, 0]
        self.trace, self.updates = [], []
        self.step_done = {}
        self.thread = threading.Thread(target=self._run, name="aircolibri-probability-warmup", daemon=True)
        self.thread.start()

    def _check(self):
        if self.fatal is not None:
            raise RuntimeError("Probability warmup worker failed") from self.fatal

    def submit(self, layer, ids, *, region=None, depth=None, lane=None, step=None):
        with self.cv:
            self._check()
            if self.closed:
                raise RuntimeError("Warmup is closed")
            if lane is not None and lane not in (0, 1, 2):
                raise ValueError("Lane must identify one of the three banks")
            mode = "latest" if lane is None else "lanes"
            if self.queue_mode not in (None, mode):
                raise ValueError("Cannot mix a latest-only queue and independent bank lanes")
            self.queue_mode = mode
            ids = tuple(dict.fromkeys(int(e) for e in ids))
            if len(ids) > self.capacity or any(e < 0 for e in ids):
                raise ValueError("Prediction exceeds its expert budget")
            self.serial += 1
            region = self.serial if region is None else region
            old = self.requests.get(lane)
            if old is not None and old["serial"] > self.settled.get(lane, 0):
                self.counters["superseded"] += 1
            retained = 0
            if old is not None and (old["epoch"], old["region"]) == (self.epoch, region):
                self.counters["refined"] += 1
                retained = sum((layer, eid) in b["keys"] for b in self.banks
                               if (b["epoch"], b["region"]) == (self.epoch, region) for eid in ids)
                self.counters["retained_on_refinement"] += retained
            step = region if step is None else step
            self.requests[lane] = dict(epoch=self.epoch, serial=self.serial, region=region,
                                       layer=int(layer), ids=ids, depth=depth, lane=lane, step=step)
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(self.device))
            self.updates.append(dict(serial=self.serial, region=region, depth=depth, layer=int(layer),
                                     lane=lane, step=step,
                                     ids=list(ids), prepared_common=retained, event=event))
            self.counters["updates_dropped"] += max(0, len(self.updates)-256)
            self.updates = self.updates[-256:]
            self.counters["submitted"] += 1
            self.cv.notify_all()
            return self.serial

    def restrict_to_route(self, layer, allowed, step):
        """Cancel obsolete optional reads for this exact work scope, not future tokens."""
        with self.cv:
            for lane, request in list(self.requests.items()):
                if request['layer'] != layer or request.get('step') != step:
                    continue
                kept = tuple(e for e in request['ids'] if e in allowed)
                if kept == request['ids']:
                    continue
                self.counters['route_pruned_ids'] = self.counters.get('route_pruned_ids', 0)+len(request['ids'])-len(kept)
                self.submit(layer, kept, region=request['region'], depth=request.get('depth'), lane=lane, step=step)

    def finish_step(self, region):
        event = torch.cuda.Event(enable_timing=True)
        event.record(torch.cuda.current_stream(self.device))
        with self.cv:
            self.step_done[region] = event
            for key in list(self.step_done)[:-128]:
                self.step_done.pop(key)

    def _packed_cpu(self, layer, eid):
        key = (layer, eid)
        if key in self.cpu_cache:
            self.cpu_cache.move_to_end(key)
            self.counters["host_cache_hits"] += 1
            return self.cpu_cache[key]
        g, u, d = self.load_cpu(layer, eid)
        packed = (pack_weight(torch.cat((g, u)), self.bits, self.group_size),
                  pack_weight(d, self.bits, self.group_size))
        while self.cpu_cache_size + self.expert_bytes > self.host_cache_bytes:
            self.cpu_cache.popitem(last=False)
            self.cpu_cache_size -= self.expert_bytes
        self.cpu_cache[key] = packed
        self.cpu_cache_size += self.expert_bytes
        self.counters["cpu_conversions"] += 1
        return packed

    @staticmethod
    def _fences(bank):
        return [e for e in (bank["ready"], bank["used"]) if e is not None and not e.query()]

    def _select_bank(self, request):
        choices = self.banks if request["lane"] is None else [self.banks[request["lane"]]]
        for bank in choices:
            if (bank["epoch"], bank["region"]) == (request["epoch"], request["region"]):
                return bank, []
        order = ([self.banks[(self.cursor+offset) % 3] for offset in range(3)]
                 if request["lane"] is None else choices)
        for bank in order:
            if not self._fences(bank):
                bank.update(keys={}, epoch=request["epoch"], region=request["region"], ready=None, used=None)
                self.cursor = (bank["id"]+1) % 3
                self.rotations[bank["id"]] += 1
                self.counters["bank_reuses"] += 1
                return bank, []
        # Only the producer waits for a real completion event. Keep the latest
        # request so a busy ring cannot silently discard a refinement.
        self.counters["bank_busy"] += 1
        return None, self._fences(order[0])

    def _pending(self):
        return [lane for lane, req in self.requests.items()
                if req["serial"] > self.settled.get(lane, 0)]

    def _run(self):
        with torch.cuda.device(self.device), torch.inference_mode():
            while True:
                with self.cv:
                    self.cv.wait_for(lambda: self.closed or bool(self._pending()))
                    if self.closed:
                        return
                    lane = self._pending()[0]
                    request = self.requests[lane]
                    # One expert per turn lets three independent areas advance
                    # without the latest update erasing the other two areas.
                    self.requests.move_to_end(lane)
                    bank, waits = self._select_bank(request)
                    if bank is not None:
                        desired = {(request["layer"], eid) for eid in request["ids"]}
                        missing = [(request["layer"], eid) for eid in request["ids"]
                                   if (request["layer"], eid) not in bank["keys"]]
                        if not missing:
                            self.settled[lane] = request["serial"]
                            self.counters["regions_completed"] += 1
                            self.cv.notify_all()
                            continue
                        key = missing[0]
                        used = {entry["pos"] for entry in bank["keys"].values()}
                        pos = next((i for i in range(self.capacity) if i not in used), None)
                        if pos is None:
                            victim = next(k for k in bank["keys"] if k not in desired)
                            pos = bank["keys"].pop(victim)["pos"]
                            # Remove this key before capturing its last reader;
                            # future consumers can only use the unaffected keys.
                            waits = self._fences(bank)
                    self.active = True
                    self.cv.notify_all()
                try:
                    for event in waits:
                        event.synchronize()
                    if bank is None:
                        continue
                    packed = self._packed_cpu(*key)
                    with self.cv:
                        latest = self.requests.get(lane)
                        if self.closed or request["epoch"] != self.epoch:
                            self.counters["stale"] += 1
                            continue
                        if (latest is None or latest["region"] != request["region"]
                                or key[0] != latest["layer"] or key[1] not in latest["ids"]):
                            self.counters["obsolete_after_pack"] += 1
                            continue
                        for dst, src in zip(bank["host"], packed):
                            for name in src:
                                dst[name][pos].copy_(src[name])
                        start, ready = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                        with torch.cuda.stream(self.stream):
                            start.record(self.stream)
                            for dst, src in zip(bank["gpu"], bank["host"]):
                                for name in src:
                                    dst[name][pos].copy_(src[name][pos], non_blocking=True)
                            ready.record(self.stream)
                        bank["ready"] = ready
                        bank["keys"][key] = dict(pos=pos, ready=ready, serial=latest["serial"])
                        self.counters["published"] += 1
                        self.trace.append(dict(serial=latest["serial"], region=request["region"],
                                               depth=latest["depth"], step=latest["step"],
                                               layer=key[0], bank=bank["id"], expert=key[1],
                                               start=start, ready=ready))
                        self.counters["trace_dropped"] += max(0, len(self.trace)-1024)
                        self.trace = self.trace[-1024:]
                except OSError as exc:
                    with self.cv:
                        self.counters["errors"] += 1
                        self.last_error = repr(exc)
                        # Already published complete experts remain valid. A
                        # partial expert is never admitted, and no retry spins.
                        self.settled[lane] = max(self.settled.get(lane, 0), request["serial"])
                except Exception as exc:
                    with self.cv:
                        self.fatal = exc
                        self.requests.clear()
                finally:
                    with self.cv:
                        self.active = False
                        self.cv.notify_all()

    def try_copy(self, layer, eid, destinations):
        with self.cv:
            self._check()
            candidates = [(b, b["keys"][(layer, eid)]) for b in self.banks
                          if b["epoch"] == self.epoch and (layer, eid) in b["keys"]]
            if not candidates:
                self.counters["unavailable_at_demand"] += 1
                return False
            bank, entry = max(candidates, key=lambda pair: pair[1]["serial"])
            stream = torch.cuda.current_stream(self.device)
            ready = entry["ready"].query()
            self.counters["ready_at_demand" if ready else "late_at_demand"] += 1
            stream.wait_event(entry["ready"])
            if bank["used"] is not None:
                stream.wait_event(bank["used"])
            for dst, src in zip(destinations, bank["gpu"]):
                for name in dst:
                    dst[name].copy_(src[name][entry["pos"]])
            done = torch.cuda.Event()
            done.record(stream)
            bank["used"] = done
            self.counters["copied_experts"] += 1
            self.consumptions[bank["id"]] += 1
            return True

    def reset(self):
        with self.cv:
            self.epoch += 1
            self.requests.clear()
            self.settled.clear()
            for bank in self.banks:
                bank["keys"] = {}
            self.cv.notify_all()

    def drain(self, timeout=30):
        """Diagnostic/teardown only; not used between generation steps."""
        with self.cv:
            if not self.cv.wait_for(lambda: not self.active and not self._pending(), timeout):
                raise TimeoutError("Warmup producer did not drain")
            self._check()
        self.stream.synchronize()

    def stats(self):
        with self.cv:
            timings = []
            for row in self.trace:
                finish = self.step_done.get(row["step"])
                start, ready = row["start"], row["ready"]
                if finish is not None and all(e.query() for e in (start, ready, finish)):
                    a, b, c = [self.origin.elapsed_time(e) for e in (start, ready, finish)]
                    timings.append({k: v for k, v in row.items() if k not in ("start", "ready")} |
                                   dict(transfer_start_ms=a, transfer_done_ms=b, step_done_ms=c,
                                        transfer_began_before_step_end=a < c, ready_before_step_end=b < c))
            updates = []
            for row in self.updates:
                if row["event"].query():
                    updates.append({k: v for k, v in row.items() if k != "event"} |
                                   dict(observed_ms=self.origin.elapsed_time(row["event"])))
            return {**self.counters, "bank_rotations": list(self.rotations), "banks": 3,
                    "bank_consumptions": list(self.consumptions),
                    "gpu_bytes": len(self.banks)*self.capacity*self.expert_bytes,
                    "pinned_bytes": len(self.banks)*self.capacity*self.expert_bytes,
                    "host_cache_bytes": self.cpu_cache_size, "host_cache_cap": self.host_cache_bytes,
                    "worker_alive": self.thread.is_alive(), "epoch": self.epoch, "closed": self.closed,
                    "last_error": getattr(self, "last_error", None), "timeline": timings,
                    "region_updates": updates}

    def close(self):
        with self.cv:
            if self.closed:
                return
            self.closed = True
            self.epoch += 1
            self.requests.clear()
            self.cv.notify_all()
        self.thread.join()
        self.stream.synchronize()
        for bank in self.banks:
            if bank["used"] is not None:
                bank["used"].synchronize()
        self.close_cpu()
        self.cpu_cache.clear()
        self.cpu_cache_size = 0
        self.banks.clear()
