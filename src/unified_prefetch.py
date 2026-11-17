"""Prefill-map producer publishing directly into the common weight arena."""
from collections import OrderedDict
import threading

import torch

from src.progressive_prefetch import ThreeBankExpertPrefetch


class UnifiedExpertPrefetch(ThreeBankExpertPrefetch):
    """Reuse bounded CPU packing; the unified allocator owns all GPU entries.

Logical map lanes are not physical banks. An expert may be retained in any
area, and a demand miss uses the same directory as a prepared expert.
"""
    def __init__(self, memory, cache, capacity, load_cpu, close_cpu, host_cache_bytes):
        self.memory = memory
        self.device, self.bits = cache.device, cache.bits
        self.capacity, self.group_size = capacity, cache.group_size
        self.expert_bytes = cache.expert_bytes
        if host_cache_bytes < self.expert_bytes:
            raise ValueError('Host budget must fit at least one expert')
        self.load_cpu, self.close_cpu = load_cpu, close_cpu
        self.host_cache_bytes = int(host_cache_bytes)
        self.cpu_cache, self.cpu_cache_size = OrderedDict(), 0
        self.cv = memory.cv
        self.epoch = memory.epoch
        self.serial = 0
        self.requests, self.settled = OrderedDict(), {}
        self.active = self.closed = False
        self.fatal = None
        self.stream = torch.cuda.Stream(device=self.device)
        self.origin = torch.cuda.Event(enable_timing=True)
        self.origin.record(torch.cuda.current_stream(self.device))
        self.stream.wait_event(self.origin)
        self.trace, self.updates, self.step_done = [], [], {}
        self.counters = dict(submitted=0, superseded=0, published=0, stale=0,
            cpu_conversions=0, host_cache_hits=0, errors=0, retained_on_refinement=0,
            obsolete_after_pack=0, trace_dropped=0, updates_dropped=0)
        self.thread = threading.Thread(target=self._run,
            name='aircolibri-unified-warmup', daemon=True)
        self.thread.start()

    def submit(self, layer, ids, *, region=None, depth=None, lane=None, step=None):
        ids = tuple(dict.fromkeys(int(e) for e in ids))
        if lane not in (0, 1, 2) or len(ids) > self.capacity or any(e < 0 for e in ids):
            raise ValueError('Invalid unified map region')
        with self.cv:
            self._check()
            if self.closed:
                raise RuntimeError('Warmup is closed')
            old = self.requests.get(lane)
            if old and old['serial'] > self.settled.get(lane, 0):
                self.counters['superseded'] += 1
            self.serial += 1
            if getattr(self.memory, 'dynamic_regions', False):
                # B holds just the next region; C owns no queued GPU payload.
                self.requests.clear()
            self.requests[lane] = dict(layer=int(layer), ids=ids, region=region,
                serial=self.serial, epoch=self.epoch, lane=lane, step=step, depth=depth)
            wanted = {(r['layer'], eid) for r in self.requests.values() for eid in r['ids']}
            self.counters['retained_on_refinement'] += len(wanted & self.memory.directory.keys())
            self.memory.set_desired(wanted)
            self.counters['submitted'] += 1
            self.cv.notify_all()
            return self.serial

    def _pending(self):
        return [lane for lane, req in self.requests.items()
                if req['serial'] > self.settled.get(lane, 0)]

    def _run(self):
        with torch.cuda.device(self.device), torch.inference_mode():
            while True:
                with self.cv:
                    self.cv.wait_for(lambda: self.closed or bool(self._pending()))
                    if self.closed:
                        return
                    lane = self._pending()[0]
                    request = self.requests[lane]
                    self.requests.move_to_end(lane)
                    missing = [(request['layer'], eid) for eid in request['ids']
                               if (request['layer'], eid) not in self.memory.directory]
                    if not missing:
                        self.settled[lane] = request['serial']
                        self.cv.notify_all()
                        continue
                    key = missing[0]
                    self.active = True
                try:
                    packed = self._packed_cpu(*key)
                    with self.cv:
                        latest = self.requests.get(lane)
                        if self.closed or request['epoch'] != self.epoch:
                            self.counters['stale'] += 1
                            continue
                        if latest is None or latest['layer'] != key[0] or key[1] not in latest['ids']:
                            self.counters['obsolete_after_pack'] += 1
                            continue
                        status, record = self.memory.prepare(key, packed, self.stream, self.epoch)
                        if status == 'published':
                            self.counters['published'] += 1
                            self.trace.append(dict(serial=latest['serial'], region=latest['region'],
                                layer=key[0], expert=key[1], step=latest['step'], depth=latest['depth'],
                                bank=self.memory.area_for(record), start=record['transfer_start'], ready=record['ready']))
                            self.counters['trace_dropped'] += max(0, len(self.trace)-1024)
                            self.trace = self.trace[-1024:]
                    if status is None:
                        self.memory.wait_for_space(lambda: self.closed)
                except OSError as exc:
                    with self.cv:
                        self.counters['errors'] += 1
                        self.last_error = repr(exc)
                        self.settled[lane] = request['serial']
                except Exception as exc:
                    with self.cv:
                        self.fatal = exc
                        self.requests.clear()
                finally:
                    with self.cv:
                        self.active = False
                        self.cv.notify_all()

    def reset(self):
        with self.cv:
            self.memory.reset_task()
            self.epoch = self.memory.epoch
            self.requests.clear()
            self.settled.clear()
            self.cv.notify_all()

    def stats(self):
        with self.cv:
            timeline = []
            for row in self.trace:
                finish = self.step_done.get(row['step'])
                if finish is not None and all(e.query() for e in (row['start'], row['ready'], finish)):
                    a, b, c = [self.origin.elapsed_time(e) for e in (row['start'], row['ready'], finish)]
                    timeline.append({k:v for k,v in row.items() if k not in ('start','ready')} |
                        dict(transfer_start_ms=a, transfer_done_ms=b, step_done_ms=c,
                             transfer_began_before_step_end=a<c, ready_before_step_end=b<c))
            common = self.memory.stats()
            return dict(**self.counters, policy='direct_shared_arena', banks=3,
                bank_consumptions=[common.get(f'warm_area_{i}',0) for i in range(3)],
                bank_rotations=list(self.memory.area_writes), gpu_bytes=0, pinned_bytes=0,
                copied_experts=0, shared_gpu_bytes=self.memory.gpu_bytes,
                direct_hits=common.get('warm_direct_hits',0),
                host_cache_bytes=self.cpu_cache_size, host_cache_cap=self.host_cache_bytes,
                worker_alive=self.thread.is_alive(), closed=self.closed, epoch=self.epoch,
                last_error=getattr(self,'last_error',None), timeline=timeline, region_updates=[])

    def close(self):
        with self.cv:
            if self.closed:
                return
            self.closed = True
            self.requests.clear()
            self.memory.reset_task()
            self.cv.notify_all()
        self.thread.join()
        self.stream.synchronize()
        self.close_cpu()
        self.cpu_cache.clear()
        self.cpu_cache_size = 0
