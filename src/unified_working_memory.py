"""One bounded three-area allocation for dense streaming and routed experts.

Dense addresses and expert addresses are laid out once. Demand and speculative
preparation share the same expert directory and storage: there is no warm-bank
to-cache copy. Leases protect readers until their actual CUDA events complete.
The first implementation reserves dense capacity in each area; it does not yet
lend that space to experts or allocate KV/SSM state from this weight arena.
"""
from collections import Counter, OrderedDict
from contextlib import contextmanager
import threading

import torch

from src.bounded_memory_manager import SlotArenaLayout, align_up
from src.quantized_projection import packed_specs


class UnifiedWorkingMemory:
    def __init__(self, pipeline, cache, extra_budget_bytes=0, min_region_experts=0):
        if pipeline.num_slots != 3 or pipeline.device.type != 'cuda':
            raise ValueError('Unified working memory requires a three-slot CUDA ring')
        if pipeline.total_loads or cache._lru or cache.num_slots < 8:
            raise ValueError('Enable unified memory before generation with a supported expert cache')
        if getattr(pipeline, 'working_memory', None) is not None:
            raise ValueError('The ring already has unified working memory')
        if extra_budget_bytes < 0:
            raise ValueError('Negative preparation budget')
        self.device = pipeline.device
        self.cv = threading.Condition(threading.RLock())
        self.closed = False
        self.epoch = 0
        self.desired = set()
        self.directory = OrderedDict()
        self.access_clock = 0
        self.counts = Counter()
        self.area_reads = [0, 0, 0]
        self.area_writes = [0, 0, 0]
        self.area_dense = [Counter() for _ in range(3)]
        self.dense_active = [False]*3
        self.dense_bytes = align_up(max(pipeline.slot_strides), 256)
        old_expert_bytes = cache.num_slots * cache.expert_bytes
        self.budget_bytes = pipeline.arena.numel() + old_expert_bytes + int(extra_budget_bytes)
        self.temporary_reserve_bytes = cache.temporary_reserve_bytes
        specs = (packed_specs('weight', (2*cache.I, cache.H), cache.bits, cache.group_size),
                 packed_specs('weight', (cache.H, cache.I), cache.bits, cache.group_size))
        per_area = (self.budget_bytes//3 - self.dense_bytes)//cache.expert_bytes
        while per_area > 0:
            templates = {f'{prefix}.{k}': ((per_area, *sh), dt)
                         for prefix, spec in zip(('gu', 'dn'), specs)
                         for k, (sh, dt) in spec.items()}
            layout = SlotArenaLayout.from_templates(templates, slot_alignment=256, tensor_alignment=256)
            stride = align_up(self.dense_bytes + layout.slot_stride, 256)
            if 3*stride <= self.budget_bytes:
                break
            per_area -= 1
        if per_area < 1 or 3*per_area < 8:
            raise ValueError('Unified budget cannot fit dense slots and eight routed experts')
        if (3*per_area-8)//3 < min_region_experts:
            raise ValueError('Aligned arena cannot fit warm regions plus eight demand/probe slots')
        self.per_area = per_area
        self.num_slots = 3*per_area
        self.area_stride = stride
        self.expert_layout = layout
        self.expert_bytes = cache.expert_bytes
        self.gpu_bytes = 3*stride
        self.pinned_bytes = 3*layout.slot_stride

        # No old+new live weight arenas at the transition. Activation is only
        # permitted before the first forward, so no model hook holds old views.
        pipeline.reset_in_flight()
        for slot in pipeline.slots:
            slot.tensors.clear()
            slot.active_tensors.clear()
        pipeline.arena = None
        cache.gu = cache.dn = None
        self.gpu = torch.empty(self.gpu_bytes, dtype=torch.uint8, device=self.device)
        self.host = torch.empty(self.pinned_bytes, dtype=torch.uint8, pin_memory=True)
        self.banks, self.host_banks, self.records = [], [], []
        for bank in range(3):
            views = layout.bind_slot_views(self.gpu, 0,
                base_offset_bytes=bank*stride+self.dense_bytes)
            host_views = layout.bind_slot_views(self.host, bank)
            self.banks.append(tuple({k.removeprefix(prefix+'.'): v for k, v in views.items()
                                     if k.startswith(prefix+'.')} for prefix in ('gu', 'dn')))
            self.host_banks.append(tuple({k.removeprefix(prefix+'.'): v for k, v in host_views.items()
                                          if k.startswith(prefix+'.')} for prefix in ('gu', 'dn')))
            slot = pipeline.slots[bank]
            slot.tensors = slot.layout.bind_slot_views(self.gpu, 0,
                                                       base_offset_bytes=bank*stride)
            slot.active_tensors = dict(slot.tensors)
            for pos in range(per_area):
                self.records.append(dict(bank=bank, pos=pos, key=None, generation=0,
                    ready=None, readers=[], leases=0, warm_epoch=-1, origin=None))
        pipeline.arena = self.gpu
        pipeline.slot_offsets = tuple(i*stride for i in range(3))
        pipeline.working_memory = self
        cache.shared_memory = self
        cache.num_slots = self.num_slots

    def _check(self):
        if self.closed:
            raise RuntimeError('Unified working memory is closed')

    def _fences(self, record):
        record['readers'] = [e for e in record['readers'] if not e.query()]
        return [e for e in [record['ready'], *record['readers']]
                if e is not None and not e.query()]

    def _views(self, record, host=False):
        cache_key = '_host_views' if host else '_gpu_views'
        if cache_key in record:
            return record[cache_key]
        banks = self.host_banks if host else self.banks
        views = tuple({k: v[record['pos']] for k, v in projection.items()}
                      for projection in banks[record['bank']])
        record[cache_key] = views
        return views

    def _stage_packed(self, packed, record):
        staged = self._views(record, host=True)
        for dst, src in zip(staged, packed):
            for name in dst:
                dst[name].copy_(src[name])
        return staged

    def _staging_complete(self, event):
        pass

    def area_for(self, record):
        return record['bank']

    def _victim(self, key, protected, optional, auxiliary=False):
        preferred = int(key[0]) % 3
        choices = sorted(self.records, key=lambda r: (r['bank'] != preferred, r['pos']))
        empty = [r for r in choices if r['key'] is None and not r['leases']]
        cold = [self.records[idx] for k, idx in self.directory.items()
                if k not in protected and k not in self.desired and not self.records[idx]['leases']]
        hot = [] if optional or auxiliary else [self.records[idx] for k, idx in self.directory.items()
                if k not in protected and k in self.desired and not self.records[idx]['leases']]
        for record in empty + cold + hot:
            if not optional or not self._fences(record):
                return record
        return None

    def _write(self, key, packed, record, stream, origin):
        assert record['leases'] == 0
        destinations = self._views(record)
        if len(packed) != len(destinations) or any(
            set(src) != set(dst) or any(src[k].shape != dst[k].shape or src[k].dtype != dst[k].dtype
                                       for k in dst)
            for src,dst in zip(packed,destinations)):
            raise ValueError('Packed expert shape/dtype mismatch')
        for event in self._fences(record):
            stream.wait_event(event)
        if record['key'] is not None:
            self.directory.pop(record['key'])
            self.counts['evictions'] += 1
        if origin == 'prefetch':
            # Optional writers are admitted only after prior H2D has finished,
            # so the pinned staging views can be overwritten on the CPU.
            packed = self._stage_packed(packed, record)
        with torch.cuda.stream(stream):
            start = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            for dst, src in zip(destinations, packed):
                for name in dst:
                    dst[name].copy_(src[name], non_blocking=True)
            event = torch.cuda.Event(enable_timing=True)
            event.record(stream)
        if origin == 'prefetch':
            self._staging_complete(event)
        record.update(key=key, generation=record['generation']+1, ready=event,
                      readers=[], origin=origin, transfer_start=start,
                      last_used=-1,
                      warm_epoch=self.epoch if key in self.desired else -1)
        self.directory[key] = record['bank']*self.per_area + record['pos']
        self.area_writes[self.area_for(record)] += 1
        self.counts[origin+'_writes'] += 1
        self.counts[origin+'_bytes'] += self.expert_bytes
        return record

    def set_desired(self, keys):
        with self.cv:
            self._check()
            desired = set(keys)
            if len(desired) > self.num_slots-8:
                raise ValueError('Preparation must leave eight demand/probe slots')
            self.desired = desired
            for key in self.desired:
                if key in self.directory:
                    self.records[self.directory[key]]['warm_epoch'] = self.epoch
            self.cv.notify_all()

    def prepare(self, key, packed, stream, epoch):
        """Optional publisher; returns None when actual readers/DMA occupy space."""
        with self.cv:
            self._check()
            if epoch != self.epoch or key not in self.desired:
                return 'stale', None
            if key in self.directory:
                r = self.records[self.directory[key]]
                r['warm_epoch'] = self.epoch
                self.counts['retained_prepared'] += 1
                return 'retained', r
            record = self._victim(key, set(), optional=True)
            if record is None:
                self.counts['producer_busy'] += 1
                return None, None
            return 'published', self._write(key, packed, record, stream, 'prefetch')

    def wait_for_space(self, cancelled=lambda: False):
        """Wait on actual completion, never an estimate of work duration."""
        with self.cv:
            self._check()
            events = [e for r in self.records if not r['leases'] and r['key'] not in self.desired
                      for e in self._fences(r)]
            if not events:
                self.cv.wait_for(lambda: self.closed or cancelled() or any(not r['leases'] and
                    (r['key'] is None or r['key'] not in self.desired) for r in self.records))
                return
        events[0].synchronize()

    def _acquire(self, key, loader, protected, allow_evict, auxiliary):
        stream = torch.cuda.current_stream(self.device)
        packed = None
        with self.cv:
            self._check()
            if key in self.directory:
                return self._take(key, stream, allow_evict, auxiliary, hit=True)
        packed = loader(*key)
        with self.cv:
            self._check()
            if key in self.directory:
                return self._take(key, stream, allow_evict, auxiliary, hit=True)
            else:
                self.counts['aux_misses' if auxiliary else 'misses'] += 1
                record = self._victim(key, protected, optional=False, auxiliary=auxiliary)
                if record is None or (record['key'] is not None and not allow_evict):
                    if packed is None:
                        raise RuntimeError('Missing packed expert')
                    return dict(gu=packed[0], dn=packed[1], record=None, bank=None, pos=None)
                self._write(key, packed, record, stream, 'demand')
            return self._take(key, stream, allow_evict, auxiliary, hit=False)

    def _take(self, key, stream, allow_evict, auxiliary, hit):
        record = self.records[self.directory[key]]
        if hit:
            self.counts['aux_hits' if auxiliary else 'hits'] += 1
        if allow_evict and not auxiliary:
            self.directory.move_to_end(key)
            self.access_clock += 1
            record['last_used'] = self.access_clock
        record['leases'] += 1
        ready = record['ready']
        if ready is not None:
            if ready.query():
                record['ready'] = None
            else:
                stream.wait_event(ready)
        self.counts['aux_leases' if auxiliary else 'leases'] += 1
        self.area_reads[self.area_for(record)] += 1
        if not auxiliary and record['warm_epoch'] == self.epoch:
            self.counts['warm_direct_hits'] += 1
            self.counts[f'warm_area_{self.area_for(record)}'] += 1
        gu, dn = self._views(record)
        return dict(gu=gu, dn=dn, record=record, bank=record['bank'], pos=record['pos'])

    @contextmanager
    def lease(self, layer, ids, loader, allow_evict=True, auxiliary=False):
        keys = [(int(layer), int(eid)) for eid in ids]
        if len(keys) > 8:
            raise ValueError('At most eight simultaneous expert readers')
        acquired = []
        try:
            for key in keys:
                acquired.append(self._acquire(key, loader, set(keys), allow_evict, auxiliary))
            yield acquired
        finally:
            with self.cv:
                event = None
                for item in acquired:
                    record = item['record']
                    if record is None:
                        continue
                    if event is None:
                        event = torch.cuda.Event()
                        event.record(torch.cuda.current_stream(self.device))
                    record['readers'].append(event)
                    record['leases'] -= 1
                self.cv.notify_all()

    def dense_event(self, bank, kind):
        with self.cv:
            self._check()
            if kind == 'load' and self.dense_active[bank]:
                raise RuntimeError('Dense overwrite with an active reader')
            if kind == 'acquire':
                if self.dense_active[bank]:
                    raise RuntimeError('Dense area already has a reader')
                self.dense_active[bank] = True
            elif kind == 'release':
                if not self.dense_active[bank]:
                    raise RuntimeError('Dense release without a reader')
                self.dense_active[bank] = False
            self.area_dense[bank][kind] += 1

    def reset_task(self):
        with self.cv:
            self.epoch += 1
            self.desired.clear()
            self.cv.notify_all()

    def stats(self):
        with self.cv:
            return dict(policy='unified_three_area_weights', **dict(self.counts),
                budget_bytes=self.budget_bytes, gpu_bytes=self.gpu_bytes,
                pinned_bytes=self.pinned_bytes, temporary_reserve_bytes=self.temporary_reserve_bytes,
                num_slots=self.num_slots, expert_slots_per_area=self.per_area,
                area_stride=self.area_stride, dense_bytes_per_area=self.dense_bytes,
                area_reads=list(self.area_reads), area_writes=list(self.area_writes),
                dense_events=[dict(c) for c in self.area_dense],
                live_experts=len(self.directory), active_leases=sum(r['leases'] for r in self.records),
                warm_bank_to_cache_copies=0, allocation_ptr=self.gpu.data_ptr() if self.gpu is not None else None,
                epoch=self.epoch, closed=self.closed)

    def close(self):
        with self.cv:
            if self.closed:
                return
            if any(r['leases'] for r in self.records):
                raise RuntimeError('Cannot close unified memory with active leases')
            self.closed = True
            self.cv.notify_all()
        torch.cuda.synchronize(self.device)
        self.directory.clear()
        # Cached tensor views must not keep the arena alive after close.
        for record in self.records:
            record.pop('_gpu_views', None)
            record.pop('_host_views', None)
        self.banks.clear()
        self.host_banks.clear()
        self.gpu = self.host = None
