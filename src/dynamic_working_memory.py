"""Elastic A (current/cache), B (next preparation), C (empty standby).

One fixed-address byte allocation is shared by dense payloads and experts.
Pages change ownership, not addresses. Retired CUDA readers remain fenced;
their bytes are reported separately until actual completion, never called free.
"""
from collections import Counter, OrderedDict
import math
import threading

import torch

from src.bounded_memory_manager import SlotArenaLayout
from src.quantized_projection import packed_specs
from src.unified_working_memory import UnifiedWorkingMemory


class DynamicWorkingMemory(UnifiedWorkingMemory):
    dynamic_regions = True

    def __init__(self, pipeline, cache, extra_budget_bytes=0, min_region_experts=0):
        if pipeline.num_slots != 3 or pipeline.device.type != 'cuda':
            raise ValueError('Dynamic regions require a three-slot CUDA ring')
        if pipeline.total_loads or cache._lru or pipeline.working_memory is not None:
            raise ValueError('Enable dynamic regions before the first forward')
        if extra_budget_bytes < 0:
            raise ValueError('Negative preparation budget')
        self.device, self.pipeline = pipeline.device, pipeline
        self.cv = threading.Condition(threading.RLock())
        self.closed = False
        self.epoch = 0
        self.desired = set()
        self.directory = OrderedDict()
        self.access_clock = 0
        self.counts = Counter()
        self.area_reads, self.area_writes = [0]*3, [0]*3
        self.area_dense = [Counter() for _ in range(3)]
        self.dense_active = [False]*3
        self.dense_blocks = {}
        self._dense_layouts = {}
        self.active_area, self.warm_area = 0, 1
        self.current_layer = None
        self.budget_bytes = pipeline.arena.numel() + cache.num_slots*cache.expert_bytes + int(extra_budget_bytes)
        self.temporary_reserve_bytes = cache.temporary_reserve_bytes
        self.expert_bytes = cache.expert_bytes
        specs = (packed_specs('weight', (2*cache.I, cache.H), cache.bits, cache.group_size),
                 packed_specs('weight', (cache.H, cache.I), cache.bits, cache.group_size))
        templates = {f'{prefix}.{name}': spec for prefix, fields in zip(('gu','dn'),specs)
                     for name,spec in fields.items()}
        self.expert_layout = SlotArenaLayout.from_templates(templates, slot_alignment=256, tensor_alignment=256)
        self.page_bytes = self.expert_layout.slot_stride
        self.num_slots = self.budget_bytes//self.page_bytes
        self.max_dense_pages = max(math.ceil(max((s.offset_bytes+s.nbytes for s in layout.specs.values()), default=0)
                                             /self.page_bytes) for layout in pipeline.slot_layouts)
        self.warm_capacity = min(8, self.num_slots-2*self.max_dense_pages-8)
        if self.warm_capacity < 1 or self.num_slots < 2*self.max_dense_pages+8+min(min_region_experts,8):
            raise ValueError('Dynamic budget cannot fit current, next payload and expert readers')
        self.per_area = self.num_slots  # one strided expert address space, not three fixed banks
        self.gpu_bytes = self.num_slots*self.page_bytes
        # Only in-flight H2D needs pinned backing. The GPU expert cache does not
        # need a second, equally large, pinned-RAM cache.
        self.staging_slots = 2
        self.pinned_bytes = self.staging_slots*self.page_bytes
        self._staging_ready = [None]*self.staging_slots
        self._staging_next = 0
        pipeline.reset_in_flight()
        for slot in pipeline.slots:
            slot.tensors.clear()
            slot.active_tensors.clear()
        pipeline.arena = None
        cache.gu = cache.dn = None
        self.gpu = torch.empty(self.gpu_bytes, dtype=torch.uint8, device=self.device)
        self.host = torch.empty(self.pinned_bytes, dtype=torch.uint8, pin_memory=True)
        self.banks, self.host_banks = [], []
        for arena, banks in ((self.gpu,self.banks),(self.host,self.host_banks)):
            slots = self.num_slots if arena is self.gpu else self.staging_slots
            fields = {}
            for name,spec in self.expert_layout.specs.items():
                element_bytes = torch.empty((),dtype=spec.dtype).element_size()
                tail = arena[spec.offset_bytes:].view(spec.dtype)
                fields[name] = tail.as_strided((slots,*spec.shape),
                    (self.page_bytes//element_bytes,spec.shape[1],1))
            banks.append(tuple({k.removeprefix(prefix+'.'):v for k,v in fields.items()
                                if k.startswith(prefix+'.')} for prefix in ('gu','dn')))
        self.records = [dict(bank=0,pos=i,key=None,generation=0,ready=None,readers=[],
            leases=0,warm_epoch=-1,origin=None,kind='free') for i in range(self.num_slots)]
        pipeline.arena = self.gpu
        pipeline.working_memory = self
        pipeline.slot_offsets = (None,None,None)
        cache.shared_memory = self
        cache.num_slots = self.num_slots

    def area_for(self, record):
        return self.warm_area if record['key'] in self.desired else self.active_area

    def _stage_packed(self, packed, record):
        slot = self._staging_next
        ready = self._staging_ready[slot]
        if ready is not None and not ready.query():
            # The CPU must not overwrite bytes still being read by H2D. This
            # waits for a real completion event, not an estimated transfer time.
            self.counts['staging_waits'] += 1
            ready.synchronize()
        staged = tuple({k: v[slot] for k, v in bank.items()} for bank in self.host_banks[0])
        for dst, src in zip(staged, packed):
            for name in dst:
                dst[name].copy_(src[name])
        return staged

    def _staging_complete(self, event):
        self._staging_ready[self._staging_next] = event
        self._staging_next = (self._staging_next+1)%self.staging_slots

    def begin_layer(self, layer):
        with self.cv:
            self._check()
            self.current_layer = int(layer)
            self.active_area = int(layer)%3
            self.warm_area = (self.active_area+1)%3
            self.counts['role_rotations'] += 1

    def set_desired(self, keys):
        keys = set(keys)
        if len(keys)>self.warm_capacity:
            raise ValueError('B exceeds its bounded next-region preparation allowance')
        super().set_desired(keys)

    def _victim(self, key, protected, optional, auxiliary=False):
        empty = (r for r in self.records if r['kind']=='free' and not r['leases'])
        cold = (self.records[i] for k,i in self.directory.items()
                if k not in protected and k not in self.desired and not self.records[i]['leases'])
        hot = () if optional or auxiliary else (self.records[i] for k,i in self.directory.items()
                if k not in protected and k in self.desired and not self.records[i]['leases'])
        for group in (empty,cold,hot):
            for record in group:
                if not optional or not self._fences(record):
                    return record
        return None

    def _write(self, key, packed, record, stream, origin):
        result = super()._write(key,packed,record,stream,origin)
        record['kind'] = 'expert'
        return result

    def wait_for_space(self, cancelled=lambda:False):
        with self.cv:
            self._check()
            eligible = [r for r in self.records if r['kind']!='dense' and
                        not r['leases'] and r['key'] not in self.desired]
            if not eligible:
                self.cv.wait_for(lambda:self.closed or cancelled() or any(r['kind']!='dense' and
                    not r['leases'] and r['key'] not in self.desired for r in self.records))
                return
            fences = self._fences(eligible[0])
        if fences:
            fences[0].synchronize()

    def bind_dense(self, slot, payload, stream):
        # Actual payload geometry, without reserving the old 2 MiB slot padding.
        for name,src in payload.items():
            spec = slot.layout.specs.get(name)
            if spec is None or src.dtype!=spec.dtype or tuple(src.shape) not in (spec.allowed_shapes or (spec.shape,)):
                raise ValueError(f'GPU payload shape/dtype/key mismatch: {name}')
        signature = tuple((name,tuple(src.shape),src.dtype) for name,src in payload.items())
        layout = self._dense_layouts.get(signature)
        if layout is None:
            layout = SlotArenaLayout.from_templates(payload, slot_alignment=256, tensor_alignment=256)
            self._dense_layouts[signature] = layout
        count = math.ceil(layout.slot_stride/self.page_bytes)
        with self.cv:
            self._check()
            if slot.slot_id in self.dense_blocks:
                raise RuntimeError('Dense role still owns a payload')
            if len(self.dense_blocks)>=2:
                raise RuntimeError('Standby must stay empty: at most current and next dense payloads')
            # A single sliding scan; stop as soon as a free contiguous span is
            # found. Never construct/sort every possible span on every layer.
            best = None
            busy = warm = occupied = heat = 0
            for end, record in enumerate(self.records):
                busy += int(record['kind']=='dense' or bool(record['leases']))
                warm += int(record['key'] in self.desired)
                occupied += int(record['key'] is not None)
                heat += max(0,record.get('last_used',0)) if record['key'] is not None else 0
                if end >= count:
                    old = self.records[end-count]
                    busy -= int(old['kind']=='dense' or bool(old['leases']))
                    warm -= int(old['key'] in self.desired)
                    occupied -= int(old['key'] is not None)
                    heat -= max(0,old.get('last_used',0)) if old['key'] is not None else 0
                if end+1 >= count and not busy:
                    candidate = (warm,occupied,heat,end-count+1)
                    if best is None or candidate < best:
                        best = candidate
                    if occupied == 0:
                        break
            if best is None:
                raise RuntimeError('No contiguous dense space without overwriting a live reader')
            start = best[3]
            for record in self.records[start:start+count]:
                for event in self._fences(record):
                    stream.wait_event(event)
                if record['key'] is not None:
                    self.directory.pop(record['key'])
                    self.counts['dense_reclaims_expert'] += 1
                record.update(key=None,kind='dense',warm_epoch=-1)
            self.dense_blocks[slot.slot_id] = dict(start=start,count=count,payload_bytes=layout.slot_stride,
                                                  layer=slot.current_layer_idx)
            slot.tensors = layout.bind_slot_views(self.gpu,0,base_offset_bytes=start*self.page_bytes)
            self.pipeline.slot_offsets = tuple(self.dense_blocks[s]['start']*self.page_bytes
                                              if s in self.dense_blocks else None for s in range(3))
            self.counts['dense_bindings'] += 1
            self.counts['peak_dense_blocks'] = max(self.counts['peak_dense_blocks'],len(self.dense_blocks))
            self.cv.notify_all()

    def dense_event(self, bank, kind):
        super().dense_event(bank,kind)
        if kind=='release':
            with self.cv:
                slot = self.pipeline.slots[bank]
                block = self.dense_blocks.pop(bank)
                for record in self.records[block['start']:block['start']+block['count']]:
                    record.update(kind='free',key=None,ready=slot.transfer_done_event,
                                  readers=[slot.compute_done_event],leases=0)
                slot.tensors.clear()
                slot.active_tensors.clear()
                slot.loaded_keys.clear()
                self.pipeline.slot_offsets = tuple(self.dense_blocks[s]['start']*self.page_bytes
                                                  if s in self.dense_blocks else None for s in range(3))
                self.counts['dense_returns_to_pool'] += 1
                self.cv.notify_all()

    def reset_dense(self):
        # Caller has synchronized CUDA before clearing cancelled ownership.
        with self.cv:
            for record in self.records:
                if record['kind']=='dense':
                    record.update(kind='free',key=None,ready=None,readers=[],leases=0)
            self.dense_blocks.clear()
            self.dense_active = [False]*3
            self.pipeline.slot_offsets = (None,None,None)
            for slot in self.pipeline.slots:
                slot.tensors.clear()
            self.cv.notify_all()

    def stats(self):
        with self.cv:
            expert_b = sum(r['key'] in self.desired for r in self.records)*self.page_bytes
            expert_a = len(self.directory)*self.page_bytes-expert_b
            dense_a = sum(b['count']*self.page_bytes for s,b in self.dense_blocks.items() if self.dense_active[s])
            dense_b = sum(b['count']*self.page_bytes for s,b in self.dense_blocks.items() if not self.dense_active[s])
            pending = sum(r['kind']=='free' and bool(self._fences(r)) for r in self.records)*self.page_bytes
            free = self.gpu_bytes-expert_a-expert_b-dense_a-dense_b-pending
            assert min(free,pending)>=0
            roles = dict(A=dict(area=self.active_area,used_bytes=expert_a+dense_a,
                                available_bytes=expert_a+dense_a+free),
                         B=dict(area=self.warm_area,used_bytes=expert_b+dense_b,
                                expert_limit_bytes=self.warm_capacity*self.page_bytes),
                         C=dict(area=3-self.active_area-self.warm_area,used_bytes=0,reserved_bytes=0))
            return dict(policy='dynamic_A_current_B_prepare_C_standby',**dict(self.counts),
                budget_bytes=self.budget_bytes,gpu_bytes=self.gpu_bytes,pinned_bytes=self.pinned_bytes,
                temporary_reserve_bytes=self.temporary_reserve_bytes,page_bytes=self.page_bytes,
                num_slots=self.num_slots,live_experts=len(self.directory),roles=roles,
                free_bytes=free,retired_waiting_cuda_bytes=pending,
                dense_blocks=len(self.dense_blocks),dense_events=[dict(c) for c in self.area_dense],
                area_reads=list(self.area_reads),area_writes=list(self.area_writes),
                active_leases=sum(r['leases'] for r in self.records),warm_bank_to_cache_copies=0,
                allocation_ptr=self.gpu.data_ptr() if self.gpu is not None else None,
                epoch=self.epoch,closed=self.closed)
