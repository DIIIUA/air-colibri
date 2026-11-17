"""Selective Q4/Q6 expert cache: only routed experts are read and converted.

The byte budget includes space for bounded conversion/dequantization temporaries.
Prefill uses the same packed representation as decode and preserves the existing
anti-thrashing rule. Unsupported expert mathematics stays on the source path.
"""
from collections import OrderedDict
import torch
from src.quantized_projection import pack_weight, unpack_weight, packed_specs


class QuantizedExpertCache:
    def __init__(self, base, bits, budget_bytes, group_size=64):
        self.base, self.bits, self.group_size = base, bits, group_size
        self.device = base.device
        self.enabled_layers = set()
        self._lru = OrderedDict()
        self.hits = self.misses = self.conversions = self.fused_calls = self.prefill_batches = 0
        self.fallback_calls = 0
        self.warmup = None
        self.shared_memory = None
        self.auxiliary_depth = 0
        self.warm_hits = 0
        self.gu = self.dn = None
        self.num_slots = 0
        self.inter_buf = self.part_buf = None
        if hasattr(base, "_layer_meta"):
            shapes = {(tuple(m["gate"]["torch_shape"][-2:]), tuple(m["up"]["torch_shape"][-2:]), tuple(m["down"]["torch_shape"][-2:])) for m in base._layer_meta.values()}
            total = sum(m["num_experts"] for m in base._layer_meta.values())
            if len(shapes) != 1:
                return
            gate, up, down = shapes.pop()
            self.I, self.H = gate
            if up != gate or down != (self.H, self.I):
                return
        else:
            if not getattr(base, "uniform_shapes", False) or not base.has_gate:
                return
            self.I, self.H = base.inter_dim, base.hidden_dim
            total = len(base._spans)
        if self.device.type != "cuda" or self.I % group_size or self.H % group_size:
            return
        specs_gu = packed_specs("weight", (2*self.I, self.H), bits, group_size)
        specs_dn = packed_specs("weight", (self.H, self.I), bits, group_size)
        self.expert_bytes = sum(torch.empty((), dtype=dt).element_size() * shape[0] * shape[1]
                                for shape, dt in list(specs_gu.values()) + list(specs_dn.values()))
        # Eight BF16 expert results plus bounded FP32 conversion tiles/workspaces.
        self.temporary_reserve_bytes = 8 * 6 * self.I * self.H + 32 * 1024**2
        available = max(0, int(budget_bytes) - self.temporary_reserve_bytes)
        slots = min(total, available // self.expert_bytes)
        if slots < 8:
            return
        self.num_slots = slots
        self.gu = {k: torch.empty((slots, *sh), dtype=dt, device=self.device) for k, (sh, dt) in specs_gu.items()}
        self.dn = {k: torch.empty((slots, *sh), dtype=dt, device=self.device) for k, (sh, dt) in specs_dn.items()}
        self.inter_buf = torch.empty((8, self.I), dtype=torch.float32, device=self.device)
        self.part_buf = torch.empty((8, self.H), dtype=torch.float32, device=self.device)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def configure_layer(self, layer, module, has_bias=False, input_weighted=False):
        from transformers.activations import SiLUActivation
        act = getattr(module, "act_fn", None)
        silu = type(act) in (torch.nn.SiLU, SiLUActivation) or act is torch.nn.functional.silu
        custom = getattr(type(module), "_apply_gate", None)
        custom_ok = custom is None or getattr(custom, "__name__", "") == "_default_apply_gate"
        if self.num_slots and silu and custom_ok and not has_bias and not input_weighted and not hasattr(module, "alpha") and not getattr(module, "has_post_expert_norm", False):
            self.enabled_layers.add(layer)

    def _get(self, layer, eid, allow_evict, protected=frozenset()):
        key = (layer, eid)
        if key in self._lru:
            self.hits += 1
            if allow_evict:
                self._lru.move_to_end(key)
            slot = self._lru[key]
            return ({k: v[slot] for k, v in self.gu.items()}, {k: v[slot] for k, v in self.dn.items()}, slot)
        self.misses += 1
        # The producer only writes its own bounded banks. Admission and LRU
        # mutation remain on the normal compute thread, behind a reader fence.
        if self.warmup is not None and allow_evict:
            victim = None
            target = len(self._lru) if len(self._lru) < self.num_slots else None
            if target is None:
                victim = next((k for k in self._lru if k not in protected), None)
                target = self._lru.get(victim)
            if target is not None:
                gu_dst = {k:v[target] for k,v in self.gu.items()}
                dn_dst = {k:v[target] for k,v in self.dn.items()}
                if self.warmup.try_copy(layer,eid,(gu_dst,dn_dst)):
                    if victim is not None:self._lru.pop(victim)
                    self._lru[key] = target
                    self.warm_hits += 1
                    return gu_dst,dn_dst,target
        g, u, d = self.base.load_batch(layer, [eid], self.device, torch.bfloat16, False)
        gu = pack_weight(torch.cat((g[0], u[0])), self.bits, self.group_size)
        dn = pack_weight(d[0], self.bits, self.group_size)
        self.conversions += 1
        slot = None
        if len(self._lru) < self.num_slots:
            slot = len(self._lru)
        elif allow_evict:
            victim = next((k for k in self._lru if k not in protected), None)
            if victim is not None:
                slot = self._lru.pop(victim)
        if slot is not None:
            for k, v in gu.items():
                self.gu[k][slot].copy_(v)
            for k, v in dn.items():
                self.dn[k][slot].copy_(v)
            self._lru[key] = slot
        return gu, dn, slot

    def load_batch(self, layer, ids, device, dtype, allow_evict):
        if layer not in self.enabled_layers or torch.device(device) != self.device:
            self.fallback_calls += 1
            return self.base.load_batch(layer, ids, device, dtype, allow_evict)
        if len(ids) > 8:
            raise ValueError("Expert dequantization batch exceeds the bounded workspace")
        self.prefill_batches += 1
        g = torch.empty((len(ids), self.I, self.H), dtype=dtype, device=device)
        u = torch.empty_like(g)
        d = torch.empty((len(ids), self.H, self.I), dtype=dtype, device=device)
        if self.shared_memory is not None:
            with self.shared_memory.lease(layer, ids, self._load_packed, allow_evict,
                                          auxiliary=bool(self.auxiliary_depth)) as entries:
                for i, entry in enumerate(entries):
                    gu, dn = entry['gu'], entry['dn']
                    g[i].copy_(unpack_weight({k:v[:self.I] for k,v in gu.items()}, self.group_size, dtype))
                    u[i].copy_(unpack_weight({k:v[self.I:] for k,v in gu.items()}, self.group_size, dtype))
                    d[i].copy_(unpack_weight(dn, self.group_size, dtype))
            return g, u, d
        protected = {(layer, eid) for eid in ids}
        for i, eid in enumerate(ids):
            gu, dn, _ = self._get(layer, eid, allow_evict, protected)
            # Decode one projection at a time, not the complete expert batch.
            g[i].copy_(unpack_weight({k: v[:self.I] for k, v in gu.items()}, self.group_size, dtype))
            u[i].copy_(unpack_weight({k: v[self.I:] for k, v in gu.items()}, self.group_size, dtype))
            d[i].copy_(unpack_weight(dn, self.group_size, dtype))
        return g, u, d

    def _load_packed(self, layer, eid):
        g, u, d = self.base.load_batch(layer, [eid], self.device, torch.bfloat16, False)
        self.conversions += 1
        return (pack_weight(torch.cat((g[0], u[0])), self.bits, self.group_size),
                pack_weight(d[0], self.bits, self.group_size))

    def _fused_shared(self, layer, hidden, valid, weights):
        from src.colibri_nvrtc import fused_routed_moe_q4, fused_routed_moe_q6, reduce_routed_parts
        memory = self.shared_memory
        with memory.lease(layer, [eid for _, eid in valid], self._load_packed,
                          auxiliary=bool(self.auxiliary_depth)) as entries:
            # Keep the original routing order and FP32 contributions. Only the
            # addresses change; reduce once, with the original native kernel.
            start = 0
            while start < len(entries):
                bank = entries[start]['bank']
                end = start+1
                while end < len(entries) and entries[end]['bank'] == bank:
                    end += 1
                if bank is None:
                    # Bounded uncached fallback (e.g. a protected auxiliary
                    # working set). A standalone expert is a one-entry bank.
                    end = start+1
                    gu = {k:v.unsqueeze(0) for k,v in entries[start]['gu'].items()}
                    dn = {k:v.unsqueeze(0) for k,v in entries[start]['dn'].items()}
                    positions = [0]
                else:
                    gu, dn = memory.banks[bank]
                    positions = [e['pos'] for e in entries[start:end]]
                idx = torch.tensor(positions, dtype=torch.int64, device=self.device)
                route = weights[0, [pos for pos,_ in valid[start:end]]].contiguous()
                args = (hidden, idx, route, gu['packed_lo'])
                if self.bits == 4:
                    fused_routed_moe_q4(*args, gu['qscale'], gu['qzero'],
                        dn['packed_lo'], dn['qscale'], dn['qzero'],
                        self.inter_buf[start:end], self.part_buf[start:end], self.group_size, reduce=None)
                else:
                    fused_routed_moe_q6(*args, gu['packed_hi'], gu['qscale'], gu['qzero'],
                        dn['packed_lo'], dn['packed_hi'], dn['qscale'], dn['qzero'],
                        self.inter_buf[start:end], self.part_buf[start:end], self.group_size, reduce=None)
                memory.counts['fused_groups'] += 1
                start = end
            return reduce_routed_parts(self.part_buf[:len(entries)])

    def try_fused(self, layer, hidden, indices, weights, num_experts):
        if layer not in self.enabled_layers or hidden.shape[0] != 1 or hidden.dtype != torch.bfloat16:
            return None
        if (self.H//self.group_size)*4 + self.H*2 > 49152 or (self.I+self.I//self.group_size)*4 > 49152:
            return None
        ids = indices[0].tolist()
        valid = [(pos, int(eid)) for pos, eid in enumerate(ids) if 0 <= int(eid) < num_experts]
        if not valid:
            return torch.zeros_like(hidden)
        if len(valid) > 8:
            return None
        if self.shared_memory is not None:
            self.fused_calls += 1
            return self._fused_shared(layer, hidden, valid, weights)
        protected = {(layer, eid) for _, eid in valid}
        slots = [self._get(layer, eid, True, protected)[2] for _, eid in valid]
        slot_idx = torch.tensor(slots, device=self.device, dtype=torch.int64)
        route_weights = weights[0, [pos for pos, _ in valid]].contiguous()
        from src.colibri_nvrtc import fused_routed_moe_q4, fused_routed_moe_q6
        self.fused_calls += 1
        gu, dn = self.gu, self.dn
        if self.bits == 4:
            return fused_routed_moe_q4(hidden, slot_idx, route_weights, gu["packed_lo"], gu["qscale"], gu["qzero"], dn["packed_lo"], dn["qscale"], dn["qzero"], self.inter_buf, self.part_buf, self.group_size)
        return fused_routed_moe_q6(hidden, slot_idx, route_weights, gu["packed_lo"], gu["packed_hi"], gu["qscale"], gu["qzero"], dn["packed_lo"], dn["packed_hi"], dn["qscale"], dn["qzero"], self.inter_buf, self.part_buf, self.group_size)

    def stats(self):
        size = sum(v.numel()*v.element_size() for bank in (self.gu, self.dn) if bank for v in bank.values())
        return {"format": f"colibri_q{self.bits}", "slots": self.num_slots, "arena_mib": round(size/1024**2, 2),
                "enabled_layers": sorted(self.enabled_layers), "hits": self.hits, "misses": self.misses,
                "hit_rate": self.hits/max(1, self.hits+self.misses), "conversions": self.conversions,
                "fused_calls": self.fused_calls, "prefill_batches": self.prefill_batches,
                "fallback_calls": self.fallback_calls, "warm_hits": self.warm_hits,
                "shared_memory": self.shared_memory.stats() if self.shared_memory is not None else None,
                "source": self.base.stats()}

    def close(self):
        if self.warmup is not None:
            self.warmup.close()
            self.warmup = None
        self._lru.clear()
        self.gu = self.dn = self.inter_buf = self.part_buf = None
        if self.shared_memory is not None:
            self.shared_memory.close()
            self.shared_memory = None
        self.base.close()
