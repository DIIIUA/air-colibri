"""Thin bindings to the original AirLLM reader and Colibri C admission policy.

No tensor loader or admission algorithm is reimplemented here. The native
component controls RAM admission; native model execution is a separate backend.
"""
from pathlib import Path
from collections import Counter, OrderedDict, deque
import ctypes as C
import hashlib
import importlib
import json
import sys
from functools import partial
from types import FunctionType


class _CachedPreadContext:
    """Reusable context wrapper over a safetensors pread handle (zero mmap bytes)."""
    def __init__(self, raw_cm):
        self._raw_cm = raw_cm
        self._handle = raw_cm.__enter__()

    def __enter__(self):
        return self._handle

    def __exit__(self, exc_type, exc, tb):
        return False

    def close(self):
        if self._raw_cm is not None:
            raw, self._raw_cm, self._handle = self._raw_cm, None, None
            raw.__exit__(None, None, None)


class AirLLMReader:
    def __init__(self, source, io_backend=None, retain_pread_handles=False, max_pread_handles=4):
        if io_backend not in (None, "mmap", "pread"):
            raise ValueError("Unsupported safetensors I/O backend")
        if retain_pread_handles and io_backend != "pread":
            raise ValueError("Persistent shard handles are permitted only for pread (never mmap)")
        self.source = Path(source).resolve()
        package = self.source / "air_llm"
        if not (package / "airllm/utils.py").is_file():
            raise FileNotFoundError("Expected original AirLLM checkout")
        sys.path.insert(0, str(package))
        try:
            module = importlib.import_module("airllm.utils")
        finally:
            sys.path.remove(str(package))
        if Path(module.__file__).resolve() != (package / "airllm/utils.py").resolve():
            raise RuntimeError("A different AirLLM version is already loaded")
        self.read = module.load_layer_subset
        self.io_backend = io_backend or "upstream-default"
        self._pread_handles: OrderedDict[str, _CachedPreadContext] = OrderedDict()
        self._max_pread_handles = max(1, int(max_pread_handles))
        if io_backend is not None:
            # Keep upstream bytecode and source intact. Bind safe_open in a
            # private function namespace, so other readers/threads are not
            # affected by process-global monkeypatching. No parsing or tensor
            # loading algorithm is reimplemented here.
            original = self.read
            if not isinstance(original, FunctionType) or "safe_open" not in original.__code__.co_names:
                raise TypeError("Expected original Python subset reader using safe_open")
            namespace = dict(original.__globals__)
            base_safe_open = partial(namespace["safe_open"], backend=io_backend)
            if retain_pread_handles:
                def cached_safe_open(filepath, *args, **kwargs):
                    key = str(filepath)
                    ctx = self._pread_handles.get(key)
                    if ctx is not None:
                        self._pread_handles.move_to_end(key)
                        return ctx
                    ctx = _CachedPreadContext(base_safe_open(filepath, *args, **kwargs))
                    self._pread_handles[key] = ctx
                    while len(self._pread_handles) > self._max_pread_handles:
                        _, old = self._pread_handles.popitem(last=False)
                        old.close()
                    return ctx
                namespace["safe_open"] = cached_safe_open
            else:
                namespace["safe_open"] = base_safe_open
            self.read = FunctionType(original.__code__, namespace, original.__name__,
                                     original.__defaults__, original.__closure__)
            self.read.__kwdefaults__ = original.__kwdefaults__
        self.calls = 0
        self.logical_bytes = 0

    def __call__(self, folder, stem, keys):
        result = self.read(folder, stem, keys)
        self.calls += 1
        self.logical_bytes += sum(t.numel() * t.element_size() for t in result.values())
        return result

    def close(self):
        while self._pread_handles:
            _, ctx = self._pread_handles.popitem(last=False)
            ctx.close()


class NativeTierPolicy:
    """Use upstream tier_pick_lfru to admit loaded layers into bounded RAM.

    A rejected miss still executes from its transient buffer. It does not evict
    resident weights. Shadow mode records the same decision without applying it.
    """
    def __init__(self, dll, num_layers, capacity, mode="apply", decay_interval=100):
        if mode not in ("shadow", "apply") or num_layers < 1 or capacity < 0 or decay_interval < 1:
            raise ValueError("Invalid native tier parameters")
        path = Path(dll).resolve()
        manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["dll_sha256"]:
            raise ValueError("Native library checksum differs from build provenance")
        self.lib = C.CDLL(str(path))
        self.lib.ac_tier_abi.restype = C.c_int
        if self.lib.ac_tier_abi() != 1:
            raise ValueError("Unsupported native tier ABI")
        u32p = C.POINTER(C.c_uint32)
        intp = C.POINTER(C.c_int)
        self.lib.ac_tier_pick.argtypes = [u32p, u32p, C.c_uint32, C.c_int, intp, C.c_int, intp, intp]
        self.lib.ac_tier_pick.restype = C.c_int
        self.lib.ac_tier_decay.argtypes = [u32p, C.c_int]
        self.lib.ac_tier_decay.restype = None
        self.lib.ac_tier_promote.argtypes = [C.c_uint32, C.c_uint32]
        self.lib.ac_tier_promote.restype = C.c_int
        self.heat = (C.c_uint32 * num_layers)()
        self.last = (C.c_uint32 * num_layers)()
        self.n = num_layers
        self.capacity = capacity
        self.mode = mode
        self.decay_interval = decay_interval
        self.clock = 0
        self.stats = dict(accesses=0, native_calls=0, accepted=0, rejected=0, decisions_returned=0,
                          backend_acks=0, backend_admissions=0, backend_rejections=0)
        self.acknowledgements = deque(maxlen=1024)
        self.provenance = manifest

    def record_access(self, layer):
        if not 0 <= layer < self.n:
            raise ValueError("Layer out of range")
        self.clock = (self.clock + 1) & 0xffffffff
        self.stats["accesses"] += 1
        if self.stats["accesses"] % self.decay_interval == 0:
            self.lib.ac_tier_decay(self.heat, self.n)
            self.stats["native_calls"] += 1
        self.heat[layer] = min(0xffffffff, self.heat[layer] + 1)
        self.last[layer] = self.clock

    def admit(self, layer, residents):
        victim = None
        if layer in residents or len(residents) < self.capacity:
            accepted = self.capacity > 0
        elif not residents:
            accepted = False
        else:
            pinned = (C.c_int * len(residents))(*residents)
            slot, candidate = C.c_int(), C.c_int()
            changed = self.lib.ac_tier_pick(self.heat, self.last, self.clock, self.n,
                pinned, len(residents), C.byref(slot), C.byref(candidate))
            self.stats["native_calls"] += 1
            accepted = bool(changed and candidate.value == layer)
            if accepted:
                victim = residents[slot.value]
        self.stats["accepted" if accepted else "rejected"] += 1
        if self.mode == "shadow":
            return True, None
        self.stats["decisions_returned"] += 1
        return accepted, victim

    def acknowledge(self, event):
        self.acknowledgements.append(event)
        self.stats["backend_acks"] += 1
        self.stats["backend_admissions"] += event["status"] == "applied"
        self.stats["backend_rejections"] += event["status"] == "rejected"


class PerLayerNativeTierPolicy:
    """Per-layer Colibri C LFRU policy (c/tier.h) with prefill route warmup and T2 epoch decay.

    Matches upstream Colibri's per-layer repin_pick organization: each decoder
    layer owns an independent 64-expert heat/recency table and per-layer slot
    quota so early layers never starve later layers, prefill token frequencies
    populate the hottest experts per layer in a single pass, and tier_decay
    runs on token epochs (T2) rather than sub-token expert call counts.
    """
    def __init__(self, dll, num_layers, num_experts, capacity, mode="apply", epoch_tokens=64):
        if (mode not in ("shadow", "apply") or num_layers < 1 or num_experts < 1
                or capacity < 0 or capacity > num_layers * num_experts or epoch_tokens < 1):
            raise ValueError("Invalid per-layer native tier parameters")
        path = Path(dll).resolve()
        manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["dll_sha256"]:
            raise ValueError("Native library checksum differs from build provenance")
        self.lib = C.CDLL(str(path))
        self.lib.ac_tier_abi.restype = C.c_int
        if self.lib.ac_tier_abi() != 1:
            raise ValueError("Unsupported native tier ABI")
        u32p = C.POINTER(C.c_uint32)
        intp = C.POINTER(C.c_int)
        self.lib.ac_tier_pick.argtypes = [u32p, u32p, C.c_uint32, C.c_int, intp, C.c_int, intp, intp]
        self.lib.ac_tier_pick.restype = C.c_int
        self.lib.ac_tier_decay.argtypes = [u32p, C.c_int]
        self.lib.ac_tier_decay.restype = None
        self.lib.ac_tier_promote.argtypes = [C.c_uint32, C.c_uint32]
        self.lib.ac_tier_promote.restype = C.c_int

        self.num_layers = int(num_layers)
        self.num_experts = int(num_experts)
        self.n = self.num_layers * self.num_experts
        self.capacity = int(capacity)
        self.mode = mode
        self.epoch_tokens = int(epoch_tokens)
        self.layer_capacity = self._distribute_capacity(self.capacity)
        self.heat = [(C.c_uint32 * self.num_experts)() for _ in range(self.num_layers)]
        self.last = [(C.c_uint32 * self.num_experts)() for _ in range(self.num_layers)]
        self.layer_clock = [0] * self.num_layers
        self.prefill_targets: dict[int, set[int]] = {}
        self._batch_recorded_layers: set[int] = set()
        self.tokens_processed = 0
        self._last_decay_tokens = 0
        self.stats = dict(
            kind="per-layer-native-lfru-v1",
            accesses=0,
            route_tokens_observed=0,
            prefill_warmed_layers=0,
            epoch_decays=0,
            native_calls=0,
            accepted=0,
            rejected=0,
            decisions_returned=0,
            backend_acks=0,
            backend_admissions=0,
            backend_rejections=0,
        )
        self.acknowledgements = deque(maxlen=1024)
        self.provenance = manifest

    def _distribute_capacity(self, total: int) -> list[int]:
        base, rem = divmod(total, self.num_layers)
        return [base + (1 if i < rem else 0) for i in range(self.num_layers)]

    def shrink_capacity(self, new_capacity: int) -> list[int]:
        if not 0 <= new_capacity <= self.capacity:
            raise ValueError("New capacity must be between 0 and current capacity")
        self.capacity = int(new_capacity)
        self.layer_capacity = self._distribute_capacity(self.capacity)
        return list(self.layer_capacity)

    def record_route_batch(self, layer: int, token_expert_ids: list[list[int]]):
        """Ingest router selections right after layer.mlp.gate before experts run."""
        if not 0 <= layer < self.num_layers:
            raise ValueError("Layer out of range")
        if not token_expert_ids:
            return
        n_tokens = len(token_expert_ids)
        if n_tokens > 1:
            counts = Counter(int(e) for row in token_expert_ids for e in row)
            clk = (self.layer_clock[layer] + 1) & 0xffffffff
            self.layer_clock[layer] = clk
            h_arr = self.heat[layer]
            l_arr = self.last[layer]
            for e, cnt in counts.items():
                if not 0 <= e < self.num_experts:
                    raise ValueError("Expert out of range")
                h_arr[e] = min(0xffffffff, int(h_arr[e]) + int(cnt))
                l_arr[e] = clk
            self.stats["route_tokens_observed"] += sum(counts.values())
            self._batch_recorded_layers.add(layer)
            k_cap = self.layer_capacity[layer]
            if k_cap > 0:
                ranked = sorted(range(self.num_experts), key=lambda e: (-int(h_arr[e]), e))
                # Select top-k_cap experts that actually have positive heat on this layer
                targets = {e for e in ranked[:k_cap] if int(h_arr[e]) > 0}
                self.prefill_targets[layer] = targets
                self.stats["prefill_warmed_layers"] += 1
        else:
            self._batch_recorded_layers.discard(layer)
            self.prefill_targets.pop(layer, None)

    def finish_layer_route(self, layer: int):
        self._batch_recorded_layers.discard(layer)
        self.prefill_targets.pop(layer, None)

    def on_token_step(self, new_tokens: int = 1):
        """T2 boundary clock: decay per-layer heats only on token epochs."""
        if new_tokens > 0:
            self.tokens_processed += int(new_tokens)
        while self.tokens_processed - self._last_decay_tokens >= self.epoch_tokens:
            self._last_decay_tokens += self.epoch_tokens
            for l in range(self.num_layers):
                self.lib.ac_tier_decay(self.heat[l], self.num_experts)
            self.stats["native_calls"] += self.num_layers
            self.stats["epoch_decays"] += 1

    def record_access(self, global_expert: int):
        if not 0 <= global_expert < self.n:
            raise ValueError("Global expert index out of range")
        layer, expert = divmod(int(global_expert), self.num_experts)
        self.stats["accesses"] += 1
        if layer in self._batch_recorded_layers:
            # Exact per-token counts were already recorded in record_route_batch.
            return
        clk = (self.layer_clock[layer] + 1) & 0xffffffff
        self.layer_clock[layer] = clk
        self.heat[layer][expert] = min(0xffffffff, int(self.heat[layer][expert]) + 1)
        self.last[layer][expert] = clk
        self.stats["route_tokens_observed"] += 1

    def admit(self, global_expert: int, residents: list[int], exclude: set[int] | None = None):
        if not 0 <= global_expert < self.n:
            raise ValueError("Global expert index out of range")
        layer, local_e = divmod(int(global_expert), self.num_experts)
        k_cap = self.layer_capacity[layer]
        victim = None
        layer_res = [g for g in residents if g // self.num_experts == layer]
        if k_cap <= 0:
            accepted = False
        elif global_expert in layer_res:
            accepted = True
        elif layer in self.prefill_targets:
            if local_e not in self.prefill_targets[layer]:
                accepted = False
            elif len(layer_res) < k_cap:
                accepted = True
            else:
                evictable = [g for g in layer_res if (g % self.num_experts) not in self.prefill_targets[layer]
                             and (not exclude or g not in exclude)]
                if evictable:
                    # Evict the coldest non-target resident in this layer
                    victim = min(evictable, key=lambda g: int(self.heat[layer][g % self.num_experts]))
                    accepted = True
                else:
                    accepted = False
        elif len(layer_res) < k_cap:
            accepted = True
        else:
            pinned = (C.c_int * len(layer_res))(*(g % self.num_experts for g in layer_res))
            slot, candidate = C.c_int(), C.c_int()
            changed = self.lib.ac_tier_pick(
                self.heat[layer], self.last[layer], self.layer_clock[layer],
                self.num_experts, pinned, len(layer_res), C.byref(slot), C.byref(candidate)
            )
            self.stats["native_calls"] += 1
            if changed and candidate.value == local_e:
                cand_victim = layer_res[slot.value]
                if exclude and cand_victim in exclude:
                    accepted = False
                else:
                    accepted = True
                    victim = cand_victim
            else:
                accepted = False
        self.stats["accepted" if accepted else "rejected"] += 1
        if self.mode == "shadow":
            return True, None
        self.stats["decisions_returned"] += 1
        return accepted, victim

    def acknowledge(self, event):
        self.acknowledgements.append(event)
        self.stats["backend_acks"] += 1
        self.stats["backend_admissions"] += event["status"] == "applied"
        self.stats["backend_rejections"] += event["status"] == "rejected"
