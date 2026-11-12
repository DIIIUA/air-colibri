"""Thin bindings to the original AirLLM reader and Colibri C admission policy.

No tensor loader or admission algorithm is reimplemented here. The native
component controls RAM admission; native model execution is a separate backend.
"""
from pathlib import Path
from collections import Counter, deque
import ctypes as C
import hashlib
import importlib
import json
import sys


class AirLLMReader:
    def __init__(self, source):
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
        self.calls = 0
        self.logical_bytes = 0

    def __call__(self, folder, stem, keys):
        result = self.read(folder, stem, keys)
        self.calls += 1
        self.logical_bytes += sum(t.numel() * t.element_size() for t in result.values())
        return result


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
