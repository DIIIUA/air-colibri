"""Conservative startup budget from CUDA and the matching GPU's NVML counter.

On the tested Windows/WDDM host CUDA's free counter did not reflect a second
process's allocation. Use the lower valid reading; never match NVML by an ordinal
that CUDA_VISIBLE_DEVICES can remap. NVML remains optional for CPU installations.
"""
import ctypes
import os
from pathlib import Path
import threading

_LOCK = threading.Lock()


class _Memory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


def _nvml_memory(uuid):
    library = (str(Path(os.environ.get("SystemRoot", r"C:\Windows"))/"System32"/"nvml.dll")
               if os.name == "nt" else "libnvidia-ml.so.1")
    lib = ctypes.CDLL(library)
    def check(code):
        if code:
            raise OSError(f"NVML returned status {code}")
    with _LOCK:
        check(lib.nvmlInit_v2())
        try:
            handle = ctypes.c_void_p()
            get_handle = lib.nvmlDeviceGetHandleByUUID
            get_handle.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
            get_handle.restype = ctypes.c_int
            check(get_handle(str(uuid).encode("ascii"), ctypes.byref(handle)))
            get_memory = lib.nvmlDeviceGetMemoryInfo
            get_memory.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Memory)]
            get_memory.restype = ctypes.c_int
            memory = _Memory()
            check(get_memory(handle, ctypes.byref(memory)))
            return int(memory.free), int(memory.total)
        finally:
            lib.nvmlShutdown()


def measure_gpu_memory(device=None):
    import torch
    free, total = (int(n) for n in torch.cuda.mem_get_info(device))
    result = dict(cuda_free_bytes=free, cuda_total_bytes=total,
                  free_bytes=free, total_bytes=total, source="cuda_only")
    try:
        uuid = getattr(torch.cuda.get_device_properties(device), "uuid", None)
        if uuid is None:
            raise OSError("CUDA did not expose a GPU UUID for NVML matching")
        physical_free, physical_total = _nvml_memory(uuid)
        if physical_total <= 0 or not 0 <= physical_free <= physical_total:
            raise OSError("Invalid NVML memory counters")
        result.update(physical_free_bytes=physical_free, physical_total_bytes=physical_total,
                      free_bytes=min(free, physical_free), total_bytes=min(total, physical_total),
                      source="cuda_nvml_min")
    except (OSError, AttributeError, UnicodeError) as exc:
        result["physical_counter_unavailable"] = str(exc)
    return result
