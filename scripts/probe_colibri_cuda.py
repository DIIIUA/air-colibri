"""Small native Colibri CUDA ABI/compute probe, without Torch or model weights."""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import traceback


def main():
    root = Path(__file__).resolve().parents[1]
    path = Path(r"D:\Colibri\coli_cuda.dll")
    result = dict(dll=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), status="STARTED")
    lib = tensor = None
    with os.add_dll_directory(str(path.parent)):
        try:
            lib = C.CDLL(str(path))
            lib.coli_cuda_init.argtypes = [C.POINTER(C.c_int), C.c_int]
            lib.coli_cuda_init.restype = C.c_int
            lib.coli_cuda_shutdown.argtypes = []
            lib.coli_cuda_shutdown.restype = None
            lib.coli_cuda_device_count.restype = C.c_int
            devices = (C.c_int * 1)(0)
            result["init"] = lib.coli_cuda_init(devices, 1)
            result["device_count"] = lib.coli_cuda_device_count()
            if not result["init"]:
                raise RuntimeError("Original Colibri CUDA init returned failure")
            ptr = C.POINTER(C.c_float)
            lib.coli_cuda_matmul.argtypes = [C.POINTER(C.c_void_p), ptr, ptr, C.c_void_p, ptr,
                                            C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int]
            lib.coli_cuda_matmul.restype = C.c_int
            lib.coli_cuda_tensor_free.argtypes = [C.c_void_p]
            lib.coli_cuda_tensor_free.restype = None
            size = 32
            weights = (C.c_float * (size*size))(*[(i%13-6)/16 for i in range(size*size)])
            x = (C.c_float * size)(*[i/32 for i in range(size)])
            y = (C.c_float * size)()
            tensor = C.c_void_p()
            ok = lib.coli_cuda_matmul(C.byref(tensor), y, x, weights, None, 0, 1, size, size, 0, 0)
            expected = [sum(float(weights[r*size+k])*float(x[k]) for k in range(size)) for r in range(size)]
            result["matmul_success"] = ok
            result["max_abs_error"] = max(abs(float(y[i])-expected[i]) for i in range(size))
            if not ok or result["max_abs_error"] > 1e-5:
                raise RuntimeError("Native matrix computation failed comparison")
            result["status"] = "PASS"
        except BaseException as exc:
            result.update(status="FAIL", error=str(exc), traceback=traceback.format_exc())
        finally:
            if lib is not None:
                if tensor:
                    lib.coli_cuda_tensor_free(tensor)
                lib.coli_cuda_shutdown()
    (root / "data/integration-20261112/native-cuda-probe.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
