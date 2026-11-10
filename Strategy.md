# Project Strategy: AirColibri (`10_ML_AND_TENSORS/air-colibri`)

```text
Outcome_(t-1) + ObservedBehavior_(t-1) = Strategy_t = ObservedBehavior_(t+1) + Outcome_(t+1)
```

## 1. Context & Objective
* **Objective:** Combine the layer-wise sequential execution mechanics of [AirLLM](https://github.com/DIIIUA/airllm) with the high-speed native C/CUDA expert-streaming runtime of [Colibri](https://github.com/DIIIUA/colibri).
* **Target Hardware:** Consumer GPUs (4GB–12GB VRAM, starting with NVIDIA RTX 3060 Ti 8GB) and standard system RAM (16GB).
* **GitHub Repository:** [DIIIUA/air-colibri](https://github.com/DIIIUA/air-colibri)

## 2. Completed Milestones (Verified)
- [x] Authenticated GitHub access verified for user `DIIIUA`.
- [x] Created public fork of `lyogavin/airllm` -> [DIIIUA/airllm](https://github.com/DIIIUA/airllm).
- [x] Created public fork of `JustVugg/colibri` -> [DIIIUA/colibri](https://github.com/DIIIUA/colibri).
- [x] Created public repository [DIIIUA/air-colibri](https://github.com/DIIIUA/air-colibri).
- [x] Injected foundational architecture specification and roadmap in `README.md`.
- [x] Initialized workspace in `D:\Antigravity Lair\10_ML_AND_TENSORS\air-colibri`.
- [x] Conducted in-depth source code audit of AirLLM and Colibrì (`docs/audit_and_design.md`).
- [x] Implemented Colibrì `tier.h` LFRU admission logic with 25% margin anti-thrashing (`src/lfru_cache.py`).
- [x] Developed AirColibri Reusable Slot Pools (`src/memory_pool.py`):
  - `SingleSlotPool` (zero-malloc in-place buffer reuse)
  - `DoubleBufferPool` (dual CUDA streams `compute_stream`/`transfer_stream` + `torch.cuda.Event` overlap)
  - `RingPoolWithCache` (Dense backbone ring buffer + MoE resident LFRU cache)
- [x] Created and verified 5 unit tests (`tests/test_memory_pool.py`) proving exact pointer immutability and mathematical correctness.
- [x] Ran empirical benchmarks on NVIDIA GeForce RTX 3060 Ti 8GB:
  - Naive AirLLM: 528.61 ms/step (1.89 tok/s, 2560 MB)
  - Single Slot: 25.59 ms/step (39.08 tok/s, 20.66x speedup, 24.1 MB VRAM)
  - Double Buffer: 24.49 ms/step (40.84 tok/s, 21.59x speedup, 48.3 MB VRAM)
  - Ring Pool + LFRU Cache: 16.96 ms/step (58.97 tok/s, 31.17x speedup, 73.75% transfer reduction, 80.3 MB VRAM)

## 3. Next Steps
- [ ] Implement C/CUDA native extension bindings (`c/air_slot.cu`) to bypass Python runtime overhead.
- [ ] Hook into Colibri's async `FILE_FLAG_OVERLAPPED` Windows prefetching engine for direct NVMe-to-Pinned-RAM streaming.
- [ ] Implement MTP speculative decoding verification on resident draft heads.
