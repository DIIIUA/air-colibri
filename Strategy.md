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

## 3. Next Steps
- [ ] Deconstruct AirLLM's layer-swapping logic into a high-performance C pipeline.
- [ ] Hook into Colibri's async `FILE_FLAG_OVERLAPPED` prefetching engine.
- [ ] Test layer-wise streaming with hybrid DeltaNet + Attention checkpoints.
