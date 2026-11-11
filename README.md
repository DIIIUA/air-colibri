# AirColibri 🦅⚡

> [!WARNING]
> **🚧 Work in Progress — Early Experimental Stage**
>
> This project is currently under active development and architectural experimentation. **Do not expect good results, high speed, or production-ready stability yet** — components are still being refactored, benchmarked, and assembled (active engineering work happens on the [`feature/hybrid-engine-mvp`](https://github.com/DIIIUA/air-colibri/tree/feature/hybrid-engine-mvp) branch).

> **Unified Low-Resource LLM Inference Engine:** Fusing the layer-wise streaming of [AirLLM](https://github.com/lyogavin/airllm) with the native high-speed C/CUDA expert-streaming architecture of [Colibri](https://github.com/JustVugg/colibri).

---

## 🎯 Motivation & Vision

Running modern massive Language Models (70B+ dense models or 200B–744B Mixture-of-Experts architectures) on consumer hardware has historically faced a hard memory wall:

* **AirLLM** solved the memory barrier for dense models by introducing **Layer-Wise Sequential Execution**: loading one transformer layer into VRAM at a time, computing its activations, and replacing it with the next layer.
* **Colibri** solved the memory barrier for massive MoE architectures (DeepSeek-V4, Qwen3.5/3.6, GLM-5.2/5.3) by **Streaming Routed Experts from NVMe SSDs**: keeping only a minimal dense backbone in RAM/VRAM and streaming active expert weights on demand via a high-performance C runtime with CUDA/Vulkan acceleration and MTP speculative decoding.

**AirColibri** combines these two breakthroughs into a unified, high-performance hybrid inference runtime:

```
┌────────────────────────────────────────────────────────────────────────┐
│                              AirColibri                                │
├────────────────────────────────────────────────────────────────────────┤
│  Dense Transformer Layers               Mixture-of-Experts (MoE)       │
│  Layer-wise sequential execution        Fine-grained expert streaming   │
│  (AirLLM Layer Swapper)                (Colibri C/CUDA Engine)         │
├────────────────────────────────────────────────────────────────────────┤
│  ⚡ Overlapped Async Disk Prefetch (FILE_FLAG_OVERLAPPED / io_uring)   │
│  ⚡ Hardware Accelerators: CUDA (sm_80+) / Vulkan 1.3+ / CPU (AVX2)    │
│  ⚡ Speculative Decoding via Multi-Token Prediction (MTP) Heads        │
│  ⚡ Native GGUF & Safetensors Container Support                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Architecture Synthesis

### 1. Hierarchical Two-Tier Streaming
* **Tier 1 (Dense & Attention):** Layer-by-layer sequential execution for backbone projections (Q/K/V/O, norms, and DeltaNet linear attention states) with double-buffered background disk readahead.
* **Tier 2 (Sparse Experts):** Dynamic on-demand expert streaming via Colibri's LRU cache and GPU expert tiers, preventing memory bloat on large-count MoE models.

### 2. High-Speed Native C & CUDA Core
* Replaces pure Python weight swapping with native C readahead routines (`FILE_FLAG_OVERLAPPED` on Windows, direct asynchronous prefetch), maximizing PCIe and NVMe throughput.
* Custom CUDA int8/FP4 matrix kernels (Ampere/Ada/Blackwell architectures) for instant in-VRAM token generation.

### 3. Integrated MTP Speculative Decoding
* Employs Multi-Token Prediction (MTP) heads to verify multiple draft tokens per forward pass, reducing the number of disk accesses per generated token and substantially boosting throughput.

---

## 🚀 Quick Setup & Roadmap

### Upstream Foundations:
* 🔗 [AirLLM Upstream Fork](https://github.com/DIIIUA/airllm)
* 🔗 [Colibri Upstream Fork](https://github.com/DIIIUA/colibri)

### Milestones:
- [x] Initial synthesis architecture & specification.
- [ ] Hybrid layer-wise loader with Colibri async C readahead.
- [ ] DeltaNet + Attention memory-efficient streaming pipeline.
- [ ] OpenAI-compatible API gateway and CLI interactive chat.
- [ ] Benchmarking against standalone AirLLM and Colibri on RTX 3060 Ti / 4060 hardware.

---

## 📜 License & Acknowledgments

* AirLLM is created by Gavin Li ([lyogavin/airllm](https://github.com/lyogavin/airllm)).
* Colibri is created by JustVugg ([JustVugg/colibri](https://github.com/JustVugg/colibri)).
* AirColibri is licensed under the Apache 2.0 License.
