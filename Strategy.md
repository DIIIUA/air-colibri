# AirColibri — current engineering strategy

**Status: active experimental integration.** The canonical local workspace is
`C:\air-colibri`; this repository publishes the implementation and selected
research evidence. The local knowledge corpus and private history remain local.

## Implemented path

Original AirLLM subset reader → bounded RAM cache → original Transformers
forward, with original Colibrì C `tier.h` controlling RAM admission. Dense
Qwen3.5-9B uses the GPU Ring. OLMoE-1B-7B now streams each layer's Dense trunk
and selected experts through the same pinned pool, GPU Ring and lease ACK
mechanism. Placement is fixed to two slots; globals remain resident and native
RAM admission still covers experts. This is not a global dynamic planner.

The [shared MoE result](docs/SHARED_MOE_RESULTS_2026-11-13.md) matched an independent
direct-safetensors/HF reference in two short contexts and fit a 0.75 GiB Torch
allocator limit. Two mmap reader crashes are preserved; their path was localized
and Windows OLMoE now uses the existing safetensors pread backend. The underlying
native defect and long-run reliability are not established.

The [13 November experiment report](docs/EXPERIMENTS_2026-11-13.md) separates
model correctness, C-policy actuation, memory lifetime and limited timing
measurements. A small measured speed difference is not a general guarantee.

## Next decisions

1. Extend independent-reference coverage beyond the short OLMoE test to Qwen
   and longer state growth. Stress the pread mitigation and preserve native
   crash evidence; successful reruns do not establish the underlying defect.
2. Measure expert reuse, physical I/O, full memory and cache-size sensitivity;
   connect the catalog to real byte/lifetime reservations and GPU placement ACKs.
3. In a separate quality track, localize the measured
   [Q4-g64 failure](docs/Q4_RESULTS_2026-11-13.md) on development data before a
   fresh held-out evaluation. Keep Q4 experimental; valid BF16 integration can continue.
4. Verify a model-specific native executor/format/state seam. Only then test
   progressive CPU/GPU migration or different prefill/decode placement policies.
5. Test longer workloads, largest-object limits, larger checkpoints and
   independent installations before packaging and broader hardware claims.

See the [P0–P9 roadmap](docs/ROADMAP.md) and [developer guide](docs/DEVELOPMENT.md).
No upstream capability is inherited by name alone.

## Historical material

The previous strategy and its early benchmark claims remain in Git history.
Historical `AirColibriEngine`, routing simulations and old benchmark JSONs
are retained; their simplified computations do not establish full-model speed.
The active product entrypoint is `aircolibri.py`.
