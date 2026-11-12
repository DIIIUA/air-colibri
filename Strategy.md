# AirColibri — current engineering strategy

**Status: active experimental integration.** The canonical local workspace is
`C:\air-colibri`; this repository publishes the implementation and selected
research evidence. The local knowledge corpus and private history remain local.

## Implemented path

Original AirLLM subset reader → bounded cache / GPU Ring → original
Transformers forward, with original Colibrì C `tier.h` controlling RAM
admission. Dense Qwen3.5-9B and OLMoE-1B-7B are the tested checkpoints.

The [13 November experiment report](docs/EXPERIMENTS_2026-11-13.md) separates
model correctness, C-policy actuation, memory lifetime and limited timing
measurements. A small measured speed difference is not a general guarantee.

## Next decisions

1. Close strict Q4 quality acceptance before promoting Q4 beyond experimental.
2. Measure expert reuse, physical I/O and cache-size sensitivity.
3. Extend actual byte/lifetime reservations and ACKs to GPU placement.
4. Verify a model-specific native executor seam and CUDA ownership separately.
5. Test longer workloads, larger checkpoints and independent installations.

See the [P0–P9 roadmap](docs/ROADMAP.md) and [developer guide](docs/DEVELOPMENT.md).
No upstream capability is inherited by name alone.

## Historical material

The previous strategy and its early benchmark claims remain in Git history.
Historical `AirColibriEngine`, routing simulations and old benchmark JSONs
are retained; their simplified computations do not establish full-model speed.
The active product entrypoint is `aircolibri.py`.
