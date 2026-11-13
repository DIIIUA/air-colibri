# AirColibri — current engineering strategy

**Status: active experimental integration.** The canonical local workspace is
`C:\air-colibri`; this repository publishes the implementation and selected
research evidence. The local knowledge corpus and private history remain local.

## Implemented path

Original AirLLM subset reader → bounded RAM cache → original Transformers
forward, with original Colibrì C `tier.h` controlling RAM admission. Dense
Qwen3.5-9B uses the GPU Ring. OLMoE-1B-7B uses a separate expert-copy adapter
and keeps its non-expert weights resident on GPU. Both checkpoints are tested;
a shared resource backend that streams trunk and experts in one MoE is not yet
implemented.

The [13 November experiment report](docs/EXPERIMENTS_2026-11-13.md) separates
model correctness, C-policy actuation, memory lifetime and limited timing
measurements. A small measured speed difference is not a general guarantee.

## Next decisions

1. Check the current runtime against an independent reference, then unify
   ownership of Dense trunk, experts and state under a constrained memory budget.
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
