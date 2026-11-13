# 🚧 Integration roadmap

**Active experiment. No stable release or general performance guarantee.**

| Package | Current state | Remaining work |
|---|---|---|
| P0 — baseline | Source revisions, local backups and evidence captured | Independent installation and reproducible checkpoint/cache preparation |
| P1 — shared path | Dense contracts, original readers, GPU Ring and persistent Q4 cache integrated | Common resource catalog; expose cache disk quota; broader format/provider coverage |
| P2 — ownership | Per-slot stream/initialization fences; 3,072 reuses and in-flight prefetch teardown tested | Common resource reservations, long-duration pressure/soak, asynchronous prefetch cancellation |
| P3 — full Dense | Qwen 9B tested with logits and recurrent/KV state | Independent reference against the current revision; longer prefill/decode and contexts |
| P4 — original native mechanism | Original C `tier.h` affects real RAM retention | Complete native CUDA executor/interoperability |
| P5 — common placement | Native RAM admission across the cache of one runtime, with backend ACKs | Catalog-to-plan-to-actuation loop; shared Dense/expert/state byte budgets; GPU residency and cancellation |
| P6 — full MoE | OLMoE BF16 with real router and original forward; non-expert weights resident on GPU | Stream the Dense trunk and experts in the same model through shared resource ownership; other families |
| P7 — quality | BF16 repeatability; 960-position Q4-g64 evaluation failed strict acceptance | Localize quantization error; improve on separate development data; fresh held-out reevaluation |
| P8 — practical benefit | Five repeated local policy A/B pairs | Physical I/O, cold/warm, TTFT, standalone comparisons |
| P9 — product path | One experimental CLI, provenance and resource guards | Largest-object limits, larger checkpoints, hardware portability, packaging and API |

## Order of work

1. Validate the current revision against an independent reference. Historical
   three-prompt HF comparisons and current same-runtime BF16 repeatability are
   different evidence; neither substitutes for a current matched reference run.
2. Unify resource ownership for the Dense trunk and experts within one supported
   MoE model. Use a constrained VRAM budget that actually requires trunk streaming.
   Reuse the existing readers and lifetime mechanisms; preserve router choices.
3. Measure expert reuse, memory and physical I/O. Introduce common byte/lifetime
   reservations and plan generations in shadow mode, then apply one real GPU
   placement operation with backend ACKs and off/on logits/state parity.
4. Resolve Q4-g64 quality on separate development data and a fresh held-out set.
   This is a separate acceptance track; it does not block valid BF16 integration
   work. The 960-position evaluation is 12 sequences of 80 tokens, not a
   960-token context or a long-generation test.
5. Validate a family-specific native executor/format/state seam, then test actual
   migration and any phase-dependent execution policy. Expand workloads and
   checkpoints only after the relevant correctness and resource gates pass.

## Requirements recovered by the historical review

The original goal extends beyond two model adapters behind one launcher.
These items remain explicit work, rather than capabilities inferred from names:

- **One Dense/MoE resource backend.** The OLMoE adapter currently loads every
  non-expert parameter onto the GPU. Expert `prepare()` uses `.to()` and
  `torch.cat`; it does not use the Dense GPU Ring or its GPU lease ACK adapter.
  Its RAM cache does not avoid repeated expert H2D. A hot GPU expert cache needs
  its own verified reservations, eviction and last-consumer fences.
- **A closed placement loop.** `catalog_bridge` reuses `CrossLayerPlanMap` for
  metadata and observations. It explicitly reports no physical residency change.
  `shadow_policy.shadow_replan` is outside the CLI import graph; CLI `--policy
  shadow` refers to native RAM admission. Model/run/phase/plan identities,
  bounded pending bytes, backpressure and stale-result disposal remain open.
- **Measured hardware and phase policies.** Slot count is selected explicitly.
  The old FLOPs-based `hierarchical_cost_model` does not drive the CLI. A useful
  replacement needs measured read/DMA/compute costs and contention, not fixed
  hardware constants. Prefill-paid residency and CPU/GPU placement are proposals
  to evaluate, not accepted performance results.
- **Progressive acceleration.** The original idea of starting generation and
  attaching GPU capacity gradually remains unimplemented. The historical
  `ProgressiveLiveAttacher` uses a sleep/callback and flags; it is not connected
  to the CLI and does not migrate KV/DeltaNet state. A real implementation needs
  a compatible CPU/native path, complete state/position transfer, event ordering
  and measured startup-inclusive TTFT and migration cost.
- **Objects larger than a slot.** Current Dense staging loads a whole layer.
  Q4 tiled dequantization is not SSD-to-GPU tiled streaming. Large layers,
  experts and output heads need explicit fit limits or a verified tiled path.
  Historical direct-I/O prototypes are evidence to reuse, not an automatically
  selected replacement loader.
- **Cache disk limits.** `PrepackedLayerCache.max_bytes` exists, but its default
  is zero (unlimited) and the main manager/CLI does not supply a quota. Wiring
  configuration, handling temporary disk peaks and testing multiple format
  variants remain product work.

The resource plan is a startup estimate plus runtime guards, not an operating
system hard quota or a complete live ledger of RAM, pinned, VRAM, native
allocations, activations and growing state. An independent setup and hardware
matrix, including smaller GPUs, remain necessary before broader claims.

The review used the existing source corpus and current code/evidence. It did
not run another GPU experiment. Historical reports and private chat material
remain local; published findings are limited to implementation scope and gates.

MTP, GGUF, Vulkan, arbitrary models and an OpenAI-compatible service are
possible future directions, **not current AirColibri capabilities**. Upstream
support does not automatically become integration support.

See [the experiment report](EXPERIMENTS_2026-11-13.md) and
[follow-up CUDA tests](TESTING_2026-11-13.md) for evidence and limits.
