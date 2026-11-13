# 🚧 Integration roadmap

**Active experiment. No stable release or general performance guarantee.**

| Package | Current state | Remaining work |
|---|---|---|
| P0 — baseline | Source revisions, local backups and evidence captured | Reproduce on an independent machine |
| P1 — shared path | Contracts, original readers, GPU Ring and persistent Q4 cache integrated | Broader format/provider coverage |
| P2 — ownership | Per-slot stream/initialization fences; 3,072 reuses and in-flight prefetch teardown tested | Long-duration soak, asynchronous prefetch cancellation |
| P3 — full Dense | Qwen 9B tested with logits and recurrent/KV state | Longer and more varied workloads |
| P4 — original native mechanism | Original C `tier.h` affects real RAM retention | Complete native CUDA executor/interoperability |
| P5 — common placement | One global RAM admission policy and backend ACKs | General dynamic VRAM placement and cancellation |
| P6 — full MoE | OLMoE BF16 with real router and original forward | Other families and checkpoints |
| P7 — quality | BF16 repeatability; 960-position Q4-g64 evaluation failed strict acceptance | Localize quantization error; improve on separate development data; fresh held-out reevaluation |
| P8 — practical benefit | Five repeated local policy A/B pairs | Physical I/O, cold/warm, TTFT, standalone comparisons |
| P9 — product path | One experimental CLI, provenance and resource guards | Independent setup, larger models, packaging and API |

## Order of work

1. Resolve the measured Q4-g64 quality failure using separate development data,
   then confirm on a fresh held-out teacher-forcing set.
2. Measure expert reuse and physical I/O before changing residency policy.
3. Introduce GPU byte/lifetime reservations in shadow mode; apply only after
   actual backend ACKs and off/on output/state parity.
4. Validate a family-specific native executor seam separately from cache policy.
5. Expand to longer contexts and a larger supported checkpoint, then package.

MTP, GGUF, Vulkan, arbitrary models and an OpenAI-compatible service are
possible future directions, **not current AirColibri capabilities**. Upstream
support does not automatically become integration support.

See [the experiment report](EXPERIMENTS_2026-11-13.md) and
[follow-up CUDA tests](TESTING_2026-11-13.md) for evidence and limits.
