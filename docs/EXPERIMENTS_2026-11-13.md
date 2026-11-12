# Integration experiments — 13 November 2026

**Experimental engineering snapshot, not a stable release.** These results cover
specific local checkpoints, hardware and short workloads. They do not establish
general model support, language quality or a universal speedup.

## What was actually connected

1. The original AirLLM `load_layer_subset` reader, imported from its checkout.
2. The original Transformers Qwen3.5 and OLMoE mathematical forward.
3. AirColibri's GPU Ring, bounded host cache, lifecycle contract and CUDA fences.
4. A compiled ABI-v1 shim calling Colibrì's original `tier.h` admission policy.

The native C policy changes which real weights remain in RAM. A rejected
admission still executes the requested weights through a transient buffer.
It does **not** replace PyTorch matrix operations. Native Colibrì CUDA math,
shared cross-engine device pointers and its complete executor are outside
the integrated path.

## Checkpoints and environment

| Item | Tested configuration |
|---|---|
| Dense | Qwen3.5-9B, `qwen3_5_text`, 32 decoder layers |
| MoE | OLMoE-1B-7B-0125-Instruct, 16 layers, 64 experts/layer, top-8 |
| Storage / compute | Qwen source F16 → BF16 baseline or experimental Q4; OLMoE BF16 |
| GPU | NVIDIA RTX 3060 Ti, 8 GiB, Windows x64 |
| Runtime | Python 3.12.11; PyTorch 2.6.0+cu124; Transformers 5.22.2 |
| AirLLM checkout | `e496ed3c5c8f614df943b9401cf17e156ef7aca0` |
| Colibrì checkout | `0a5504a1dc64f86fcd0fe7db298b0448c730e7e6` |

The lab's legacy Qwen directory is named `deepseek-v4-9b-colibri`; its
**config**, not the directory name, identifies the model. This does not
constitute a DeepSeek V4 validation.

## Correctness results

| Comparison | Scope | Result |
|---|---|---|
| Dense Q4, off/observe/native sequential/native ring | 4 runs × 2 output steps | Full logits and KV/recurrent state bitwise equal |
| Dense BF16, off/native | 2 runs × 2 steps | Full logits and state bitwise equal |
| MoE full-expert reference/sparse/native | 3 runs × 2 steps | Logits, state and router choices equal |
| MoE repeated off/native | 5 pairs × 8 steps | All corresponding logits, state and routes equal |
| MoE mixed RU/EN/code prompt | 34 input tokens, 2 output steps, off/apply/shadow | Logits, state and routes equal |
| Checkpoint tokenizer preservation | 8 Unicode/text/code examples | Original token IDs retained |

Maximum absolute logit difference in these comparisons: **0**.
Dense Q4 transferred 64 layers for two steps; the ring performed one
tail-to-head prefetch without adding a transfer. Ready cache hits did not
requantize weights. Dense tracing covered 248 linear modules and 64 state
tensors per generated step.

The MoE all-expert reference materializes all 64 experts of each layer and
runs the original eager expert forward. The sparse adapter changes weight
indexing only and retains the original router and expert math.

Evidence: [comparison summary](evidence/2026-11-13/validation-summary.json)
and [tokenizer checks](evidence/2026-11-13/tokenizer-preservation.json).

## Small causal policy comparison

Same sparse MoE backend, BF16, `Hello`, eight output steps, a 256 MiB RAM
cache, five fresh-process pairs, alternating off/apply order. Source hashes
were identical throughout this series. The OS file cache was neither flushed
nor controlled. This is a local repeated observation, not statistical proof
of a broadly applicable performance advantage.

| Metric | Policy off | Native C apply |
|---|---:|---:|
| Median total forward time, 8 steps | 18.004 s | 17.569 s |
| Median of each run's mean decode time | 2.186 s/step | 2.140 s/step |
| Expert reads | 1,024 | 983 |
| Total original AirLLM reader calls | 1,173 | 1,132 |
| Logical expert H2D bytes | 12 GiB | 12 GiB |
| Peak Torch CUDA allocation | 938.5 MiB | 938.5 MiB |

The policy reduced expert reads by **4.00%** and the observed median forward
time by **2.42%** in this series. H2D traffic did not decrease. The actual
cache held 21 experts, 252 MiB, under the 256 MiB limit.

Reader calls and tensor-copy bytes are **logical counters**, not physical SSD
telemetry. Selecting top-8 instead of loading all 64 experts is already part
of the sparse baseline; its savings are not attributed to the native policy.
Startup-inclusive TTFT, cold-cache comparisons, longer distributions and
standalone AirLLM/Colibrì comparisons remain open.

## Failures found and fixes

- Sequential execution now uses the same load/DMA/compute/release contract.
  Recording a completion event is distinguished from GPU completion.
- Pinned overlay payloads are checked before any write. Missing keys,
  incompatible shapes/dtypes and active slot reuse fail explicitly.
- Cache admission checks byte budgets before mutation. Rejected replacement
  retains resident weights; each applied/rejected placement has an ACK.
- Cold LFRU misses no longer evict a resident unconditionally. The consumer
  stream, rather than the transfer stream itself, waits for expert DMA.
- Large globals mappings are released after staging; embeddings remain lazy.
  Mmap handles and diagnostic tails have bounded retention.
- A native Windows MoE read crash was reproduced with and without C policy.
  Retaining owned copies of expert tensors instead of large-shard mmap views
  eliminated that crash in the reported runs. Mapping/commit pressure is the
  supported mechanism hypothesis, not proof of an identical upstream bug.
- Failed initial loading and interrupted computation invalidate the runtime.
  Double close and same-process reopen restore reference logits and state.

The first complete pytest process returned **53 passed, 1 failed, 1 skipped**.
The failure was a legacy RAM-residency assertion after preceding tests caused
the RAM budget to be clamped to 0.02 GiB. Without relaxing its assertion, the
entire affected file then returned **4 passed, 1 skipped** in a fresh process.
The final targeted contracts returned **25 passed**. Thus 54 distinct tests
passed across these runs; the initial full-suite invocation was not all green.
The remaining zero-disk-reread test requires full residency unavailable under
the effective budgets.

[Lifecycle evidence](evidence/2026-11-13/lifecycle-validation.json),
[12 release probes](evidence/2026-11-13/release-validation.json),
[final targeted/CLI checks](evidence/2026-11-13/final-checks.json).
After the timing series, bounded router diagnostics, Dense error handling and
CLI provenance were hardened; final correctness/lifecycle checks were rerun.

## Native CUDA: a separate result

The installed original Colibrì CUDA DLL initialized the RTX 3060 Ti and
performed a 32×32 FP32 matrix multiplication with maximum absolute error 0.
This was an isolated ABI probe, not full native model inference or Torch
pointer/stream interoperability.

The current original `coli plan` sees the GPU but assigns no expert VRAM tier
to this Dense checkpoint: its expert bytes and recognized int8 trunk bytes
are both zero, yielding a zero placement budget. That explains this current
plan; the exact environment of an older CPU-only run was not reconstructed.

## Quality boundary

Q4/off–on parity is **not** Q4/BF16 quality acceptance. Earlier g64/g32/g16
experiments failed the strict acceptance gate. The benchmark now enforces
at least 300 confident positions; thresholds were not relaxed. A new complete
Q4 quality/held-out evaluation was not performed in this integration pass.
BF16 remains the default. Q4 is an explicitly selected experiment.

## Research direction

The observed benefit is retention of reusable expert weights, not reduced
GPU transfers. The next useful experiments are a longer, diverse route trace,
cache-size sweeps and measured physical I/O. Extending admission to real GPU
residency requires a byte/lifetime reservation and acknowledgement mechanism,
including concurrent consumer ownership and cancellation semantics.

Dense state streaming and MoE expert selection have different constraints;
one generic 'component count' is not a memory plan. A model-specific native
executor seam must prove layout, pointer ownership and CUDA event ordering
before passing raw tensors across engines. See the [roadmap](ROADMAP.md).

This public evidence bundle contains measured summaries. Large checkpoints,
Q4 payloads, raw logits, host-specific logs and private chat/research archives
are deliberately not shipped. The original local records remain preserved.
