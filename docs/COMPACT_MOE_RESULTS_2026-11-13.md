# Compact MoE slots — 13 November 2026

**Experimental; GPU expert retention remains off by default.** This experiment changes physical
slot layout in the existing shared pinned pool and GPU Ring. It does not introduce a weight reader,
replace HF math, or connect the full native Colibrì model executor.

The previous [GPU-cache report](GPU_EXPERT_CACHE_RESULTS_2026-11-13.md) used a union template:
every slot reserved space for both Dense and expert tensors. The new `compact` layout reserves
slot 0 for the Dense trunk and the remaining slots for experts. `union` remains available for
comparison/rollback. Both arenas still use one allocation each, with validated, disjoint byte offsets.

## Physical reservations

For the tested OLMoE-1B-7B BF16 checkpoint:

| Configuration | Historical union | Compact | Reduction in each arena |
|---|---:|---:|---:|
| No retained GPU experts: Dense + scratch expert | 92 MiB | 46 MiB | 46 MiB |
| Two retained GPU experts: four slots total | 184 MiB | 70 MiB | 114 MiB |

Compact geometry is **34 MiB Dense + 12 MiB per expert slot**. The four-slot layout is
34 + 12 + 12 + 12 MiB; optional retention adds 24 MiB GPU and 24 MiB pinned RAM.
These are weight-arena reservations, not total process memory. Globals, activations, KV,
allocator reservations and native/runtime allocations remain additional memory.

The shared MoE CLI defaults to `--moe-slot-layout compact`; `--gpu-expert-slots` still defaults
to zero. `plan` and runtime use the same geometry builder. Wrong-role, missing or mismatched
payloads are checked against the actual destination slot before any write/DMA.

## Correctness and memory

| Workload / mode | Forwards | Last-position logits checked | Torch peak allocated | GPU cache hits |
|---|---:|---:|---:|---:|
| Short / off | 6 | 301,824 | 450.25 MiB | disabled |
| Short / apply, two retained slots | 6 | 301,824 | 474.25 MiB | 4 / 1,180 |
| Extended / off | 17 | 855,168 | 476.06 MiB | disabled |
| Extended / shadow, two retained slots | 17 | 855,168 | 500.06 MiB | 0 predicted |
| Extended / apply, two retained slots | 17 | 855,168 | 500.06 MiB | 0 / 2,950 |

All listed candidates matched the independent direct-safetensors reference **bit-for-bit** on
last-position logits, router selections and raw KV-state hashes. The comparison now compares
tensor bytes, including signed zero, and rejects known checkpoint metadata / shard-stat /
HF-source mismatches. The loader/materialization reference is independent; mathematical
equations still come from the same installed Transformers implementation.

Extended workload: **128-token prefill + 16 fixed teacher-forced inputs**, 17 forwards, final
context 144 tokens. KV bytes grew from **16 to 18 MiB**. This is a state-growth integration
test, not long-context, free-generation quality or soak certification. Reference ran all
272 layer calls and reported 228,657,991,680 logical input bytes.

On the short workload, cache apply excluded the same four expert transfers / **48 MiB H2D**
as the prior union experiment. The historical Torch peaks were 496.25 MiB (off) and 588.25 MiB
(apply); their compact counterparts fell by 46 and 114 MiB respectively.

On the extended workload, all three modes transferred **37,119,590,400 expert bytes** and
**9,202,565,120 Dense bytes**. Apply admitted the first two entries and had no subsequent hits;
the extra 24 MiB arena reservation bought no transfer reduction on that trace. This does not
establish the cause or general usefulness of the policy. No paired timing/acceleration result
is claimed. Counts are logical transfers, not a hardware PCIe/SSD measurement.

All extended candidates ended with zero active leases/violations and 3,222 observed DMA and
compute completions. Post-close Torch allocated memory was 8.125 MiB; this is not a claim that
the whole Python process returned all RAM to the OS.

## Validation

- **82 targeted tests passed** in 15.08 s: mixed slot geometry, CPU/CUDA source lifetime,
  wrong-role rejection before writes, native admission/eviction, delayed consumers across
  streams, uniform Ring compatibility, comparison identity and resource-plan boundaries.
- 76 Python files compiled; `git diff --check` passed at that checkpoint.
- Normal CLI: two steps passed with compact selected by default, two retained experts,
  64 MiB native RAM cache and a 1 GiB VRAM budget. Startup VRAM estimate was 0.9522 GiB;
  measured Torch peak allocated was 471.72 MiB.

The later original-topology audit changed documentation only. Its findings are recorded in
[the separate source maps](ORIGINALS_TOPOLOGY_RU.md); no new scheduler behavior was added during
that audit.

## Preserved incomplete attempts

1. Extended reference v1 was **SAFE_SKIP** below the unchanged 2.8 GiB startup available-RAM gate.
2. Extended reference v2 stopped after four completed forwards. The shell reported exit 1;
   the original Python exit code was not captured. No Python traceback or matching Windows
   application fault event was found. **Cause unknown.** Its raw `STARTED` report and four
   completed states are retained; it is not counted as PASS.
3. Extended reference v3 completed all 17 forwards with a captured Python exit code 0. A
   successful rerun does not explain or resolve v2's process termination.

This event is not relabelled as the earlier diagnosed mmap failure. Existing pread selection
and memory guards were preserved. No unrelated processes were terminated.

## Evidence and remaining work

Portable evidence: [compact-moe.json](evidence/2026-11-13/compact-moe.json), including source
hashes, identities, comparisons, per-step state/memory, actual geometry, CLI and prior attempts.
Raw local reports, logits and logs remain in `data/shared-moe-20261113/` and are excluded from Git.
Source-weight metadata/stat agreement is not a full content rehash. Historical short-reference
metadata that was not recorded remains unknown instead of being invented.

Still open: policy event/cadence equivalence, capacity sensitivity, measured full memory and
physical I/O, common live budgets with growing state, long-run stability, native executor/state
migration and the separate Q4 quality failure. See the [originals topology](ORIGINALS_TOPOLOGY_RU.md)
before extending the integration.
