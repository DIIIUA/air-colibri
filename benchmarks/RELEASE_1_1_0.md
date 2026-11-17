# AirColibri 1.1.0 validation — 2026-11-18

This release stabilizes the default supported HF/GGUF inference paths and packages
optional memory/thought experiments with explicit boundaries. It does not certify
all models, arbitrary model size, reasoning quality or every combination of features.

## Regression checks

The full local suite and targeted reruns after test-fixture corrections cover 181
unique tests: **171 passed, 10 skipped, zero unresolved failures**. The skipped tests
need the compatible native Qwen3.5-9B fixture, absent from the moved model folder.
See `test_results_20261118.json` for the latest outcome of each test. This is a union
of runs, not a claim that the first full run was green. An earlier run also hit a
Windows pytest temporary-directory cleanup error; subsequent runs use an isolated
temporary directory. `uv pip check` found no incompatible installed dependencies.

A separate full run with CUDA hidden passed **78 tests and skipped 103 GPU/local-model
checks**. Preflight confirmed no CUDA devices. On this Windows host, an earlier empty
CUDA mask did not produce a consistent CPU-only environment; the corrected run used
`CUDA_VISIBLE_DEVICES=-1`. See `cpu_test_results_20261118.json`. The CPU workflow uses
the same mask and a CPU PyTorch wheel.

A real Gemma-3-1B GGUF regression used 712 prompt tokens with a sliding window of
512 and a three-token auxiliary probe. Before the fix the audit observed state
corruption; after the fix **zero of 22 sliding layers changed and next-logit max
absolute difference was 0.0**, with alpha=0 isolating rollback from logit steering.
Tiny real Transformer implementations of Mamba, Mamba2, LFM2 and Gemma2 also match
uninterrupted next-token logits exactly after repeated probe rollback.

Additional checks cover context rejection before cache unpark, HTTP context errors
and subsequent server recovery, ring ownership/cancellation, dense/expert Q4/Q6,
retained initial probabilities, probes/retries and dynamic memory together.

## Benchmark protocol

Windows x64; Ryzen 5 5600X, 16 GiB system RAM, RTX 3060 Ti 8 GiB; driver 617.42,
CUDA 12.4, PyTorch 2.6.0+cu124, six PyTorch CPU threads. Models and project are on
D: (ST1000DM010 hard disk). GPU power limit stays at 200 W. One worker runs at a time.

Each configuration starts a new process, performs one unscored warm-up and three
48-token greedy requests with the same prompt and no session reuse. MTP is off.
EOS stopping is disabled for timing; two separate short answer probes use EOS.
Context budget is 2048, prefill chunks 64. Timed requests contain no per-layer
observers, full-logit CPU copies or KV hashing. Constructor time includes loading
and packing; it excludes process/import time. First visible delta is a UI timing,
not necessarily the first internally generated token. Decode rate excludes prefill.

OS file cache is not flushed. Process read-byte counters include cached reads and
are not physical HDD traffic. These experiments do not isolate HDD against SSD or
measure cold-cache storage throughput. Repeated generation of resident small models
and streamed/cache-missing workloads must be interpreted separately.

The granite-ring and granite-dynamic modes are **controlled diagnostic placements**:
18 dense layers resident, six streamed, three logical slots and a 700-expert starting
cache budget. Both use the same planning override and 256 MiB host-layer cache.
The dynamic mode reuses the combined ring/expert allocation and adds prefill-map
warmup. Anchor experiments apply the same optional probability anchor on both sides.
These placements are not default CLI speeds. The injected planner snapshot retains
`slot_bytes=0` from the original full-residency plan; the actual separate ring allocated
three 2 MiB slots. Dynamic memory derives its budget from the actual ring and expert
buffers (596.625 MiB budget in these runs), not that inherited descriptive field. Changes to output from the anchor are
intentional; its correctness/quality is not established by runtime tests.

Raw cases include the source-tree hash.
The source manifest includes the measured Windows worktree hashes and hashes with
CRLF normalized to LF, so Git source archives can be checked without treating line
ending conversion as a code change.

## Measured results

Rates are medians of three requests; ranges show all three samples.

| Model | Runtime | Placement | Anchor | Decode tok/s: median (range) | Prefill ms | Constructor s |
|---|---|---|---|---:|---:|---:|
| Llama 3.2 1B GGUF | Q4 | auto | off | 26.94 (25.58-28.32) | 368.52 | 12.45 |
| Llama 3.2 1B GGUF | Q6 | auto | off | 22.17 (21.40-25.83) | 851.47 | 11.31 |
| Qwen3.5 0.8B GGUF | Q4 | auto | off | 19.24 (19.07-19.26) | 313.73 | 23.66 |
| Qwen3.5 0.8B GGUF | Q6 | auto | off | 25.12 (19.64-25.82) | 553.33 | 19.66 |
| Granite MoE 1B HF | Q4 | auto | off | 15.84 (13.61-17.94) | 1763.32 | 9.97 |
| Granite MoE 1B HF | Q6 | auto | off | 15.99 (15.36-16.50) | 3365.12 | 45.49 |
| Granite MoE 1B HF | Q4 | ring | off | 10.43 (10.05-10.52) | 2329.18 | 18.49 |
| Granite MoE 1B HF | Q4 | dynamic | off | 7.46 (5.60-8.69) | 3699.97 | 11.87 |
| Granite MoE 1B HF | Q4 | ring | on | 13.10 (12.94-13.83) | 2710.27 | 13.40 |
| Granite MoE 1B HF | Q4 | dynamic | on | 6.85 (6.84-7.51) | 4170.18 | 11.23 |

All ten configurations completed, with 30 measured requests plus warm-ups and separate answer probes. Repeated text was identical within every configuration. Ring/dynamic output hashes match within each matched anchor setting; this is text parity, not a full-logit comparison. All six automatic configurations and the two unanchored ring/dynamic controls answered the short probes with 42 and Tokyo. **Both anchor-on configurations answered the Japan-capital probe with `N/A` instead of Tokyo**, while retaining 42 on arithmetic. This is a concrete quality regression of the experimental anchor; its output is not certified and it stays disabled by default.

Dynamic counters showed all three physical areas participating in changing roles, zero C payload/reserve, at most two dense payloads and no warm-bank-to-cache copies. This is consistent with the separate correctness tests; it is not continuous per-layer observation in these timed runs.

**No dynamic-memory speedup is claimed.** At the matched 700-expert budget the dynamic mode was slower than the ring in both anchor settings. Anchor-on and anchor-off produce different continuations, so their rates must not be interpreted as a controlled measurement of anchor overhead.

The separate 400-expert stress attempt was stopped during its first long warm-up, before any measured trials completed. Sampling showed repeated expert loading/packing. Its partial record is included separately in the JSON and excluded from the table; it is not a completed benchmark.

For Granite Q6 auto, constructor time was 45.49 s and the first unscored request 66.87 s; the three measured requests took 6.22-6.50 s including prefill. Granite auto reported zero process file-read bytes during measured requests. The ring at 700 experts read about 675 MiB per request, while dynamic plain read about 639-849 MiB. These are logical process reads, not physical disk reads. Warm residency, cache misses, packing and startup have very different costs; HDD impact cannot be inferred from model parameter count alone.

The host was running its normal applications; available RAM is recorded per case. This is a small Windows acceptance benchmark, not an exclusive-machine comparison or proof of general throughput gains. Native Qwen3.5-9B/MTP speed figures in the README remain historical and were not remeasured here.

Raw cases: [results_20261118.json](results_20261118.json). Code fingerprints: [source_manifest_1_1_0.json](source_manifest_1_1_0.json).
