# AirColibri 1.2.0 validation — 2026-11-18

This release brings independent CPU execution, common CPU/GPU preparation, bounded
streamed speculation, more conservative startup VRAM accounting and model-management
commands into the supported runtime. Optional probability/memory experiments retain
their narrower scope. Model support is checkpoint-specific, not universal.

## Regression and model coverage

The final GPU-available and CUDA-hidden test outcomes are recorded per test in
[test_results_1_2_0.json](test_results_1_2_0.json). Skipped local-weight fixtures
are listed as skips, not model passes. CPU masking uses `CUDA_VISIBLE_DEVICES=-1`.
The GPU-available full run passed **261 tests, 10 skipped**. The first CPU-only
run exposed an incomplete mocked-GPU fixture: it claimed CUDA was present while
leaving device-property lookup real. The UUID property was stubbed consistently,
then the entire CPU configuration was rerun. Production probing was not weakened.
The final CUDA-hidden full run passed **142 tests, 130 skipped** with no failures;
it includes the newly added census-controller regression. The different skip counts
reflect GPU availability, not hundreds of additional model launches.

The fresh [46-checkpoint census](model_support_1_2_0.json) and
[support table](../MODEL_SUPPORT.md) record every attempted local roster entry,
including failures and bounded stops. Each pass means finite raw logits, two
chat tokens and one session-continuation token, ordinary Q6 execution, context 512,
prefill chunk 8. A 600-second per-model watchdog bounds the experiment. Failed
adaptation is distinct from failed execution and timeout. These few tokens are
insufficient for a model-quality or throughput claim.

Final census: **31 passed, 15 failed**, no actual per-model timeout. Source file
size/mtime stayed unchanged for all 46 attempts. During receipt review, a controller
bug was found: after a long batch completed, its watchdog overwrote the first
already-complete Llama pass with `bounded_stop`. Complete token/cleanup metrics and
the worker's pass log remained intact. The controller was fixed, its long-batch
regression passed, and that record was restored with an explicit recovery note;
the original incorrectly labelled report is retained locally. The regression was
added after the GPU suite collected tests and is also included in the final CPU run.
The [source manifest](source_manifest_1_2_0.json) distinguishes the census snapshot
from final packaging: one trailing empty line was removed from the CPU executor,
with identical Python AST. The exact measured source is retained locally.

Deleted DeepSeek9B and OLMoE7B checkpoints were excluded rather than redownloaded.
The large uncensored Qwen stays in the roster. The separate fresh SmolLM2-135M HF
download, automatic preparation, CPU answer and offline reuse are described in
[CLI validation](CLI_MODELS_20261118.md), with its
[download receipt](cli_download_1_2_0.json). It is additional to the 46-entry census.
The 25.62 GB Qwen35B MoE completed in 199 seconds; its 40 dense trunk layers were
resident while routed experts remained separately demand-loaded. This is not a
25 GB dense-model residency measurement.

## Shared execution and optional mechanisms

CPU and GPU preparation are released by the same event. Ready devices join eligible
work on one graph; there is no forced CPU-first layer or prompt replay at adoption.
`--device cpu` keeps execution on CPU; `--no-progressive-start` selects the GPU-first
reference. Preparation failure can return partially adopted layers to CPU at a
safe boundary while retaining state and a visible diagnostic. Layer caches, staging
and GPU queues remain bounded; neither path requires a second full backbone.

The common speculative loop can reuse each streamed layer across verification rows.
Native Qwen3.5 text MTP shares embeddings/output projection and keeps only its
bounded trained head on CPU. Prompt lookup is explicitly a separate fallback.
Rejected attention/recurrent rows are rolled back; unknown row/state contracts
continue with ordinary one-token decode. The optional snapshot estimate limits
speculation depth, not the permitted user context.

The [development integration receipt](joint_streaming_20261118.json) records:

- Real Qwen0.8 native MTP: the same 12 generated tokens as control, 4/9 drafts
  accepted, decode backbone passes 10 → 6, bounded cache with real evictions.
- Real Granite1B lookup with shared dynamic memory, map warming and probability
  anchor: the same 12 tokens, 5/7 drafts accepted, passes 10 → 5. The expert arena
  held 95 of 768 experts; all three physical areas consumed prepared data, standby
  owned zero payload/reserve, and the original probability distribution/map remained.
- Scoped route exclusions omit only exactly zero output-weighted HF Granite/OLMoE
  contributions before reads. Zero-route read prevention is verified synthetically;
  the real Granite sample did not contain a zero coefficient. Nonzero actual routes
  remain required even if a forecast missed them.

These are narrow correctness checks. They do not establish a general speedup,
semantic memory, every intended physical A/B/C transition or all-family integration.
The full [MTP report](MTP_STREAMING_20261118.md) gives arithmetic, state and memory
boundaries. Llama prefill wavefront is checked separately with speculation off.

The thought buffer retains the original full FP32 distribution separately from the
working field. Bounded prior influence and explicit corrections address the earlier
Granite anchor regression; the [paired development checks](DEVELOPMENT_20261118.md)
are not a general reasoning-quality certificate. It stays off by default. The
dynamic arena also stays opt-in: matched small-model measurements remain slower
than the ordinary ring.

## Decode measurements and startup memory

The new [resident/streaming comparison](STREAMING_DECODE_20261118.md) uses Windows,
Ryzen5600X/16 GiB, RTX3060Ti8 GiB, driver617.42 and unchanged 200 W limit. Source
weights are Llama3.2-1B GGUF Q4_K_M; all cases execute runtime Q6 with MTP/anchor off.
Each layout has four 64-token measurements in reversed block order, then the full
experiment repeats with a separate idle 2 GiB allocation. One inference worker
runs at a time; OS caches are not flushed.

| Resident decoder layers | No added allocation | Separate 2 GiB allocation |
|---|---:|---:|
| 16/16 | 30.55 tok/s | 31.64 tok/s |
| 8/16 | 11.98 tok/s | 11.92 tok/s |
| 0/16, ring | 7.34 tok/s | 7.34 tok/s |

All 24 output hashes matched, with zero host layer-cache misses during timing.
These are forced placements and warm RAM→GPU streaming, not cold-disk throughput
or a remeasurement of the historical Qwen9B ~31 tok/s result. Resident sample ranges
overlap across the pressure settings; no gain from pressure is claimed.

An initial diagnostic run found that this Windows CUDA free counter barely changed
for another process's allocation. The fix queries the same GPU by UUID through NVML
and uses the lower valid CUDA/NVML reading. In the corrected experiment median
effective free VRAM changed 6.542 → 4.160 GiB, and every plan received that reading.
Status exposes optional NVML failure rather than concealing it. Startup sampling
does not implement continuous global rebalancing after later surprise allocations.

## Remaining limits

No 90–350 GB checkpoint was tested. Size is not an exclusion rule, but active dense
operators and the output head still need their planned allocations. Arbitrary cube
tiling, general fragmentation recovery, universal trained MTP, arbitrary dense
probability pruning and a common KV/SSM arena are not implemented by this release.
Quantization is lossy; placement/token parity is not BF16 equivalence. Multimodal
fixtures cover language only. The source archive contains program/tests/evidence,
without model weights, local caches or private chat histories.
