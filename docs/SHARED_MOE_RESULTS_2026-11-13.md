# Shared OLMoE Dense/expert resources — 13 November 2026

This is the first-stage snapshot. A later [bounded GPU expert cache experiment](GPU_EXPERT_CACHE_RESULTS_2026-11-13.md)
adds optional retention and measures excluded H2D; the original results below
remain the history of the two-slot implementation.

**Experimental integration.** The default OLMoE CLI path now streams the Dense
weights of each decoder layer and its selected experts through the existing
AirColibri pinned pool, GPU Ring and CUDA lease acknowledgement adapter.
Embedding, final norm and output head remain resident. This is the first
bounded shared execution path, not a global adaptive placement planner.

## Mechanism and scope

- Original AirLLM `load_layer_subset` still supplies the checkpoint tensors.
- One 92 MiB pinned arena and one 92 MiB GPU arena each contain two 46 MiB slots.
  Slot 0 holds the current layer's Dense parameters. Slot 1 holds the current
  expert's fused gate/up and down projections; gate/up packing happens on CPU.
- Original Transformers OLMoE attention, router, eager expert forward and cache
  code remain unchanged. Layer parameters return to meta after their consumers.
- Every submission has a resource identity and monotonically increasing ticket.
  Replacing an active lease or releasing a stale lease is rejected. Original
  CUDA transfer/consumer events protect both GPU reuse and pinned-source reuse.
- Colibrì's original C tier policy still controls **expert RAM admission**.
  Dense host staging is transient. Globals, activations and KV state are outside
  these weight arenas; the startup planner estimates their budgets separately.

The low-level adapter preserves `stream_dense=False` for existing callers.
The CLI defaults to `--moe-placement streamed`; `resident` selects the earlier
Dense-resident/expert-copy adapter for comparisons. Sequential/enforce remains
the only supported MoE scheduling/control combination.

## Independent reference and workload

`scripts/reference_olmoe.py` imports no AirColibri reader, cache, Ring or inference
adapter. It reads safetensors directly and materializes **all 64 experts** of
each layer in the original HF modules. It unloads that layer after consumption.
Thus the loading, materialization and indexing path is independent; both sides
deliberately use the same installed Transformers mathematical implementation.
This is not a second independent implementation of the model equations.

The real checkpoint is OLMoE-1B-7B-0125-Instruct, BF16, 16 layers. Each case runs
in a fresh process, with two raw prompts (`Hello` and `The answer is 42. Код: x = 1`),
then two fixed input tokens, 13 and 42, after each prompt. Initial lengths are
1 and 11 tokens; the longest final context is **13 tokens**. Six last-position
logit vectors contain **301,824 values**. This is neither long-context validation
nor a free-generation quality evaluation.

The reference completes 96 decoder-layer calls. Comparisons require exact input
IDs, logit shapes/dtypes/values, all recorded KV-state tensor hashes and router
choices. Incomplete or crashed cases cannot pass the comparison.

## Measured results

| Complete case | Torch peak allocated | Torch peak reserved | Reference parity |
|---|---:|---:|---|
| Independent all-expert HF reference | 1,204.46 MiB | 1,228 MiB | Reference |
| Historical resident-Dense adapter, successful diagnostic rerun | 940.42 MiB | 1,064 MiB | Exact logits, KV and routes |
| Shared Ring, policy off, pread | 496.25 MiB | 512 MiB | Exact logits, KV and routes |
| Shared Ring, native policy, pread | 496.25 MiB | 512 MiB | Exact logits, KV and routes |

The shared probes use a **0.75 GiB / 768 MiB Torch allocator limit**. The
resident adapter's permanent Dense weights alone exceed that limit; a full
GPU copy of the trunk cannot be hidden inside this test. The measured peak
allocation decreases by **47.23%** relative to the successful resident run.
This excludes driver/context/other-process memory and is not a claim about
total device use or about speed. The CLI retains its larger conservative
state/scratch reservation; the 0.75 GiB probe does not lower CLI safety gates.

Each shared case records **96 Dense and 1,180 expert leases**, 1,276 loads and
1,276 releases. All corresponding DMA and compute completion events are
acknowledged, with zero contract violations and no active slot at the end.
Logical H2D totals are 3,247,964,160 Dense bytes and 14,847,836,160 expert bytes.
Expert H2D is unchanged by RAM policy: there is still no hot GPU expert cache.
After close, CUDA allocated memory returns to the existing 8.125 MiB baseline.

The initial shared/native mmap run also passed with the same output and memory
results. It remains historical evidence; the shipped Windows default uses pread.
Portable result summaries and source identities are in
[shared-moe.json](evidence/2026-11-13/shared-moe.json).

## Native crashes found during verification

Two attempts terminated with Windows `0xc0000005` in `torch_cpu.dll`:

1. `resident-v1`, during the fourth forward. The successful `resident-v2` rerun
   does not erase or explain this failure.
2. `streamed-off-v1`, during the sixth forward; five completed observations were
   preserved. Python faulthandler identifies `torch.storage.__getitem__`, reached
   by `safetensors.get_tensor` inside original AirLLM `load_layer_subset`, before
   the tensor enters the Ring.

The native defect's underlying cause is **not established**. The failure path
is localized to mmap tensor construction. The Windows OLMoE adapter now selects
safetensors' existing **pread** backend, which reads individual requested
tensors without that mmap storage path. Both off/native full probes passed
after this change. This is a tested mitigation, not proof that every possible
native crash has been eliminated.

AirLLM's function bytecode is preserved: the bridge binds `safe_open` with the
chosen backend in a private per-reader namespace. It changes neither the
upstream file nor module globals nor other reader instances. The default
AirLLM bridge used by other adapters retains its previous behavior. On other
platforms OLMoE defaults to mmap; `--moe-read-backend` can select either backend.

The backend is documented in the
[safetensors 0.8 source](https://github.com/safetensors/safetensors/blob/v0.8.0/bindings/python/src/lib.rs).
A [related upstream Windows report](https://github.com/Comfy-Org/ComfyUI/issues/15424)
helped identify the alternative; resemblance of stacks does not establish that
the native root cause is identical.

## Verification and remaining work

The final local targeted run passed **42 tests in 8.64 seconds**, including
CPU/CUDA cases. A normal CLI invocation also passed two greedy steps with
native policy, a 64 MiB expert RAM cache, a 4 GiB RAM budget and a 1.2 GiB VRAM
budget. It selected the shared Ring and pread by default, reported completed
leases and returned to 8.125 MiB CUDA allocation after close.

Targeted tests cover original full HF graph parity on CPU/CUDA, prefill/decode
state, router choices, active/stale lease rejection, invalid payloads, error
cleanup, existing concurrent-stream fences and resource-plan budget rejection.
Backend-binding tests verify that only requested tensors are read, upstream
bytecode is reused and module globals are untouched. CI exercises synthetic
CPU tests; local real-checkpoint/CUDA evidence is separate.

No source weights or original research archives were changed. Raw runs, partial
crash artifacts and logs remain locally under `data/shared-moe-20261113`.
The initial reference lacks a pre-run checkpoint metadata snapshot; later
candidate runs record config/index/tokenizer hashes and shard sizes/mtimes.
No full source-weight rehash was performed in this pass.

Still open: longer pressure/soak and contexts, independent current Qwen
reference, physical SSD I/O and matched timing, dynamic GPU expert residency,
full RAM/pinned/VRAM/state reservation policy, asynchronous cancellation,
native Colibrì execution, progressive CPU/GPU migration and Q4 quality acceptance.
Earlier Q4 failures remain failures.
