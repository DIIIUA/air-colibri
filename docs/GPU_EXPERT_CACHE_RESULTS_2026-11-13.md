# Bounded GPU expert retention — 13 November 2026

**Experimental, disabled by default.** OLMoE can now retain one or two experts
in the existing shared GPU Ring. A hit skips the host weight supplier, CPU
gate/up packing and H2D submission. All three tested modes matched the existing
independent-loader reference exactly. The measured benefit is small: **4 hits
among 1,180 expert requests**, excluding **48 MiB H2D** at the cost of **92 MiB
extra GPU arena and 92 MiB extra pinned RAM**. No speed improvement is claimed.

## Mechanism

The default remains two slots: layer Dense trunk and streaming expert. Optional
`--gpu-expert-slots 1|2` adds dedicated expert slots to that same Ring, pinned
pool and lease ACK adapter. Original AirLLM reads and original HF OLMoE
mathematics remain in use; no new weight loader is introduced.

A separate `NativeTierPolicy` instance uses Colibrì's original C tier policy
for GPU proposals. RAM admission keeps its own policy and byte budget. Initial
empty-slot fills use the bridge's capacity check; replacement proposals invoke
the native selection mechanism. The GPU backend selects the execution mode:

- **shadow:** maintain a virtual resident set; physically stream every expert.
- **apply:** retain selected expert tensors; on a hit acquire a fresh consumer
  lease for the same bytes, without another DMA. Each reader waits for initial
  transfer completion and the preceding reader's CUDA completion event. Thus
  the last consumer fence also protects a subsequent overwrite.
- **ACK:** submitted admission and observed DMA completion are distinct. Only
  observed completion produces an applied admission ACK. An active victim,
  stale release, wrong content identity or invalid proposal is rejected.

Each expert needs 12 MiB, but the shared union layout reserves 46 MiB per slot.
Two retained experts therefore occupy 24 MiB of logical payload in 92 MiB of
extra reserved GPU storage. Matching pinned slots are also reserved. Shadow
allocates the same extra arenas, although it does not reuse their contents.
The startup planner accounts for full strides in both RAM and VRAM estimates.
Single-host-forward ownership is supported; arbitrary concurrent host mutation
is not a scheduling mode of this cache.

## Real-model comparison

Three fresh processes used the real OLMoE-1B-7B-0125-Instruct BF16 checkpoint,
native RAM admission with a 64 MiB cache, Windows pread and a **0.75 GiB Torch
allocator limit**. GPU cache modes were off, two-slot shadow and two-slot apply.
The existing independent direct-safetensors/HF reference was reused unchanged;
its loading/materialization is independent, its HF equations are shared.

Each process ran `Hello` and `The answer is 42. Код: x = 1`, followed by fixed
tokens 13 and 42 for each prompt: six forwards, at most 13 context tokens,
96 Dense layer calls and 1,180 selected expert calls. Comparisons require exact
inputs, **301,824 last-position logit values**, KV tensor hashes and router
choices. All comparisons passed with maximum absolute logit error zero.

| Observation | Off | Shadow, 2 slots | Apply, 2 slots |
|---|---:|---:|---:|
| Actual GPU cache hits | 0 | 0 | 4 |
| Shadow predicted hits | — | 4 | — |
| Ring H2D loads, Dense + experts | 1,276 | 1,276 | 1,272 |
| Consumer completions | 1,276 | 1,276 | 1,276 |
| Expert H2D bytes | 14,847,836,160 | 14,847,836,160 | 14,797,504,512 |
| Expert weight-reader calls | 1,170 | 1,170 | 1,170 |
| Applied GPU admission ACKs | 0 | 0 | 2 |
| GPU / pinned arena, each | 92 MiB | 184 MiB | 184 MiB |
| Torch peak allocated | 496.25 MiB | 588.25 MiB | 588.25 MiB |
| Torch peak reserved | 512 MiB | 604 MiB | 604 MiB |
| Allocated after close | 8.125 MiB | 8.125 MiB | 8.125 MiB |
| Contract violations / active slots at end | 0 / 0 | 0 / 0 | 0 / 0 |
| Reference logits / state / routes | Exact | Exact | Exact |

Dense H2D remained **3,247,964,160 bytes** in each case. Apply's four extra
consumer acquisitions account exactly for the four missing loads:
`loads + resident_acquires == computes`. All 1,272 DMA and 1,276 compute
completion events were observed in apply. Counters represent tensor-copy
submissions, not hardware PCIe or physical SSD measurements.

The actual trace filled expert IDs 17 and 20 and retained them throughout;
it did not cause eviction. Subsequent native policy calls rejected 1,174
replacement proposals. A separate synthetic test using the **real native DLL**
on CPU and CUDA heats a replacement, verifies initial rejection, then observes
physical eviction, exact computed values and two completed admission ACKs.
Cross-stream delayed-consumer eviction is covered separately with controlled
synthetic proposals. These scopes are deliberately distinct.

## Validation and provenance

- Main targeted suite: **57 passed in 11.12 s**, including full synthetic HF
  graph parity, CPU/CUDA prefill/decode state, router choices, existing Ring
  fences, malformed admission rejection and physical-completion ACK gating.
- Additional native CPU/CUDA eviction tests: **2 passed in 0.27 s**. These were
  added after the main suite; they are separate runs, not a claimed monolithic
  full-suite result. CPU CI skips CUDA and missing-native-library cases.
- Ordinary CLI: two greedy steps passed with explicit GPU apply, 64 MiB RAM
  cache, 4 GiB RAM / 1.2 GiB VRAM budgets; close returned to 8.125 MiB allocated.
- All three candidate runtime source hashes, checkpoint metadata and shard
  sizes/mtimes agree. The reference lacks a pre-run checkpoint metadata snapshot;
  no full weight-file rehash was performed. No source weights were changed.
- DLL checksum and ABI match its existing build manifest. Both policy instances
  use the same upstream Colibrì revision; the full C/CUDA executor is still absent.

Portable metrics, source hashes, native build identities, raw report hashes and
CLI details: [gpu-expert-cache.json](evidence/2026-11-13/gpu-expert-cache.json).
Local raw runs and logs are retained under `data/shared-moe-20261113`, named
`gpu-cache-off-v1`, `gpu-cache-shadow-v1`, `gpu-cache-apply-v1` and
`cli-gpu-cache-v1`. Earlier crashes and failed experiments remain unchanged.
Commands and the opt-in flags are in [developer setup](DEVELOPMENT.md).

## Decision and remaining work

The mechanism is connected and the missing H2D is measured, but a **0.339% hit
rate** on this short trace does not justify enabling it by default. It did not
reduce weight-reader calls: those four accesses would otherwise have hit RAM.
No paired timing experiment was conducted. Peak Torch allocation excludes
driver/context/other-process usage and is not total device memory.

Next work should assess compact expert storage and workload-dependent reuse
before reserving more slots. A general byte/lifetime planner, growing KV/state
ledger, pressure/soak tests, asynchronous cancellation and independent Qwen
coverage remain open. The pread mitigation passed these additional short runs;
it is not a diagnosis of the earlier mmap defect or long-run certification.
Native model execution, progressive CPU/GPU migration and Q4 quality remain
separate unfinished tracks. The prior Q4 quality failure remains a failure.
