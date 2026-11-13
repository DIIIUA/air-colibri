<p align="center">
  <img src="assets/aircolibri-logo.png" width="300" alt="AirColibri — AirLLM-inspired wings meet the Colibrì hummingbird">
</p>

# AirColibri 🦅⚡

<p>
  <a href="docs/ROADMAP.md"><img src="https://img.shields.io/badge/status-active_WIP-orange" alt="Active work in progress"></a>
  <a href="#-current-scope"><img src="https://img.shields.io/badge/stability-experimental-red" alt="Experimental"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Apache 2.0"></a>
  <a href="https://github.com/DIIIUA/air-colibri/actions/workflows/cpu-contract.yml"><img src="https://github.com/DIIIUA/air-colibri/actions/workflows/cpu-contract.yml/badge.svg" alt="Synthetic CPU contract checks"></a>
</p>

> [!WARNING]
> **🚧 Work in Progress — Early Experimental Stage**
>
> Active engineering and architectural experimentation are ongoing.
> **Do not expect production-ready stability, consistently good results or
> high speed.** Short local tests have passed; broad model support and
> performance claims have not. Q4 quality is still unresolved.
> `main` contains the current experimental snapshot; ongoing integration work
> also lives on [`feature/hybrid-engine-mvp`](https://github.com/DIIIUA/air-colibri/tree/feature/hybrid-engine-mvp).

> **Experimental low-memory LLM inference:** bringing together layer-wise
> streaming from [AirLLM](https://github.com/lyogavin/airllm), native cache
> admission from [Colibrì](https://github.com/JustVugg/colibri), and the original
> Transformers mathematical forward.

[**Current scope**](#-current-scope) · [**Original topology maps**](docs/ORIGINALS_TOPOLOGY_RU.md) · [**Experiments**](docs/EXPERIMENTS_2026-11-13.md) · [**Developer setup**](docs/DEVELOPMENT.md) · [**Roadmap**](docs/ROADMAP.md)

---

## 🎯 Motivation & Vision

The aim is to run models whose weights exceed the available memory budget
by preparing only what the next computation needs and retaining what is
likely to be reused.

* **AirLLM** provides the layer-streaming and selective weight-reading foundation.
* **Colibrì** provides native resource-management ideas and mechanisms for expert streaming.
* **AirColibri** connects selected original mechanisms through explicit buffer
  ownership, CUDA dependencies, byte budgets and placement acknowledgements.

The long-term target includes larger Dense and MoE models on consumer hardware.
**The integration has currently been tested on Qwen3.5-9B and OLMoE-1B-7B,
not 70B+ or hundreds-of-billions models.** Upstream model lists are not an
AirColibri support matrix.

## 🏗️ Architecture Synthesis

```text
Original AirLLM subset reader
          │
          ▼
Bounded RAM cache ◄──── original Colibrì C tier.h policy
          │                   │
          │              admission + ACK
          ▼
Dense: pinned slots / GPU Ring     MoE: shared Ring for trunk + experts
          │                                   │
          └──────── original Transformers forward ────────┘
                              │
                       logits + KV/state
```

The C bridge makes real RAM-cache admission decisions and can separately control
an optional bounded GPU expert cache. Computation still uses
**PyTorch/Transformers**. An isolated Colibrì CUDA-library probe passed, but
the full native Colibrì CUDA executor is **not** the integrated math backend.

The [separate original-program maps](docs/ORIGINALS_TOPOLOGY_RU.md) trace startup,
prefill/decode, memory ownership, prefetch, repeated requests and shutdown in
AirLLM and Colibrì. They distinguish our existing Ring from upstream AirLLM and
the native scoring function from its original callers' event/cadence rules.
These are source-backed maps of pinned revisions, not runtime certification of
all upstream engines. An offline [HTML atlas](docs/ORIGINALS_TOPOLOGY_RU.html)
contains the diagrams and source index.
The [functional overlay](docs/FUNCTIONAL_OVERLAY_RU.md) identifies shared
responsibilities and conflicting owners; the [integration plan](docs/INTEGRATION_INTERSECTIONS_PLAN_RU.md)
assigns contracts, temporal windows and acceptance gates. Graphify's extracted
edges are cross-checked against source, with known resolution errors preserved.

## 🔬 Current Scope

| Path | Implemented and locally checked | Boundary |
|---|---|---|
| Dense Qwen3.5-9B | Full forward, BF16 / experimental Q4, 1–3 ring slots, tail-to-head prefetch | Only the tested `qwen3_5_text` checkpoint/layout |
| OLMoE-1B-7B | Original router/forward; Dense trunk and selected experts share pinned slots, GPU Ring and lease ACKs; optional 1–2 retained experts | BF16, sequential placement; globals remain resident; GPU cache off by default |
| Native Colibrì C | ABI-v1 bridge to original `tier.h`; RAM admission and separate bounded GPU expert admission | Not native model math or a global placement planner |
| Lifecycle | CUDA fences, pre-write guards, byte budgets, placement ACKs, error/reopen checks | No integrated asynchronous prefetch cancellation |
| CLI | `doctor`, `plan`, `run`; policy off/shadow/apply; JSON report | Research/developer interface, not a packaged service |

**Not integrated:** native CUDA forward, global dynamic VRAM placement, MTP,
GGUF, Vulkan, arbitrary models, an API server or interactive chat service.

### Latest measured snapshot — 13 November 2026

* [Compact per-role slots](docs/COMPACT_MOE_RESULTS_2026-11-13.md) reduce the
  four-slot arena from **184 to 70 MiB** in each of GPU and pinned memory.
  A **128-token prefill + 16 fixed inputs** matched the independent reference
  on **855,168 last-position logits**, routes and KV hashes in off/shadow/apply.
  GPU retention had **0 hits / 2,950 requests** on that trace and added 24 MiB
  per arena. This is a correctness/memory result, with no acceleration claim.
  82 targeted local tests passed; one incomplete reference attempt is preserved
  with its cause unresolved. A 144-token context is not a long-context test.
* The first [GPU expert retention experiment](docs/GPU_EXPERT_CACHE_RESULTS_2026-11-13.md)
  passed off/shadow/apply reference parity. Apply excluded **4 H2D transfers /
  48 MiB** across 1,180 expert requests, while adding **92 MiB GPU + 92 MiB pinned
  RAM** reservation in the historical union layout. It stays **off by default**;
  acceleration is not established.
* The earlier [shared Dense/expert Ring](docs/SHARED_MOE_RESULTS_2026-11-13.md) matched an
  independent direct-safetensors/HF reference: all **301,824 logits**, router
  choices and KV-state hashes were exact across six forwards in two short contexts.
  Torch peak allocated memory was **496.25 MiB**, versus **940.42 MiB** in the
  resident-Dense adapter. This is an allocator-memory result, not a speed claim.
* Full logits and recurrent/KV state matched bit-for-bit between the tested
  off/observe/apply configurations. MoE router choices also matched the reference.
* In five paired MoE runs of eight steps, native admission reduced expert reads
  from **1,024 to 983 (4.00%)**. Median forward time was **18.004 → 17.569 s**.
* In that earlier RAM-only test, expert CPU→GPU bytes were unchanged. Its **2.42% local timing change**
  is not a general speed promise; OS file-cache state was uncontrolled.
* Q4/off–on correctness does **not** mean Q4/BF16 quality acceptance. BF16 is
  the default. A [960-position quality evaluation](docs/Q4_RESULTS_2026-11-13.md)
  confirmed bitwise BF16 repeatability; Q4-g64 **failed** the strict gate.

See the [full experiment report](docs/EXPERIMENTS_2026-11-13.md) for workloads,
test failures and reruns, source revisions, measurements and limitations.
The CI badge covers **synthetic CPU contracts only**, not these GPU/model runs.

---

## 🚀 Developer Setup & Roadmap

This is a source-based experiment. The verified environment is **Windows x64,
Python 3.12.11, PyTorch 2.6.0+cu124 and RTX 3060 Ti 8 GiB**. Other machines
and fresh installations are not yet end-to-end certified.

After preparing the environment, checkpoint and upstream checkout described in
[Developer setup](docs/DEVELOPMENT.md):

```powershell
.\.venv\Scripts\python.exe aircolibri.py doctor --airllm-source "D:\path\to\airllm"
.\.venv\Scripts\python.exe aircolibri.py plan --model olmoe --model-dir "D:\models\OLMoE-1B-7B-0125-Instruct"
.\.venv\Scripts\python.exe aircolibri.py run --model olmoe --model-dir "D:\models\OLMoE-1B-7B-0125-Instruct" --airllm-source "D:\path\to\airllm" --policy off --prompt "Hello" --max-new-tokens 8
```

Use `--policy apply` after building the native tier bridge. Budgets default to
`--ram-gib 4 --vram-gib 5 --cache-mib 256`. Startup checks RAM, VRAM and Windows
commit headroom; runtime RAM checks occur at layer boundaries. These are not
an OS-level hard memory quota. Results are written to `data/latest-run.json`
or the explicit `--report` path.

### Milestones

- [x] Connect original AirLLM reads, full Transformers forward and native C admission.
- [x] Verify full Dense 9B and MoE 7B logits, state and actual routing.
- [x] Add a single experimental CLI, budgets, lifecycle guards and repeatable probes.
- [ ] Pass strict Q4 quality acceptance on a sufficient held-out set.
- [x] Stream the Dense trunk and selected experts of OLMoE through a shared Ring and lease ACKs.
- [x] Add optional bounded GPU expert retention with shadow/apply and completion ACKs.
- [ ] Integrate a general byte/lifetime planner for GPU residency and cancellation.
- [ ] Connect and verify the complete native Colibrì executor where compatible.
- [ ] Measure cold/warm I/O, startup latency, long runs and standalone baselines.
- [ ] Validate larger checkpoints and an independent setup.

The default MoE CLI path streams layer Dense weights and experts through two
shared slots, with compact role-specific sizes. Embeddings, final norm and output head remain resident. The
historical adapter is available with `--moe-placement resident` for comparisons.
`--gpu-expert-slots 1|2` additionally enables bounded expert retention through
the same Ring; the default is zero. A separate native policy proposes GPU
admissions, and `--gpu-cache-mode shadow|apply` selects virtual or physical
retention. Compact slots reserve 12 MiB per retained expert on the tested
checkpoint; `--moe-slot-layout union` restores the historical 46 MiB slots.
This is still an experiment with a small cache capacity.
Progressive CPU/GPU migration and phase-based
placement remain research work. The [roadmap](docs/ROADMAP.md) records these
original goals separately from the integration already tested.

[Detailed P0–P9 roadmap](docs/ROADMAP.md).

## 📚 Research & Reproducibility

* [Bounded GPU expert cache: correctness, actual H2D reduction and memory cost](docs/GPU_EXPERT_CACHE_RESULTS_2026-11-13.md).
* [Integration experiment report and machine-readable evidence](docs/EXPERIMENTS_2026-11-13.md).
* [Follow-up CUDA ordering fixes and 3,072 slot reuses](docs/TESTING_2026-11-13.md).
* [Strict Q4/BF16 quality protocol](docs/QUALITY_EVALUATION.md) and [completed results](docs/Q4_RESULTS_2026-11-13.md) — Q4-g64 did not pass acceptance.
* [Developer setup, layouts and validation commands](docs/DEVELOPMENT.md).
* [Original architecture audit](docs/audit_and_design.md) — historical design context.
* [Current engineering strategy](Strategy.md).

Earlier demo engines and benchmark scripts are retained as research history.
Their simplified-graph timings are **not full-model inference throughput**.
Checkpoints, generated weight caches, raw chat archives and machine-local
working material are not part of the public repository.

## 📜 License & Acknowledgments

* **AirLLM:** Gavin Li and contributors — [original project](https://github.com/lyogavin/airllm) · [working fork](https://github.com/DIIIUA/airllm).
* **Colibrì:** Vincenzo Fornaro / JustVugg and contributors — [original project](https://github.com/JustVugg/colibri) · [working fork](https://github.com/DIIIUA/colibri).
* **Transformers / PyTorch:** the original model graph and tensor runtime used by this integration.

AirColibri is an independent experimental integration, licensed under
[Apache-2.0](LICENSE). The combined project mark draws on the original
AirLLM rocket and Colibrì hummingbird; [asset sources and attribution](assets/README.md).
Upstream names and marks do not imply endorsement. See [NOTICE](NOTICE).
