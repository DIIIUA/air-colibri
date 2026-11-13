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

[**Current scope**](#-current-scope) · [**Experiments**](docs/EXPERIMENTS_2026-11-13.md) · [**Developer setup**](docs/DEVELOPMENT.md) · [**Roadmap**](docs/ROADMAP.md)

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
Dense: pinned slots / GPU Ring     MoE: selected expert weights
          │                                   │
          └──────── original Transformers forward ────────┘
                              │
                       logits + KV/state
```

The C bridge makes real RAM-cache admission decisions. Computation still uses
**PyTorch/Transformers**. An isolated Colibrì CUDA-library probe passed, but
the full native Colibrì CUDA executor is **not** the integrated math backend.

## 🔬 Current Scope

| Path | Implemented and locally checked | Boundary |
|---|---|---|
| Dense Qwen3.5-9B | Full forward, BF16 / experimental Q4, 1–3 ring slots, tail-to-head prefetch | Only the tested `qwen3_5_text` checkpoint/layout |
| OLMoE-1B-7B | Original router and eager expert forward; selective expert reads | BF16, sequential expert adapter |
| Native Colibrì C | ABI-v1 bridge to original `tier.h`; bounded global RAM admission | Not native model math or general GPU residency |
| Lifecycle | CUDA fences, pre-write guards, byte budgets, placement ACKs, error/reopen checks | No integrated asynchronous prefetch cancellation |
| CLI | `doctor`, `plan`, `run`; policy off/shadow/apply; JSON report | Research/developer interface, not a packaged service |

**Not integrated:** native CUDA forward, global dynamic VRAM placement, MTP,
GGUF, Vulkan, arbitrary models, an API server or interactive chat service.

### Latest measured snapshot — 13 November 2026

* Full logits and recurrent/KV state matched bit-for-bit between the tested
  off/observe/apply configurations. MoE router choices also matched the reference.
* In five paired MoE runs of eight steps, native admission reduced expert reads
  from **1,024 to 983 (4.00%)**. Median forward time was **18.004 → 17.569 s**.
* Expert CPU→GPU bytes were unchanged. This small **2.42% local timing change**
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
- [ ] Stream the Dense trunk and selected experts of one MoE through shared resource ownership.
- [ ] Integrate a general byte/lifetime planner for GPU residency and cancellation.
- [ ] Connect and verify the complete native Colibrì executor where compatible.
- [ ] Measure cold/warm I/O, startup latency, long runs and standalone baselines.
- [ ] Validate larger checkpoints and an independent setup.

The current MoE adapter keeps non-expert weights resident on GPU and copies
experts on demand. Native admission controls RAM retention; it does not yet
provide a hot-expert GPU cache. Progressive CPU/GPU migration and phase-based
placement remain research work. The [roadmap](docs/ROADMAP.md) records these
original goals separately from the integration already tested.

[Detailed P0–P9 roadmap](docs/ROADMAP.md).

## 📚 Research & Reproducibility

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
