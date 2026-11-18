<p align="center">
  <img src="assets/aircolibri-logo.png" width="300" alt="AirColibri — AirLLM-inspired wings meet the Colibrì hummingbird">
</p>

# AirColibri

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Apache 2.0"></a>
</p>

Local inference of supported **Hugging Face and GGUF language models** with bounded layer
streaming, BF16 or packed Q4/Q6 execution and an OpenAI-compatible LAN server. The
CPU executor can run on its own; an NVIDIA GPU can join when ready.

**Version 1.2.0 does not support every model.** See [model support](MODEL_SUPPORT.md) for dated
per-checkpoint evidence, unsupported cases and experimental mechanisms.

AirColibri combines ideas from [AirLLM](https://github.com/lyogavin/airllm) (layer streaming,
selective weight reads) and [Colibrì](https://github.com/JustVugg/colibri) (tiered placement,
native MTP speculation) with its own packed Q4/Q6 weights and NVRTC CUDA kernels.

<a href="https://buymeacoffee.com/silentfish">
  <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" width="217" height="60">
</a>


## Requirements

- Optional NVIDIA GPU with CUDA (tested: RTX 3060 Ti 8 GB; smaller budgets require streaming and sufficient room for individual operators). CPU-only execution is available with `--device cpu`.
- Python 3.12, PyTorch 2.6 with CUDA 12.4 (`requirements.txt`)
- Windows x64 or Linux

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv/Scripts/python -m pip install -r requirements.txt
```

## Model

Choose a tested checkpoint from [model support](MODEL_SUPPORT.md). Standard Hugging
Face shards (`model-0000X-of-0000N.safetensors` + `model.safetensors.index.json`)
are read directly. The common path packs eligible projections into Q4/Q6 on demand;
it needs no manual converter or second full checkpoint. The specialized Qwen3.5-9B
path builds its persistent Q4 cache in `data/q4-g64-cache/`.

```bash
aircolibri download https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct --device cpu
```

You can point `--model` at a local folder or place it in `models/`. The loader reads
single or sharded Hugging Face `.safetensors`, supported single/split `.gguf` files,
and Colibri per-layer containers. Original checkpoint files are not modified.
Selected MoE expert slices are read on demand rather than loading all experts.

The architecture registry contains adapters for dense attention, MoE and several
recurrent/hybrid families. An adapter or recognized architecture ID is not a guarantee
that every checkpoint of that family works. Multimodal adapters, where usable, expose
only the language portion; image/audio inference is not provided. Pickle `.bin`, arbitrary
remote model code and the unresolved checkpoints listed in the support matrix are outside
the validated release scope.

`--weight-format auto` retains the specialized Qwen3.5 Q4/MTP path when its geometry
matches; otherwise it selects universal Q6. Explicit `bf16`, `q4` and `q6` are available
on `ask`, `chat`, `serve` and `doctor`. Eligible projections use packed CUDA kernels;
routing, normalization, direct recurrent consumers and the generic LM head retain source
compute precision. Compatible SwiGLU experts use a bounded packed cache. Other expert
math keeps its original path. Q4/Q6 packing adds quantization error, including when the
source is already GGUF; BF16 execution of GGUF does not recover original unquantized weights.

Hugging Face FP8 block, AWQ GEMM INT4 and GPTQ INT4 projection adapters are included.
They read bounded source slices and convert eligible projections to the selected runtime
format. This does not establish support for every quantized checkpoint layout.
Self-contained NVRTC kernels handle supported DeltaNet, causal convolution and RMSNorm
operations without Triton. The specialized Qwen MTP head is not enabled on unrelated models.

## Usage

```bash
aircolibri doctor                      # GPU, model, VRAM plan, LAN addresses
aircolibri models                      # numbered local model catalog
aircolibri ask --model 1 "Hello"        # select the current catalog's first entry
aircolibri ask "Привет! Кто ты?"       # one answer, streamed
aircolibri chat                        # multi-turn chat (Ctrl+C stops the current answer)
aircolibri serve                       # web chat + OpenAI API on :8787, UDP beacon on :8788
aircolibri discover                    # find AirColibri servers on the LAN
```

On Windows use `aircolibri.cmd` (it picks `.venv` automatically) or `python aircolibri.py`.
In PowerShell from the program directory, for example: `.\aircolibri.cmd models`.

### Model downloads and preparation

`models` accepts `--json` and `--check`. The latter checks adaptation metadata;
it does not run generation or certify support. Choose with `--model NUMBER`, a
catalog folder name or a full path. Numbers refer to the current sorted catalog.
Plain listing and command help do not initialize PyTorch or CUDA.

```bash
aircolibri download https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct --directory "D:/Models/SmolLM2-135M" --device cpu --weight-format q6
aircolibri ask --model "D:/Models/SmolLM2-135M" --device cpu "What is 6 times 7?"
aircolibri prepare --model "D:/Models/SmolLM2-135M" --device cpu --weight-format q6
```

`download` defaults to `models/OWNER__REPO`, registers external destinations in the
local catalog, and performs a finite first-forward check including runtime packing.
Its `aircolibri-source.json` records `ready`, an unverified download or a failure;
downloaded files alone are not a support claim. `--no-verify` skips that forward.
An unsupported checkpoint reports the error and retains its source files.

Repository IDs, model-page URLs, revision pages and GGUF file URLs are accepted.
Use `--file path/model.gguf` or `OWNER/REPO::path/model.gguf` to choose a quantization;
numbered split parts are downloaded together. Repositories with several GGUF variants
require a selection. `--revision` selects a revision in a separate destination.
Completed compatible downloads can be reused offline. Partial Hub caches trigger
completion rather than being mistaken for runnable models. A nonempty unrelated
directory is never reused for a new download.

`ask`, `chat`, `serve` and `doctor` also accept `--download-dir DIRECTORY` with an
HF reference. Without it, explicit remote references retain the standard Hub-cache
behavior. Automatic preparation is validated on a fresh SmolLM2 HF download and
synthetic HF/GGUF selection cases; [CLI evidence](benchmarks/CLI_MODELS_20261118.md)
records the scope.

### VRAM modes

AirColibri measures the **free** VRAM at start and plans with it — there are no fixed card profiles.
When NVML is available it matches the CUDA device by UUID and uses the lower CUDA/NVML
free reading, so another process's allocation enters the startup budget. Status and
`doctor` expose the counters and any NVML fallback diagnostic. This samples startup
memory; it does not continuously rebalance after later desktop allocations.

| Mode | What it takes |
|---|---|
| `--vram-mode normal` (default) | most free VRAM, keeping a reserve for the OS, the KV/DeltaNet state of `--max-context` tokens and the prefill workspace; estimates do not guarantee every workload will fit |
| `--vram-mode turbo` | all free VRAM with a minimal reserve |

If the model fits, every layer stays resident. Otherwise the remaining budget is split into equal
slots that rotate roles — computing → clean → receiving the next layer's weights — so the next layer
is already on the GPU when the current one finishes, and as many layers as fit stay resident.

Requests are rejected before prefill/session unpark when prompt tokens plus requested output
exceed `--max-context`. Shorten the request or restart with a larger planned context; the
runtime does not silently truncate conversation history.

Prompts are prefilled in chunks (`--prefill-chunk`, default 512 tokens). Where the model
supports it, only the last position is projected onto its output vocabulary.

For Llama models, `--prefill-wavefront --prefill-chunk 64` enables experimental three-stage
prefill. The next layer consumes a completed prompt chunk while the previous layer works on
another chunk. Four chunks share each loaded layer; CUDA events protect activations, causal KV
updates and weight-slot reuse. The planner reserves extra memory for concurrent workspaces.
This works with BF16 and universal Q4/Q6, session reuse, cancellation and the optional thought
buffer. It is off by default and currently limited to unpadded Llama input. Autoregressive decode
retains its existing dependencies. Performance depends on weight format and hardware; compare
against normal chunked prefill and against the same wave scheduled on one CUDA stream.

### LAN server

| Endpoint | |
|---|---|
| `POST /v1/chat/completions` | OpenAI-compatible, JSON or `stream: true` SSE (`reasoning_content` for `<think>`) |
| `POST /v1/chat/stop` | stops the running generation immediately |
| `GET /v1/models`, `GET /api/info` | model list, context length (`GET /api/v0/models`, `/props`, `/api/extra/true_max_context_length`), node status |
| `GET /` | built-in web chat |

- One generation runs at a time. A new request — or a client that closes its connection, detected
  even during prefill — cancels the running one, and its VRAM is released before the next starts
  (`--no-preempt` queues requests instead).
- Request bodies above `--max-request-bytes` (default 8 MiB) are rejected with 413.
- `--require-auth` enables a Bearer API key (the server is open on the LAN by default).
- The server log is written to `logs/server.log` (rotating, 10 MB × 5; `--log-file`, `--no-log-file`).

## Accuracy and speed

- **Q4 and Q6 are lossy.** Accuracy depends on the model, source quantization, and prompt. The
  earlier **0.987–0.993** logit cosine range was measured on the specialized Qwen path, not every
  architecture. Kernel tests compare against independently decoded affine weights with explicit
  tolerances; they do not establish equivalence to original BF16 weights. Q6 is the universal
  default to reduce additional quantization error.
- **Specialized Qwen MTP without the experimental thought anchor**: with greedy decoding the text is identical with
  and without MTP (covered by `tests/test_engine_gpu.py`).
- Historical specialized-path measurement (not a new 1.1.0 benchmark): RTX 3060 Ti 8 GB, Qwen3.5-9B, all layers resident, 400-token answers, **one answer
  at a time**: ~31 tok/s without MTP; 30–37 tok/s with MTP, depending on how many drafts are accepted
  (43–71 % in our runs — on new text MTP may give no gain). Throughput summed over several parallel
  requests is a different number and is not quoted here.
- `--fast-vocab` is experimental and off by default: it can drop rare tokens on long outputs.

Earlier 1.1.0 small-model measurements on the same RTX 3060 Ti, with models on HDD:

| Model | Q4 median tok/s | Q6 median tok/s |
|---|---:|---:|
| Llama 3.2 1B GGUF | 26.94 | 22.17 |
| Qwen3.5 0.8B GGUF | 19.24 | 25.12 |
| Granite MoE 1B HF | 15.84 | 15.99 |

These are three 48-token requests after a warm-up, automatic placement, MTP off.
They exclude prefill and loading. See [full ranges and protocol](benchmarks/RELEASE_1_1_0.md).
The optional dynamic arena was slower than the matched streaming-ring control in
these experiments and remains opt-in; no dynamic-memory speedup is claimed.

The new controlled Llama 3.2 1B Q6 comparison measured **30.55 tok/s resident,
11.98 with half the layers streamed and 7.34 with all layers streamed**. Four
64-token samples per placement, warm host cache, MTP off; all output hashes matched.
A separate 2 GiB allocation was reflected in the corrected startup budget. These
are different placements on a small model, not a repeat of the historical Qwen9B
measurement or a cold-disk test. See [protocol and pressure results](benchmarks/STREAMING_DECODE_20261118.md).

## Common CPU/GPU startup

CPU and GPU preparation share one start event after the common graph geometry is
known. CPU global-weight preparation and GPU context/kernel preparation then proceed
independently. No CPU layer has to finish before GPU startup. Each ready device takes
eligible work; there is no forced device priority, startup delay or token-count ramp.

The foreground reads weights once and offers them to a bounded GPU-transfer queue
before computation. Prepared dense layers can join the same running graph; their live
per-layer cache follows them. At a safe chunk/token boundary the completed GPU setup
adopts that graph, state and the existing streaming ring. The prompt is not replayed.
Readiness includes the original Colibri NVRTC kernels and completed weight transfers.
Construction and CPU arithmetic still cost time; this does not guarantee instant output
or a lower time to first token.

This is the default (`progressive_start=True` in Python). `--no-progressive-start`
selects the previous GPU-first reference. The `--cpu-first` / `cpu_first=`
spelling remains an alias for compatibility; it imposes no CPU-first ordering.
Without CUDA, the same graph streams layers on CPU. `--device cpu` (Python:
`device="cpu"`) selects that path explicitly and starts no GPU helper. GPU planning,
compilation or preparation failure before full adoption retires the helper and
continues on CPU at a forward boundary, retaining the live cache and diagnostic.
Already adopted layers return to CPU without replaying the prefix. CPU sessions
park and restore state without CUDA synchronization or pinned allocation.
CPU startup retains the slot geometry without allocating a staging arena; GPU
transfer buffers are allocated when the GPU ring is prepared.
Failures during execution after full GPU adoption are still reported normally.
`doctor --device cpu` does not require a VRAM plan. This is not a guarantee that
every model fits its CPU working buffer: whole layer/projection limits still apply;
cluster/"cube" scheduling is not implemented by this change.
The common path now has a bounded draft/verify/commit loop over the same HF executor.
`--speculation auto` uses compatible trained MTP tensors when present, otherwise prompt
lookup. `--speculation native --mtp-head PATH` loads a separate Qwen3.5 text head
(default weight cap: 128 MiB); it shares the base embeddings and output projection.
`--speculation off` disables drafts. Lookup is not a trained MTP head. Each verification
row uses ordinary one-token layer arithmetic; rejected rows are removed from attention
and recurrent state before commit. A 64 MiB optional state-snapshot estimate limits
draft depth and falls back to ordinary decoding. No second backbone is loaded.
See [streamed MTP validation](benchmarks/MTP_STREAMING_20261118.md) for tested scope.
The specialized GPU-first Qwen reference and experimental vocabulary pruner remain
separate paths. MoE expert-cache adoption is atomic: those layers remain on CPU until their
cache can follow them. Up to 64 MiB of prepared CPU experts can be reused by the GPU
without a second read or quantization. Optional common memory retains the initial
prefill map during adoption, and the probability anchor follows the device while
preserving original values. These changes are included in 1.2.0.

`--region-pruning` (default) excludes exactly zero output-weighted Granite/OLMoE
routes before expert reads and cancels obsolete optional preparation for the same
layer/work scope. Later tokens can select those experts again. Actual nonzero routes
outside the prefill forecast are always loaded. This does not certify skipping arbitrary
dense weights based on early probabilities; models without a supported exclusion
contract keep their normal computation. Verification retains the warming producer,
but only accepted route rows may revise the task map.

## Experimental common weight arena

Before the first generation on an HF Granite/OLMoE Q4/Q6 engine with a three-slot
streaming plan, `engine.enable_unified_memory(region_experts=12)` joins the dense
ring, expert cache and prefill-map producer in one bounded GPU allocation. The
same method is available on `AirColibriUnifiedEngine`. By default it reuses the
existing ring/cache weight budget; an explicit `gpu_budget_bytes` adds an allowance.

The default uses elastic roles: A owns current work and retained cache, B prepares
only the next small region, and C stays empty with zero reserved payload bytes.
Dense weights and experts borrow pages from the same allocation. Promotion changes
ownership without copying weights. The dense scheduler keeps at most current and
next payloads, rather than preloading all three slots. Retired CUDA readers are
reported separately until their completion fences permit reuse.

Prepared experts are consumed in place. Reader leases and CUDA events prevent
overwrite while weights are in use. The prefill map and initial probability
anchor remain separate, and thought-buffer probes preserve the task map and
desired weights. `probability_warmup.stats()['shared_memory']` reports allocation,
budget, per-area activity and direct warm hits. The old separate-bank path remains
available through `enable_probability_warmup()` for comparison.

This is opt-in and currently limited to the checked HF MoE route. B uses the
retained prefill map for successive layers; changing its target does not rebuild
that map. Region ranking mixes normalized initial activity (25%) and revised activity (75%),
so the initial map contributes beyond resolving ties. Use `dynamic_regions=False` for the previous equal-area experiment.
KV/SSM allocation, CPU neural-compute handoff, fragmentation recovery for larger
operators and wider family support remain separate work. A complete dense payload
still needs contiguous pages. It is not a speed or reasoning-quality guarantee.

## Experimental thought buffer

The Python `ColibriThoughtBuffer` API keeps the original full FP32 log-probability distribution
and its contrastive prior separate from the revisable working field. It is off by default and
changes the model's output distribution. Version 1.2.0 bounds the prior's pairwise
log-odds contribution to 0.125 and retains FP32 sampling logits. New user input may revise the
working field while the original distribution remains available throughout waves and synthesis.
Direct mixing of the original recurrent state into current state is now explicit opt-in
(`rec_anchor_lambda > 0`); snapshots and rollback remain active independently.

Lexical alignment is diagnostic only: it neither verifies notes nor triggers retries with
stronger bias. In the **1.1.0 release** probes Granite answered the Japan-capital question with
`N/A` under the older anchor, versus `Tokyo` without it. The regression checks cover
that failure and explicit premise corrections; they do not establish generally improved
reasoning or answer correctness. Use a new anchor for a new task. Unsupported state containers
fail explicitly. Native MTP plus anchoring is not certified as lossless.

See [current development validation](benchmarks/DEVELOPMENT_20261118.md) for the
paired checks and the distinction between probability retention and semantic memory.

## Reproducing small-model measurements

`benchmarks/small_models.py` measures constructor time, an untimed warm-up and repeated
fixed-length decode requests, plus separate short answer probes. Example:

```bash
python benchmarks/small_models.py --model models/01_gguf_Llama-3.2-1B-Instruct-Q4_K_M --weight-format q6 --output benchmark.json
```

See [release validation](benchmarks/RELEASE_1_1_0.md) for hardware, results and limitations.
OS file caches are not flushed; constructor time must not be described as a cold-disk test.

## Tests

```bash
python -m pytest tests -q
```

CPU tests run anywhere; `tests/test_engine_gpu.py` and the model-dependent Q4 tests run only when a
CUDA GPU and a local model are available.
`tests/test_universal_quant.py` covers dense Q4/Q6, expert cache eviction, prefill/decode consistency,
resident-versus-three-slot-ring parity, tail prefetch, session reuse, cancellation and recovery.

## License & acknowledgments

- **AirLLM:** Gavin Li and contributors — [original project](https://github.com/lyogavin/airllm).
- **Colibrì:** Vincenzo Fornaro / JustVugg and contributors — [original project](https://github.com/JustVugg/colibri).

AirColibri is an independent project licensed under [Apache-2.0](LICENSE). Upstream names and marks
do not imply endorsement; see [NOTICE](NOTICE) and [asset sources](assets/README.md).
Changes are listed in [CHANGELOG.md](CHANGELOG.md).
