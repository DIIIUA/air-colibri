<p align="center">
  <img src="assets/aircolibri-logo.png" width="300" alt="AirColibri — AirLLM-inspired wings meet the Colibrì hummingbird">
</p>

# AirColibri

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Apache 2.0"></a>
</p>

Local inference of supported **Hugging Face and GGUF language models** on a single consumer
NVIDIA GPU, with packed Q4/Q6 execution, bounded layer streaming and an OpenAI-compatible LAN server.

**Version 1.1.0 does not support every model.** See [model support](MODEL_SUPPORT.md) for dated
per-checkpoint evidence, unsupported cases and experimental mechanisms.

AirColibri combines ideas from [AirLLM](https://github.com/lyogavin/airllm) (layer streaming,
selective weight reads) and [Colibrì](https://github.com/JustVugg/colibri) (tiered placement,
native MTP speculation) with its own packed Q4/Q6 weights and NVRTC CUDA kernels.

## Requirements

- NVIDIA GPU with CUDA (tested: RTX 3060 Ti 8 GB; smaller budgets require streaming and sufficient room for individual operators)
- Python 3.12, PyTorch 2.6 with CUDA 12.4 (`requirements.txt`)
- Windows x64 or Linux

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv/Scripts/python -m pip install -r requirements.txt
```

## Model

The recommended model is **[Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)**.
Standard Hugging Face shards (`model-0000X-of-0000N.safetensors` + `model.safetensors.index.json`)
are read directly — no converter and no second copy on disk. On first start the Q4 weights are
built once into `data/q4-g64-cache/`.

```bash
aircolibri doctor --model Qwen/Qwen3.5-9B      # downloads once (~19 GB) and checks the setup
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
aircolibri ask "Привет! Кто ты?"       # one answer, streamed
aircolibri chat                        # multi-turn chat (Ctrl+C stops the current answer)
aircolibri serve                       # web chat + OpenAI API on :8787, UDP beacon on :8788
aircolibri discover                    # find AirColibri servers on the LAN
```

On Windows use `aircolibri.cmd` (it picks `.venv` automatically) or `python aircolibri.py`.

### VRAM modes

AirColibri measures the **free** VRAM at start and plans with it — there are no fixed card profiles.

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

Fresh 1.1.0 small-model measurements on the same RTX 3060 Ti, with models on HDD:

| Model | Q4 median tok/s | Q6 median tok/s |
|---|---:|---:|
| Llama 3.2 1B GGUF | 26.94 | 22.17 |
| Qwen3.5 0.8B GGUF | 19.24 | 25.12 |
| Granite MoE 1B HF | 15.84 | 15.99 |

These are three 48-token requests after a warm-up, automatic placement, MTP off.
They exclude prefill and loading. See [full ranges and protocol](benchmarks/RELEASE_1_1_0.md).
The optional dynamic arena was slower than the matched streaming-ring control in
these experiments and remains opt-in; no dynamic-memory speedup is claimed.

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

The Python `ColibriThoughtBuffer` API keeps the initial probability field separate from the
working field. It is off by default and changes the model's output distribution. Its lexical
alignment score is a heuristic, not an answer verifier. Improved reasoning and reliable handling
of corrections to an earlier premise have not been established. In the release probes,
Granite answered the Japan-capital question with `N/A` under anchoring, versus `Tokyo`
without it. This experimental mode can worsen factual answers. Use a new anchor for a new task.
The release fixes probe rollback for sliding-window and conv/recurrent caches; unsupported
state containers fail explicitly. Native MTP plus anchoring is not certified as lossless.

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
