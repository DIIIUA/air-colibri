# Changelog

## 1.2.0 — 2026-11-18

- Add a numbered local model catalog, selection by number/name/path, Hugging Face
  page/file links, managed downloads into a chosen directory and `prepare`.
  Downloads verify a finite first forward and automatic runtime packing by default;
  complete local snapshots can be reused offline. Split GGUF selection is explicit.
- Keep help/catalog independent of Torch initialization. Register external model
  directories, isolate revisions and retain partial downloads for retry.
- Use the lower CUDA/NVML startup free-memory reading for the GPU matched by UUID;
  expose counter/fallback diagnostics in status and doctor. Live two-process testing
  caught and corrected a Windows accounting discrepancy. This is startup sampling,
  not continuous global memory rebalancing.
- Add a fresh local roster census and a controlled resident/half/full-streaming
  comparison. Llama1B Q6 medians: 30.55 / 11.98 / 7.34 tok/s, warm RAM cache,
  MTP off; all 24 output hashes matched across two pressure settings. See
  `MODEL_SUPPORT.md` and `benchmarks/RELEASE_1_2_0.md` for results and limits.

### Common CPU/GPU execution

- Keep the streamed CPU graph running without CUDA or when GPU preparation fails;
  retire partially adopted GPU layers at a forward boundary without losing live cache.
- Add `--device cpu` / `device="cpu"` and CPU-aware doctor/session parking. Keep GPU
  preparation diagnostics visible; do not retry failed preparation on every token.
- Defer the unused CPU-start staging arena; retain its geometry until GPU transfers
  need real buffers, rather than reserving three complete layer-sized RAM slots.
- Release CPU and GPU preparation together; ready devices take work without a forced CPU-first dependency. GPU transfers use a bounded background queue.
- Adopt ready dense layers and their live cache on the same graph, then hand off to
  the existing GPU ring without replaying the prompt. Keep `--no-progressive-start` as a GPU-first reference.
- Carry the original CPU prefill map and probability-anchor values through GPU adoption;
  protect auxiliary probes against a device change and drain preparation on cancellation.
- Add common streamed draft/verify/commit: prompt lookup and an optional bounded native
  Qwen3.5 text MTP head sharing base embeddings/output projection. Verify token rows
  with decode arithmetic under one layer lease; roll back rejected KV/recurrent rows.
- Preserve retained-map preparation during verification; commit only accepted routes.
  Add execution-scoped Granite/OLMoE zero-route exclusions before reads and current-work
  preparation cuts. Probability forecasts cannot suppress nonzero required routes.
- Add `--speculation`, `--mtp-head`, `--mtp-head-budget-mib` and `--[no-]region-pruning`.
  General large-model, arbitrary dense pruning and universal native-MTP coverage remain unverified.

### Memory and probability experiments

- Bound thought-anchor log-odds influence, retain FP32 logits and the complete original probability distribution, and allow explicit input corrections. Lexical alignment remains diagnostic; it no longer verifies notes or triggers stronger-bias retries. Direct recurrent-state mixing is opt-in.
- Reduce dynamic-memory pinned staging to two expert slots, reuse views/layouts, preserve recently used experts during dense-page reclamation, share observed router rows with dispatch, and wake the producer only for missing preparation work. Dynamic placement remains experimental; a general speedup is not established.
- Use bounded positioned reads for HF globals and layer metadata, avoiding whole-shard mappings and the reproduced Windows error 1455. Global staging and physical read sizes obey the configured cap.
- Add a resumable current-model census and paired thought-buffer acceptance probes. See [development evidence](benchmarks/DEVELOPMENT_20261118.md) for scope and results; probability retention is not a semantic-memory certification.

## 1.1.0 — 2026-11-18

- Universal Q4/Q6 packed projections and compatible SwiGLU expert kernels; Q6 is the generic default. Source GGUF and runtime affine formats remain distinct and lossy.
- Bounded adapters for HF FP8 block, AWQ GEMM INT4 and GPTQ INT4; repaired SentencePiece/ByteLevel tokenizer selection and conv/recurrent/sliding session parking.
- Request context admission now checks prompt plus requested output against the planned and declared model limits before prefill or session unpark. LAN clients receive a context-length error instead of entering an oversized generation.
- Experimental shared dynamic weight arena: A owns active work/cache, B prepares a bounded next region, C reserves no payload; dense and expert pages share CUDA reader fences and transfer ownership without extra copies.
- Retained prefill-map prediction now combines normalized initial and recent activity (25%/75%). Actual routing always determines required expert computation.
- Experimental thought-buffer probes now restore complete sliding-window and conv/recurrent state. Unknown state containers fail explicitly; recurrent telemetry no longer counts nonexistent recurrent states. Original probability fields remain preserved. Lexical alignment is not a correctness verifier and reasoning improvements remain unproven.
- Added reproducible small-model benchmarking, a dated 48-checkpoint support matrix, and regression coverage for the combined memory/probability mechanisms. Not all architectures or checkpoints are supported; see `MODEL_SUPPORT.md` and `benchmarks/RELEASE_1_1_0.md`.

Changes accumulated since 1.0.0 (architecture adapters below are implementation coverage, not blanket checkpoint certification):

- Broadened Hugging Face CausalLM and VLM checkpoint support (`aircolibri_cli/model_source.py`, `src/aircolibri_integrated.py`, `src/bounded_memory_manager.py`, `src/gpu_ring_pipeline.py`): unwraps `text_config` from multimodal VLM checkpoints while ignoring vision/audio towers, supports single custom-named `.safetensors` files and all decoder root/norm/head conventions (`model.layers`, `transformer.h`, `gpt_neox.layers`, `model.decoder.layers`, `backbone.layers`, `rwkv.blocks`, `transformer.blocks`), handles Hub leaf renamings (`Mixtral`, `PhiMoE`, `Nemotron-H`, `Ernie-4.5-MoE`), streams 2D per-expert and 3D packed standard/transposed/biased MoE experts (`Qwen3/3.5-MoE`, `Mixtral`, `DeepSeek-V2/V3`, `Llama-4`, `GPT-OSS`, `OLMoE`, `GLM-4-MoE`, `MiniMax-M2`) via positioned reads + VRAM LRU cache, supports variable-shape same-name layer views (`Gemma 4` sliding vs global attention), co-existing parallel mixers (`Falcon-H1`), scaled input embeddings (`Gemma 1–4`), SSM `cache_params` / `RWKV` `state`, and base-model chat-template fallback while keeping `Qwen3.5-9B` on its fast Q4 + MTP path as default.
- Broadened zero-`mmap` GGUF support (`aircolibri_cli/gguf_arch_registry.py`, `aircolibri_cli/model_source.py`) from 3 architectures to 35+ model families (47+ GGUF architecture IDs), added multi-shard split GGUF (`*-00001-of-0000N.gguf`) discovery while ignoring `mmproj*.gguf`, added `Llama`/`DeepSeek` interleaved RoPE Q/K un-permutation, `GPT-2` `Conv1D` 2D transpose, `BLOOM` QKV un-interleaving, `Phi-2` fused QKV virtual row slicing, `Q8_1` and ternary `TQ1_0`/`TQ2_0` block dequantization, and extended pure-GGUF tokenizer building without `gguf` PyPI or sidecar files.
- Multi-turn KV and GatedDeltaNet session prefix reuse (`aircolibri chat` and LAN web chat) now works across BF16 Hugging Face and GGUF checkpoints as well as Q4 Qwen3.5.
- Self-contained NVRTC CUDA kernels (`src/colibri_deltanet_nvrtc.py`) accelerate `causal_conv1d_fn`, `causal_conv1d_update`, `chunk_gated_delta_rule`, `fused_recurrent_gated_delta_rule`, and gated/zero-centered `RMSNorm` on Windows and Linux without `causal-conv1d`, `flash-linear-attention`, or Triton, eliminating `transformers` PyTorch reference fallback warnings and multi-kernel `solve_triangular` overhead.
- Coalesced positioned disk reads and batched H2D token/expert routing transfers accelerate MoE multi-token prefill across GGUF and sharded Hugging Face checkpoints.
- LAN server exposes context-window metadata on `GET /v1/models`, `GET /api/v0/models`, `GET /props`, and `GET /api/extra/true_max_context_length` for OpenAI, LM Studio, llama.cpp, and KoboldCPP clients, and logs SSE client disconnects accurately as `cancelled`.
- Cooperative cancellation (`POST /v1/chat/stop`, client disconnect, preemption) now polls at every streamed decoder layer boundary and resets in-flight ring slots cleanly.

## 1.0.0 — 2026-11-14

- Public release: the repository now contains only the program, the CLI and its tests.
- Reads standard Hugging Face shards directly; recommended model `Qwen/Qwen3.5-9B`.
- VRAM modes `normal` and `turbo` replace the fixed `4gb`/`8gb` profiles.
- Fixed CUDA out-of-memory on long prompts: chunked prefill, last-position logits only.
- LAN server: immediate stop (`POST /v1/chat/stop`, closed connections), one generation at a time,
  guaranteed VRAM cleanup, request size limit, rotating log `logs/server.log`.
