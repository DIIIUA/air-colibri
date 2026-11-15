# Changelog

## Unreleased

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
