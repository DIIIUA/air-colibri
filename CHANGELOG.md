# Changelog

## Unreleased

- The engine reads the checkpoint and turns Q4 and MTP on only when the geometry matches the fused kernels. Other Hugging Face causal LMs (including per-expert MoE shards such as OLMoE) run in BF16 on the same ring with on-demand expert routing and a budgeted VRAM hot-expert cache.
- Supported GGUF checkpoints (`qwen35moe` / `qwen35`) run on the BF16 ring directly from the `.gguf` file with zero `mmap` calls and zero sidecar files; routed MoE experts are fetched by parallel positioned reads, dequantized on GPU via NVRTC (`Q4_0`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, plus extended `Q5_0`/`Q5_1`/`Q8_K`/`MXFP4` support), and cached in spare VRAM while `Qwen3.5-9B` remains the default.
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
