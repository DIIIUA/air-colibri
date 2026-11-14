# Changelog

## 1.0.0 — 2026-11-14

- Public release: the repository now contains only the program, the CLI and its tests.
- Reads standard Hugging Face shards directly; recommended model `Qwen/Qwen3.5-9B`.
- VRAM modes `normal` and `turbo` replace the fixed `4gb`/`8gb` profiles.
- Fixed CUDA out-of-memory on long prompts: chunked prefill, last-position logits only.
- LAN server: immediate stop (`POST /v1/chat/stop`, closed connections), one generation at a time,
  guaranteed VRAM cleanup, request size limit, rotating log `logs/server.log`.
