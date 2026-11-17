# Model support in AirColibri 1.1.0

**Not every model is supported.** The 2026-11-16 local matrix contains 48 checkpoints:
32 completed at least one short technical cycle; 16 did not. These are mixed, dated
development runs, not 48 reruns of this release. Passing means a bounded runtime cycle,
not certified answer quality, long-context correctness, image/audio support or all optional
mechanisms working together. The fresh release subset is recorded in
[release validation](benchmarks/RELEASE_1_1_0.md).

Q4/Q6 comparisons of resident and streamed execution test placement consistency. They do
not establish equivalence to original BF16 weights. Large model size is a scaling goal,
not an exclusion criterion; execution of an arbitrary 350 GB checkpoint is not demonstrated.
A whole dense payload and the LM head must currently fit their planned device allocations.

| Checkpoint | Short cycle | Latest recorded scenarios |
|---|---|---|
| 01_gguf_Llama-3.2-1B-Instruct-Q4_K_M | completed | baseline: passed; ring: passed |
| 02_gguf_SmolLM3-3B-IQ3_XXS | completed | baseline: passed; ring: passed |
| 03_gguf_Phi-4-mini-instruct-IQ3_XXS | completed | baseline: passed; ring: passed |
| 04_gguf_granite-3.3-2b-instruct-Q4_K_M | completed | baseline: passed; ring: passed |
| 05_gguf_gemma-3-1b-it-Q4_K_M | completed | baseline: passed; ring: passed |
| 06_gguf_gemma-2-2b-it-Q4_K_M | completed | baseline: passed; ring: passed |
| 07_gguf_OLMo-2-0425-1B-Q4_K_M | completed | baseline: passed; ring: passed |
| 08_gguf_command-r7b-12-2024-IQ3_XS | not completed | baseline: bounded_stop; ring: bounded_stop |
| 09_gguf_gemma-3n-E2B-it-Q2_K_L | not completed | baseline: failed |
| 10_gguf_MiniCPM3-4B-MLA-Q4_K_M | not completed | baseline: failed |
| 11_gguf_Nemotron-Mini-4B-IQ3_M | completed | baseline: passed; ring: passed |
| 12_gguf_Qwen3.5-0.8B-Q4_K_M | completed | baseline: passed; ring: passed |
| 13_gguf_Falcon-H1-0.5B-Q4_K_M | not completed | baseline: failed |
| 14_gguf_granite-4.0-tiny-hybrid-IQ3_XXS | not completed | baseline: failed |
| 15_gguf_LFM2-350M-Q4_K_M | completed | baseline: passed; ring: passed |
| 16_gguf_Quant_SmolLM2-135M-Q2_K | completed | baseline: passed; ring: passed |
| 17_gguf_Quant_Qwen2.5-0.5B-Q3_K_M | completed | baseline: passed; ring: passed |
| 18_gguf_Quant_Qwen2.5-0.5B-Q4_0 | completed | baseline: passed; ring: passed |
| 19_gguf_Quant_Llama-3.2-1B-Q4_1 | completed | baseline: passed; ring: passed |
| 20_gguf_Quant_SmolLM2-135M-IQ4_XS | completed | baseline: passed; ring: passed |
| 21_gguf_Quant_Llama-3.2-1B-IQ4_NL | completed | baseline: passed; ring: passed |
| 22_gguf_Quant_Llama-3.2-1B-IQ2_XXS | completed | baseline: passed; ring: passed |
| 23_gguf_Quant_Llama-3.2-1B-IQ1_S | completed | baseline: passed; ring: passed |
| 24_gguf_BitNet-b1.58-2B-TQ2_0 | not completed | baseline: failed |
| 25_gguf_BitNet-b1.58-2B-i2_s | not completed | baseline: failed |
| 26_hf_Qwen3-0.6B-FP8 | completed | baseline: passed; ring: passed |
| 27_hf_ERNIE-4.5-0.3B-PT | completed | baseline: passed; ring: passed |
| 28_hf_granite-3.1-1b-a400m-MoE | completed | baseline: passed; ring: passed |
| 29_hf_mamba-130m | completed | baseline: passed; ring: passed |
| 30_hf_mamba2-370m | completed | baseline: passed; ring: passed |
| 31_hf_Zamba2-1.2B-instruct-v2 | not completed | baseline: failed |
| 32_hf_rwkv7-0.1B-g1 | not completed | baseline: failed |
| 33_hf_recurrentgemma-2b-it-8bit | not completed | baseline: failed |
| 34_hf_xlstm-german-wikipedia | not completed | baseline: failed |
| 35_hf_RWKV4-169m-bin | not completed | baseline: failed |
| 36_hf_OpenELM-270M-Instruct | not completed | baseline: failed |
| 37_hf_gpt2-124m | completed | baseline: passed; ring: passed |
| 38_hf_bloom-560m | completed | baseline: passed; ring: passed |
| 39_hf_pythia-410m | completed | baseline: passed; ring: passed |
| 40_hf_SmolVLM2-256M | completed | baseline: passed; ring: passed |
| 41_hf_Qwen2-VL-2B-Instruct-AWQ | not completed | baseline: failed |
| 42_hf_Qwen2.5-0.5B-AWQ | completed | baseline: passed; ring: passed |
| 43_hf_Qwen2.5-0.5B-GPTQ-Int4 | completed | baseline: passed; ring: passed |
| 44_hf_internlm2_5-1_8b-chat | not completed | baseline: failed |
| 45_hf_gpt-oss-20b-MXFP4 | not completed | baseline: failed |
| deepseek-v4-9b-colibri | completed | baseline: cancel_not_observed; ring: bounded_stop; compact: passed; ring_compact: passed |
| OLMoE-1B-7B-0125-Instruct | completed | baseline: bounded_stop; compact: passed; ring_compact: passed |
| Qwen3.6-35B-A3B-uncensored-heretic-APEX-Balanced | completed | baseline: bounded_stop; compact: passed; ring_compact: passed |

## Known boundaries

- `bounded_stop` is an exhausted experiment time/resource budget, not proof of incompatibility.
- Gemma3n, MiniCPM3, BitNet/i2_s, Zamba2, RWKV7, recurrentgemma 8-bit, xLSTM, OpenELM, RWKV4 `.bin`, the tested Qwen2-VL AWQ, InternLM2 and GPT-OSS MXFP4 fixtures remain unresolved in this matrix. Falcon-H1 and Granite hybrid GGUF fixtures also failed tensor/config adaptation.
- Mamba/LFM2 session-state coverage does not imply the same coverage for every recurrent architecture.
- Dynamic shared A/B/C memory and prefill-map warmup are opt-in HF Granite/OLMoE Q4/Q6 experiments. Broad GGUF integration, CPU neural-compute handoff, large-operator tiling and fragmentation recovery are pending.
- Thought-buffer probability retention and rollback tests do not prove answer correctness or improved reasoning.
- New families require config, tensor layout, tokenizer and state-adapter validation; recognizing a family name alone is insufficient.
