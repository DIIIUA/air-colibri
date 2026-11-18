# Model support in AirColibri 1.2.0

**Support is checkpoint-specific.** The fresh 2026-11-18 local census attempted all 46
remaining roster entries on the release implementation: **31 passed**, **15 failed**.
Each successful entry completed finite raw logits, two chat tokens and one continuation
token from the session. Context was 512, prefill chunk 8, runtime Q6; MTP, thought anchor
and the optional dynamic arena were off. CPU/GPU preparation used the default common
start. This is a startup/state smoke check, not answer-quality, speed, full-context,
multimodal or all-feature certification. Exact outcomes and source fingerprints are in
[the machine-readable report](benchmarks/model_support_1_2_0.json).

| Checkpoint | Fresh result | Failure stage / detail |
|---|---|---|
| 01_gguf_Llama-3.2-1B-Instruct-Q4_K_M | passed |  |
| 02_gguf_SmolLM3-3B-IQ3_XXS | passed |  |
| 03_gguf_Phi-4-mini-instruct-IQ3_XXS | passed |  |
| 04_gguf_granite-3.3-2b-instruct-Q4_K_M | passed |  |
| 05_gguf_gemma-3-1b-it-Q4_K_M | passed |  |
| 06_gguf_gemma-2-2b-it-Q4_K_M | passed |  |
| 07_gguf_OLMo-2-0425-1B-Q4_K_M | passed |  |
| 08_gguf_command-r7b-12-2024-IQ3_XS | passed |  |
| 09_gguf_gemma-3n-E2B-it-Q2_K_L | failed | load: UnsupportedModelError: [project]\models\09_gguf_gemma-3n-E2B-it-Q2_K_L\gemma-3n-E2B-it-Q2_K_L.gguf: GGUF architecture 'gemma3n' is not supported |
| 10_gguf_MiniCPM3-4B-MLA-Q4_K_M | failed | load: UnsupportedModelError: [project]\models\10_gguf_MiniCPM3-4B-MLA-Q4_K_M\minicpm3-4b-q4_k_m.gguf: GGUF architecture 'minicpm3' is not supported |
| 11_gguf_Nemotron-Mini-4B-IQ3_M | passed |  |
| 12_gguf_Qwen3.5-0.8B-Q4_K_M | passed |  |
| 13_gguf_Falcon-H1-0.5B-Q4_K_M | failed | raw_forward: ValueError: Trying to set a tensor of shape torch.Size([128, 1024]) in "weight" (which has shape torch.Size([256, 1024])), this looks incorrect. |
| 14_gguf_granite-4.0-tiny-hybrid-IQ3_XXS | failed | raw_forward: ValueError: `get_seq_length` can only be called on Attention layers, and the current Cache seem to only contain LinearAttention layers. |
| 15_gguf_LFM2-350M-Q4_K_M | passed |  |
| 16_gguf_Quant_SmolLM2-135M-Q2_K | passed |  |
| 17_gguf_Quant_Qwen2.5-0.5B-Q3_K_M | passed |  |
| 18_gguf_Quant_Qwen2.5-0.5B-Q4_0 | passed |  |
| 19_gguf_Quant_Llama-3.2-1B-Q4_1 | passed |  |
| 20_gguf_Quant_SmolLM2-135M-IQ4_XS | passed |  |
| 21_gguf_Quant_Llama-3.2-1B-IQ4_NL | passed |  |
| 22_gguf_Quant_Llama-3.2-1B-IQ2_XXS | passed |  |
| 23_gguf_Quant_Llama-3.2-1B-IQ1_S | passed |  |
| 24_gguf_BitNet-b1.58-2B-TQ2_0 | failed | load: UnsupportedModelError: [project]\models\24_gguf_BitNet-b1.58-2B-TQ2_0\bitnet-2b4t-tq2_0.gguf: GGUF architecture 'bitnet' is not supported |
| 25_gguf_BitNet-b1.58-2B-i2_s | failed | load: UnsupportedModelError: [project]\models\25_gguf_BitNet-b1.58-2B-i2_s\ggml-model-i2_s.gguf: [project]\models\25_gguf_BitNet-b1.58-2B-i2_s\ggml-model-i2_s.gguf: tensor 'blk.0.ffn_down.weight': unsupported ggml type 36 (T36) |
| 26_hf_Qwen3-0.6B-FP8 | passed |  |
| 27_hf_ERNIE-4.5-0.3B-PT | passed |  |
| 28_hf_granite-3.1-1b-a400m-MoE | passed |  |
| 29_hf_mamba-130m | passed |  |
| 30_hf_mamba2-370m | passed |  |
| 31_hf_Zamba2-1.2B-instruct-v2 | failed | raw_forward: RuntimeError: Tensor on device cpu is not on the expected device meta! |
| 32_hf_rwkv7-0.1B-g1 | failed | load: ValueError: The repository [project]\models\32_hf_rwkv7-0.1B-g1 contains custom code which must be executed to correctly load the model. You can inspect the repository content at [project]\models\32_hf_rwkv7-0.1B-g1 .  You can inspect the repository content at https://hf.co |
| 33_hf_recurrentgemma-2b-it-8bit | failed | load: UnsupportedModelError: [project]\models\33_hf_recurrentgemma-2b-it-8bit: checkpoint misses global tensors ['model.norm.weight'] |
| 34_hf_xlstm-german-wikipedia | failed | load: UnsupportedModelError: [project]\models\34_hf_xlstm-german-wikipedia: config has no decoder shape (num_hidden_layers, hidden_size) |
| 35_hf_RWKV4-169m-bin | failed | load: UnsupportedModelError: [project]\models\35_hf_RWKV4-169m-bin: no Colibri container, Hugging Face safetensors checkpoint, or GGUF file found |
| 36_hf_OpenELM-270M-Instruct | failed | load: UnsupportedModelError: [project]\models\36_hf_OpenELM-270M-Instruct: config has no decoder shape (num_hidden_layers, hidden_size) |
| 37_hf_gpt2-124m | passed |  |
| 38_hf_bloom-560m | passed |  |
| 39_hf_pythia-410m | passed |  |
| 40_hf_SmolVLM2-256M | passed |  |
| 41_hf_Qwen2-VL-2B-Instruct-AWQ | failed | load: UnsupportedModelError: [project]\models\41_hf_Qwen2-VL-2B-Instruct-AWQ: model_type='qwen2_vl' (Qwen2VLConfig) has no CausalLM class |
| 42_hf_Qwen2.5-0.5B-AWQ | passed |  |
| 43_hf_Qwen2.5-0.5B-GPTQ-Int4 | passed |  |
| 44_hf_internlm2_5-1_8b-chat | failed | load: UnsupportedModelError: [project]\models\44_hf_internlm2_5-1_8b-chat: checkpoint misses global tensors ['model.embed_tokens.weight', 'lm_head.weight'] |
| 45_hf_gpt-oss-20b-MXFP4 | failed | load: UnsupportedModelError: [project]\models\45_hf_gpt-oss-20b-MXFP4: layer 0 packed experts miss projections (['down_proj_bias', 'gate_up_proj_bias']) |
| Qwen3.6-35B-A3B-uncensored-heretic-APEX-Balanced | passed |  |

## Additional fresh download

`HuggingFaceTB/SmolLM2-135M-Instruct` was downloaded through the CLI, automatically
packed to Q6, checked with a finite first forward and run on CPU. It answered the
short arithmetic smoke probe correctly, then passed offline download/preparation reuse.
This extra HF checkpoint is separate from the 46-entry census and used a different
[CLI protocol](benchmarks/CLI_MODELS_20261118.md).

## Boundaries

- `failed` records an actual unresolved error; its stage distinguishes adaptation from execution failures.
- `bounded_stop` is the 600-second experiment limit, not proof of architectural incompatibility.
- DeepSeek9B and OLMoE7B were deleted locally and excluded; their older passes in the
  [historical 48-entry matrix](benchmarks/model_matrix_20261116.json) are not fresh release passes.
- Source GGUF, FP8, AWQ and GPTQ are adapted to runtime arithmetic; Q4/Q6 are lossy.
  Placement parity does not prove equivalence to original BF16 weights.
- Multimodal fixtures exercise language only; images/audio are not supported by these checks.
- Model size is not a rejection criterion. Arbitrary 90–350 GB checkpoints were not tested.
  The active dense payload/operator and LM head still require planned contiguous memory;
  operator tiling into arbitrary cubes and general fragmentation recovery remain pending.
- Native common-stream MTP currently targets a compatible trained Qwen3.5 text head.
  Other contracts use lookup or plain decode; the support table does not imply native MTP.
- The shared A/B/C arena, retained-map warming and thought buffer were combined in narrow
  HF Granite/OLMoE Q4/Q6 tests, not across this entire roster. The anchor remains opt-in;
  preserved probabilities are not proof of improved reasoning. See
  [MTP/memory validation](benchmarks/MTP_STREAMING_20261118.md) and
  [release validation](benchmarks/RELEASE_1_2_0.md).
