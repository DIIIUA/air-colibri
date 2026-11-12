# Q4 quality protocol

This is an experimental Q4/BF16 **distribution agreement** test on the real
Qwen3.5-9B checkpoint. It does not establish general task accuracy, human-rated
answer quality or independence from the model's training data.

## Fixed inputs and original mathematics

`tests/fixtures/quality_prompts_v1.json` contains 12 newly authored texts:
English, Russian, Python, SQL, JSON and mixed-language material. The corpus
was frozen before inspecting candidate Q4 results. It is held out from this
quantizer's development, not a standard external benchmark.

The default protocol uses five chunks of 16 input tokens per text, giving
960 prediction positions. One additional token supplies the last next-token
target. Both modes consume exactly the same original tokenizer IDs, without
padding, generated-token feedback or repeated EOS positions. Recurrent/KV
state persists between chunks, separately within each mode.

The existing Transformers forward is unchanged. A read-only `lm_head` hook
copies all position logits to the CPU. Each chunk verifies that the observed
last-position logits equal the runtime's ordinary output bit for bit. The
observer adds synchronization and copying, so these timings must not be used
as inference-throughput measurements.

BF16 reference and Q4 candidate run in separate processes. Reference files
are saved per chunk and verified by SHA-256 before comparison. The runner,
runtime source, tokenizer, configuration, corpus and original AirLLM reader
must match. Checkpoint identity includes current file sizes/timestamps and
the historical trusted layer-hash manifest; this is not a new full rehash of
every model weight.

## Unchanged strict gate

All requirements from the earlier `bench_q4_top1.py` gate remain:

- At least 300 positions with BF16 top-1/top-2 logit margin >= 1.0.
- Zero top-1 flips at those confident positions.
- Candidate top-1 is within BF16 top-3 at every evaluated position.
- Top-5 overlap mean >= 4.7 **or** median >= 5.0.
- Mean KL(BF16 || candidate) <= 0.03.
- Mean cosine similarity >= 0.99.

Violations yield `FAIL`; agreement on too few confident positions yields
`INSUFFICIENT_EVIDENCE`. Neither is a pass. Mean next-token NLL on the fixed
texts is supplementary and does not replace this gate. `lm_head` remains BF16.

## Run locally

Use the prepared checkpoint, original AirLLM checkout and trusted prepacked
cache described in [DEVELOPMENT.md](DEVELOPMENT.md). All outputs stay within
the canonical checkout. Use a new output directory for each attempt; the
runner refuses to overwrite earlier evidence.

```powershell
.\.venv\Scripts\python.exe scripts/validate_q4_quality.py --mode reference --steps 5 --output data/quality-20261113/text-reference-v2
.\.venv\Scripts\python.exe scripts/validate_q4_quality.py --mode bf16-check --reference data/quality-20261113/text-reference-v2 --output data/quality-20261113/bf16-repeat
.\.venv\Scripts\python.exe scripts/validate_q4_quality.py --mode q4 --reference data/quality-20261113/text-reference-v2 --output data/quality-20261113/q4-g64
```

Startup requires CUDA, at least 2.8 GiB available RAM, declared RAM/VRAM
reservations and Windows commit headroom. The plan includes explicit batch
recurrent/KV/logit reserves. Runtime layer guards stop under RAM pressure.
Resource rejection is `SAFE_SKIP` (exit 3), a runtime error is `ERROR` (exit 1),
and a completed Q4 run that does not pass the gate exits 2. `COMPLETED` refers
to experiment execution; inspect `metrics.gate` for Q4 acceptance.

The initial two-step, 12-prefix BF16 smoke run completed. A first fixed-text
launch was rejected by the 2.8 GiB RAM guard before model initialization.
Full fixed-text results will be recorded separately after actual execution.
See [the follow-up test report](TESTING_2026-11-13.md) for the actual guarded
attempts and the separately verified CUDA ordering fixes.
