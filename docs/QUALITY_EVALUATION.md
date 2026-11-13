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
- Candidate greedy `argmax` is within BF16 top-3 at every evaluated position;
  equality with the third-largest logit counts as a tied rank within top-3.
- Top-5 overlap mean >= 4.7 **or** median >= 5.0.
- Mean KL(BF16 || candidate) <= 0.03.
- Mean cosine similarity >= 0.99.

Violations yield `FAIL`; agreement on too few confident positions yields
`INSUFFICIENT_EVIDENCE`. Neither is a pass. Mean next-token NLL on the fixed
texts is supplementary and does not replace this gate. `lm_head` remains BF16.

Protocol v2 corrects the earlier use of ordered `topk` indices as greedy tokens.
Actual token selection uses `argmax`, matching the runtime; the old ordered-topk
statistics are retained as explicit comparison fields. The numeric thresholds
are unchanged. See [the results](Q4_RESULTS_2026-11-13.md) for the measured effect
of ties and the independently unchanged FAIL verdict.

## Run locally

Use the prepared checkpoint, original AirLLM checkout and trusted prepacked
cache described in [DEVELOPMENT.md](DEVELOPMENT.md). All outputs stay within
the canonical checkout. Use a new output directory for each attempt; the
runner refuses to overwrite earlier evidence.

```powershell
.\.venv\Scripts\python.exe scripts/validate_q4_quality.py --mode reference --steps 5 --output data/quality-20261113/my-new-run/reference
.\.venv\Scripts\python.exe scripts/validate_q4_quality.py --mode bf16-check --reference data/quality-20261113/my-new-run/reference --output data/quality-20261113/my-new-run/bf16-repeat
.\.venv\Scripts\python.exe scripts/validate_q4_quality.py --mode q4 --reference data/quality-20261113/my-new-run/reference --output data/quality-20261113/my-new-run/q4-g64
```

Startup requires CUDA, at least 2.8 GiB available RAM, declared RAM/VRAM
reservations and Windows commit headroom. The plan includes explicit batch
recurrent/KV/logit reserves. Runtime layer guards stop under RAM pressure.
Resource rejection is `SAFE_SKIP` (exit 3), a runtime error is `ERROR` (exit 1),
and a completed Q4 run that does not pass the gate exits 2. `COMPLETED` refers
to experiment execution; inspect `metrics.gate` for Q4 acceptance.

The full evaluation completed on 960 positions / 623 confident positions.
BF16 repeatability passed bit for bit; Q4-g64 failed the strict agreement gate.
See [the quality results](Q4_RESULTS_2026-11-13.md). Earlier guarded attempts and
the separately verified CUDA ordering fixes remain in the
[follow-up test report](TESTING_2026-11-13.md).
