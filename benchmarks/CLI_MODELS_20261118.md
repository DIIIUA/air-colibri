# Model-management CLI validation — 2026-11-18

The command line now lists and selects local models, accepts Hugging Face model
and GGUF-file links, downloads into a chosen directory and verifies preparation.
Listing files is separate from metadata compatibility and successful execution.
The [live download receipt](cli_download_1_2_0.json) includes the exact Hub revision,
source SHA256, packing counters and offline answer. The weight SHA256 matched the
Hub's cached download metadata.

## Actual downloaded checkpoint

`HuggingFaceTB/SmolLM2-135M-Instruct` was downloaded through the public `download`
command into a new external directory on D: (HDD). The original
`model.safetensors` is **269,060,552 bytes**. No manual converter was invoked and
no second complete weight checkpoint was created.

`--device cpu --weight-format q6` automatically adapted and packed the model.
The preparation receipt reports finite logits, **210 packed projection modules**,
6-bit weights and **210 tiled projection calls**. The receipt reached `ready`.
The directory was registered in the local numbered catalog.

A new process with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` generated
**“6 times 7 is 42.”** (10 tokens, EOS) from that folder, entirely on CPU,
with reported peak VRAM 0. Another offline `download` of the same URL/destination
reused the files and passed preparation again. This verifies download → automatic
packing → generation → offline reuse for this checkpoint. The short answer is a
smoke test, not an answer-quality or throughput evaluation.

The test also caught an unnecessary HTTP-client import in a new error handler;
it was removed, and the successful offline retry uses the pinned Hub dependencies.

## Local model selection and contracts

The existing GGUF SmolLM2-135M Q2_K was selected with `ask --model 16` from the
then-current 46-entry catalog and generated three tokens using CPU/Q6. A catalog
number is relative to the current sorted list, so scripts should prefer paths.

The model-management/download/CLI subset passed **26 tests** before the live
download. The release's final regression run covers the final implementation.
Cases include help/catalog with Torch imports blocked, number/name/path resolution,
external registration, nested GGUF directories, download ownership, revision
isolation, page/file URL parsing, exact split-GGUF selection, rejection of ambiguous
variants, completion of partial caches and actual Q6 execution after a simulated
download of a tiny real Transformer checkpoint.

Fresh network transfer was exercised for HF safetensors. GGUF variant and split
selection were verified with controlled Hub fixtures; a fresh real multi-shard GGUF
network download was not repeated. First-forward readiness does not certify every
architecture, optional mechanism, long-context behavior or model quality.
