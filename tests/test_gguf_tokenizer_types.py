"""GGUF's default pre-tokenizer name must respect the vocabulary encoding."""
import pytest
from transformers import Gemma2Config, Gemma3TextConfig, NemotronConfig
from aircolibri_cli.gguf_arch_registry import build_extended_gguf_tokenizer


@pytest.mark.parametrize("arch,config_type", [("gemma2", Gemma2Config),
                                               ("gemma3", Gemma3TextConfig),
                                               ("nemotron", NemotronConfig)])
def test_sentencepiece_default_keeps_unicode_and_whitespace(arch, config_type):
    text = "Hello world\nПривет, мир!"
    tokens = ["<unk>", "<s>", "</s>"] + [f"<0x{i:02X}>" for i in range(256)]
    tokens += sorted(set(text.replace(" ", "▁")))
    meta = {"tokenizer.ggml.model": "llama", "tokenizer.ggml.pre": "default",
            "tokenizer.ggml.tokens": tokens, "tokenizer.ggml.scores": [0.0]*len(tokens),
            "tokenizer.ggml.token_type": [2, 3, 3] + [6]*256 + [1]*(len(tokens)-259),
            "tokenizer.ggml.unknown_token_id": 0, "tokenizer.ggml.bos_token_id": 1,
            "tokenizer.ggml.eos_token_id": 2}
    tok = build_extended_gguf_tokenizer(arch, meta, config_type())
    assert tok.decode(tok.encode(text, add_special_tokens=False)) == text
    assert tok.bos_token_id == 1 and tok.eos_token_id == 2
