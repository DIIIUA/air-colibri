"""Correctness checks of the real engine on the GPU (skipped without CUDA or a local model).

These cover what the CPU tests cannot: chunked prefill, the absence of full-vocabulary prefill
logits, MTP speculation with rollback, multi-row GatedDeltaNet steps, KV + DeltaNet prefix reuse
between turns and VRAM cleanup after cancellation.
"""
import copy
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _model_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from aircolibri_cli.model_source import resolve_model_dir

        resolve_model_dir(None, ROOT)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _model_available(), reason="needs CUDA and a supported model in models/")

MiB = 1024 ** 2
FACT_PROMPT = [{"role": "user", "content": "Перечисли пять планет Солнечной системы по порядку от Солнца, через запятую."}]


@pytest.fixture(scope="module")
def eng():
    from aircolibri_cli.unified_runtime import AirColibriUnifiedEngine

    e = AirColibriUnifiedEngine(vram_mode="normal", verbose=False)
    yield e
    e.close()


def _answer(eng, messages, n=40, **kw):
    done = [ev for ev in eng.stream_chat(messages, max_new_tokens=n, enable_thinking=False, **kw) if ev["type"] == "done"]
    return done[0]


def _long_messages(repeats: int):
    text = "Кольцевой буфер видеопамяти держит слоты одинакового размера: пока один считает, соседний загружается. "
    return [
        {"role": "system", "content": text * repeats},
        {"role": "user", "content": "Сколько будет 17 + 25? Ответь только числом."},
    ]


def test_chunked_prefill_matches_single_pass(eng):
    msgs = _long_messages(25)
    saved = eng.prefill_chunk_tokens
    try:
        eng.prefill_chunk_tokens = 4096
        single = _answer(eng, msgs, n=12)
        eng.prefill_chunk_tokens = 96
        chunked = _answer(eng, msgs, n=12)
    finally:
        eng.prefill_chunk_tokens = saved
    assert single["metrics"]["prompt_tokens"] > 500
    assert chunked["answer_text"] == single["answer_text"]
    assert "42" in single["answer_text"]


def test_prefill_does_not_materialise_full_vocab_logits(eng):
    msgs = _long_messages(120)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated() / MiB
    done = _answer(eng, msgs, n=4)
    m = done["metrics"]
    assert m["prompt_tokens"] > 3000
    # [1, S, 248320] BF16 logits alone would need ~0.47 MiB per prompt token (> 1.4 GiB here).
    assert m["peak_vram_mib"] - base < 700, m


def test_mtp_speculation_matches_plain_greedy(eng):
    if not eng.mtp_enabled:
        pytest.skip("checkpoint has no MTP head")
    with_mtp = _answer(eng, FACT_PROMPT, n=48)
    eng.mtp_enabled = False
    try:
        plain = _answer(eng, FACT_PROMPT, n=48)
    finally:
        eng.mtp_enabled = True
    assert with_mtp["metrics"]["avg_tokens_per_backbone_pass"] > 1.0  # drafts were accepted
    assert with_mtp["answer_text"] == plain["answer_text"]


def test_multirow_deltanet_step_matches_sequential_steps(eng):
    from aircolibri_cli.unified_runtime import forward_step_all_logits

    tok = eng.tokenizer
    ids = tok("Колибри — самая маленькая птица на Земле, и её сердце", add_special_tokens=False).input_ids
    prompt, tail = ids[:-3], ids[-3:]
    with torch.no_grad():
        _, past = forward_step_all_logits(eng.engine, torch.tensor([prompt]), past_key_values=None, logits_to_keep=1)
        past_b = copy.deepcopy(past)
        lg_multi, _ = forward_step_all_logits(eng.engine, torch.tensor([tail]), past_key_values=past)
        lg_seq = None
        for t in tail:
            lg_seq, past_b = forward_step_all_logits(eng.engine, torch.tensor([[t]]), past_key_values=past_b)
    a = lg_multi[0, -1].float()
    b = lg_seq[0, -1].float()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    assert int(a.argmax()) == int(b.argmax())
    assert cos > 0.999, cos
    del past, past_b, lg_multi, lg_seq
    eng.reset_session()


def test_session_prefix_reuse_matches_fresh_prompt(eng):
    eng.reset_session()
    conv = [{"role": "user", "content": "Назови столицу Японии одним словом."}]
    first = _answer(eng, conv, n=24, use_session_cache=True)
    conv = conv + [{"role": "assistant", "content": first["answer_text"]},
                   {"role": "user", "content": "А столицу Италии? Одним словом."}]
    reused = _answer(eng, conv, n=24, use_session_cache=True)
    eng.reset_session()
    fresh = _answer(eng, conv, n=24, use_session_cache=False)
    assert reused["metrics"]["cached_prefix_tokens"] > 0
    assert reused["answer_text"] == fresh["answer_text"]
    eng.reset_session()


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_cancellation_releases_all_generation_vram(eng, phase):
    from aircolibri_cli.unified_runtime import GenerationCancelled

    eng.reset_session()
    _answer(eng, FACT_PROMPT, n=8)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    calls = {"n": 0}

    def cancel_check():
        calls["n"] += 1
        return calls["n"] > (2 if phase == "prefill" else 12)

    saved = eng.prefill_chunk_tokens
    eng.prefill_chunk_tokens = 64
    try:
        with pytest.raises(GenerationCancelled) as exc:
            for _ in eng.stream_chat(_long_messages(10), max_new_tokens=200, enable_thinking=True,
                                     use_session_cache=True, cancel_check=cancel_check):
                pass
    finally:
        eng.prefill_chunk_tokens = saved
    assert exc.value.phase == phase
    torch.cuda.synchronize()
    assert abs(torch.cuda.memory_allocated() - base) < 8 * MiB
    assert eng.session.parked_backbone is None  # partial state is never kept for the next turn
