from types import SimpleNamespace

import pytest
import torch

from aircolibri_cli.unified_runtime import AirColibriUnifiedEngine, ContextBudgetExceeded


class Tokenizer:
    eos_token_id = 0
    chat_template = None

    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=torch.tensor([[1] * 8]))


def test_context_rejects_requested_output_before_allocating_cache():
    engine = AirColibriUnifiedEngine.__new__(AirColibriUnifiedEngine)
    engine.engine = SimpleNamespace(tokenizer=Tokenizer())
    engine.plan = SimpleNamespace(max_context_tokens=16)
    # No model or CUDA context is needed to reject an oversized request.
    messages = [{"role": "user", "content": "hello"}]
    with pytest.raises(ContextBudgetExceeded, match="8 prompt.*9 requested"):
        engine._prepare_prompt_and_cache(messages, False, False, max_new_tokens=9)
    ids, reused, past, _ = engine._prepare_prompt_and_cache(messages, False, False, max_new_tokens=8)
    assert len(ids) == 8 and reused == 0 and past is None
    engine.engine.config = SimpleNamespace(n_positions=12)
    with pytest.raises(ContextBudgetExceeded, match="tokens > 12"):
        engine._prepare_prompt_and_cache(messages, False, False, max_new_tokens=5)


def test_cached_chat_is_checked_before_unparking():
    engine = AirColibriUnifiedEngine.__new__(AirColibriUnifiedEngine)
    engine.engine = SimpleNamespace(tokenizer=Tokenizer())
    engine.plan = SimpleNamespace(max_context_tokens=16)
    engine._session_messages = [{"role": "user", "content": "old"}, {"role": "assistant", "content": "reply"}]
    class Session:
        fed = [1] * 8
        unfed_tail = []
        parked_backbone = {}
        chatml_think_format = False
        def kv_prefix_reuse(self, ids):
            raise AssertionError("Oversized request reached session reuse")
    engine.session = Session()
    with pytest.raises(ContextBudgetExceeded):
        engine._prepare_prompt_and_cache(engine._session_messages + [{"role": "user", "content": "next"}],
                                         False, True, max_new_tokens=9)
