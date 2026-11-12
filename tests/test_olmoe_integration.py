import copy
import pytest
import torch
from transformers import OlmoeConfig
from transformers.models.olmoe.modeling_olmoe import OlmoeExperts
from src.olmoe_integrated import ExpertStore, install_expert_views


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_original_olmoe_forward_matches_streamed_expert_views(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(31)
    c = OlmoeConfig(hidden_size=16, intermediate_size=16, num_experts=4,
                   num_experts_per_tok=2, num_hidden_layers=1, num_attention_heads=2)
    c._experts_implementation = "eager"
    original = OlmoeExperts(c).to(device)
    with torch.no_grad():
        original.gate_up_proj.normal_(std=.1)
        original.down_proj.normal_(std=.1)
    candidate = copy.deepcopy(original)
    weights = {}
    for i in range(4):
        prefix = f"model.layers.0.mlp.experts.{i}."
        weights[prefix + "gate_proj.weight"] = original.gate_up_proj[i, :16].detach().cpu().clone()
        weights[prefix + "up_proj.weight"] = original.gate_up_proj[i, 16:].detach().cpu().clone()
        weights[prefix + "down_proj.weight"] = original.down_proj[i].detach().cpu().clone()
    class Provider:
        def read(self, names):
            return {name:weights[name] for name in names}
    store = ExpertStore(Provider(), c, torch.device(device), torch.float32, ram_bytes=2*3*16*16*4)
    owned = store.host_weights(0, 0)
    assert owned["gate_proj"].data_ptr() != weights["model.layers.0.mlp.experts.0.gate_proj.weight"].data_ptr()
    install_expert_views(candidate, store, 0)
    x = torch.randn(3, 16, device=device)
    ids = torch.tensor([[0, 2], [1, 3], [0, 1]], device=device)
    scores = torch.softmax(torch.randn(3, 2, device=device), dim=-1)
    with torch.no_grad():
        expected = original(x, ids, scores)
        actual = candidate(x, ids, scores)
    assert torch.equal(expected, actual)
    assert store.expert_calls == 4 and store.cache.current_bytes <= store.cache.max_bytes
    store.close()
