"""Behavioral checks against an unmodified full HF graph, not just counters."""
import pytest
import torch
from transformers import OlmoeConfig, OlmoeForCausalLM

from src.shared_resources import SharedWeightResources
from src import olmoe_integrated as adapter
from scripts.verify_shared_moe import state_hashes


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_nested_resources_reject_stale_and_live_overwrite(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    backend = SharedWeightResources({"weight":((4,4),torch.float32)}, device, torch.float32)
    try:
        dense, d = backend.acquire(0, ("dense",0), {"weight":torch.eye(4)})
        expert, e = backend.acquire(1, ("expert",0,1), {"weight":torch.full((4,4),2.)})
        out = d["weight"] @ e["weight"]
        with pytest.raises(RuntimeError, match="live"):
            backend.acquire(0,("dense",1),{"weight":torch.eye(4)})
        backend.release(expert)
        newer, e2 = backend.acquire(1,("expert",0,2),{"weight":torch.full((4,4),3.)})
        with pytest.raises(RuntimeError, match="stale"):
            backend.release(expert)
        out2 = d["weight"] @ e2["weight"]
        backend.release(newer)
        backend.release(dense)
        assert torch.equal(out.cpu(),torch.full((4,4),2.))
        assert torch.equal(out2.cpu(),torch.full((4,4),3.))
        assert backend.snapshot()["peak_active_slots"] == 2
    finally:
        backend.close()
        backend.close()


def make_fixture(tmp_path, monkeypatch, device):
    torch.manual_seed(413)
    config = OlmoeConfig(hidden_size=16, intermediate_size=16, num_experts=4,
        num_experts_per_tok=2, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=2, vocab_size=32, pad_token_id=0)
    config._attn_implementation = config._experts_implementation = "eager"
    original = OlmoeForCausalLM(config).eval().to(device)
    config.save_pretrained(tmp_path)
    weights = {}
    for name, p in original.named_parameters():
        if ".experts." not in name:
            weights[name] = p.detach().cpu().clone()
            continue
        prefix, projection = name.rsplit(".",1)
        for expert in range(config.num_experts):
            if projection == "gate_up_proj":
                weights[f"{prefix}.{expert}.gate_proj.weight"] = p[expert,:16].detach().cpu().clone()
                weights[f"{prefix}.{expert}.up_proj.weight"] = p[expert,16:].detach().cpu().clone()
            else:
                weights[f"{prefix}.{expert}.down_proj.weight"] = p[expert].detach().cpu().clone()
    class Provider:
        def __init__(self,*args,**kwargs): pass
        def read(self,names): return {n:weights[n] for n in names}
    monkeypatch.setattr(adapter,"IndexedAirLLMProvider",Provider)
    monkeypatch.setattr(adapter.AutoTokenizer,"from_pretrained",lambda *a,**k:None)
    return original


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("gpu_mode",[None,"shadow","apply"])
def test_shared_full_graph_prefill_decode_state_and_router(tmp_path,monkeypatch,device,gpu_mode):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    original = make_fixture(tmp_path,monkeypatch,device)
    routes = []
    handles = [layer.mlp.gate.register_forward_hook(lambda m,a,o: routes.append(o[2].detach().cpu().tolist())) for layer in original.model.layers]
    if gpu_mode:
        from test_gpu_expert_residency import TestAdmission
        monkeypatch.setattr(adapter,"NativeTierPolicy",lambda dll,n,capacity,mode:TestAdmission(capacity,replace=False))
    runtime = adapter.AirColibriOLMoE(tmp_path,"unused",device=device,dtype=torch.float32,
        ram_bytes=2*3*16*16*4,stream_dense=True,gpu_expert_slots=2 if gpu_mode else 0,
        gpu_cache_mode=gpu_mode or "apply",gpu_native_dll="synthetic" if gpu_mode else None)
    state_a = state_b = None
    try:
        for ids in ([[1,2,3,4]],[[5]],[[6]],[[7]]):
            ids = torch.tensor(ids,device=device)
            with torch.no_grad():
                ref = original(input_ids=ids,past_key_values=state_a,use_cache=True)
                actual,state_b = runtime.forward(ids,state_b)
            state_a=ref.past_key_values
            assert torch.equal(ref.logits[:,-1,:],actual)
            assert state_hashes(state_a)==state_hashes(state_b)
            assert all(p.device.type=="meta" for layer in runtime.model.model.layers for p in layer.parameters())
            assert runtime.resources.active=={}
        assert routes == [r["ids"] for r in runtime.router_choices]
        assert runtime.resources.counts["dense"]==8
        assert runtime.resources.counts["expert"]==runtime.store.expert_calls
        assert runtime.resources.snapshot()["violations"]==0
        if gpu_mode:
            stats=runtime.resources.snapshot()["expert_residency"]["stats"]
            assert stats.get("hits" if gpu_mode=="apply" else "shadow_hits",0)>0
    finally:
        for h in handles:h.remove()
        runtime.close()
    assert runtime.resources.ring.arena is None
    assert runtime.resources.pool.arena is None


def test_forward_error_is_fail_closed_and_drains_resources(tmp_path,monkeypatch):
    make_fixture(tmp_path,monkeypatch,"cpu")
    runtime = adapter.AirColibriOLMoE(tmp_path,"unused",device="cpu",dtype=torch.float32,stream_dense=True)
    def fail(*args):raise RuntimeError("injected consumer failure")
    h=runtime.model.model.layers[0].mlp.experts.register_forward_pre_hook(fail)
    with pytest.raises(RuntimeError, match="injected"):
        runtime.forward(torch.tensor([[1]]))
    with pytest.raises(RuntimeError,match="failed"):
        runtime.forward(torch.tensor([[1]]))
    h.remove()
    runtime.close()
    runtime.close()
    assert runtime.resources.ring.arena is None


def test_bad_payload_cannot_replace_live_values_or_submit_dma():
    backend=SharedWeightResources({"weight":((4,4),torch.float32)},"cpu",torch.float32)
    try:
        lease,views=backend.acquire(0,("dense",0),{"weight":torch.eye(4)})
        for payload in ({},{"weight":torch.ones(1,4)},{"weight":torch.ones(4,4,dtype=torch.float64)}):
            with pytest.raises(ValueError):
                backend.acquire(1,("expert",0,0),payload)
        assert backend.ring.total_loads==1
        assert torch.equal(views["weight"],torch.eye(4))
        backend.release(lease)
    finally:
        backend.close()
