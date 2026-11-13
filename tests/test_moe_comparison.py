"""Parity includes bytes and observed source identities, not just float equality."""
import json

import pytest
import torch

from scripts.compare_shared_moe import compare


@pytest.mark.parametrize("change",[None,"signed_zero","checkpoint","hf_source"])
def test_parity_rejects_byte_or_source_changes(tmp_path,change):
    common=dict(status="PASS",runs=[dict(input_ids=[[1]],state={})],router_choices=[],
        oracle_source_sha256="oracle",hf_olmoe_source_sha256="hf",checkpoint_metadata_sha256={"config":"same"},
        checkpoint_shards={"shard":{"bytes":1}},peak_vram=0,peak_reserved=0,
        vram_budget_gib=1,allocated_after_close=0,native=False,runtime_sources={})
    base=dict(common,mode="reference")
    candidate=dict(common,mode="streamed")
    values=torch.tensor([[0.,1.]])
    actual=values.clone()
    if change=="signed_zero":actual[0,0]=-0.
    if change=="checkpoint":candidate["checkpoint_metadata_sha256"]={"config":"changed"}
    if change=="hf_source":candidate["hf_olmoe_source_sha256"]="changed"
    for name,report,tensor in (("base",base,values),("candidate",candidate,actual)):
        (tmp_path/(name+".json")).write_text(json.dumps(report),encoding="utf-8")
        torch.save([tensor],tmp_path/(name+"-logits.pt"))
    result=compare(tmp_path,"base",["candidate"])
    assert result["status"]==("PASS" if change is None else "FAIL")
    if change=="signed_zero":
        assert torch.equal(values,actual) and not result["results"][0]["bitwise_logits"]
