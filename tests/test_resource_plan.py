from pathlib import Path
import pytest
from src.resource_plan import make_plan, GIB

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("model,fmt", [("deepseek-v4-9b-colibri","q4"),("OLMoE-1B-7B-0125-Instruct","bf16")])
def test_real_checkpoint_plan_is_bounded_and_rejects_shortage(model,fmt):
    p=make_plan(ROOT/"models"/model,weight_format=fmt,ram_gib=4,vram_gib=5)
    assert p.source_weight_bytes>p.ram_budget_bytes+p.vram_budget_bytes
    assert p.ram_cache_bytes==256*2**20
    p.validate(available_vram=8*GIB,available_ram=8*GIB,commit_available=32*GIB)
    with pytest.raises(MemoryError,match="RAM"):
        p.validate(available_vram=8*GIB,available_ram=GIB,commit_available=32*GIB)
    with pytest.raises(MemoryError,match="commit"):
        p.validate(available_vram=8*GIB,available_ram=8*GIB,commit_available=GIB)


def test_moe_rejects_unimplemented_quantization():
    with pytest.raises(ValueError,match="BF16"):
        make_plan(ROOT/"models/OLMoE-1B-7B-0125-Instruct",weight_format="q4")


def test_impossible_budget_rejected_before_model_allocation():
    p=make_plan(ROOT/"models/deepseek-v4-9b-colibri",weight_format="bf16",slots=3,ram_gib=1,vram_gib=1)
    with pytest.raises(MemoryError):
        p.validate(available_vram=8*GIB,available_ram=8*GIB,commit_available=32*GIB)


def test_shared_moe_plan_fits_budget_that_rejects_resident_dense():
    model=ROOT/"models/OLMoE-1B-7B-0125-Instruct"
    shared=make_plan(model,vram_gib=1.2,cache_mib=64,moe_placement="streamed")
    resident=make_plan(model,vram_gib=1.2,cache_mib=64,moe_placement="resident")
    shared.validate(available_vram=8*GIB,available_ram=8*GIB,commit_available=32*GIB)
    with pytest.raises(MemoryError,match="VRAM"):
        resident.validate(available_vram=8*GIB,available_ram=8*GIB,commit_available=32*GIB)
    assert shared.pinned_bytes>0 and shared.gpu_weight_bytes<resident.gpu_weight_bytes


@pytest.mark.parametrize("layout,extra_mib,budget",[("union",92,1.05),("compact",24,0.94)])
def test_gpu_expert_slots_reserve_full_physical_stride_before_startup(layout,extra_mib,budget):
    model=ROOT/"models/OLMoE-1B-7B-0125-Instruct"
    base=make_plan(model,vram_gib=budget,cache_mib=64,moe_slot_layout=layout)
    hot=make_plan(model,vram_gib=budget,cache_mib=64,gpu_expert_slots=2,moe_slot_layout=layout)
    assert hot.gpu_expert_reservation_bytes==extra_mib*2**20
    assert hot.pinned_bytes-base.pinned_bytes==hot.gpu_weight_bytes-base.gpu_weight_bytes==hot.gpu_expert_reservation_bytes
    base.validate(available_vram=8*GIB,available_ram=8*GIB,commit_available=32*GIB)
    with pytest.raises(MemoryError,match="VRAM"):
        hot.validate(available_vram=8*GIB,available_ram=8*GIB,commit_available=32*GIB)
