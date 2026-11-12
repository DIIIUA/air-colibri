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
