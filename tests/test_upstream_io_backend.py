"""Backend selection must retain original bytecode and isolate module globals."""
from pathlib import Path
from types import ModuleType

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from src import upstream_bridge


def test_per_reader_backend_preserves_bytecode_globals_and_selected_reads(tmp_path,monkeypatch):
    package=tmp_path/"source/air_llm/airllm"
    package.mkdir(parents=True)
    file=package/"utils.py"
    file.write_text("# Test fixture for the external function binding seam.\n")
    module=ModuleType("test_original_reader")
    module.__file__=str(file)
    module.__dict__.update(Path=Path,safe_open=safe_open)
    exec('def load_layer_subset(folder, stem, keys):\n    with safe_open(Path(folder)/(stem+".safetensors"), framework="pt") as f:\n        return {k:f.get_tensor(k) for k in keys}\n',module.__dict__)
    original=module.load_layer_subset
    monkeypatch.setattr(upstream_bridge.importlib,"import_module",lambda name:module)
    source=tmp_path/"source"
    default=upstream_bridge.AirLLMReader(source)
    pread=upstream_bridge.AirLLMReader(source,io_backend="pread")
    mmap=upstream_bridge.AirLLMReader(source,io_backend="mmap")
    assert default.read is original
    assert pread.read.__code__ is mmap.read.__code__ is original.__code__
    assert original.__globals__["safe_open"] is safe_open
    assert pread.read.__globals__ is not mmap.read.__globals__
    expected=torch.arange(24,dtype=torch.bfloat16).reshape(4,6)
    save_file({"wanted":expected,"unrequested":torch.ones(20)},tmp_path/"weights.safetensors")
    for reader in (pread,mmap,default):
        out=reader(tmp_path,"weights",["wanted"])
        assert list(out)==["wanted"] and torch.equal(out["wanted"],expected)
        assert reader.logical_bytes==expected.nbytes and reader.calls==1
