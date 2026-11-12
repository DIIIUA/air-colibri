"""Resource guards must stop before allocating model weights."""
import json
from types import SimpleNamespace

import pytest

from scripts import validate_q4_quality as runner


@pytest.mark.parametrize("cuda,available", [(False, 8*2**30), (True, 2*2**30)])
def test_resource_rejection_is_not_a_model_failure(tmp_path, monkeypatch, cuda, available):
    output = tmp_path/"attempt"
    monkeypatch.setattr("sys.argv", ["validate_q4_quality.py", "--mode", "reference", "--output", str(output)])
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(runner.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(runner.torch.cuda, "memory_allocated", lambda: 0)
    monkeypatch.setattr(runner.psutil, "virtual_memory", lambda: SimpleNamespace(available=available))
    def unexpected_model(*args, **kwargs):
        raise AssertionError("Model must not be constructed after a failed startup guard")
    monkeypatch.setattr(runner, "AirColibriModel", unexpected_model)
    assert runner.main() == 3
    result = json.loads((output/"result.json").read_text(encoding="utf-8"))
    assert result["status"] == "SAFE_SKIP" and result["rows"] == []
    assert result["available_ram_before"] == available


def test_existing_evidence_is_never_overwritten(tmp_path, monkeypatch):
    output = tmp_path/"attempt"
    output.mkdir()
    evidence = output/"result.json"
    evidence.write_text('{"preserve": true}', encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["validate_q4_quality.py", "--mode", "reference", "--output", str(output)])
    with pytest.raises(FileExistsError):
        runner.main()
    assert evidence.read_text(encoding="utf-8") == '{"preserve": true}'
