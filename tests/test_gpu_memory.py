"""Memory reported by another process must affect this process's startup budget."""
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
import torch

from aircolibri_cli import gpu_memory


@pytest.mark.parametrize("driver_free,cuda_free,expected", [(4,7,4), (6,3,3), (0,7,0)])
def test_lower_valid_counter_matches_uuid_not_visible_ordinal(monkeypatch, driver_free, cuda_free, expected):
    gib = 2**30
    seen = []
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (cuda_free*gib, 8*gib))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(uuid="remapped-gpu-uuid"))
    def nvml(uuid):
        seen.append(uuid)
        return driver_free*gib, 8*gib
    monkeypatch.setattr(gpu_memory, "_nvml_memory", nvml)
    result = gpu_memory.measure_gpu_memory("cuda:0")
    assert result["source"] == "cuda_nvml_min" and result["free_bytes"] == expected*gib
    assert seen == ["remapped-gpu-uuid"]


def test_optional_nvml_failure_keeps_cuda_with_diagnostic(monkeypatch):
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (700, 800))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(uuid="uuid"))
    def unavailable(uuid):
        raise OSError("NVML library absent")
    monkeypatch.setattr(gpu_memory, "_nvml_memory", unavailable)
    result = gpu_memory.measure_gpu_memory("cuda:0")
    assert result["source"] == "cuda_only" and result["free_bytes"] == 700
    assert "absent" in result["physical_counter_unavailable"]


def test_physical_budget_changes_residency_without_changing_kv_reserve(monkeypatch):
    from aircolibri_cli.unified_runtime import plan_vram, GiB, MiB
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (7*GiB, 8*GiB))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(uuid="uuid"))
    plans = []
    for free in (6*GiB,4*GiB):
        monkeypatch.setattr(gpu_memory, "_nvml_memory", lambda uuid: (free, 8*GiB))
        m = gpu_memory.measure_gpu_memory("cuda:0")
        plans.append(plan_vram("normal", total_bytes=m["total_bytes"], free_bytes=m["free_bytes"],
            layer_bytes=[192*MiB]*20, lm_head_bytes=256*MiB, mtp_bytes=0,
            kv_bytes_per_token=4096, recurrent_state_bytes=0, hidden_size=1024, intermediate_size=4096,
            max_context_tokens=2048, prefill_chunk_tokens=64))
    assert plans[0].full_residency and not plans[1].full_residency
    assert plans[0].reserve_bytes == plans[1].reserve_bytes
    assert plans[0].resident_layers > plans[1].resident_layers


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_separate_two_gib_allocation_reduces_physical_startup_budget(tmp_path):
    before = gpu_memory.measure_gpu_memory(0)
    if before["source"] != "cuda_nvml_min" or before["free_bytes"] < 3*2**30:
        pytest.skip("NVML and at least 3 GiB available required")
    root = Path(__file__).resolve().parents[1]
    ready, stop = tmp_path/"ready.json", tmp_path/"stop"
    with (tmp_path/"holder.log").open("w") as log:
        child = subprocess.Popen([sys.executable, str(root/"benchmarks/streaming_decode.py"),
            "--hold-vram", "2048", "--ready", str(ready), "--stop", str(stop)], stdout=log,
            stderr=subprocess.STDOUT, creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        try:
            deadline = time.monotonic()+120
            while not ready.exists():
                assert child.poll() is None and time.monotonic() < deadline, "VRAM holder startup failed"
                time.sleep(.1)
            state = json.loads(ready.read_text("utf-8"))
            assert state.get("allocated_bytes") == 2*2**30, state
            after = gpu_memory.measure_gpu_memory(0)
            assert after["source"] == "cuda_nvml_min"
            assert before["physical_free_bytes"]-after["physical_free_bytes"] > 1.5*2**30
            assert before["free_bytes"]-after["free_bytes"] > 1.5*2**30
        finally:
            stop.touch()
            try:
                child.wait(20)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait()
