import gc
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import psutil
import pytest
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

_TEST_MEMORY_METRICS: List[Tuple[str, float, float, float]] = []


def _release_runtime_memory(reset_peak: bool = False) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if reset_peak:
            torch.cuda.reset_peak_memory_stats()


@pytest.fixture
def engine_factory():
    """Creates AirColibri engine/cache objects and guarantees deterministic cleanup."""
    created = []

    def make(*args, cls=None, **kwargs):
        if cls is None:
            from src.aircolibri_engine import AirColibriEngine
            cls = AirColibriEngine
        obj = cls(*args, **kwargs)
        created.append(obj)
        return obj

    yield make

    for obj in reversed(created):
        close_fn = getattr(obj, "close", None)
        if callable(close_fn):
            close_fn()
    created.clear()
    _release_runtime_memory()


@pytest.fixture(autouse=True)
def _cleanup_and_track_memory(request):
    """Resets CUDA peak stats, enforces cleanup between tests, and records RSS/VRAM peaks."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    yield

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
        res_vram_mb = torch.cuda.max_memory_reserved() / (1024**2)
    else:
        peak_vram_mb = 0.0
        res_vram_mb = 0.0

    _release_runtime_memory(reset_peak=True)

    rss_mb = psutil.Process().memory_info().rss / (1024**2)

    _TEST_MEMORY_METRICS.append((request.node.nodeid, rss_mb, peak_vram_mb, res_vram_mb))


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not _TEST_MEMORY_METRICS:
        return
    terminalreporter.write_sep("-", "Per-Test Memory Summary (after cleanup)")
    for nodeid, rss_mb, peak_vram_mb, res_vram_mb in _TEST_MEMORY_METRICS:
        terminalreporter.write_line(
            f"{nodeid:<85} | RSS: {rss_mb:7.1f} MB | PeakVRAM: {peak_vram_mb:7.1f} MB (res: {res_vram_mb:7.1f} MB)"
        )
