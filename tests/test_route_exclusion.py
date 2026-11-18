"""A scoped exclusion must prevent I/O without becoming a permanent forecast ban."""
from types import SimpleNamespace
import threading

import pytest
import torch

from src.route_exclusion import RouteExclusion


@pytest.mark.parametrize("rows", [1, 3])
@torch.no_grad()
def test_exact_zero_routes_do_not_request_weights_and_next_scope_can_read_them(rows):
    from aircolibri_cli.model_source import _fast_moe_dispatch
    torch.manual_seed(321)
    bank = torch.randn(4, 3, 16, 16)*.05
    calls = []
    def load(ids, device, dtype, allow_evict):
        calls.extend(ids)
        weights = bank[ids].to(device=device, dtype=dtype)
        return weights[:, 0], weights[:, 1], weights[:, 2]
    class Experts(torch.nn.Module):
        act_fn = staticmethod(torch.nn.functional.silu)
        def forward(self, hidden, indices, coefficients):
            return _fast_moe_dispatch(self, hidden, indices, coefficients, 4, load)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = Experts()
        def forward(self, *args):
            return self.experts(*args)
    model = Model()
    engine = SimpleNamespace(model=model, config=SimpleNamespace(model_type="granitemoe"),
                             model_source=SimpleNamespace(_expert_loader=SimpleNamespace(num_experts_by_layer={0:4})),
                             _decoder_layers=lambda: [SimpleNamespace(block_sparse_moe=model)])
    hidden = torch.randn(rows, 16)
    indices = torch.tensor([[0, 1]]*rows)
    weights = torch.tensor([[1., 0.]]*rows)
    expected = model(hidden, indices, weights)
    assert set(calls) == {0, 1}
    calls.clear()
    guard = RouteExclusion(engine)
    try:
        actual = model(hidden, indices, weights)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert set(calls) == {0}
        assert guard.decisions[0]["allowed"] == [0]
        assert guard.stats()["zero_weight_pairs_excluded"] == rows
        calls.clear()
        model(hidden, indices, torch.tensor([[0., 1.]]*rows))
        assert set(calls) == {1}  # Previous exclusion is invalid in this work scope.
        assert guard.decisions[0]["allowed"] == [1]
    finally:
        guard.close()


def test_rejected_route_rows_do_not_change_retained_map():
    from src.probability_warmup import PrefillRegionMap
    model = torch.nn.Identity()
    engine = SimpleNamespace(model=model, config=SimpleNamespace(model_type="llama"), model_source=None)
    guard = RouteExclusion(engine)
    region = PrefillRegionMap({0: 4}, 2, drift_patience=1)
    region.seed(0, [10, 9, 0, 0])
    initial = region.stats()["initial_sha256"]
    engine.probability_warmup = SimpleNamespace(region_map=region)
    guard.commit_capture({0:[[0,1], [2,3], [2,3]]}, 1)
    assert region.regions[0] == (0, 1)
    assert region.stats()["initial_sha256"] == initial
    assert guard.stats()["discarded_route_rows"] == 2
    guard.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_route_cut_stops_unread_optional_experts_but_preserves_next_scope():
    from src.progressive_prefetch import ThreeBankExpertPrefetch
    entered, release = threading.Event(), threading.Event()
    reads = []
    weights = torch.randn(3, 128, 128, dtype=torch.bfloat16)*.01
    def load(layer, eid):
        reads.append(eid)
        if eid == 0 and len(reads) == 1:
            entered.set()
            assert release.wait(10)
        return tuple(weights)
    worker = ThreeBankExpertPrefetch(128, 128, 6, 3, load, lambda: None, "cuda", host_cache_bytes=2**20)
    try:
        worker.submit(0, [0, 1, 2], region=(0, 1), lane=0, step=1)
        assert entered.wait(10)
        worker.restrict_to_route(0, {0}, 1)
        release.set()
        worker.drain()
        assert reads == [0]
        assert worker.stats()["route_pruned_ids"] == 2
        worker.submit(0, [1, 2], region=(0, 1), lane=0, step=2)
        worker.drain()
        assert {1, 2}.issubset(reads)
    finally:
        release.set()
        worker.close()
