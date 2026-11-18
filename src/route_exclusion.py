"""Execution-scoped MoE exclusions, distinct from a probability forecast.

Only known output-weighted Granite/OLMoE contracts are changed. A zero route
coefficient removes its expert before I/O. All other actual routes remain legal,
even outside the retained prefill map. No decision is carried to a later token.
"""
from collections import Counter

import torch


class RouteExclusion:
    def __init__(self, engine):
        self.engine = engine
        self.counts = Counter()
        self.handles = []
        self.capture = None
        self.decisions = {}
        self.scope = 0
        self.handles.append(engine.model.register_forward_pre_hook(self._begin))
        loader = getattr(engine.model_source, "_expert_loader", None)
        self.kind = engine.config.model_type
        if self.kind not in {"granitemoe", "olmoe"} or loader is None:
            return
        numbers = getattr(loader, "num_experts_by_layer", {})
        for i, layer in enumerate(engine._decoder_layers()):
            if i not in numbers:
                continue
            block = layer.block_sparse_moe if self.kind == "granitemoe" else layer.mlp
            self.handles.append(block.experts.register_forward_pre_hook(self._observe(i, numbers[i]), prepend=True))

    def _begin(self, module, args):
        if self._isolated_probe():
            return
        self.scope += 1
        self.decisions.clear()

    def _isolated_probe(self):
        progressive = getattr(self.engine, "progressive", None)
        warmup = getattr(self.engine, "probability_warmup", None)
        return bool((getattr(progressive, "auxiliary_depth", 0) and not getattr(progressive, "verification_depth", 0))
                    or getattr(warmup, "_auxiliary_depth", 0) or getattr(warmup, "auxiliary_depth", 0))

    def _observe(self, layer, universe):
        def observe(module, args):
            hidden, indices, weights = args[:3]
            # The sentinel is already understood by the on-demand dispatcher.
            # A probability prediction never supplies this mask.
            masked = indices.masked_fill(weights == 0, -1)
            rows = masked.detach().cpu().tolist()
            loader = getattr(self.engine.model_source, "_expert_loader", None)
            if hasattr(loader, "observed_route_rows"):
                loader._observed_routes = (layer, masked, rows)
            zero_pairs = int(((indices >= 0) & (masked < 0)).sum().item())
            self.counts["zero_weight_pairs_excluded"] += zero_pairs
            if self.capture is not None:
                self.capture.setdefault(layer, []).extend(rows)
            observed_rows = self.capture[layer] if self.capture is not None else rows
            allowed = {int(e) for row in observed_rows for e in row if 0 <= int(e) < universe}
            isolated = self._isolated_probe()
            if not isolated:
                self.decisions[layer] = {"scope": self.scope, "allowed": sorted(allowed),
                                         "excluded_count": universe-len(allowed)}
            warmup = getattr(self.engine, "probability_warmup", None)
            prefetch = getattr(warmup, "prefetch", None)
            verification = getattr(self.engine, "_rowwise_verification", None)
            routes_complete = verification is None or verification.rows < 2 or verification.current_row == verification.rows-1
            if prefetch is not None and not isolated and routes_complete and not getattr(warmup, "is_prefill", True):
                prefetch.restrict_to_route(layer, allowed, warmup.step)
            if self.capture is None and not isolated:
                self.counts["committed_route_rows"] += len(rows)
                self.counts["unrouted_experts_excluded"] += universe-len(allowed)
            return (hidden, masked, weights, *args[3:])
        return observe

    def start_capture(self):
        if self.capture is not None:
            raise RuntimeError("Nested route verification is not supported")
        self.capture = {}

    def finish_capture(self):
        result, self.capture = self.capture, None
        return result

    def cancel_capture(self):
        self.capture = None

    def commit_capture(self, trace, keep):
        warmup = getattr(self.engine, "probability_warmup", None)
        region_map = getattr(warmup, "region_map", None)
        for layer, rows in (trace or {}).items():
            committed = rows[:keep]
            self.counts["discarded_route_rows"] += len(rows)-keep
            self.counts["committed_route_rows"] += len(committed)
            allowed = {e for row in committed for e in row if e >= 0}
            decision = self.decisions.get(layer)
            if decision is not None:
                universe = decision["excluded_count"]+len(decision["allowed"])
                decision.update(allowed=sorted(allowed), excluded_count=universe-len(allowed))
            if region_map is not None:
                changed = False
                for row in committed:
                    valid = [e for e in row if e >= 0]
                    if valid:
                        changed = region_map.observe(layer, valid) or changed
                if changed and hasattr(warmup, "_submit"):
                    warmup._submit(layer, "verified_route_shift")

    def stats(self):
        return dict(self.counts, policy="current_router_scope", scope=self.scope,
                    eligible_layers=len(self.handles)-1,
                    decisions={str(i): dict(v) for i, v in self.decisions.items()})

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.cancel_capture()
