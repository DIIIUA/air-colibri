"""Non-invasive AirColibri control-plane contract.

The backend retains ownership of the real forward, tensors, memory slots,
CUDA events, expert routing, I/O and state. This module only records events,
checks mandatory decoder order and creates *advisory-only* placement intents.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from collections import Counter, deque
from typing import Any, Callable, Optional
import time


class ContractViolation(RuntimeError):
    pass


@dataclass(frozen=True)
class BackendCapabilities:
    name: str
    supports_layer_hooks: bool
    supports_expert_hooks: bool
    supports_live_prefetch_cancel: bool = False
    supports_residency_lease: bool = False
    supports_live_cache_stats: bool = False
    supports_native_cuda: bool = False
    proof: str = "source-inspected"


@dataclass(frozen=True)
class RuntimeEvent:
    seq: int
    kind: str
    step: int | None
    layer: int | None
    component: str | None
    expert: int | None
    elapsed_ms: float | None
    logical_bytes: int | None
    timestamp_ns: int
    source: str


@dataclass(frozen=True)
class PlacementIntent:
    layer: int
    component: str
    target: str
    estimated_bytes: int
    reason: str
    advisory: bool = True
    dispatched: bool = False


class TraceContract:
    """Strict event grammar: complete 0..N-1 order, no physical actuation."""
    def __init__(self, num_layers: int, on_access: Optional[Callable] = None,
                 max_events: int = 32768):
        if num_layers < 1:
            raise ValueError("Need at least one layer")
        self.num_layers = num_layers
        self.on_access = on_access
        self.max_events = max_events
        if max_events < 1:
            raise ValueError("Trace capacity must be positive")
        self.events = deque(maxlen=max_events)
        self.event_counts = Counter()
        self.event_sequence = 0
        self.current_step: int | None = None
        self._next_step = 0
        self._expected_layer = 0
        self._active_layer: int | None = None
        self.completed_steps = 0
        self.io_calls = Counter()
        self.advisory_intents: list[PlacementIntent] = []

    def _emit(self, kind, layer=None, component=None, expert=None,
              elapsed_ms=None, logical_bytes=None, source="runtime"):
        event = RuntimeEvent(self.event_sequence, kind, self.current_step, layer,
                             component, expert, elapsed_ms, logical_bytes,
                             time.perf_counter_ns(), source)
        self.events.append(event)
        self.event_sequence += 1
        self.event_counts[kind] += 1
        return event

    def start_step(self):
        if self.current_step is not None or self._active_layer is not None:
            raise ContractViolation("Nested or incomplete token forward")
        self.current_step = self._next_step
        self._next_step += 1
        self._expected_layer = 0
        self._emit("step_begin")
        return self.current_step

    def enter_layer(self, index: int):
        if self.current_step is None or self._active_layer is not None:
            raise ContractViolation("Layer begin without step or with overlap")
        if index != self._expected_layer:
            raise ContractViolation(
                f"Layer order violation: expected {self._expected_layer}, got {index}")
        self._active_layer = index
        self._emit("layer_begin", layer=index)

    def component_called(self, index: int, component: str,
                         expert: int | None = None):
        if self._active_layer != index:
            raise ContractViolation("Component outside active decoder layer")
        self._emit("component_call", layer=index, component=component, expert=expert)
        if self.on_access is not None:
            self.on_access(index, component, self.current_step)

    def leave_layer(self, index: int):
        if self._active_layer != index:
            raise ContractViolation("Layer end unmatched or out of order")
        self._emit("layer_end", layer=index)
        self._active_layer = None
        self._expected_layer += 1

    def finish_step(self):
        if self.current_step is None or self._active_layer is not None:
            raise ContractViolation("Step end without begin or active layer remains")
        if self._expected_layer != self.num_layers:
            raise ContractViolation(
                f"Missing mandatory layers: {self._expected_layer}/{self.num_layers}")
        self._emit("step_end")
        self.current_step = None
        self.completed_steps += 1

    def loader_observed(self, kind: str, layer: int, elapsed_ms: float,
                        logical_bytes: int | None, source: str):
        """Callback from an existing backend; not physical disk attribution."""
        if kind not in ("layer_read", "pinned_stage", "expert_read"):
            raise ValueError("Unknown backend data event")
        self.io_calls[kind] += 1
        self._emit(kind, layer=layer, elapsed_ms=elapsed_ms,
                   logical_bytes=logical_bytes, source=source)

    def suggest(self, layer: int, component: str, target: str,
                estimated_bytes: int, reason: str) -> PlacementIntent:
        if target not in ("vram_stand", "ram_pinned", "disk_stream"):
            raise ValueError("Invalid target tier")
        if not 0 <= layer < self.num_layers or estimated_bytes < 0:
            raise ValueError("Invalid placement proposal")
        proposal = PlacementIntent(layer, component, target,
                                   estimated_bytes, reason)
        self.advisory_intents.append(proposal)
        return proposal

    def snapshot(self):
        counts = self.event_counts
        return dict(completed_steps=self.completed_steps,
                    current_step=self.current_step,
                    event_counts=dict(counts), io_calls=dict(self.io_calls),
                    dropped_trace_events=max(0, self.event_sequence - len(self.events)),
                    placement_intents=[asdict(x) for x in self.advisory_intents],
                    no_advisory_writes=True)

    def json_events(self):
        return [asdict(x) for x in self.events]



def declared_backend_capabilities() -> dict[str, BackendCapabilities]:
    """Source-verified capabilities, not a claim of a connected live backend."""
    return {
        "airllm_hf": BackendCapabilities(
            "airllm_hf", supports_layer_hooks=True, supports_expert_hooks=True,
            supports_live_prefetch_cancel=False, supports_residency_lease=False,
            proof="AirLLMBaseModel._install_streaming_hooks and _setup_expert_streaming"),
        "aircolibri_hf": BackendCapabilities(
            "aircolibri_hf", supports_layer_hooks=True, supports_expert_hooks=False,
            supports_live_cache_stats=True,
            proof="AirColibriModel._install_pipeline_hooks + StableWeightManager"),
        "colibri_native": BackendCapabilities(
            "colibri_native", supports_layer_hooks=False,
            supports_expert_hooks=False, supports_residency_lease=False,
            supports_native_cuda=True,
            proof="c/tier.h and expert_store.h are C-native, not a Python ABI"),
    }
