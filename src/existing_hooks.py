"""Attach telemetry to EXISTING inference hooks, not to a new loader.

No new forward, no new quantizer, no new memory manager and no hooks removed
from the original backend until the instrumented object is explicitly closed.
"""
from __future__ import annotations
from functools import wraps
from pathlib import Path
import time
from typing import Any

import torch
from torch import nn

from src.catalog_bridge import ExistingCrossLayerCatalog, component_for_module
from src.control_contract import TraceContract, ContractViolation


def tensor_payload_bytes(result: Any) -> int | None:
    if not isinstance(result, dict):
        return None
    if not all(isinstance(t, torch.Tensor) for t in result.values()):
        return None
    return sum(t.numel() * t.element_size() for t in result.values())


class ExistingForwardHookAdapter:
    """Read-only execution trace on the REAL decoder modules used by a backend.

    Hooks are registered after the pre-existing backend hooks. Thus they cannot
    preempt loading, reorder the layers, schedule transfers or mutate buffers.
    """
    def __init__(self, engine, catalog: ExistingCrossLayerCatalog,
                 backend_name="aircolibri_hf", trace_limit=32768):
        self.engine = engine
        self.catalog = catalog
        self.backend_name = backend_name
        self.contract = TraceContract(catalog.num_layers,
                                      catalog.record_actual_access,
                                      max_events=trace_limit)
        self.handles = []
        self._original_methods = []
        self._attached = False
        self.hooked_linears = 0
        self.hooked_norms = 0
        self.hooked_experts = 0

    def _runtime_layers(self):
        if self.backend_name == "airllm_hf":
            # AirLLM's layer_names[0] is embedding. Real decoder layers
            # are determined by the configured layer-prefix, not guessed.
            prefix = self.engine.layer_names_dict["layer_prefix"] + "."
            result = {}
            for mod, name in zip(self.engine.layers, self.engine.layer_names):
                if name.startswith(prefix):
                    suffix = name[len(prefix):]
                    if suffix.isdigit():
                        result[int(suffix)] = mod
            layers = [result[i] for i in range(self.catalog.num_layers)]
            root_name = self.engine.layer_names_dict["layer_prefix"].rsplit(".", 1)[0]
            root = self.engine.model.get_submodule(root_name)
            return root, layers
        if self.backend_name == "aircolibri_hf":
            root = self.engine.model.model
            return root, list(root.layers)
        raise ValueError("Unsupported backend for execution hooks")

    def install(self):
        if self._attached:
            raise RuntimeError("Already instrumented")
        root, layers = self._runtime_layers()
        if len(layers) != self.catalog.num_layers:
            raise ContractViolation("Real layer count and checkpoint differ")
        self._attached = True
        try:
            # A model-level hook tracks EVERY real forward, including prefill
            # and decode. Register after the original GPU-loading hooks.
            def root_begin(_module, _args, _kwargs):
                self.contract.start_step()
            def root_end(_module, _args, _kwargs, _output):
                self.contract.finish_step()

            self.handles.append(root.register_forward_pre_hook(
                root_begin, with_kwargs=True))
            self.handles.append(root.register_forward_hook(
                root_end, with_kwargs=True))

            for layer_id, layer in enumerate(layers):
                def before(_module, _args, idx=layer_id):
                    self.contract.enter_layer(idx)
                def after(_module, _args, output, idx=layer_id):
                    self.contract.leave_layer(idx)
                    return None

                self.handles.append(layer.register_forward_pre_hook(before))
                self.handles.append(layer.register_forward_hook(after))

                for name, submodule in layer.named_modules():
                    if not name:
                        continue
                    group = component_for_module(name)
                    if group not in self.catalog.entries[layer_id]:
                        continue
                    eligible = (isinstance(submodule, nn.Linear)
                                or type(submodule).__name__ == "ColibriQ4Linear")
                    is_norm = (group == "norm"
                               and ("Norm" in type(submodule).__name__
                                    or "norm" in type(submodule).__name__))
                    if not eligible and not is_norm:
                        continue
                    def capture(_module, _args, idx=layer_id, comp=group):
                        self.contract.component_called(idx, comp)
                    self.handles.append(
                        submodule.register_forward_pre_hook(capture))
                    self.hooked_linears += int(eligible)
                    self.hooked_norms += int(is_norm)

            if self.backend_name == "airllm_hf":
                self._install_airllm_expert_observers()
            if self.backend_name == "aircolibri_hf":
                self._wrap_existing_manager_calls()
        except BaseException:
            self.close()
            raise
        return self


    def _wrap_existing_manager_calls(self):
        manager = self.engine.weight_manager
        for name, event_type in (
                ("read_layer_tensors", "layer_read"),
                ("load_layer_to_pinned_slot", "pinned_stage")):
            original = getattr(manager, name)
            @wraps(original)
            def traced(*args, _original=original,
                       _event=event_type, **kwargs):
                if args:
                    layer = int(args[0])
                else:
                    layer = int(kwargs.get("layer_idx", -1))
                started = time.perf_counter()
                result = _original(*args, **kwargs)
                elapsed = (time.perf_counter() - started) * 1000
                self.contract.loader_observed(
                    _event, layer, elapsed, tensor_payload_bytes(result),
                    source=f"{self.backend_name}.{_original.__name__}")
                return result
            self._original_methods.append((manager, name, original))
            setattr(manager, name, traced)

    def _install_airllm_expert_observers(self):
        # AirLLM already executes selected expert pre/post hooks. We only
        # register an additional observer on modules which it actually hooked.
        for layer in self.engine.layers:
            for submodule in layer.modules():
                payload = getattr(submodule, "_airllm_expert", None)
                if payload is None:
                    continue
                source_idx, expert_idx = payload
                name = self.engine.layer_names[source_idx]
                prefix = self.engine.layer_names_dict["layer_prefix"] + "."
                if not name.startswith(prefix):
                    continue
                layer_id = int(name.removeprefix(prefix))
                group = f"expert:{expert_idx}"
                if group not in self.catalog.entries[layer_id]:
                    raise ContractViolation(
                        f"AirLLM expert {layer_id}/{expert_idx} missing in checkpoint")
                def observe(_module, _args, i=layer_id, c=group, e=expert_idx):
                    self.contract.component_called(i, c, expert=e)
                self.handles.append(
                    submodule.register_forward_pre_hook(observe))
                self.hooked_experts += 1

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for instance, name, original in reversed(self._original_methods):
            setattr(instance, name, original)
        self._original_methods.clear()
        self._attached = False

    def __enter__(self):
        return self.install()

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False

    def snapshot(self):
        return {
            "backend": self.backend_name,
            "existing_backend_hooks_preserved": True,
            "hooks_on_real_q4_or_linear_modules": self.hooked_linears,
            "hooks_on_real_norm_modules": self.hooked_norms,
            "hooks_on_selected_airllm_experts": self.hooked_experts,
            "trace": self.contract.snapshot(),
            "component_access": self.catalog.plan_snapshot(),
            "external_loader_implemented": False,
            "original_forward_replaced": False,
            "physical_residency_actuated": False,
        }
