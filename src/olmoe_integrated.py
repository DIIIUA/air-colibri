"""OLMoE adapter preserving the installed Transformers eager mathematical graph.

Only the expert weight indexing surface is adapted. Original AirLLM subset
reads supply selected experts; original Colibri C decides bounded RAM admission.
This does not claim native Colibri CUDA math or a new fused MoE kernel.
"""
from collections import defaultdict, deque
import gc
import json
import os
from pathlib import Path

import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from src.bounded_memory_manager import BoundedRAMCache
from src.control_contract import TraceContract
from src.upstream_bridge import AirLLMReader, NativeTierPolicy


class IndexedAirLLMProvider:
    """Map checkpoint-index keys to original AirLLM load_layer_subset calls."""
    def __init__(self, model_dir, source):
        self.folder = Path(model_dir)
        self.index = json.loads((self.folder / "model.safetensors.index.json").read_text())["weight_map"]
        self.reader = AirLLMReader(source)

    def read(self, keys):
        by_shard = defaultdict(list)
        for key in keys:
            by_shard[self.index[key]].append(key)
        result = {}
        for shard, names in by_shard.items():
            if os.environ.get("AIRCOLIBRI_IO_DEBUG") == "1":
                print("AirLLM subset", shard, names, flush=True)
            # Reject path escape before passing external metadata to the reader.
            path = (self.folder / shard).resolve()
            if path.parent != self.folder.resolve() or path.suffix != ".safetensors":
                raise ValueError("Invalid checkpoint shard path")
            result.update(self.reader(str(path.parent), path.stem, names))
        return result


class ExpertStore:
    def __init__(self, provider, config, device, dtype, ram_bytes, native_dll=None, native_mode="apply"):
        self.provider, self.config = provider, config
        self.device, self.dtype = device, dtype
        self.expert_bytes = 3 * config.hidden_size * config.intermediate_size * torch.empty((), dtype=dtype).element_size()
        capacity = ram_bytes // self.expert_bytes
        self.cache = BoundedRAMCache(capacity, "lru", max_bytes=ram_bytes)
        self.policy = NativeTierPolicy(native_dll, config.num_hidden_layers * config.num_experts, capacity,
                                      mode=native_mode) if native_dll else None
        if self.policy:
            self.cache.on_access, self.cache.admission = self.policy.record_access, self.policy.admit
            self.cache.on_placement = self.policy.acknowledge
        self.active_key = None
        self.active = None
        self.reads = self.logical_h2d_bytes = self.expert_calls = 0
        self.on_access = None

    def host_weights(self, layer, expert):
        key = layer * self.config.num_experts + expert
        data = self.cache.get(key)
        if data is None:
            prefix = f"model.layers.{layer}.mlp.experts.{expert}."
            names = [prefix + p + ".weight" for p in ("gate_proj", "up_proj", "down_proj")]
            raw = self.provider.read(names)
            # Own only the expert's bytes. Retaining safetensors views would
            # retain a separate multi-GB Windows COW mapping for every expert,
            # far exceeding the apparent tensor-byte RAM budget/commit limit.
            data = {p: raw[prefix + p + ".weight"].to(dtype=self.dtype, copy=True)
                    for p in ("gate_proj", "up_proj", "down_proj")}
            self.cache.put(key, data)
            self.reads += 1
        return data

    def prepare(self, layer, expert):
        if not 0 <= expert < self.config.num_experts:
            raise ValueError("Router expert out of range")
        self.release()
        if self.on_access:
            self.on_access(layer, expert)
        host = self.host_weights(layer, expert)
        # Copies and consumers use the same current stream. No slot is
        # overwritten; PyTorch owns allocations until queued consumers finish.
        gate = host["gate_proj"].to(self.device)
        up = host["up_proj"].to(self.device)
        self.active = {"gate_up_proj":torch.cat((gate, up), dim=0),
                       "down_proj":host["down_proj"].to(self.device)}
        self.active_key = (layer, expert)
        self.expert_calls += 1
        self.logical_h2d_bytes += self.expert_bytes

    def projection(self, layer, expert, kind):
        if kind == "gate_up_proj":
            self.prepare(layer, expert)
        elif self.active_key != (layer, expert):
            raise RuntimeError("Expert down projection without matching active lease")
        return self.active[kind]

    def release(self):
        self.active = None
        self.active_key = None

    def close(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.release()
        self.cache.clear()


class _Projection:
    def __init__(self, store, layer, kind):
        self.store, self.layer, self.kind = store, layer, kind

    def __getitem__(self, expert):
        return self.store.projection(self.layer, int(expert), self.kind)


def install_expert_views(experts, store, layer):
    """Adapt data indexing only; retain the original decorated forward method."""
    if type(experts).__name__ != "OlmoeExperts" or experts.config._experts_implementation != "eager":
        raise ValueError("Expert views require original OlmoeExperts eager backend")
    for name in ("gate_up_proj", "down_proj"):
        del experts._parameters[name]
        setattr(experts, name, _Projection(store, layer, name))


class AirColibriOLMoE:
    def __init__(self, model_dir, airllm_source, native_dll=None, ram_bytes=256 * 2**20,
                 device="cuda", dtype=torch.bfloat16, full_expert_baseline=False, native_mode="apply"):
        self.native_mode = native_mode
        self.device, self.dtype = torch.device(device), dtype
        self.handles = []
        self.model = self.store = None
        self.failed = False
        try:
            self.config = AutoConfig.from_pretrained(model_dir)
            if self.config.model_type != "olmoe":
                raise ValueError("OLMoE checkpoint required")
            self.config._experts_implementation = "eager"
            self.config._attn_implementation = "eager"
            self.provider = IndexedAirLLMProvider(model_dir, airllm_source)
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            with init_empty_weights():
                self.model = AutoModelForCausalLM.from_config(self.config)
            self.model.eval()
            self.store = ExpertStore(self.provider, self.config, self.device, dtype, ram_bytes, native_dll, native_mode)
            self.trace = TraceContract(self.config.num_hidden_layers)
            self.store.on_access = lambda layer, expert: self.trace.component_called(layer, f"expert:{expert}", expert=expert)
            self._router_choices = deque(maxlen=4096)
            self.router_records_total = 0
            # Dense attention, routers, norms, embedding and output head retain
            # their original names/math; read one tensor at a time through AirLLM.
            for name, parameter in list(self.model.named_parameters()):
                if ".experts." in name:
                    continue
                value = self.provider.read([name])[name].to(device=self.device, dtype=dtype)
                set_module_tensor_to_device(self.model, name, self.device, value=value, clear_cache=False)
            del value
            for i, layer in enumerate(self.model.model.layers):
                self.handles.append(layer.register_forward_pre_hook(lambda m, a, idx=i: self.trace.enter_layer(idx)))
                def leave(m, a, o, idx=i):
                    self.store.release()
                    self.trace.leave_layer(idx)
                self.handles.append(layer.register_forward_hook(leave))
                def routed(m, a, output, idx=i):
                    self.router_records_total += 1
                    self._router_choices.append({"step":self.trace.current_step, "layer":idx,
                                                "ids":output[2].detach().cpu().tolist()})
                self.handles.append(layer.mlp.gate.register_forward_hook(routed))
                if full_expert_baseline:
                    self.handles.append(layer.mlp.experts.register_forward_pre_hook(
                        lambda m, a, idx=i: self._materialize_all(m, idx)))
                    self.handles.append(layer.mlp.experts.register_forward_hook(self._release_all))
                else:
                    install_expert_views(layer.mlp.experts, self.store, i)
        except BaseException:
            self.close()
            raise

    @property
    def router_choices(self):
        """Bounded diagnostic tail; aggregate counts remain in the trace."""
        return list(self._router_choices)

    def _materialize_all(self, module, layer):
        c = self.config
        gate_up = torch.empty((c.num_experts, 2*c.intermediate_size, c.hidden_size), device=self.device, dtype=self.dtype)
        down = torch.empty((c.num_experts, c.hidden_size, c.intermediate_size), device=self.device, dtype=self.dtype)
        for expert in range(c.num_experts):
            host = self.store.host_weights(layer, expert)
            gate_up[expert, :c.intermediate_size].copy_(host["gate_proj"])
            gate_up[expert, c.intermediate_size:].copy_(host["up_proj"])
            down[expert].copy_(host["down_proj"])
            self.store.logical_h2d_bytes += self.store.expert_bytes
        module.gate_up_proj = torch.nn.Parameter(gate_up, requires_grad=False)
        module.down_proj = torch.nn.Parameter(down, requires_grad=False)

    def _release_all(self, module, args, output):
        for name in ("gate_up_proj", "down_proj"):
            p = getattr(module, name)
            setattr(module, name, torch.nn.Parameter(torch.empty(p.shape, dtype=self.dtype, device="meta"), requires_grad=False))

    def forward(self, ids, state=None):
        if self.failed or self.model is None:
            raise RuntimeError("Runtime is closed or failed")
        self.trace.start_step()
        try:
            with torch.no_grad():
                output = self.model(input_ids=ids.to(self.device), past_key_values=state, use_cache=True)
            self.trace.finish_step()
            return output.logits[:, -1, :], output.past_key_values
        except BaseException:
            self.failed = True
            raise

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if self.store:
            self.store.close()
        self.model = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
