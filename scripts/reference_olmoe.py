"""Independent OLMoE oracle: direct safetensors and unmodified HF modules.

The reference stages every expert in each layer, regardless of router output.
It imports no AirColibri reader, cache, ring, lease or inference adapter. This
is a bounded-memory reference harness, not a second production weight loader.
"""
from pathlib import Path
import gc
import json

import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


class ReferenceOLMoE:
    def __init__(self, model_dir, device="cuda", dtype=torch.bfloat16):
        self.folder = Path(model_dir).resolve()
        self.device, self.dtype = torch.device(device), dtype
        self.handles, self.router_choices = [], []
        self.model = None
        self.step = -1
        self.loaded_layers = 0
        self.read_bytes = 0
        try:
            self.config = AutoConfig.from_pretrained(self.folder, local_files_only=True)
            if self.config.model_type != "olmoe":
                raise ValueError("OLMoE checkpoint required")
            self.config._attn_implementation = "eager"
            self.config._experts_implementation = "eager"
            self.index = json.loads((self.folder / "model.safetensors.index.json").read_text())["weight_map"]
            self.tokenizer = AutoTokenizer.from_pretrained(self.folder, local_files_only=True)
            with init_empty_weights():
                self.model = AutoModelForCausalLM.from_config(self.config)
            self.model.eval()
            for name, _ in list(self.model.named_parameters()):
                if not name.startswith("model.layers."):
                    self._assign(self.model, name, self._read(name).to(self.device, dtype=dtype))
            for i, layer in enumerate(self.model.model.layers):
                self.handles.append(layer.register_forward_pre_hook(lambda m, a, idx=i: self._load(m, idx)))
                self.handles.append(layer.register_forward_hook(self._unload))
                def routed(m, a, output, idx=i):
                    self.router_choices.append(dict(step=self.step, layer=idx, ids=output[2].detach().cpu().tolist()))
                self.handles.append(layer.mlp.gate.register_forward_hook(routed))
        except BaseException:
            self.close()
            raise

    def _read(self, key):
        path = (self.folder / self.index[key]).resolve()
        if path.parent != self.folder or path.suffix != ".safetensors":
            raise ValueError("Invalid checkpoint path")
        with safe_open(path, framework="pt", device="cpu") as f:
            value = f.get_tensor(key)
        self.read_bytes += value.nbytes
        return value

    def _assign(self, module, name, value):
        set_module_tensor_to_device(module, name, self.device, value=value, clear_cache=False)

    def _load(self, layer, index):
        prefix = f"model.layers.{index}."
        for name, _ in list(layer.named_parameters()):
            if ".experts." not in name:
                self._assign(layer, name, self._read(prefix + name).to(self.device, dtype=self.dtype))
        c = self.config
        # Fully materialized original expert tensors; no adapted indexing.
        gate_up = torch.empty((c.num_experts, 2*c.intermediate_size, c.hidden_size), device=self.device, dtype=self.dtype)
        down = torch.empty((c.num_experts, c.hidden_size, c.intermediate_size), device=self.device, dtype=self.dtype)
        for expert in range(c.num_experts):
            ep = prefix + f"mlp.experts.{expert}."
            gate_up[expert, :c.intermediate_size].copy_(self._read(ep + "gate_proj.weight"))
            gate_up[expert, c.intermediate_size:].copy_(self._read(ep + "up_proj.weight"))
            down[expert].copy_(self._read(ep + "down_proj.weight"))
        self._assign(layer.mlp.experts, "gate_up_proj", gate_up)
        self._assign(layer.mlp.experts, "down_proj", down)
        self.loaded_layers += 1

    def _unload(self, layer, args, output):
        for name, _ in list(layer.named_parameters()):
            set_module_tensor_to_device(layer, name, "meta", clear_cache=False)

    def forward(self, ids, state=None):
        self.step += 1
        with torch.no_grad():
            out = self.model(input_ids=ids.to(self.device), past_key_values=state, use_cache=True)
        return out.logits[:, -1, :], out.past_key_values

    def close(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.model = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
