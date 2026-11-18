"""Bounded, device-independent native Qwen3.5 draft head over the shared base model.

Only the trained head is resident here. Embeddings and vocabulary projection use
the existing base executor; no second backbone or checkpoint is materialized.
"""
import copy
import math
from pathlib import Path

import torch
from torch import nn
from accelerate.utils import set_module_tensor_to_device


class NativeMTPHead(nn.Module):
    def __init__(self, engine, path=None, budget_bytes=128*2**20):
        super().__init__()
        from aircolibri_cli.model_source import PreadSafetensorsReader, read_safetensors_header
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5DecoderLayer, Qwen3_5RMSNorm, Qwen3_5TextRotaryEmbedding)
        if engine.config.model_type != "qwen3_5_text":
            raise NotImplementedError("Native MTP adapter currently supports Qwen3.5 text geometry")
        self.engine, self.dtype = engine, engine.dtype
        self.device = torch.device("cpu")
        cfg = copy.deepcopy(engine.config)
        cfg.num_hidden_layers, cfg.layer_types = 1, ["full_attention"]
        cfg._attn_implementation = "sdpa"
        self.config = cfg
        source = getattr(engine, "model_source", None)
        entries = source.mtp_entries() if path is None and hasattr(source, "mtp_entries") else {}
        if path is not None:
            header, _ = read_safetensors_header(path)
            entries = {name: (str(Path(path)), name) for name in header if name.startswith("mtp.")}
        if not entries:
            raise FileNotFoundError("Checkpoint has no trained mtp.* tensors")
        shapes = {}
        for name, (filename, stored) in entries.items():
            header, _ = read_safetensors_header(filename)
            shapes[name] = tuple(header[stored]["shape"])
        item_bytes = torch.empty((), dtype=self.dtype).element_size()
        required_bytes = sum(math.prod(shape)*item_bytes for shape in shapes.values())
        if required_bytes > budget_bytes:
            raise ValueError(f"Native MTP weights need {required_bytes} bytes; head budget is {budget_bytes}")
        with torch.device("meta"):
            self.layer = Qwen3_5DecoderLayer(cfg, layer_idx=0)
            self.fc = nn.Linear(2*cfg.hidden_size, cfg.hidden_size, bias=False)
            self.embedding_norm = Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
            self.hidden_norm = Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
            self.output_norm = Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.rotary = Qwen3_5TextRotaryEmbedding(cfg)
        aliases = {"mtp.fc.weight": "fc.weight", "mtp.pre_fc_norm_embedding.weight": "embedding_norm.weight",
                   "mtp.pre_fc_norm_hidden.weight": "hidden_norm.weight", "mtp.norm.weight": "output_norm.weight"}
        mapped = {aliases.get(name, name.replace("mtp.layers.0.", "layer.")): entry
                  for name, entry in entries.items()}
        expected = dict(self.named_parameters())
        if set(mapped) != set(expected):
            raise ValueError("Native MTP tensor names do not match the configured trained head")
        self.reader = PreadSafetensorsReader(max_handles=2)
        try:
            for name, entry in mapped.items():
                value = self.reader.read_tensor(*entry)
                if value.shape != expected[name].shape:
                    raise ValueError(f"Native MTP shape mismatch: {name}")
                set_module_tensor_to_device(self, name, "cpu", value=value, dtype=self.dtype, clear_cache=False)
        finally:
            self.reader.close()
        self.weight_bytes = sum(p.nbytes for p in self.parameters())
        self.eval()
        self.reset()

    def reset(self):
        from transformers.cache_utils import DynamicCache
        self.cache = DynamicCache(config=self.config)
        self.committed = 0
        self.last_hidden = self.last_normed = None

    def _trim(self, length):
        remove = int(self.cache.get_seq_length())-length
        if remove > 0:
            self.cache.crop(-remove)

    @torch.no_grad()
    def _pairs(self, hidden, tokens):
        count = len(tokens)
        hidden = hidden.to(device=self.device, dtype=self.dtype)
        embedding = self.engine._embed_step_tokens(torch.tensor([tokens], dtype=torch.long)).to(self.device)
        x = self.fc(torch.cat((self.embedding_norm(embedding), self.hidden_norm(hidden)), dim=-1))
        start = self.cache.get_seq_length()
        positions = torch.arange(start, start+count, device=self.device)
        rotary = self.rotary(x, positions.view(1, 1, count).expand(3, 1, count))
        mask = None
        if count > 1:
            mask = torch.zeros((count, start+count), dtype=self.dtype, device=self.device)
            mask[:, start:] = torch.triu(torch.full((count, count), float("-inf"), dtype=self.dtype), diagonal=1)
            mask = mask[None, None]
        result = self.layer(x, attention_mask=mask, position_ids=positions[None],
                            past_key_values=self.cache, use_cache=True, position_embeddings=rotary)
        return self.output_norm(result[0] if isinstance(result, tuple) else result)

    @torch.no_grad()
    def resume(self, prefix, next_token):
        if not prefix:
            self.reset()
            return True
        if self.committed != prefix or self.last_hidden is None:
            self.reset()
            return False
        # The next user turn can re-tokenize the unfed output. Repair the final
        # pair without replaying the backbone or retaining all prefix hiddens.
        self._trim(prefix-1)
        self.last_normed = self._pairs(self.last_hidden, [next_token])[:, -1:]
        return True

    @torch.no_grad()
    def commit(self, hidden, next_tokens):
        self._trim(self.committed)
        result = self._pairs(hidden, next_tokens)
        self.committed += len(next_tokens)
        self.last_hidden = hidden[:, -1:].detach().to(self.device).clone()
        self.last_normed = result[:, -1:].detach()

    @torch.no_grad()
    def draft(self, count):
        if not count or self.last_normed is None:
            return [], []
        self._trim(self.committed)
        hidden = self.last_normed
        tokens, confidences = [], []
        for i in range(count):
            logits = self.engine.model.get_output_embeddings()(hidden.to(self.engine.device)).float()[0, -1]
            probability, token = torch.softmax(logits, dim=-1).max(dim=-1)
            tokens.append(int(token.item()))
            confidences.append(float(probability.item()))
            if i+1 < count:
                hidden = self._pairs(hidden, [tokens[-1]])[:, -1:]
        return tokens, confidences

    def close(self):
        self.reset()
