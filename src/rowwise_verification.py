"""Verify several tokens per loaded layer while preserving one-row arithmetic.

The HF graph still owns dependencies and model state. Its unpadded single-stream
decode layers run one row at a time under one weight lease. Recurrent/sliding
cache states are retained at row boundaries for rejection, not model weights.
"""
from contextlib import contextmanager
from types import SimpleNamespace

import torch

from aircolibri_cli.thought_buffer import capture_cache_snapshot


class RowwiseVerification:
    # Other HF decoders can use different positional arguments, ALiBi/bias or
    # cache semantics. Keep ordinary decode until their row contract is covered.
    SUPPORTED_TYPES = frozenset({"llama", "qwen2", "qwen3", "qwen3_5_text", "mamba", "mamba2",
                                 "lfm2", "gemma2", "granitemoe", "olmoe"})

    def __init__(self, engine):
        self.engine = engine
        self.supported = engine.config.model_type in self.SUPPORTED_TYPES
        self.rows = 0
        self.current_row = None
        self.cache = None
        self.saved = {}
        self.originals = []
        for i, layer in enumerate(engine._decoder_layers()):
            self._wrap(layer, i)
        self._wrap(engine.model.get_output_embeddings(), None)
        engine._rowwise_verification = self

    def _slice(self, value, name, row):
        if name in {"past_key_values", "past_key_value", "cache_params", "state"}:
            return value
        if name == "attention_mask":
            # Runtime inputs are one unpadded stream. A one-row decode sees only
            # the keys already appended, including a sliding cache's live window.
            return None
        if isinstance(value, torch.Tensor):
            if name in {"position_ids", "cache_position"}:
                return value[..., row:row+1]
            if name == "position_embeddings" and value.ndim >= 2 and value.shape[-2] == self.rows:
                return value[..., row:row+1, :]
            if value.ndim >= 2 and value.shape[0] == 1 and value.shape[1] == self.rows:
                return value[:, row:row+1]
        if isinstance(value, tuple):
            return tuple(self._slice(v, name, row) for v in value)
        if isinstance(value, list):
            return [self._slice(v, name, row) for v in value]
        return value

    @staticmethod
    def _join(outputs):
        first = outputs[0]
        if isinstance(first, torch.Tensor):
            return torch.cat(outputs, dim=1)
        if isinstance(first, tuple):
            return tuple(RowwiseVerification._join([x[i] for x in outputs]) for i in range(len(first)))
        return outputs[-1]

    def _wrap(self, module, layer):
        original = module.forward
        def wrapped(*args, **kwargs):
            hidden = args[0] if args else kwargs.get("hidden_states")
            if (self.rows < 2 or not isinstance(hidden, torch.Tensor) or hidden.ndim != 3
                    or tuple(hidden.shape[:2]) != (1, self.rows)):
                return original(*args, **kwargs)
            outputs = []
            for row in range(self.rows):
                self.current_row = row
                row_args = tuple(self._slice(v, "", row) for v in args)
                row_kwargs = {k: self._slice(v, k, row) for k, v in kwargs.items()}
                outputs.append(original(*row_args, **row_kwargs))
                if layer is not None and row+1 < self.rows and layer < len(self.cache.layers):
                    cache_layer = self.cache.layers[layer]
                    from transformers.cache_utils import DynamicLayer
                    if type(cache_layer) is not DynamicLayer:
                        snapshot = capture_cache_snapshot(SimpleNamespace(layers=[cache_layer]))
                        self.saved.setdefault(layer, []).append(snapshot)
            self.current_row = None
            return self._join(outputs)
        module.forward = wrapped
        self.originals.append((module, original, wrapped))

    @contextmanager
    def execution(self, cache, rows):
        self.rows, self.cache, self.saved = rows, cache, {}
        try:
            yield self
        finally:
            self.rows, self.current_row, self.cache = 0, None, None
            self.saved.clear()

    def restore(self, cache, keep):
        from transformers.cache_utils import DynamicLayer
        for i, layer in enumerate(cache.layers):
            if type(layer) is not DynamicLayer and len(self.saved.get(i, ())) < keep:
                return False
        for i, layer in enumerate(cache.layers):
            if type(layer) is not DynamicLayer:
                # These row snapshots are single-use. Transfer their tensors
                # into the live cache instead of allocating a second clone.
                state = self.saved[i][keep-1].layer_states[0]
                vars(layer).clear()
                vars(layer).update(state)
        # Attention rows have identical one-row arithmetic and are append-only.
        for i, layer in enumerate(cache.layers):
            if type(layer) is DynamicLayer:
                remove = int(layer.get_seq_length())-self.position-keep
                if remove > 0:
                    layer.crop(-remove)
        return True

    def close(self):
        for module, original, wrapped in self.originals:
            if module.forward is wrapped:
                module.forward = original
        self.originals.clear()
        self.saved.clear()
