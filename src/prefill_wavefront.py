"""Three CUDA stages for causal prefill, sharing the engine's weight ring.

Layer L+1 may consume chunk c while L processes c+1. A layer's KV updates
remain ordered on one stream; a consumer waits for its particular input,
not for all chunks of the preceding layer. No speculative tokens/pruning.

Initially restricted to the verified HF Llama decoder contract. It deliberately
does not claim that sequential autoregressive tokens become independent.
"""
from __future__ import annotations

from functools import lru_cache
import torch
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import sdpa_mask


def supports_wavefront(engine) -> bool:
    cfg = engine.config
    return (getattr(cfg, "model_type", None) == "llama"
            and not getattr(cfg, "sliding_window", None)
            and not engine.is_q4  # legacy Q4 shares one dequantization scratch
            and getattr(cfg, "_attn_implementation", None) in ("sdpa", "eager")
            and engine.device.type == "cuda"
            and (engine.vram_resident_layers >= len(engine._decoder_layers())
                 or engine.num_slots == 3))


def _record(tensor, stream):
    if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
        tensor.record_stream(stream)


@lru_cache(maxsize=None)
def _compute_streams(device_index):
    # cuBLAS keeps a workspace per stream. Reusing a fixed trio also across
    # engine reopenings prevents successive prompt windows growing that cache.
    return tuple(torch.cuda.Stream(device=device_index) for _ in range(3))


@torch.no_grad()
def run_prefill_wavefront(engine, tokens, chunk_size, past=None, *,
                          concurrent=True, collect_timing=False):
    if engine._failed or engine.model is None:
        raise RuntimeError("Model is closed or failed")
    if not supports_wavefront(engine):
        raise ValueError("Wavefront prefill currently supports Llama with full residency or a three-slot ring")
    if engine._wavefront_active:
        raise RuntimeError("Wavefront execution is not reentrant")
    if tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.shape[1] == 0 or chunk_size < 1:
        raise ValueError("Expected nonempty, unpadded [1, tokens] input and positive chunk_size")
    if past is not None and not isinstance(past, DynamicCache):
        raise ValueError("Wavefront prefill requires a DynamicCache")

    device = engine.device
    caller = torch.cuda.current_stream(device)
    streams = _compute_streams(device.index) if concurrent else (caller,) * 3
    layers = engine._decoder_layers()
    base = engine.model.model
    prefix = past.get_seq_length() if past is not None else 0
    if past is not None and (getattr(past, "offloading", False)
                            or any(layer.get_seq_length() != prefix for layer in past.layers)):
        raise ValueError("Wavefront requires a synchronized, non-offloaded prefix cache")
    past = past if past is not None else DynamicCache(config=engine.config)
    # Discard any preceding tail prefetch before taking ownership of the ring.
    # Preserve the caller's cancellation callback.
    cancel = engine._step_cancel_check
    engine.reset_after_cancel()
    engine._step_cancel_check = cancel
    engine._prefetch_next_step = False
    engine._wavefront_active = True
    loads_before = getattr(engine.pipeline, "total_loads", 0)
    records = []
    root = torch.cuda.Event(enable_timing=collect_timing)
    done = []

    def submit_weights(index):
        if index < engine.vram_resident_layers or index >= len(layers):
            return
        pinned = engine.weight_manager.load_layer_to_pinned_slot(index, target_slot_idx=index % 3)
        slot = engine.pipeline.get_slot_for_loading(index)
        engine.pipeline.load_tensors_async(slot, pinned, pinned_pool=engine.weight_manager.pinned_pool)

    try:
        hidden = engine._embed_step_tokens(tokens)
        chunks = list(hidden.split(chunk_size, dim=1))
        positions, rotations, masks = [], [], []
        offset = prefix
        for h in chunks:
            length = h.shape[1]
            pos = torch.arange(offset, offset + length, device=device).unsqueeze(0)
            positions.append(pos)
            rotations.append(base.rotary_emb(h, position_ids=pos))
            # An explicit rectangular causal mask avoids relying on layer 0's
            # cache length while other layers are at different chunk positions.
            if engine.config._attn_implementation == "sdpa":
                # Match HF's boolean-mask/is_causal choices as well as causality;
                # an always-float mask can select a numerically different kernel.
                mask = sdpa_mask(batch_size=1, q_length=length, kv_length=offset + length,
                                 q_offset=offset, kv_offset=0, device=device)
            else:
                allowed = torch.arange(offset + length, device=device)[None, :] <= pos[0, :, None]
                mask = torch.zeros((length, offset + length), device=device, dtype=h.dtype)
                mask.masked_fill_(~allowed, torch.finfo(h.dtype).min)
                mask = mask[None, None]
            masks.append(mask)
            offset += length
        root.record(caller)
        done = [root] * len(chunks)
        for s in streams:
            s.wait_event(root)
        first = engine.vram_resident_layers
        for i in range(first, min(first + 3, len(layers))):
            submit_weights(i)

        # A three-layer window bounds live activations and weight ownership.
        # Finished slots immediately upload their next layer (L+3), overlapping
        # the tail of the current window. There is no CPU barrier between windows.
        for start in range(0, len(layers), 3):
            width = min(3, len(layers) - start)
            values = [chunks] + [[None] * len(chunks) for _ in range(width)]
            events = [done] + [[None] * len(chunks) for _ in range(width)]
            for diagonal in range(len(chunks) + width - 1):
                for stage in reversed(range(width)):
                    c = diagonal - stage
                    if not 0 <= c < len(chunks):
                        continue
                    if cancel is not None:
                        cancel()
                    i = start + stage
                    stream = streams[i % 3]
                    h = values[stage][c]
                    with torch.cuda.stream(stream):
                        stream.wait_event(events[stage][c])
                        _record(h, stream)
                        _record(masks[c], stream)
                        _record(positions[c], stream)
                        for t in rotations[c]:
                            _record(t, stream)
                        if i < len(past.layers):
                            for attr in ("keys", "values"):
                                _record(getattr(past.layers[i], attr, None), stream)
                        began = torch.cuda.Event(enable_timing=True) if collect_timing else None
                        marker = None
                        if began is not None:
                            # Register after the engine's DMA/binding hook: this
                            # timestamp must not count waiting for weights as compute.
                            marker = layers[i].register_forward_pre_hook(
                                lambda _m, _a, ev=began, st=stream: ev.record(st))
                        try:
                            out = layers[i](h, attention_mask=masks[c], position_ids=positions[c],
                                            position_embeddings=rotations[c], past_key_values=past, use_cache=True)
                        finally:
                            if marker is not None:
                                marker.remove()
                        if isinstance(out, tuple):
                            out = out[0]
                        end = torch.cuda.Event(enable_timing=collect_timing)
                        end.record(stream)
                        values[stage + 1][c] = out
                        values[stage][c] = None
                        events[stage + 1][c] = end
                        if collect_timing:
                            records.append((i, c, began, end))
                        if c == len(chunks) - 1 and i >= engine.vram_resident_layers:
                            slot = engine.active_slot_per_layer[i]
                            engine.pipeline.release_slot_after_compute(slot)
                            engine.active_slot_per_layer[i] = None
                            submit_weights(i + 3)
            chunks, done = values[-1], events[-1]

        # The caller gets both logits and a completely ordered cache, ready for
        # normal decode/session reuse. Other chunks do not need lm_head work.
        for s in streams:
            caller.wait_stream(s)
        last = chunks[-1]
        _record(last, caller)
        logits = engine.model.lm_head(base.norm(last[:, -1:, :]))[:, -1, :]
        stats = {"mode": "concurrent" if concurrent else "serial", "stages": 3,
                 "chunks": len(chunks), "layers": len(layers),
                 "layer_chunk_calls": len(layers) * len(chunks),
                 "weight_loads": getattr(engine.pipeline, "total_loads", 0) - loads_before}
        if collect_timing:
            caller.synchronize()
            stats["layer_intervals_ms"] = [
                {"layer": i, "chunk": c, "start": root.elapsed_time(a), "end": root.elapsed_time(b)}
                for i, c, a, b in records]
        engine.last_wavefront_stats = stats
        return logits, past
    except BaseException as exc:
        # A partially extended cache is caller-owned and must be discarded.
        # Synchronize all readers before resetting/reusing any weight slot.
        for s in streams:
            s.synchronize()
        engine.reset_after_cancel()
        engine._step_cancel_check = cancel
        if not getattr(exc, "_is_cooperative_cancel", False):
            engine._failed = True
        raise
    finally:
        engine._wavefront_active = False
