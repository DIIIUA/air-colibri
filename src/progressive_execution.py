"""Common CPU/GPU startup of one HF graph, with readiness-driven work adoption.

Only the foreground reads checkpoints. The bounded worker queue contains immutable
CPU weight payloads already used by that graph, never a second checkpoint loader.
Both branches share a start event. No CPU arithmetic must finish before GPU setup.
"""
from __future__ import annotations

import queue
import inspect
import threading
import time
from contextlib import contextmanager

import torch
from accelerate.utils import set_module_tensor_to_device


def move_tensors(value, device, memo=None):
    """Move cache/activation trees without losing container metadata or tensor aliases."""
    memo = {} if memo is None else memo
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, torch.device):
        return torch.device(device)
    if isinstance(value, torch.Tensor):
        result = value.to(device)
        memo[id(value)] = result
        return result
    if isinstance(value, tuple):
        items = tuple(move_tensors(v, device, memo) for v in value)
        result = type(value)(*items) if hasattr(value, "_fields") else items
        memo[id(value)] = result
        return result
    if isinstance(value, list):
        memo[id(value)] = value
        value[:] = [move_tensors(v, device, memo) for v in value]
    elif isinstance(value, dict):
        memo[id(value)] = value
        for k, v in list(value.items()):
            value[k] = move_tensors(v, device, memo)
    elif hasattr(value, "__dict__") and not isinstance(value, (torch.nn.Module, type)):
        # Cache objects own tensors and per-layer metadata. Configs/callables are
        # not cache state and must not be traversed or mutated.
        if type(value).__module__.startswith("transformers.cache_utils") or type(value).__name__ == "SimpleNamespace":
            memo[id(value)] = value
            for k, v in list(vars(value).items()):
                if isinstance(v, torch.device):
                    setattr(value, k, torch.device(device))
                else:
                    setattr(value, k, move_tensors(v, device, memo))
    return value


class ProgressiveExecution:
    def __init__(self, engine, plan_factory, expert_cache_bytes=0, allow_gpu=True):
        self.engine, self.plan_factory = engine, plan_factory
        self.complete = False
        self.stopped = threading.Event()
        self.bundle_ready = threading.Event()
        self.backend_ready = threading.Event()
        self.start_event = threading.Event()
        self.globals_ready = threading.Event()
        self.jobs = queue.Queue(maxsize=2)
        self.pending, self.ready, self.adopted = set(), {}, set()
        self.lock = threading.Lock()
        self.worker = None
        self.error = None
        self.gpu_enabled = bool(allow_gpu and torch.cuda.is_available())
        self.gpu_state = "preparing" if self.gpu_enabled else ("unavailable" if allow_gpu else "disabled")
        self.start_policy = "simultaneous_readiness" if self.gpu_enabled else "cpu_only"
        self.plan = None
        self.kernels_ready = False
        self.gpu = torch.device("cuda:0")
        self.cpu_layers = self.gpu_layers = self.cpu_steps = 0
        self.auxiliary_depth = 0
        self.verification_depth = 0
        self.transitions = []
        self.started_at = None
        self.global_payload = None
        self.gpu_pool = self.gpu_pipeline = None
        self._globals, self._buffers = {}, {}
        self.handles = []
        self.bound = {}
        self.deferred_warmup = None
        for i, layer in enumerate(engine._decoder_layers()):
            self.handles.append(layer.register_forward_pre_hook(self._before(i), with_kwargs=True))
            self.handles.append(layer.register_forward_hook(self._after(i)))
        engine.active_slot_per_layer = [None] * len(engine._decoder_layers())
        engine._hook_handles = []

    def stats(self):
        with self.lock:
            return {"enabled": True, "complete": self.complete,
                    "start_policy": self.start_policy,
                    "gpu_state": "failed" if self.error is not None else self.gpu_state,
                    "cpu_layer_calls": self.cpu_layers, "gpu_layer_calls": self.gpu_layers,
                    "cpu_steps": self.cpu_steps, "gpu_resident_layers_ready": self.engine.vram_resident_layers if self.complete else len(self.ready),
                    "adopted_layers": sorted(self.adopted), "queue_capacity": 2,
                    "queued_payloads": self.jobs.qsize(), "transitions": list(self.transitions),
                    "kernels_ready": self.kernels_ready,
                    "preparation_error": str(self.error) if self.error else None}

    def capture_globals(self):
        self._globals = {n: t for n, t in self.engine.model.named_parameters(remove_duplicate=False) if not t.is_meta}
        self._buffers = {n: t for n, t in self.engine.model.named_buffers(remove_duplicate=False) if not t.is_meta}
        self.globals_ready.set()

    def start(self):
        if not self.gpu_enabled or self.error is not None:
            return
        if self.worker is None:
            self.stopped.clear()
            self.start_event.clear()
            self.started_at = time.perf_counter()
            self.worker = threading.Thread(target=self._prepare, name="aircolibri-gpu-prepare", daemon=True)
            self.worker.start()
            self.start_event.set()

    def _prepare(self):
        try:
            self.start_event.wait()
            with torch.no_grad(), torch.cuda.device(self.gpu):
                if self.plan is None:
                    params = inspect.signature(self.plan_factory).parameters
                    self.plan = self.plan_factory(self.engine) if params else self.plan_factory()
                if self.stopped.is_set():
                    return
                # Compile/load the original Colibri patch modules off the
                # inference thread. A weight payload alone is not GPU readiness.
                if self.engine.quant_bits:
                    from src.colibri_nvrtc import get_kernels
                    get_kernels(self.gpu)
                if any(getattr(m, "_colibri_nvrtc_norm", False) for m in self.engine.model.modules()):
                    from src.colibri_deltanet_nvrtc import get_deltanet_kernels
                    get_deltanet_kernels(self.gpu)
                if getattr(self.engine.model_source, "kind", None) == "gguf" and getattr(self.engine.model_source, "_expert_loader", None) is not None:
                    from aircolibri_cli.model_source import _get_nvrtc_gguf_kernels
                    _get_nvrtc_gguf_kernels(self.gpu)
                self.kernels_ready = True
                while not self.globals_ready.wait(0.05):
                    if self.stopped.is_set():
                        return
                stream = torch.cuda.Stream(device=self.gpu)
                if self.global_payload is None:
                    with torch.cuda.stream(stream):
                        params = self._copy_payload(self._globals)
                        buffers = self._copy_payload(self._buffers)
                    if params is None or buffers is None:
                        return
                    self.global_payload = (params, buffers)
                stream.synchronize()
                self.backend_ready.set()
                while not self.stopped.is_set():
                    if len(self.ready) == self.plan.resident_layers:
                        from src.bounded_memory_manager import BoundedPinnedBufferPool
                        from src.gpu_ring_pipeline import GPURingPipeline
                        from src.aircolibri_integrated import _NoStreamingPipeline
                        if self.plan.full_residency:
                            self.gpu_pipeline = _NoStreamingPipeline()
                        else:
                            self.gpu_pool = BoundedPinnedBufferPool(
                                arena_layout=self.engine.weight_manager.pinned_pool.layout,
                                num_slots=self.plan.ring_slots, dtype=self.engine.dtype, device=self.gpu)
                            self.gpu_pipeline = GPURingPipeline(
                                num_slots=self.plan.ring_slots, device=self.gpu, dtype=self.engine.dtype,
                                tail_to_head_prefetch=self.plan.ring_slots > 1,
                                arena_layout=self.gpu_pool.layout)
                        self.bundle_ready.set()
                        return
                    try:
                        i, tensors = self.jobs.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    if i >= self.plan.resident_layers:
                        with self.lock:
                            self.pending.discard(i)
                        continue
                    with torch.cuda.stream(stream):
                        payload = self._copy_payload(tensors)
                    stream.synchronize()
                    if payload is None:
                        return
                    with self.lock:
                        self.ready[i] = payload
                        self.pending.discard(i)
                    del tensors, payload
        except BaseException as exc:
            # A traceback can retain partially allocated GPU payloads. Keep the
            # diagnostic, then let the foreground retire resources at a boundary.
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            if self.global_payload is not None:
                self._globals.clear()
                self._buffers.clear()

    def _copy_payload(self, tensors):
        result = {}
        for name, value in tensors.items():
            if self.stopped.is_set():
                return None
            target = torch.empty_like(value, device=self.gpu)
            if not value.numel():
                result[name] = target
                continue
            if value.ndim:
                rows = max(1, 1024**2 // max(1, value[0].numel()))
                for a in range(0, value.shape[0], rows):
                    if self.stopped.is_set():
                        return None
                    target[a:a+rows].copy_(value[a:a+rows])
            else:
                target.copy_(value)
            result[name] = target
        return result

    def _bind(self, module, payload, device):
        raw = dict(module.named_parameters())
        buffers = dict(module.named_buffers())
        for n, t in payload.items():
            if n in raw or n in buffers:
                dt = buffers[n].dtype if n in buffers else self.engine.dtype
                set_module_tensor_to_device(module, n, device, value=t, dtype=dt, clear_cache=False)
        for _, sub in module.named_modules():
            for n, t in list(sub._buffers.items()):
                if t is not None and not t.is_meta and t.device != device:
                    sub._buffers[n] = t.to(device)

    def _offer(self, i, tensors):
        if (not self.gpu_enabled or self.error is not None or (self.auxiliary_depth and not self.verification_depth)
                or (self.plan is not None and i >= self.plan.resident_layers)):
            return
        with self.lock:
            if i in self.ready or i in self.pending:
                return
            try:
                self.jobs.put_nowait((i, tensors))
                self.pending.add(i)
            except queue.Full:
                pass  # Neither device waits; offer it again at the next visit.

    def _gpu_payload(self, i):
        with self.lock:
            if not self.gpu_enabled or (self.error is not None and not self.auxiliary_depth):
                return None
            if self.auxiliary_depth and i not in self.adopted:
                return None
            return self.ready.get(i)

    def _before(self, i):
        def before(module, args, kwargs):
            eng = self.engine
            if eng._step_cancel_check is not None:
                eng._step_cancel_check()
            cache_name = getattr(eng, "_cache_kwarg", "past_key_values")
            cache = kwargs.get(cache_name) if cache_name else None
            # Monolithic legacy state and device-specific expert dispatch move
            # at atomic adoption. Dense cache layers can follow readiness alone.
            moe = getattr(eng.model_source, "_expert_loader", None) is not None
            can_move_layer = (cache is None or hasattr(cache, "layers")) and not moe
            payload = self._gpu_payload(i) if can_move_layer else None
            cpu_payload = None
            if payload is None:
                cache_store = eng.weight_manager.ram_cache
                cpu_payload = cache_store.get(i)
                if cpu_payload is None:
                    cpu_payload = eng.weight_manager.read_layer_tensors(i, target_device=torch.device("cpu"))
                    cache_store.put(i, cpu_payload)
                # Make weights available before arithmetic on either device.
                self._offer(i, cpu_payload)
                payload = self._gpu_payload(i) if can_move_layer else None
            if payload is not None:
                dev = self.gpu
                self._bind(module, payload, dev)
                if cache is not None and i < len(cache.layers):
                    move_tensors(cache.layers[i], dev)
                with self.lock:
                    if i not in self.adopted:
                        self.adopted.add(i)
                        self.transitions.append({"layer": i, "event": "gpu_layer_adopted",
                                                 "cpu_calls_before": self.cpu_layers})
                self.gpu_layers += 1
            else:
                dev = torch.device("cpu")
                self._bind(module, cpu_payload, dev)
                if cache is not None and hasattr(cache, "layers") and i < len(cache.layers):
                    move_tensors(cache.layers[i], dev)
                self.bound[i] = cpu_payload
                self.cpu_layers += 1
            converted = {k: v if k == cache_name else move_tensors(v, dev)
                         for k, v in kwargs.items()}
            return move_tensors(args, dev), converted
        return before

    def _after(self, i):
        def after(module, args, output):
            payload = self.bound.pop(i, None)
            if payload is not None:
                # The model graph retains no full CPU checkpoint outside the
                # bounded RAM cache and two in-flight payloads.
                for n, t in list(module.named_parameters()):
                    if not t.is_meta:
                        set_module_tensor_to_device(module, n, "meta", clear_cache=False)
                for n, t in list(module.named_buffers()):
                    if n.rsplit(".", 1)[-1] in ("packed_lo", "packed_hi", "qscale", "qzero"):
                        set_module_tensor_to_device(module, n, "meta", clear_cache=False)
            # Until global adoption the HF backbone's masks/rotary/norm live on CPU.
            return move_tensors(output, torch.device("cpu"))
        return after

    def _adopt_all(self, past):
        eng = self.engine
        if not self.bundle_ready.is_set():
            return past
        # Foreground token/chunk boundary: there are no live layer readers here.
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for i, payload in self.ready.items():
            self._bind(eng._decoder_layers()[i], payload, self.gpu)
        params, buffers = self.global_payload
        for n, t in params.items():
            set_module_tensor_to_device(eng.model, n, self.gpu, value=t, clear_cache=False)
        for n, t in buffers.items():
            # RoPE and other non-weight buffers may have changed on CPU since
            # background preparation. Adopt their current values and metadata.
            current = eng.model.get_buffer(n)
            live = current.to(self.gpu) if not current.is_meta else t
            set_module_tensor_to_device(eng.model, n, self.gpu, value=live, dtype=t.dtype, clear_cache=False)
        self.global_payload = None
        eng.device = self.gpu
        eng.vram_resident_layers = self.plan.resident_layers
        eng.num_slots = self.plan.ring_slots
        eng.tail_to_head_prefetch = self.plan.ring_slots > 1 and not self.plan.full_residency
        if self.gpu_pool is not None:
            eng.weight_manager.pinned_pool.close()
            eng.weight_manager.pinned_pool = self.gpu_pool
            self.gpu_pool = None
        eng.pipeline = self.gpu_pipeline
        self.gpu_pipeline = None
        if eng.model_source is not None:
            old_cache = getattr(eng.model_source, "_expert_loader", None)
            seeds = old_cache.detach_cpu_packed() if hasattr(old_cache, "detach_cpu_packed") else None
            expert_budget = max(0, self.plan.budget_bytes - self.plan.weights_bytes - 256 * 1024**2)
            eng.model_source.prepare_model(eng.model, device=self.gpu, dtype=eng.dtype,
                                           expert_cache_bytes=expert_budget, quant_bits=eng.quant_bits)
            new_cache = getattr(eng.model_source, "_expert_loader", None)
            if seeds is not None and hasattr(new_cache, "adopt_cpu_packed"):
                new_cache.adopt_cpu_packed(seeds)
        from src.colibri_deltanet_nvrtc import install_deltanet_nvrtc_kernels
        install_deltanet_nvrtc_kernels(eng.model, patch_norms=True, patch_zero_centered_norm=True)
        eng._install_pipeline_hooks()
        past = move_tensors(past, self.gpu)
        if self.deferred_warmup is not None:
            self.deferred_warmup.activate()
        self.complete = True
        self.gpu_state = "adopted"
        self.ready.clear()
        self.transitions.append({"event": "gpu_ring_adopted", "cpu_steps_before": self.cpu_steps})
        return past

    def _continue_on_cpu(self, past):
        """Retire an unsuccessful helper without replaying the model's prefix.

        Called only between forwards, while the original CPU hooks still own
        the graph. Runtime failures after full GPU adoption are not intercepted.
        """
        self.pause()
        past = move_tensors(past, torch.device("cpu"))
        for i in self.adopted:
            layer = self.engine._decoder_layers()[i]
            for name, tensor in list(layer.named_parameters()):
                if not tensor.is_meta:
                    set_module_tensor_to_device(layer, name, "meta", clear_cache=False)
            for name, tensor in list(layer.named_buffers()):
                if tensor.is_meta:
                    continue
                if name.rsplit(".", 1)[-1] in ("packed_lo", "packed_hi", "qscale", "qzero"):
                    set_module_tensor_to_device(layer, name, "meta", clear_cache=False)
                elif tensor.device.type != "cpu":
                    set_module_tensor_to_device(layer, name, "cpu", value=tensor.cpu(),
                                                dtype=tensor.dtype, clear_cache=False)
        self.adopted.clear()
        self.ready.clear()
        self.global_payload = None
        self._globals.clear()
        self._buffers.clear()
        if self.gpu_pipeline is not None:
            self.gpu_pipeline.close()
            self.gpu_pipeline = None
        if self.gpu_pool is not None:
            self.gpu_pool.close()
            self.gpu_pool = None
        self.bundle_ready.clear()
        self.backend_ready.clear()
        self.kernels_ready = False
        self.gpu_enabled = False
        self.gpu_state = "failed"
        self.transitions.append({"event": "cpu_continued", "reason": self.error,
                                 "cpu_steps": self.cpu_steps})
        return past

    def forward(self, tokens, past, prefetch, position, prefill, prefill_end, return_all_logits=False):
        if self.error is not None and self.gpu_enabled and not self.complete and not self.auxiliary_depth:
            past = self._continue_on_cpu(past)
        if not self.auxiliary_depth:
            self.start()
        if not self.auxiliary_depth and (self.deferred_warmup is None or not self.deferred_warmup.auxiliary_depth):
            past = self._adopt_all(past)
        if self.complete:
            return self.engine._forward_step(tokens, past, prefetch, position,
                                             prefill=prefill, prefill_end=prefill_end,
                                             return_all_logits=return_all_logits)
        eng = self.engine
        warmup = self.deferred_warmup
        if warmup is not None:
            warmup.begin_step(tokens, past, prefill=prefill, prefill_end=prefill_end)
        inputs = eng._embed_step_tokens(tokens)
        kwargs = {eng._cache_kwarg: past} if eng._cache_kwarg else {}
        if eng._supports_use_cache:
            kwargs["use_cache"] = True
        if position is not None and eng._supports_cache_position:
            kwargs["cache_position"] = position
        if eng._supports_logits_to_keep:
            kwargs["logits_to_keep"] = 0 if return_all_logits else 1
        ple = eng._compute_per_layer_inputs(tokens)
        if ple is not None:
            kwargs["per_layer_inputs"] = ple
        outputs = eng.model(inputs_embeds=inputs, input_ids=None, **kwargs)
        new_past = getattr(outputs, eng._cache_kwarg, None) if eng._cache_kwarg else None
        self.cpu_steps += 1
        if warmup is not None:
            warmup.end_step()
        if self.error is not None and self.gpu_enabled and not self.auxiliary_depth:
            new_past = self._continue_on_cpu(new_past)
        return (outputs.logits if return_all_logits else outputs.logits[:, -1, :]), new_past

    def pause(self):
        self.stopped.set()
        if self.worker is not None:
            self.worker.join()
            self.worker = None
        with self.lock:
            self.pending.clear()
        while not self.jobs.empty():
            self.jobs.get_nowait()

    @contextmanager
    def auxiliary_forwards(self):
        self.auxiliary_depth += 1
        try:
            yield
        finally:
            self.auxiliary_depth -= 1

    @contextmanager
    def verification_forwards(self):
        # Freeze compute placement for rollback, but allow already-read immutable
        # weights to continue preparing the optional GPU in the background.
        self.verification_depth += 1
        try:
            with self.auxiliary_forwards():
                yield
        finally:
            self.verification_depth -= 1


    def close(self):
        self.pause()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if self.gpu_pipeline is not None:
            self.gpu_pipeline.close()
            self.gpu_pipeline = None
        if self.gpu_pool is not None:
            self.gpu_pool.close()
            self.gpu_pool = None
        self.global_payload = None
        self._globals.clear()
        self._buffers.clear()
        self.error = None
        self.plan_factory = None
        self.ready.clear()
        self.bound.clear()
        while not self.jobs.empty():
            self.jobs.get_nowait()


class DeferredWarmup:
    """Collect the original CPU prefill map, then attach the existing GPU arena."""
    def __init__(self, engine, options):
        from src.probability_warmup import PrefillRegionMap
        self.engine, self.options, self.active = engine, options, None
        loader = getattr(engine.model_source, "_expert_loader", None)
        eligible = getattr(loader, "cpu_quant_layers", ())
        if engine.config.model_type not in {"granitemoe", "olmoe"} or len(eligible) < 3:
            raise ValueError("Progressive warmup requires HF Granite/OLMoE Q4/Q6")
        numbers = {i: loader.base.num_experts_by_layer[i] for i in sorted(eligible)}
        capacity = min(int(options.get("region_experts", 12)), min(numbers.values()))
        self.region_map = PrefillRegionMap(numbers, capacity)
        self.counts = {i: [0]*n for i, n in numbers.items()}
        self.collecting = False
        self.auxiliary_depth = 0
        self.is_prefill = False
        self.verification_depth = 0
        self.prefill_end = False
        self.handles = []
        for i in numbers:
            layer = engine._decoder_layers()[i]
            experts = (layer.block_sparse_moe if engine.config.model_type == "granitemoe" else layer.mlp).experts
            self.handles.append(experts.register_forward_pre_hook(self._observe(i)))

    def _observe(self, i):
        def observe(module, args):
            if self.auxiliary_depth or self.verification_depth:
                return
            if self.is_prefill:
                for eid in args[1].detach().reshape(-1).tolist():
                    if 0 <= int(eid) < len(self.counts[i]):
                        self.counts[i][int(eid)] += 1
            elif i in self.region_map.initial:
                self.region_map.observe(i, args[1].detach().reshape(-1).tolist())
        return observe

    def begin_step(self, tokens, past, *, prefill=None, prefill_end=True):
        if self.auxiliary_depth:
            return
        self.is_prefill = (past is None or tokens.shape[-1] > 1) if prefill is None else bool(prefill)
        self.prefill_end = prefill_end
        if self.is_prefill and not self.collecting:
            self.counts = {i: [0]*n for i, n in self.region_map.num_experts.items()}
            self.collecting = True

    def end_step(self):
        if self.auxiliary_depth:
            return
        if self.is_prefill and self.prefill_end:
            for i, counts in self.counts.items():
                self.region_map.seed(i, counts)
            self.collecting = False

    def activate(self):
        from src.probability_warmup import ProbabilityWarmup
        for h in self.handles:
            h.remove()
        self.handles.clear()
        active = ProbabilityWarmup(self.engine, **self.options)
        # Include all completed CPU chunks if GPU joins mid-prefill.
        active.region_map = self.region_map
        active.building_counts = self.counts
        active.collecting = self.collecting
        self.active = active
        self.engine.probability_warmup = active

    def stats(self):
        return self.active.stats() if self.active is not None else {"cpu_first_pending": True, "map": self.region_map.stats()}

    def reset(self):
        if self.active is not None:
            self.active.reset()
        self.collecting = self.is_prefill = False

    @contextmanager
    def verification_forwards(self):
        if self.active is not None:
            with self.active.verification_forwards():
                yield
            return
        self.verification_depth += 1
        loader = getattr(self.engine.model_source, "_expert_loader", None)
        if hasattr(loader, "auxiliary_depth"):
            loader.auxiliary_depth += 1
        try:
            yield
        finally:
            if hasattr(loader, "auxiliary_depth"):
                loader.auxiliary_depth -= 1
            self.verification_depth -= 1

    @contextmanager
    def auxiliary_forwards(self):
        if self.active is not None:
            with self.active.auxiliary_forwards():
                yield
            return
        self.auxiliary_depth += 1
        try:
            yield
        finally:
            self.auxiliary_depth -= 1

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        if self.active is not None:
            self.active.close()
