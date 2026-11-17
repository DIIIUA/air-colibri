"""AirColibri Integrated Inference Engine:
Соединение математического графа HuggingFace / Transformers с ярусным менеджером памяти Colibri
и кольцевым конвейером GPU Ring Pipeline («Лесенка»).

Ключевые свойства:
1. Математический forward: оригинальный Qwen3_5 / Transformers (без упрощений, бит-в-бит)
2. GPU Ring Pipeline: 3-слотовая кольцевая архитектура с перекрытием вычислений и DMA-передачи
3. Bounded Memory Manager: строгий лимит RAM-кэша + 2-3 закрепленных слота (~800-1200 МБ)
4. Tail-to-Head Prefetch: упреждающая загрузка слоя 0 следующего токена во время вычисления хвоста
5. Zero VRAM Waste: embed_tokens вынесен в Host RAM (экономия 2.03 ГБ VRAM)
"""

import collections
import gc
import inspect
import os
import sys
import time
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import safetensors
import safetensors.torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from src.bounded_memory_manager import StableWeightManager, BoundedRAMCache, BoundedPinnedBufferPool
from src.gpu_ring_pipeline import GPURingPipeline, SlotState
from src.quantization_q4 import (
    ColibriQ4Linear,
    Q4_GROUP_SIZE,
    Q4_SCRATCH_ELEMS,
    q4_subkey_names,
    quantize_tensor_q4,
    replace_linear_with_q4,
)


def _cfg_num_layers(cfg: Any) -> int:
    for attr in ("num_hidden_layers", "n_layers", "n_layer", "num_layers"):
        val = getattr(cfg, attr, None)
        if val is not None:
            return int(val)
    raise AttributeError(f"{type(cfg).__name__} has no layer count attribute")


_NORM_WEIGHT_CANDIDATES: Tuple[str, ...] = (
    "model.norm.weight",
    "model.language_model.norm.weight",
    "language_model.model.norm.weight",
    "model.embedding_norm.weight",
    "model.final_layernorm.weight",
    "model.decoder.final_layer_norm.weight",
    "model.decoder.layer_norm.weight",
    "transformer.ln_f.weight",
    "transformer.norm_f.weight",
    "gpt_neox.final_layer_norm.weight",
    "backbone.norm_f.weight",
    "backbone.norm.weight",
    "rwkv.ln_out.weight",
    "biogpt.layer_norm.weight",
    "norm.weight",
    "ln_f.weight",
)

_LM_HEAD_WEIGHT_CANDIDATES: Tuple[str, ...] = (
    "lm_head.weight",
    "language_model.lm_head.weight",
    "embed_out.weight",
    "output_projection.weight",
    "head.weight",
    "transformer.lm_head.weight",
)


def _cleanup_failed_init(init):
    @wraps(init)
    def wrapped(self, *args, **kwargs):
        try:
            init(self, *args, **kwargs)
        except BaseException:
            self.close()
            raise
    return wrapped


def _fail_runtime_on_error(forward):
    @wraps(forward)
    def wrapped(self, *args, **kwargs):
        try:
            return forward(self, *args, **kwargs)
        except BaseException as exc:
            if not getattr(exc, "_is_cooperative_cancel", False):
                # Initial embedding / ring preparation can already own resources,
                # so non-cancel failures before model.forward require close and reopen.
                self._failed = True
            raise
    return wrapped


class _NoStreamingPipeline:
    """Placeholder ring for fully VRAM-resident models: holds no VRAM and must never be streamed through."""

    num_slots = 0
    arena = None

    def _unused(self, *args, **kwargs):
        raise RuntimeError("All layers are VRAM-resident; the streaming ring is not allocated")

    get_slot_for_loading = load_tensors_async = acquire_slot_for_compute = _unused
    release_slot_after_compute = prefetch_tail_to_head = _unused

    def reset_in_flight(self) -> None:
        return None

    def close(self) -> None:
        return None


class AirColibriModel:
    """Интегрированная модель AirColibri с поддержкой GPU Ring Pipeline и Tail-to-Head Prefetch."""

    @_cleanup_failed_init
    def __init__(
        self,
        model_dir: str,
        pipeline_mode: str = "triple_ring_prefetch",  # "sequential", "double_buffer", "triple_ring", "triple_ring_prefetch"
        vram_resident_layers: int = 0,
        max_ram_layers: int = 8,
        cache_policy: str = "static_pinned",
        num_slots: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
        weight_format: str = "bf16",
        quantize_lm_head: bool = False,
        lazy_embed_lookup: Optional[bool] = None,
        q4_group_size: int = Q4_GROUP_SIZE,
        q4_cache_dir: Optional[str] = None,
        checkpoint_manifest: Optional[str] = None,
        max_ram_bytes: Optional[int] = None,
        control_mode: str = "off",
        airllm_source: Optional[str] = None,
        native_policy=None,
        use_fused_q4_gemv: bool = True,
        model_source=None,
        weight_reader=None,
        expert_cache_bytes: int = 0,
        universal_quantization: bool = False,
    ):
        if pipeline_mode not in ("sequential", "double_buffer", "triple_ring", "triple_ring_prefetch"):
            raise ValueError("Unsupported pipeline mode")
        if control_mode not in ("off", "observe", "enforce"):
            raise ValueError("Unsupported control mode")
        if weight_format not in ("bf16", "q4", "q6", "q4_g16_asym", "q4_g32_asym", "q4_g64_asym"):
            raise ValueError("Unsupported weight format")
        if q4_group_size not in (16, 32, 64):
            raise ValueError("Q4 group size must be 16, 32 or 64")
        if num_slots is not None and not 1 <= num_slots <= 4:
            raise ValueError("num_slots must be 1..4")
        self.control_mode = control_mode
        self.ack_adapter = None
        self.trace_adapter = None
        self.native_policy = native_policy
        self._failed = False
        self._wavefront_active = False
        self.weight_reader = weight_reader
        if self.weight_reader is None and airllm_source:
            from src.upstream_bridge import AirLLMReader
            self.weight_reader = AirLLMReader(airllm_source)
        self.model_source = model_source
        self.model_dir = model_dir
        self.pipeline_mode = pipeline_mode
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.vram_resident_layers = vram_resident_layers
        self.cache_policy = cache_policy
        self.quant_bits = (6 if weight_format == "q6" else 4 if weight_format.startswith("q4") else 0) if universal_quantization or weight_format == "q6" else 0
        self.is_q4 = weight_format.startswith("q4") and not self.quant_bits
        self.weight_format = f"packed_q{self.quant_bits}" if self.quant_bits else f"q4_g{q4_group_size}_asym" if self.is_q4 else "bf16"
        self.quantize_lm_head = bool(quantize_lm_head and self.is_q4)
        self.lazy_embed_lookup = True if lazy_embed_lookup is None else bool(lazy_embed_lookup)
        self.q4_group_size = q4_group_size
        self.use_fused_q4_gemv = bool(use_fused_q4_gemv and self.is_q4)
        self.fused_q4_gemv_calls = 0
        self.tiled_q4_fallback_calls = 0
        self._q4_wrapped_modules: List[Tuple[ColibriQ4Linear, Any]] = []
        self._q4_scratch: Optional[torch.Tensor] = None
        self.embed_tokens_weight: Optional[torch.Tensor] = None
        self._step_cancel_check = None

        # 1. Загрузка конфигурации и токенизатора
        if model_source is not None:
            from aircolibri_cli.model_source import load_runtime_config
            self.config = load_runtime_config(model_dir)
        else:
            self.config = AutoConfig.from_pretrained(model_dir, config_file="config.hf.json")
        self.fast_qwen = self.config.model_type == "qwen3_5_text"
        if self.is_q4 and not self.fast_qwen:
            raise ValueError("Q4 fast path supports qwen3_5_text only; other models run in BF16")
        num_hidden_layers = _cfg_num_layers(self.config)
        if not 0 <= vram_resident_layers <= num_hidden_layers:
            raise ValueError("Resident layer count outside model")
        # Preserve the checkpoint tokenizer without ever calling mmap on GGUF files.
        if getattr(model_source, "gguf_file", None):
            from aircolibri_cli.model_source import load_runtime_tokenizer
            self.tokenizer = load_runtime_tokenizer(model_dir, config=self.config)
        else:
            tok_kwargs: Dict[str, Any] = {"fix_mistral_regex": False, "local_files_only": True}
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_dir, **tok_kwargs)
            except TypeError:
                tok_kwargs.pop("fix_mistral_regex", None)
                self.tokenizer = AutoTokenizer.from_pretrained(model_dir, **tok_kwargs)

        # Определяем параметры кольца по режиму (при полной VRAM-резидентности всех слоёв стриминга нет -> 1 слот)
        all_layers_resident = (self.vram_resident_layers == num_hidden_layers)
        if num_slots is None:
            if all_layers_resident:
                self.num_slots = 1
            elif pipeline_mode in ("triple_ring", "triple_ring_prefetch"):
                self.num_slots = 3
            elif pipeline_mode == "double_buffer":
                self.num_slots = 2
            else:
                self.num_slots = 1
        else:
            self.num_slots = num_slots

        self.tail_to_head_prefetch = (
            pipeline_mode == "triple_ring_prefetch"
            and self.num_slots >= 2
            and not all_layers_resident
        )

        # Discover module-call projections before sizing the packed streaming slots.
        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(self.config, dtype=self.dtype)
        self.model.eval()
        quantized_projections = None
        if self.quant_bits:
            from pathlib import Path
            from aircolibri_cli.unified_runtime import _model_shape_inventory
            from src.quantized_projection import projection_map
            layer_shapes, _, _ = _model_shape_inventory(Path(model_dir), model_source, num_hidden_layers)
            quantized_projections = projection_map(self.model, layer_shapes, self.quant_bits, self.q4_group_size, replace=True)

        # 2. Инициализация менеджера памяти
        self.weight_manager = StableWeightManager(
            model_dir=model_dir,
            max_ram_layers=max_ram_layers,
            num_pinned_slots=self.num_slots,
            cache_policy=cache_policy,
            dtype=self.dtype,
            weight_format=self.weight_format,
            q4_group_size=self.q4_group_size,
            q4_cache_dir=q4_cache_dir,
            checkpoint_manifest=checkpoint_manifest,
            max_ram_bytes=max_ram_bytes,
            weight_reader=self.weight_reader,
            model_source=model_source,
            quantized_projections=quantized_projections,
        )

        # Bind source-specific expert loading without hiding implementation errors.
        if model_source is not None and hasattr(model_source, "prepare_model"):
            prepare_args = dict(device=self.device, dtype=self.dtype,
                                expert_cache_bytes=int(expert_cache_bytes), quant_bits=self.quant_bits)
            params = inspect.signature(model_source.prepare_model).parameters
            if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
                prepare_args = {key: value for key, value in prepare_args.items() if key in params}
            model_source.prepare_model(self.model, **prepare_args)
        if self.device.type == "cuda":
            from src.colibri_deltanet_nvrtc import install_deltanet_nvrtc_kernels

            install_deltanet_nvrtc_kernels(
                self.model,
                patch_norms=True,
                patch_zero_centered_norm=not self.is_q4,
            )
        try:
            sig_params = inspect.signature(self.model.forward).parameters
            has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_params.values())
            self._supports_logits_to_keep = "logits_to_keep" in sig_params or has_varkw
            if "past_key_values" in sig_params:
                self._cache_kwarg: Optional[str] = "past_key_values"
            elif "cache_params" in sig_params:
                self._cache_kwarg = "cache_params"
            elif "state" in sig_params:
                self._cache_kwarg = "state"
            elif has_varkw:
                self._cache_kwarg = "past_key_values"
            else:
                self._cache_kwarg = None
            self._supports_cache_position = "cache_position" in sig_params or (
                self._cache_kwarg == "past_key_values" and has_varkw
            )
            self._supports_use_cache = "use_cache" in sig_params or has_varkw
        except (TypeError, ValueError):
            self._supports_logits_to_keep = False
            self._cache_kwarg = "past_key_values"
            self._supports_cache_position = True
            self._supports_use_cache = True
        if self.is_q4:
            replace_linear_with_q4(
                self.model,
                quantize_lm_head=self.quantize_lm_head,
                scratch_provider=self._get_q4_scratch,
                group_size=self.q4_group_size,
                compute_dtype=self.dtype,
            )
            if self.use_fused_q4_gemv:
                self._install_fused_q4_gemv_wrappers()

        # 4. Постоянные веса и VRAM-резидентные слои
        self._setup_permanent_weights()
        self.weight_manager.globals.release_handle()

        # 5. Инициализация GPU Ring Pipeline на едином layout без union-раздувания.
        # When every layer is resident nothing is streamed, so no ring arena is allocated
        # (it used to waste 120-475 MiB of VRAM on 8 GB cards).
        if all_layers_resident:
            self.pipeline = _NoStreamingPipeline()
        else:
            self.pipeline = GPURingPipeline(
                template_tensors=None,
                num_slots=self.num_slots,
                device=self.device,
                dtype=self.dtype,
                tail_to_head_prefetch=self.tail_to_head_prefetch,
                arena_layout=self.weight_manager.pinned_pool.layout,
            )
        self.tail_prefetched = False

        # 6. Загружаем VRAM-резидентные слои
        for r_idx in range(self.vram_resident_layers):
            self._load_resident_layer(r_idx)

        # 7. Установка стриминговых хуков на декодер-слои
        self._install_pipeline_hooks()
        if native_policy is not None:
            self.weight_manager.ram_cache.admission = native_policy.admit
            self.weight_manager.ram_cache.on_access = native_policy.record_access
            self.weight_manager.ram_cache.on_placement = native_policy.acknowledge
        if control_mode != "off":
            from src.catalog_bridge import ExistingCrossLayerCatalog
            from src.existing_hooks import ExistingForwardHookAdapter
            self.trace_adapter = ExistingForwardHookAdapter(self, ExistingCrossLayerCatalog(model_dir)).install()
        if control_mode == "enforce":
            from src.lease_ack_bridge import ExistingRingAcknowledgementAdapter
            self.ack_adapter = ExistingRingAcknowledgementAdapter(self.pipeline).install()

        # Статистика
        self.last_step_time_ms = 0.0

    def _get_q4_scratch(self, device: torch.device) -> torch.Tensor:
        """Единый переиспользуемый scratch-буфер <= 12 MiB для тайлового деквантования Q4."""
        if self._q4_scratch is None or self._q4_scratch.device != device:
            self._q4_scratch = torch.empty(Q4_SCRATCH_ELEMS, dtype=self.dtype, device=device)
        return self._q4_scratch

    def _install_fused_q4_gemv_wrappers(self) -> int:
        """Оборачивает ColibriQ4Linear.forward для прямого запуска NVRTC W4A16 GEMV без распаковки."""
        from src.colibri_nvrtc import can_use_fused_dense_q4, fused_dense_linear_q4

        wrapped_count = 0
        engine_ref = self
        for submod in self.model.modules():
            if isinstance(submod, ColibriQ4Linear):
                orig_forward = submod.forward

                def _fused_or_tiled_forward(
                    x: torch.Tensor,
                    _m: ColibriQ4Linear = submod,
                    _orig=orig_forward,
                ) -> torch.Tensor:
                    qweight = _m._buffers["qweight"]
                    scales = _m._buffers["scales"]
                    zeros = _m._buffers["zeros"]
                    bias = _m._buffers["bias"]
                    if (
                        engine_ref.use_fused_q4_gemv
                        and x.dtype == torch.bfloat16
                        and can_use_fused_dense_q4(x, qweight, scales, zeros, _m.group_size)
                    ):
                        engine_ref.fused_q4_gemv_calls += 1
                        return fused_dense_linear_q4(
                            x,
                            qweight,
                            scales,
                            zeros,
                            bias=bias,
                            group_size=_m.group_size,
                        )
                    engine_ref.tiled_q4_fallback_calls += 1
                    return _orig(x)

                submod.forward = _fused_or_tiled_forward
                self._q4_wrapped_modules.append((submod, orig_forward))
                wrapped_count += 1
        return wrapped_count

    @staticmethod
    def _collect_layer_bindings(layer_module: torch.nn.Module) -> Tuple[List[Tuple[ColibriQ4Linear, str, str, str]], List[str]]:
        q4_bindings: List[Tuple[ColibriQ4Linear, str, str, str]] = []
        for mod_name, submod in layer_module.named_modules():
            if isinstance(submod, ColibriQ4Linear):
                q_k, s_k, z_k = q4_subkey_names(f"{mod_name}.weight")
                q4_bindings.append((submod, q_k, s_k, z_k))
        raw_params = [k for k, _ in layer_module.named_parameters()]
        return q4_bindings, raw_params

    @staticmethod
    def _collect_layer_buffers(layer_module: torch.nn.Module) -> Dict[str, torch.dtype]:
        out: Dict[str, torch.dtype] = {}
        for k, b in layer_module.named_buffers():
            if b is None or k.endswith((".qweight", ".scales", ".zeros")):
                continue
            out[k] = b.dtype
        return out

    def _decoder_layers(self):
        """Resolves the decoder block list across standard, VLM-unwrapped, GPT-2, OPT, NeoX, Mamba, and RWKV models."""
        inner = getattr(self.model, "model", None)
        layers = getattr(inner, "layers", None) if inner is not None else None
        if layers is not None:
            return layers
        from aircolibri_cli.model_source import resolve_decoder_layers

        layers = resolve_decoder_layers(self.model)
        if layers is None:
            raise RuntimeError(f"{type(self.model).__name__} has no supported decoder layer stack")
        return layers

    def _load_resident_layer(self, layer_idx: int) -> None:
        """Загружает VRAM-резидентный слой навсегда на GPU (в bf16 или packed Q4)."""
        layer_module = self._decoder_layers()[layer_idx]
        if self.is_q4 or self.quant_bits or self.model_source is not None:
            tensors = self.weight_manager.read_layer_tensors(layer_idx, target_device=self.device)
            q4_bindings, raw_params = self._collect_layer_bindings(layer_module)
            required = set(raw_params)
            for _, q, s, z in q4_bindings:
                required.update((q, s, z))
            required.update(k for k in self._collect_layer_buffers(layer_module)
                            if k.rsplit(".", 1)[-1] in ("packed_lo", "packed_hi", "qscale", "qzero"))
            missing = required - tensors.keys()
            if missing:
                raise ValueError(f"Layer {layer_idx} is missing required tensors: {sorted(missing)}")
            for submod, q_k, s_k, z_k in q4_bindings:
                submod.bind_q4_tensors(tensors[q_k], tensors[s_k], tensors[z_k])
            for k in raw_params:
                if k in tensors:
                    set_module_tensor_to_device(
                        layer_module, k, self.device, value=tensors[k], dtype=self.dtype, clear_cache=False
                    )
            for buf_k, buf_dt in self._collect_layer_buffers(layer_module).items():
                if buf_k in tensors:
                    set_module_tensor_to_device(
                        layer_module, buf_k, self.device, value=tensors[buf_k], dtype=buf_dt, clear_cache=False
                    )
            return

        layer_file = os.path.join(self.model_dir, f"model-{layer_idx:05d}.safetensors")
        prefix = f"model.layers.{layer_idx}."
        with safetensors.safe_open(layer_file, framework="pt", device="cpu") as f:
            for k in f.keys():
                short_k = k.replace(prefix, "")
                t = f.get_slice(k)[:].to(device=self.device, dtype=self.dtype)
                set_module_tensor_to_device(
                    layer_module, short_k, self.device, value=t, dtype=self.dtype, clear_cache=False
                )

    def _quantize_lm_head_from_slice(self, target_mod: Optional[Any] = None) -> None:
        """Квантует lm_head чанками из mmap прямо на GPU без материализации 1.94 ГБ в RAM."""
        sl = self.weight_manager.globals.get_tensor_slice("lm_head.weight")
        out_features, in_features = sl.get_shape()
        groups = in_features // self.q4_group_size
        q_out = torch.empty((out_features, in_features // 2), dtype=torch.uint8, device=self.device)
        s_out = torch.empty((out_features, groups), dtype=self.dtype, device=self.device)
        z_out = torch.empty((out_features, groups), dtype=self.dtype, device=self.device)

        chunk_rows = 2048
        for r0 in range(0, out_features, chunk_rows):
            r1 = min(r0 + chunk_rows, out_features)
            chunk = sl[r0:r1]
            p_c, s_c, z_c = quantize_tensor_q4(
                chunk,
                group_size=self.q4_group_size,
                compute_dtype=self.dtype,
                quant_device=self.device,
                target_device=self.device,
            )
            q_out[r0:r1].copy_(p_c)
            s_out[r0:r1].copy_(s_c)
            z_out[r0:r1].copy_(z_c)
            del chunk, p_c, s_c, z_c

        head_mod = target_mod if target_mod is not None else self.model.lm_head
        head_mod.bind_q4_tensors(q_out, s_out, z_out)
        del self.weight_manager.globals["lm_head.weight"]

    def _load_bf16_lm_head_from_slice(self, target_param_name: str = "lm_head.weight") -> None:
        """Загружает lm_head на GPU чанками из mmap без материализации 1.94 ГБ в RAM."""
        sl = self.weight_manager.globals.get_tensor_slice("lm_head.weight")
        shape = sl.get_shape()
        lm_head_w = torch.empty(shape, dtype=self.dtype, device=self.device)
        chunk_rows = 4096
        if len(shape) >= 2 and shape[0] > chunk_rows:
            for r0 in range(0, shape[0], chunk_rows):
                r1 = min(r0 + chunk_rows, shape[0])
                chunk = sl[r0:r1].to(device=self.device, dtype=self.dtype)
                lm_head_w[r0:r1].copy_(chunk)
                del chunk
        else:
            lm_head_w.copy_(sl[:].to(device=self.device, dtype=self.dtype))
        set_module_tensor_to_device(
            self.model, target_param_name, self.device, value=lm_head_w, dtype=self.dtype, clear_cache=False
        )
        del lm_head_w
        del self.weight_manager.globals["lm_head.weight"]

    def _setup_permanent_weights(self) -> None:
        """Эмбеддинг (ленивый построчный lookup или CPU), norm, lm_head и доп. глобалы на GPU."""
        if not self.lazy_embed_lookup and "model.embed_tokens.weight" in self.weight_manager.globals:
            # Полный embed на CPU только если отключен ленивый построчный lookup
            self.embed_tokens_weight = self.weight_manager.globals.pop("model.embed_tokens.weight")

        model_param_names = {n for n, _ in self.model.named_parameters(remove_duplicate=False)}
        model_buffer_dtypes = {
            n: b.dtype for n, b in self.model.named_buffers(remove_duplicate=False) if b is not None
        }

        # Norm на GPU (любое имя корня: model.norm, transformer.ln_f, gpt_neox.final_layer_norm, backbone.norm_f и др.)
        norm_w_path = next((c for c in _NORM_WEIGHT_CANDIDATES if c in model_param_names), "model.norm.weight")
        if "model.norm.weight" in self.weight_manager.globals:
            norm_w = self.weight_manager.globals.pop("model.norm.weight").to(device=self.device, dtype=self.dtype)
            set_module_tensor_to_device(
                self.model, norm_w_path, self.device, value=norm_w, dtype=self.dtype, clear_cache=False
            )
            del norm_w
        if "model.norm.bias" in self.weight_manager.globals:
            norm_b_path = norm_w_path[:-7] + ".bias"
            norm_b = self.weight_manager.globals.pop("model.norm.bias").to(device=self.device, dtype=self.dtype)
            if norm_b_path in model_param_names:
                set_module_tensor_to_device(
                    self.model, norm_b_path, self.device, value=norm_b, dtype=self.dtype, clear_cache=False
                )
            del norm_b

        # LM Head на GPU (lm_head, embed_out, head, output_projection)
        lm_w_path = next((c for c in _LM_HEAD_WEIGHT_CANDIDATES if c in model_param_names), "lm_head.weight")
        if "lm_head.weight" in self.weight_manager.globals:
            out_emb = (
                self.model.get_output_embeddings()
                if hasattr(self.model, "get_output_embeddings")
                else getattr(self.model, "lm_head", None)
            )
            if self.quantize_lm_head and isinstance(out_emb, ColibriQ4Linear):
                self._quantize_lm_head_from_slice(out_emb)
            else:
                self._load_bf16_lm_head_from_slice(lm_w_path)
        if "lm_head.bias" in self.weight_manager.globals:
            lm_b_path = lm_w_path[:-7] + ".bias"
            lm_b = self.weight_manager.globals.pop("lm_head.bias").to(device=self.device, dtype=self.dtype)
            if lm_b_path in model_param_names:
                set_module_tensor_to_device(
                    self.model, lm_b_path, self.device, value=lm_b, dtype=self.dtype, clear_cache=False
                )
            del lm_b

        # Дополнительные глобальные параметры (transformer.wpe, embed_positions, word_embeddings_layernorm и др.)
        for extra_k in list(self.weight_manager.globals.keys()):
            if extra_k == "model.embed_tokens.weight":
                continue
            if self.lazy_embed_lookup and extra_k == "model.embed_tokens_per_layer.weight":
                continue
            cands = (
                extra_k,
                f"model.{extra_k}",
                extra_k[6:] if extra_k.startswith("model.") else extra_k,
            )
            matched_param = next((c for c in cands if c in model_param_names), None)
            if matched_param is not None:
                val = self.weight_manager.globals.pop(extra_k).to(device=self.device, dtype=self.dtype)
                set_module_tensor_to_device(
                    self.model, matched_param, self.device, value=val, dtype=self.dtype, clear_cache=False
                )
                del val
                continue
            matched_buf = next((c for c in cands if c in model_buffer_dtypes), None)
            if matched_buf is not None:
                b_dt = model_buffer_dtypes[matched_buf]
                val = self.weight_manager.globals.pop(extra_k).to(device=self.device, dtype=b_dt)
                set_module_tensor_to_device(
                    self.model, matched_buf, self.device, value=val, dtype=b_dt, clear_cache=False
                )
                del val

        # Переносим не-meta буферы (rotary inv_freq, embed_scale, маски и др.) с CPU на целевое устройство
        for submod in self.model.modules():
            for buf_name, buf_val in list(submod._buffers.items()):
                if buf_val is not None and not buf_val.is_meta and buf_val.device != self.device:
                    submod._buffers[buf_name] = buf_val.to(device=self.device)

        gc.collect()

    def _install_pipeline_hooks(self) -> None:
        """Устанавливает pre/post хуки кольцевого конвейера на каждый слой."""
        num_layers = _cfg_num_layers(self.config)
        self.active_slot_per_layer = [None] * num_layers
        self._hook_handles: List[Any] = []

        # Собираем привязки Q4-подмодулей, параметров и опциональных буферов для каждого слоя
        layer_q4_bindings: List[List[Tuple[ColibriQ4Linear, str, str, str]]] = []
        layer_param_names: List[List[str]] = []
        layer_buffer_dtypes: List[Dict[str, torch.dtype]] = []
        for layer_mod in self._decoder_layers():
            q4_b, raw_p = self._collect_layer_bindings(layer_mod)
            layer_q4_bindings.append(q4_b)
            layer_param_names.append(raw_p)
            layer_buffer_dtypes.append(self._collect_layer_buffers(layer_mod))

        for idx, layer_module in enumerate(self._decoder_layers()):
            layer_module._air_layer_idx = idx

            def _pre_hook(module, args):
                if self._step_cancel_check is not None:
                    self._step_cancel_check()
                i = module._air_layer_idx
                warm = getattr(self, 'probability_warmup', None)
                dynamic = bool(warm and warm.dynamic_regions)
                if dynamic:
                    warm.before_layer(i)
                # Слой резидентен в VRAM?
                if i < self.vram_resident_layers:
                    return

                # A wave owns this layer's slot until its final prompt chunk.
                # All readers of a layer execute on the same CUDA stream.
                if self._wavefront_active and self.active_slot_per_layer[i] is not None:
                    return

                if self.num_slots == 1 and self.pipeline_mode == "sequential":
                    pinned = self.weight_manager.load_layer_to_pinned_slot(i, target_slot_idx=0)
                    target = self.pipeline.get_slot_for_loading(i)
                    self.pipeline.load_tensors_async(target, pinned, pinned_pool=self.weight_manager.pinned_pool)
                slot = self.pipeline.acquire_slot_for_compute(i)
                required = set(layer_param_names[i])
                required.update(k for k in layer_buffer_dtypes[i] if k.rsplit(".", 1)[-1] in ("packed_lo", "packed_hi", "qscale", "qzero"))
                for _, q, s, z in layer_q4_bindings[i]:
                    required.update((q, s, z))
                missing = required - slot.loaded_keys
                if missing:
                    raise ValueError(f"Layer {i} is missing required tensors: {sorted(missing)}")

                self.active_slot_per_layer[i] = slot
                slot_map = getattr(slot, "active_tensors", None) or slot.tensors
                # Zero-copy привязка указателей к буферам Q4, параметрам и буферам слоя
                for submod, q_k, s_k, z_k in layer_q4_bindings[i]:
                    submod.bind_q4_tensors(slot_map[q_k], slot_map[s_k], slot_map[z_k])
                for k in layer_param_names[i]:
                    if k in slot_map:
                        set_module_tensor_to_device(
                            module, k, self.device, value=slot_map[k], dtype=self.dtype, clear_cache=False
                        )
                for buf_k, buf_dt in layer_buffer_dtypes[i].items():
                    if buf_k in slot_map:
                        set_module_tensor_to_device(
                            module, buf_k, self.device, value=slot_map[buf_k], dtype=buf_dt, clear_cache=False
                        )
                if dynamic:
                    warm.prepare_next_dense(i)

            def _post_hook(module, args, output):
                i = module._air_layer_idx
                if self._wavefront_active:
                    return output
                if i < self.vram_resident_layers:
                    return output

                slot = self.active_slot_per_layer[i]
                self.pipeline.release_slot_after_compute(slot)
                self.active_slot_per_layer[i] = None
                warm = getattr(self, 'probability_warmup', None)
                if warm and warm.dynamic_regions:
                    if i==num_layers-1 and not self.tail_prefetched:
                        warm.prepare_next_dense(i)
                    return output
                if self.num_slots > 1 or self.pipeline_mode != "sequential":
                    # Запускаем загрузку следующего слоя для этого слота
                    next_layer = i + self.num_slots
                    if next_layer < num_layers:
                        pinned_t = self.weight_manager.load_layer_to_pinned_slot(next_layer, target_slot_idx=next_layer % self.num_slots)
                        next_slot = self.pipeline.get_slot_for_loading(next_layer)
                        self.pipeline.load_tensors_async(next_slot, pinned_t, pinned_pool=self.weight_manager.pinned_pool)
                    elif self.tail_to_head_prefetch and self._prefetch_next_step:
                        # Tail-to-Head Prefetch: загружаем первый стриминговый слой следующего токена
                        # ТОЛЬКО когда освободился именно его слот (target_slot_idx)!
                        first_s = self.vram_resident_layers
                        target_slot_idx = first_s % self.num_slots
                        if slot.slot_id == target_slot_idx and not self.tail_prefetched:
                            pinned_t = self.weight_manager.load_layer_to_pinned_slot(first_s, target_slot_idx=target_slot_idx)
                            self.pipeline.prefetch_tail_to_head(target_layer_idx=first_s, pinned_tensors=pinned_t, pinned_pool=self.weight_manager.pinned_pool)
                            self.tail_prefetched = True

                return output

            self._hook_handles.append(layer_module.register_forward_pre_hook(_pre_hook))
            self._hook_handles.append(layer_module.register_forward_hook(_post_hook))

    def _embed_step_tokens(self, step_tokens: torch.Tensor) -> torch.Tensor:
        if self.lazy_embed_lookup or self.embed_tokens_weight is None:
            embeds_cpu = self.weight_manager.globals.lookup_rows(
                "model.embed_tokens.weight",
                step_tokens,
                dtype=self.dtype,
            )
            embeds = embeds_cpu.to(device=self.device, dtype=self.dtype)
        else:
            embeds = torch.nn.functional.embedding(step_tokens, self.embed_tokens_weight).to(
                device=self.device,
                dtype=self.dtype,
            )
        emb_mod = self.model.get_input_embeddings() if hasattr(self.model, "get_input_embeddings") else None
        if emb_mod is not None:
            scale = getattr(emb_mod, "embed_scale", None)
            if scale is not None:
                if isinstance(scale, torch.Tensor):
                    embeds = embeds * scale.to(device=embeds.device, dtype=embeds.dtype)
                else:
                    embeds = embeds * float(scale)
            emb_norm = getattr(emb_mod, "embed_norm", None) or getattr(emb_mod, "norm", None)
            if callable(emb_norm):
                embeds = emb_norm(embeds)
        return embeds

    def _compute_per_layer_inputs(self, step_tokens: torch.Tensor) -> Optional[torch.Tensor]:
        """Computes Gemma-4 Per-Layer Embeddings via lazy row lookup without keeping PLE tables in VRAM."""
        inner = getattr(self.model, "model", None)
        ple_dim = int(getattr(inner, "hidden_size_per_layer_input", 0) or 0) if inner is not None else 0
        if ple_dim <= 0:
            return None
        ple_mod = getattr(inner, "embed_tokens_per_layer", None)
        if ple_mod is None:
            return None
        num_layers = _cfg_num_layers(self.config)
        if "model.embed_tokens_per_layer.weight" in self.weight_manager.globals:
            flat_cpu = self.weight_manager.globals.lookup_rows(
                "model.embed_tokens_per_layer.weight",
                step_tokens,
                dtype=self.dtype,
            )
            flat = flat_cpu.to(device=self.device, dtype=self.dtype)
            scale = getattr(ple_mod, "embed_scale", None)
            if scale is not None:
                if isinstance(scale, torch.Tensor):
                    flat = flat * scale.to(device=flat.device, dtype=flat.dtype)
                else:
                    flat = flat * float(scale)
            return flat.reshape(*step_tokens.shape, num_layers, ple_dim)
        if hasattr(inner, "get_per_layer_inputs"):
            return inner.get_per_layer_inputs(step_tokens.to(device=self.device))
        return None

    @_fail_runtime_on_error
    def _forward_step(self, step_tokens: torch.Tensor, past_key_values: Any = None,
                      prefetch_next: bool = True, cache_position: Optional[torch.Tensor] = None,
                      *, prefill: Optional[bool] = None, prefill_end: bool = True) -> Tuple[torch.Tensor, Any]:
        if self._failed or self.model is None:
            raise RuntimeError("Model is closed or failed; create a new runtime")
        self._prefetch_next_step = prefetch_next
        warmup = getattr(self, "probability_warmup", None)
        if warmup is not None:
            warmup.begin_step(step_tokens, past_key_values, prefill=prefill, prefill_end=prefill_end)
        num_layers = _cfg_num_layers(self.config)
        inputs_embeds = self._embed_step_tokens(step_tokens)
        ple_inputs = self._compute_per_layer_inputs(step_tokens)

        # Первоначальное заполнение кольца перед началом слоев
        if self.num_slots > 1 or self.pipeline_mode != "sequential":
            first_s = self.vram_resident_layers
            initial_depth = 1 if warmup and warmup.dynamic_regions else self.num_slots
            for s in range(first_s, min(first_s + initial_depth, num_layers)):
                if s == first_s and self.tail_prefetched:
                    self.tail_prefetched = False
                    continue
                pinned_t = self.weight_manager.load_layer_to_pinned_slot(s, target_slot_idx=s % self.num_slots)
                slot = self.pipeline.get_slot_for_loading(s)
                self.pipeline.load_tensors_async(slot, pinned_t, pinned_pool=self.weight_manager.pinned_pool)

        try:
            extra: Dict[str, Any] = {}
            if ple_inputs is not None:
                extra["per_layer_inputs"] = ple_inputs
            cache_kw = getattr(self, "_cache_kwarg", "past_key_values")
            if cache_kw is not None:
                extra[cache_kw] = past_key_values
            if getattr(self, "_supports_use_cache", True):
                extra["use_cache"] = True
            if cache_position is not None and getattr(self, "_supports_cache_position", True):
                extra["cache_position"] = cache_position.to(device=self.device)
            if getattr(self, "_supports_logits_to_keep", False):
                extra["logits_to_keep"] = 1
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                input_ids=None,
                **extra,
            )
        except BaseException as exc:
            if not getattr(exc, "_is_cooperative_cancel", False):
                self._failed = True
            raise
        next_cache = getattr(outputs, "past_key_values", None)
        if next_cache is None and cache_kw and cache_kw != "past_key_values":
            next_cache = getattr(outputs, cache_kw, None)
        if warmup is not None:
            warmup.end_step(outputs.logits, next_cache)
        return outputs.logits[:, -1, :], next_cache

    def enable_unified_memory(self, **kwargs):
        """Join ring/cache/preparation with elastic A/B and empty C by default.

        By default preparation is carved out of the existing weight budget.
        An explicit gpu_budget_bytes can add a separately budgeted allowance.
        dynamic_regions=False retains the previous three-equal-area experiment.
        """
        kwargs.setdefault('gpu_budget_bytes', 0)
        kwargs.setdefault('dynamic_regions', True)
        kwargs['unified_memory'] = True
        return self.enable_probability_warmup(**kwargs)

    def enable_probability_warmup(self, **kwargs):
        """Opt-in HF MoE preparation; unified_memory=True joins ring/cache/producer.

        Enable before the first forward. The unified budget is the current ring
        plus cache payload plus gpu_budget_bytes; conversion reserves are retained.
        """
        if getattr(self, "probability_warmup", None) is not None:
            raise RuntimeError("Probability warmup is already enabled")
        from src.probability_warmup import ProbabilityWarmup
        self.probability_warmup = ProbabilityWarmup(self, **kwargs)
        return self.probability_warmup

    def prefill_wavefront(self, step_tokens: torch.Tensor, chunk_size: int,
                          past_key_values: Any = None, *, concurrent: bool = True,
                          collect_timing: bool = False):
        """Experimental three-stage, causally ordered prompt-chunk pipeline."""
        from src.prefill_wavefront import run_prefill_wavefront
        return run_prefill_wavefront(self, step_tokens, chunk_size, past_key_values,
                                     concurrent=concurrent, collect_timing=collect_timing)

    def reset_after_cancel(self) -> None:
        """Restores ring slots and prefetch flags after a cooperative cancellation."""
        self._step_cancel_check = None
        if getattr(self, "probability_warmup", None) is not None:
            self.probability_warmup.reset()
        self.tail_prefetched = False
        if getattr(self, "active_slot_per_layer", None):
            for i in range(len(self.active_slot_per_layer)):
                self.active_slot_per_layer[i] = None
        if getattr(self, "pipeline", None) is not None and hasattr(self.pipeline, "reset_in_flight"):
            self.pipeline.reset_in_flight()

    def compute_next_token_logits(self, prompt: str) -> torch.Tensor:
        """Возвращает полный вектор логитов следующего токена [1, vocab_size] на CPU."""
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            logits, _ = self._forward_step(input_ids, past_key_values=None, prefetch_next=False)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            return logits.detach().cpu()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 10,
        temperature: float = 0.0
    ) -> Tuple[str, List[int], List[float]]:
        """Честная генерация токенов через сквозной вычислительный граф и кольцевой конвейер."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]

        generated_ids = input_ids[0].tolist()
        step_times = []

        print(f"\n[AirColibri Integrated] Starting genuine generation for prompt: '{prompt}'")
        print(f"[AirColibri Integrated] Input tokens: {len(generated_ids)} | Target new tokens: {max_new_tokens} | Mode: {self.pipeline_mode}")

        past_key_values = None

        with torch.no_grad():
            for step in range(max_new_tokens):
                t0 = time.perf_counter()

                # На первом шаге передаем весь prompt, затем по 1 токену с KV-кэшем
                if past_key_values is None:
                    step_tokens = torch.tensor([generated_ids], dtype=torch.long)
                else:
                    step_tokens = torch.tensor([[generated_ids[-1]]], dtype=torch.long)

                logits, past_key_values = self._forward_step(step_tokens, past_key_values=past_key_values,
                                                           prefetch_next=step + 1 < max_new_tokens)

                # Проверка корректности логитов
                if torch.isnan(logits).any():
                    raise ValueError(f"NaN detected in logits at step {step}!")

                if temperature == 0.0:
                    next_token = torch.argmax(logits, dim=-1).item()
                else:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()

                generated_ids.append(next_token)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                step_ms = (time.perf_counter() - t0) * 1000.0
                step_times.append(step_ms)

                decoded_token = self.tokenizer.decode([next_token])
                print(f"  Step {step + 1}/{max_new_tokens}: Token ID={next_token} ('{decoded_token}') | Step Time: {step_ms:.1f} ms", flush=True)

        output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return output_text, generated_ids, step_times

    def close(self) -> None:
        """Удаляет хуки и освобождает ресурсы GPU и CPU."""
        if getattr(self, "probability_warmup", None) is not None:
            self.probability_warmup.close()
            self.probability_warmup = None
        if getattr(self, "trace_adapter", None) is not None:
            self.trace_adapter.close()
            self.trace_adapter = None
        if getattr(self, "ack_adapter", None) is not None:
            self.ack_adapter.close()
            self.ack_adapter = None
        for handle in getattr(self, "_hook_handles", []):
            try:
                handle.remove()
            except Exception:
                pass
        if hasattr(self, "_hook_handles"):
            self._hook_handles.clear()
        for submod, orig_forward in getattr(self, "_q4_wrapped_modules", []):
            try:
                submod.forward = orig_forward
            except Exception:
                pass
        if hasattr(self, "_q4_wrapped_modules"):
            self._q4_wrapped_modules.clear()
        if hasattr(self, "active_slot_per_layer"):
            self.active_slot_per_layer.clear()
        if getattr(self, "pipeline", None) is not None:
            self.pipeline.close()
            self.pipeline = None
        if getattr(self, "weight_manager", None) is not None:
            self.weight_manager.close()
            self.weight_manager = None
        if getattr(self, "model_source", None) is not None and hasattr(self.model_source, "close"):
            self.model_source.close()
        self.embed_tokens_weight = None
        self._q4_scratch = None
        self.model = None
        gc.collect()
        if getattr(self, "device", torch.device("cpu")).type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> "AirColibriModel":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
