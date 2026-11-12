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
        except BaseException:
            # Initial embedding / ring preparation can already own resources,
            # so failures before model.forward also require close and reopen.
            self._failed = True
            raise
    return wrapped


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
    ):
        if pipeline_mode not in ("sequential", "double_buffer", "triple_ring", "triple_ring_prefetch"):
            raise ValueError("Unsupported pipeline mode")
        if control_mode not in ("off", "observe", "enforce"):
            raise ValueError("Unsupported control mode")
        if weight_format not in ("bf16", "q4", "q4_g16_asym", "q4_g32_asym", "q4_g64_asym"):
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
        self.weight_reader = None
        if airllm_source:
            from src.upstream_bridge import AirLLMReader
            self.weight_reader = AirLLMReader(airllm_source)
        self.model_dir = model_dir
        self.pipeline_mode = pipeline_mode
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.vram_resident_layers = vram_resident_layers
        self.cache_policy = cache_policy
        self.is_q4 = weight_format.startswith("q4")
        self.weight_format = f"q4_g{q4_group_size}_asym" if self.is_q4 else "bf16"
        self.quantize_lm_head = bool(quantize_lm_head and self.is_q4)
        self.lazy_embed_lookup = True if lazy_embed_lookup is None else bool(lazy_embed_lookup)
        self.q4_group_size = q4_group_size
        self._q4_scratch: Optional[torch.Tensor] = None
        self.embed_tokens_weight: Optional[torch.Tensor] = None

        # Определяем параметры кольца по режиму
        if num_slots is None:
            if pipeline_mode in ("triple_ring", "triple_ring_prefetch"):
                self.num_slots = 3
            elif pipeline_mode == "double_buffer":
                self.num_slots = 2
            else:
                self.num_slots = 1
        else:
            self.num_slots = num_slots

        self.tail_to_head_prefetch = (pipeline_mode == "triple_ring_prefetch" and self.num_slots >= 2)

        # 1. Загрузка конфигурации и токенизатора
        self.config = AutoConfig.from_pretrained(model_dir, config_file="config.hf.json")
        if self.config.model_type != "qwen3_5_text":
            raise ValueError("Dense adapter currently supports qwen3_5_text only")
        if not 0 <= vram_resident_layers <= self.config.num_hidden_layers:
            raise ValueError("Resident layer count outside model")
        # Preserve the checkpoint tokenizer. The installed Transformers detector
        # incorrectly treats local configs without transformers_version as Mistral.
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, fix_mistral_regex=False, local_files_only=True)

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
        )

        # 3. Создание модели на meta-устройстве (0 байт VRAM!)
        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(self.config)
        self.model.eval()
        if self.is_q4:
            replace_linear_with_q4(
                self.model,
                quantize_lm_head=self.quantize_lm_head,
                scratch_provider=self._get_q4_scratch,
                group_size=self.q4_group_size,
                compute_dtype=self.dtype,
            )

        # 4. Постоянные веса и VRAM-резидентные слои
        self._setup_permanent_weights()
        self.weight_manager.globals.release_handle()

        # 5. Инициализация GPU Ring Pipeline на едином layout без union-раздувания
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

    @staticmethod
    def _collect_layer_bindings(layer_module: torch.nn.Module) -> Tuple[List[Tuple[ColibriQ4Linear, str, str, str]], List[str]]:
        q4_bindings: List[Tuple[ColibriQ4Linear, str, str, str]] = []
        for mod_name, submod in layer_module.named_modules():
            if isinstance(submod, ColibriQ4Linear):
                q_k, s_k, z_k = q4_subkey_names(f"{mod_name}.weight")
                q4_bindings.append((submod, q_k, s_k, z_k))
        raw_params = [k for k, _ in layer_module.named_parameters()]
        return q4_bindings, raw_params

    def _load_resident_layer(self, layer_idx: int) -> None:
        """Загружает VRAM-резидентный слой навсегда на GPU (в bf16 или packed Q4)."""
        layer_module = self.model.model.layers[layer_idx]
        if self.is_q4:
            tensors = self.weight_manager.read_layer_tensors(layer_idx, target_device=self.device)
            q4_bindings, raw_params = self._collect_layer_bindings(layer_module)
            for submod, q_k, s_k, z_k in q4_bindings:
                submod.bind_q4_tensors(tensors[q_k], tensors[s_k], tensors[z_k])
            for k in raw_params:
                if k in tensors:
                    set_module_tensor_to_device(layer_module, k, self.device, value=tensors[k], clear_cache=False)
            return

        layer_file = os.path.join(self.model_dir, f"model-{layer_idx:05d}.safetensors")
        prefix = f"model.layers.{layer_idx}."
        with safetensors.safe_open(layer_file, framework="pt", device="cpu") as f:
            for k in f.keys():
                short_k = k.replace(prefix, "")
                t = f.get_slice(k)[:].to(device=self.device, dtype=self.dtype)
                set_module_tensor_to_device(layer_module, short_k, self.device, value=t, clear_cache=False)

    def _quantize_lm_head_from_slice(self) -> None:
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

        self.model.lm_head.bind_q4_tensors(q_out, s_out, z_out)
        del self.weight_manager.globals["lm_head.weight"]

    def _load_bf16_lm_head_from_slice(self) -> None:
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
        set_module_tensor_to_device(self.model, "lm_head.weight", self.device, value=lm_head_w, clear_cache=False)
        del lm_head_w
        del self.weight_manager.globals["lm_head.weight"]

    def _setup_permanent_weights(self) -> None:
        """Эмбеддинг (ленивый построчный mmap или CPU), norm и lm_head (bf16 или Q4) на GPU."""
        if not self.lazy_embed_lookup:
            # Полный embed на CPU только если отключен ленивый построчный lookup
            self.embed_tokens_weight = self.weight_manager.globals.pop("model.embed_tokens.weight")

        # Norm на GPU
        if "model.norm.weight" in self.weight_manager.globals:
            norm_w = self.weight_manager.globals.pop("model.norm.weight").to(device=self.device, dtype=self.dtype)
            set_module_tensor_to_device(self.model, "model.norm.weight", self.device, value=norm_w, clear_cache=False)
            del norm_w

        # LM Head на GPU
        if "lm_head.weight" in self.weight_manager.globals:
            if self.quantize_lm_head and isinstance(self.model.lm_head, ColibriQ4Linear):
                self._quantize_lm_head_from_slice()
            else:
                self._load_bf16_lm_head_from_slice()

        gc.collect()

    def _install_pipeline_hooks(self) -> None:
        """Устанавливает pre/post хуки кольцевого конвейера на каждый слой."""
        num_layers = self.config.num_hidden_layers
        self.active_slot_per_layer = [None] * num_layers
        self._hook_handles: List[Any] = []

        # Собираем привязки Q4-подмодулей и неквантуемых параметров для каждого слоя
        layer_q4_bindings: List[List[Tuple[ColibriQ4Linear, str, str, str]]] = []
        layer_param_names: List[List[str]] = []
        for layer_mod in self.model.model.layers:
            q4_b, raw_p = self._collect_layer_bindings(layer_mod)
            layer_q4_bindings.append(q4_b)
            layer_param_names.append(raw_p)

        for idx, layer_module in enumerate(self.model.model.layers):
            layer_module._air_layer_idx = idx

            def _pre_hook(module, args):
                i = module._air_layer_idx
                # Слой резидентен в VRAM?
                if i < self.vram_resident_layers:
                    return

                if self.num_slots == 1 and self.pipeline_mode == "sequential":
                    pinned = self.weight_manager.load_layer_to_pinned_slot(i, target_slot_idx=0)
                    target = self.pipeline.get_slot_for_loading(i)
                    self.pipeline.load_tensors_async(target, pinned, pinned_pool=self.weight_manager.pinned_pool)
                slot = self.pipeline.acquire_slot_for_compute(i)
                required = set(layer_param_names[i])
                for _, q, s, z in layer_q4_bindings[i]:
                    required.update((q, s, z))
                missing = required - slot.loaded_keys
                if missing:
                    raise ValueError(f"Layer {i} is missing required tensors: {sorted(missing)}")

                self.active_slot_per_layer[i] = slot
                # Zero-copy привязка указателей к буферам Q4 и параметрам слоя
                for submod, q_k, s_k, z_k in layer_q4_bindings[i]:
                    submod.bind_q4_tensors(slot.tensors[q_k], slot.tensors[s_k], slot.tensors[z_k])
                for k in layer_param_names[i]:
                    if k in slot.tensors:
                        set_module_tensor_to_device(module, k, self.device, value=slot.tensors[k], clear_cache=False)

            def _post_hook(module, args, output):
                i = module._air_layer_idx
                if i < self.vram_resident_layers:
                    return output

                slot = self.active_slot_per_layer[i]
                self.pipeline.release_slot_after_compute(slot)
                self.active_slot_per_layer[i] = None
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
            return embeds_cpu.to(device=self.device, dtype=self.dtype)
        return torch.nn.functional.embedding(step_tokens, self.embed_tokens_weight).to(
            device=self.device,
            dtype=self.dtype,
        )

    @_fail_runtime_on_error
    def _forward_step(self, step_tokens: torch.Tensor, past_key_values: Any = None,
                      prefetch_next: bool = True) -> Tuple[torch.Tensor, Any]:
        if self._failed or self.model is None:
            raise RuntimeError("Model is closed or failed; create a new runtime")
        self._prefetch_next_step = prefetch_next
        num_layers = self.config.num_hidden_layers
        inputs_embeds = self._embed_step_tokens(step_tokens)

        # Первоначальное заполнение кольца перед началом слоев
        if self.num_slots > 1 or self.pipeline_mode != "sequential":
            first_s = self.vram_resident_layers
            for s in range(first_s, min(first_s + self.num_slots, num_layers)):
                if s == first_s and self.tail_prefetched:
                    self.tail_prefetched = False
                    continue
                pinned_t = self.weight_manager.load_layer_to_pinned_slot(s, target_slot_idx=s % self.num_slots)
                slot = self.pipeline.get_slot_for_loading(s)
                self.pipeline.load_tensors_async(slot, pinned_t, pinned_pool=self.weight_manager.pinned_pool)

        try:
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                input_ids=None,
                past_key_values=past_key_values,
                use_cache=True,
            )
        except BaseException:
            self._failed = True
            raise
        return outputs.logits[:, -1, :], outputs.past_key_values

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
        if getattr(self, "trace_adapter", None) is not None:
            self.trace_adapter.close()
        if getattr(self, "ack_adapter", None) is not None:
            self.ack_adapter.close()
        for handle in getattr(self, "_hook_handles", []):
            try:
                handle.remove()
            except Exception:
                pass
        if hasattr(self, "_hook_handles"):
            self._hook_handles.clear()
        if hasattr(self, "active_slot_per_layer"):
            self.active_slot_per_layer.clear()
        if getattr(self, "pipeline", None) is not None:
            self.pipeline.close()
            self.pipeline = None
        if getattr(self, "weight_manager", None) is not None:
            self.weight_manager.close()
            self.weight_manager = None
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
