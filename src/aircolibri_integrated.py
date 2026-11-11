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


class AirColibriModel:
    """Интегрированная модель AirColibri с поддержкой GPU Ring Pipeline и Tail-to-Head Prefetch."""

    def __init__(
        self,
        model_dir: str,
        pipeline_mode: str = "triple_ring_prefetch",  # "sequential", "double_buffer", "triple_ring", "triple_ring_prefetch"
        vram_resident_layers: int = 0,
        max_ram_layers: int = 8,
        cache_policy: str = "static_pinned",
        num_slots: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16
    ):
        self.model_dir = model_dir
        self.pipeline_mode = pipeline_mode
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.vram_resident_layers = vram_resident_layers
        self.cache_policy = cache_policy

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
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # 2. Инициализация менеджера памяти
        self.weight_manager = StableWeightManager(
            model_dir=model_dir,
            max_ram_layers=max_ram_layers,
            num_pinned_slots=self.num_slots,
            cache_policy=cache_policy,
            dtype=self.dtype
        )

        # 3. Создание модели на meta-устройстве (0 байт VRAM!)
        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(self.config)
        self.model.eval()

        # 4. Постоянные веса и VRAM-резидентные слои
        self._setup_permanent_weights()

        # 5. Инициализация GPU Ring Pipeline
        template = {k: v for k, v in self.weight_manager.pinned_pool.slots[0].items()}
        self.pipeline = GPURingPipeline(
            template_tensors=template,
            num_slots=self.num_slots,
            device=self.device,
            dtype=self.dtype,
            tail_to_head_prefetch=self.tail_to_head_prefetch
        )
        del template
        self.tail_prefetched = False

        # 6. Загружаем VRAM-резидентные слои
        for r_idx in range(self.vram_resident_layers):
            self._load_resident_layer(r_idx)

        # 7. Установка стриминговых хуков на декодер-слои
        self._install_pipeline_hooks()

        # Статистика
        self.last_step_time_ms = 0.0

    def _load_resident_layer(self, layer_idx: int) -> None:
        """Загружает VRAM-резидентный слой навсегда на GPU."""
        layer_file = os.path.join(self.model_dir, f"model-{layer_idx:05d}.safetensors")
        prefix = f"model.layers.{layer_idx}."
        layer_module = self.model.model.layers[layer_idx]
        with safetensors.safe_open(layer_file, framework="pt", device="cpu") as f:
            for k in f.keys():
                short_k = k.replace(prefix, "")
                t = f.get_slice(k)[:].to(device=self.device, dtype=self.dtype)
                set_module_tensor_to_device(layer_module, short_k, self.device, value=t, clear_cache=False)

    def _setup_permanent_weights(self) -> None:
        """Эмбеддинг на CPU (экономия 2.03 ГБ VRAM), norm и lm_head на GPU."""
        # Embed держим на CPU: забираем pop(), чтобы не удерживать дубликат в globals
        self.embed_tokens_weight = self.weight_manager.globals.pop("model.embed_tokens.weight")

        # Norm на GPU
        if "model.norm.weight" in self.weight_manager.globals:
            norm_w = self.weight_manager.globals.pop("model.norm.weight").to(device=self.device, dtype=self.dtype)
            set_module_tensor_to_device(self.model, "model.norm.weight", self.device, value=norm_w, clear_cache=False)
            del norm_w

        # LM Head на GPU (удаляем 2.03 ГБ копию из globals для экономии RAM!)
        if "lm_head.weight" in self.weight_manager.globals:
            lm_head_w = self.weight_manager.globals.pop("lm_head.weight").to(device=self.device, dtype=self.dtype)
            set_module_tensor_to_device(self.model, "lm_head.weight", self.device, value=lm_head_w, clear_cache=False)
            del lm_head_w

        gc.collect()

    def _install_pipeline_hooks(self) -> None:
        """Устанавливает pre/post хуки кольцевого конвейера на каждый слой."""
        num_layers = self.config.num_hidden_layers
        self.active_slot_per_layer = [None] * num_layers
        self._hook_handles: List[Any] = []

        # Собираем список параметров для каждого слоя
        layer_param_names = []
        for l_idx, layer_mod in enumerate(self.model.model.layers):
            layer_param_names.append([k for k, _ in layer_mod.named_parameters()])

        for idx, layer_module in enumerate(self.model.model.layers):
            layer_module._air_layer_idx = idx

            def _pre_hook(module, args):
                i = module._air_layer_idx
                # Слой резидентен в VRAM?
                if i < self.vram_resident_layers:
                    return

                if self.num_slots == 1 and self.pipeline_mode == "sequential":
                    # Базовый последовательный режим
                    cached = self.weight_manager.ram_cache.get(i)
                    if cached is None:
                        layer_file = os.path.join(self.model_dir, f"model-{i:05d}.safetensors")
                        cached = {}
                        prefix = f"model.layers.{i}."
                        with safetensors.safe_open(layer_file, framework="pt", device="cpu") as f:
                            for k in f.keys():
                                short_k = k.replace(prefix, "")
                                cached[short_k] = f.get_slice(k)[:].to(dtype=self.dtype)
                        self.weight_manager.ram_cache.put(i, cached)
                    slot = self.pipeline.slots[0]
                    for k, t in cached.items():
                        slot.tensors[k].copy_(t, non_blocking=False)
                    for k in layer_param_names[i]:
                        if k in slot.tensors:
                            set_module_tensor_to_device(module, k, self.device, value=slot.tensors[k], clear_cache=False)
                    self.active_slot_per_layer[i] = slot
                else:
                    # Ring Pipeline: ожидаем готовности слота
                    slot = self.pipeline.acquire_slot_for_compute(i)
                    self.active_slot_per_layer[i] = slot
                    # Zero-copy привязка указателей
                    for k in layer_param_names[i]:
                        if k in slot.tensors:
                            set_module_tensor_to_device(module, k, self.device, value=slot.tensors[k], clear_cache=False)

            def _post_hook(module, args, output):
                i = module._air_layer_idx
                if i < self.vram_resident_layers:
                    return output

                slot = self.active_slot_per_layer[i]
                if self.num_slots > 1 or self.pipeline_mode != "sequential":
                    # Освобождаем слот после вычислений
                    self.pipeline.release_slot_after_compute(slot)

                    # Запускаем загрузку следующего слоя для этого слота
                    next_layer = i + self.num_slots
                    if next_layer < num_layers:
                        pinned_t = self.weight_manager.load_layer_to_pinned_slot(next_layer, target_slot_idx=next_layer % self.num_slots)
                        next_slot = self.pipeline.get_slot_for_loading(next_layer)
                        self.pipeline.load_tensors_async(next_slot, pinned_t, pinned_pool=self.weight_manager.pinned_pool)
                    elif self.tail_to_head_prefetch:
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
        num_layers = self.config.num_hidden_layers

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

                # Поиск эмбеддингов на CPU -> передача 8 КБ на GPU
                inputs_embeds = torch.nn.functional.embedding(step_tokens, self.embed_tokens_weight).to(
                    device=self.device,
                    dtype=self.dtype
                )

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

                # Истинный forward pass модели Transformers
                outputs = self.model(
                    inputs_embeds=inputs_embeds,
                    input_ids=None,
                    past_key_values=past_key_values,
                    use_cache=True
                )

                past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :]

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
        self.model = None
        gc.collect()
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> "AirColibriModel":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
