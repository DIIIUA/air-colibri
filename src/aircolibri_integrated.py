"""AirColibri Integrated Inference Engine:
Соединение математического графа HuggingFace / Transformers с ярусным менеджером памяти Colibri.

- Математический forward: оригинальный Qwen3_5 / Transformers (без упрощений!)
- Стриминг и кэширование: DynamicTieredCache + Zero Disk Re-read
- Поддержка KV-кэша и RoPE
- Генерация настоящих токенов и проверка логитов
"""

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


class AirColibriWeightProvider:
    """Менеджер весов: подгружает и удерживает веса слоев из model-XXXXX.safetensors."""

    def __init__(
        self,
        model_dir: str,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        vram_resident_layers: int = 4
    ):
        self.model_dir = model_dir
        self.device = device
        self.dtype = dtype
        self.vram_resident_layers_count = vram_resident_layers

        # Кэш весов:
        # vram_cache: {layer_idx: {param_name: tensor_on_cuda}}
        # ram_cache:  {layer_idx: {param_name: tensor_on_cpu}}
        self.vram_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self.ram_cache: Dict[int, Dict[str, torch.Tensor]] = {}

        # Глобальные веса (embed, norm, lm_head) держим в памяти
        self.globals: Dict[str, torch.Tensor] = {}
        self._load_globals()

    def _load_globals(self) -> None:
        globals_path = os.path.join(self.model_dir, "model-globals.safetensors")
        with safetensors.safe_open(globals_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                t = f.get_slice(k)[:]
                self.globals[k] = t

    def get_global_weights(self, module_name: str) -> Dict[str, torch.Tensor]:
        weights = {}
        if module_name == "embed":
            weights["weight"] = self.globals["model.embed_tokens.weight"]
        elif module_name == "norm":
            weights["weight"] = self.globals["model.norm.weight"]
        elif module_name == "lm_head":
            weights["weight"] = self.globals["lm_head.weight"]
        return weights

    def get_layer_weights(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        # 1. Если уже в VRAM
        if layer_idx in self.vram_cache:
            return self.vram_cache[layer_idx]

        # 2. Если в RAM
        if layer_idx in self.ram_cache:
            ram_tensors = self.ram_cache[layer_idx]
            # Если входит в пул VRAM-резидентности — переносим в VRAM навсегда
            if layer_idx < self.vram_resident_layers_count:
                vram_tensors = {k: v.to(device=self.device, dtype=self.dtype) for k, v in ram_tensors.items()}
                self.vram_cache[layer_idx] = vram_tensors
                return vram_tensors
            return ram_tensors

        # 3. Чтение с SSD (только в первый раз!)
        layer_file = os.path.join(self.model_dir, f"model-{layer_idx:05d}.safetensors")
        tensors = {}
        with safetensors.safe_open(layer_file, framework="pt", device="cpu") as f:
            prefix = f"model.layers.{layer_idx}."
            for k in f.keys():
                short_k = k.replace(prefix, "")
                tensors[short_k] = f.get_slice(k)[:]

        # Сохраняем в RAM-кэш для исключения повторного чтения с диска
        self.ram_cache[layer_idx] = tensors

        if layer_idx < self.vram_resident_layers_count:
            vram_tensors = {k: v.to(device=self.device, dtype=self.dtype) for k, v in tensors.items()}
            self.vram_cache[layer_idx] = vram_tensors
            return vram_tensors

        return tensors


class AirColibriModel:
    """Обертка над настоящей моделью HuggingFace с потоковой подачей весов AirColibri."""

    def __init__(
        self,
        model_dir: str,
        vram_resident_layers: int = 4,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16
    ):
        self.model_dir = model_dir
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.vram_resident_layers = vram_resident_layers

        # 1. Загрузка конфигурации
        self.config = AutoConfig.from_pretrained(model_dir, config_file="config.hf.json")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # 2. Создание модели на meta-устройстве (0 байт VRAM!)
        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(self.config)

        self.model.eval()

        # 3. Инициализация менеджера весов
        self.weight_provider = AirColibriWeightProvider(
            model_dir=model_dir,
            device=self.device,
            dtype=self.dtype,
            vram_resident_layers=vram_resident_layers
        )

        # 4. Предзагрузка глобальных весов на устройство
        self._load_permanent_globals()

        # 5. Установка стриминговых хуков на слои
        self._install_hooks()

        # Статистика
        self.last_step_time_ms = 0.0

    def _load_permanent_globals(self) -> None:
        """Эмбеддинг и финальную норму держим на GPU для быстрого прохода."""
        # Embed
        embed_w = self.weight_provider.get_global_weights("embed")["weight"]
        set_module_tensor_to_device(self.model, "model.embed_tokens.weight", self.device, value=embed_w.to(dtype=self.dtype))

        # Norm
        norm_w = self.weight_provider.get_global_weights("norm")["weight"]
        set_module_tensor_to_device(self.model, "model.norm.weight", self.device, value=norm_w.to(dtype=self.dtype))

        # LM Head (можно держать в RAM или VRAM в зависимости от доступной памяти)
        lm_head_w = self.weight_provider.get_global_weights("lm_head")["weight"]
        set_module_tensor_to_device(self.model, "lm_head.weight", self.device, value=lm_head_w.to(dtype=self.dtype))

    def _install_hooks(self) -> None:
        """Устанавливает forward pre/post хуки на каждый декодер-слой."""
        for idx, layer_module in enumerate(self.model.model.layers):
            layer_module._air_layer_idx = idx

            def _pre_hook(module, args):
                idx = module._air_layer_idx
                # Получаем веса из кэша (VRAM или RAM)
                weights = self.weight_provider.get_layer_weights(idx)
                # Переносим на GPU только если еще не там
                for param_name, tensor in weights.items():
                    target_device = tensor.device if tensor.device.type == "cuda" else self.device
                    set_module_tensor_to_device(module, param_name, target_device, value=tensor.to(device=self.device, dtype=self.dtype))

            def _post_hook(module, args, output):
                idx = module._air_layer_idx
                # Если слой не VRAM-резидентный — освобождаем слот на GPU
                if idx >= self.vram_resident_layers:
                    for name, _ in module.named_parameters():
                        set_module_tensor_to_device(module, name, "meta")
                return output

            layer_module.register_forward_pre_hook(_pre_hook)
            layer_module.register_forward_hook(_post_hook)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 10,
        temperature: float = 0.0
    ) -> Tuple[str, List[int], List[float]]:
        """Честная генерация токенов через сквозной вычислительный граф."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)

        generated_ids = input_ids[0].tolist()
        step_times = []

        print(f"\n[AirColibri Integrated] Starting genuine generation for prompt: '{prompt}'")
        print(f"[AirColibri Integrated] Input tokens: {len(generated_ids)} | Target new tokens: {max_new_tokens}")

        past_key_values = None

        with torch.no_grad():
            for step in range(max_new_tokens):
                t0 = time.perf_counter()

                # На первом шаге передаем весь prompt, затем по 1 токену с KV-кэшем
                if past_key_values is None:
                    curr_input = input_ids
                else:
                    curr_input = torch.tensor([[generated_ids[-1]]], device=self.device)

                # Истинный forward pass модели Transformers
                outputs = self.model(
                    input_ids=curr_input,
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
                step_ms = (time.perf_counter() - t0) * 1000.0
                step_times.append(step_ms)

                decoded_token = self.tokenizer.decode([next_token])
                print(f"  Step {step + 1}/{max_new_tokens}: Token ID={next_token} ('{decoded_token}') | Step Time: {step_ms:.1f} ms")

        output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return output_text, generated_ids, step_times
