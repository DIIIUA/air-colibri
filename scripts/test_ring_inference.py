"""Test script for Ring Pipeline Inference on DeepSeek-V4-9B."""

import os
import sys
import time
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from src.bounded_memory_manager import StableWeightManager
from src.gpu_ring_pipeline import GPURingPipeline, SlotState

MODEL_DIR = r"C:\air-colibri\models\deepseek-v4-9b-colibri"


def test_ring_inference(num_slots=3, tail_to_head_prefetch=True):
    print("=" * 80)
    print(f"Testing Ring Pipeline Inference: {num_slots} slots | Prefetch={tail_to_head_prefetch}")
    print("=" * 80)

    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    # 1. Config & Tokenizer
    config = AutoConfig.from_pretrained(MODEL_DIR, config_file="config.hf.json")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    # 2. Stable Weight Manager (Bounded RAM + Pinned Pool)
    print("[1/4] Initializing StableWeightManager...")
    manager = StableWeightManager(
        model_dir=MODEL_DIR,
        max_ram_layers=8,
        num_pinned_slots=num_slots,
        dtype=dtype
    )

    # 3. Model on meta device
    print("[2/4] Initializing model on meta device...")
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    model.eval()

    # Permanent globals: norm and lm_head on GPU, embed on CPU
    norm_w = manager.globals["model.norm.weight"].to(device=device, dtype=dtype)
    set_module_tensor_to_device(model, "model.norm.weight", device, value=norm_w, clear_cache=False)

    lm_head_w = manager.globals["lm_head.weight"].to(device=device, dtype=dtype)
    set_module_tensor_to_device(model, "lm_head.weight", device, value=lm_head_w, clear_cache=False)

    embed_w_cpu = manager.globals["model.embed_tokens.weight"]  # Stays on CPU

    # 4. GPU Ring Pipeline
    print(f"[3/4] Initializing GPURingPipeline ({num_slots} slots)...")
    # Union template from manager.pinned_pool.slots[0]
    template = {k: v for k, v in manager.pinned_pool.slots[0].items()}
    pipeline = GPURingPipeline(
        template_tensors=template,
        num_slots=num_slots,
        device=device,
        dtype=dtype,
        tail_to_head_prefetch=tail_to_head_prefetch
    )

    num_layers = config.num_hidden_layers
    pipeline.tail_prefetched = False

    # Store param names for each layer
    layer_param_names = []
    for l_idx, layer_mod in enumerate(model.model.layers):
        names = [k for k, _ in layer_mod.named_parameters()]
        layer_param_names.append(names)

    # Install forward pre/post hooks
    active_slot_per_layer = [None] * num_layers

    for idx, layer_mod in enumerate(model.model.layers):
        layer_mod._air_idx = idx

        def _pre(m, a):
            i = m._air_idx
            slot = pipeline.acquire_slot_for_compute(i)
            active_slot_per_layer[i] = slot
            # Zero-copy pointer binding with clear_cache=False
            for k in layer_param_names[i]:
                if k in slot.tensors:
                    set_module_tensor_to_device(m, k, device, value=slot.tensors[k], clear_cache=False)

        def _post(m, a, o):
            i = m._air_idx
            slot = active_slot_per_layer[i]
            pipeline.release_slot_after_compute(slot)

            # Schedule next layer for this slot
            next_layer = i + num_slots
            if next_layer < num_layers:
                pinned_t = manager.load_layer_to_pinned_slot(next_layer, target_slot_idx=next_layer % num_slots)
                next_slot = pipeline.get_slot_for_loading(next_layer)
                pipeline.load_tensors_async(next_slot, pinned_t)
            elif next_layer == num_layers and pipeline.tail_to_head_prefetch:
                pinned_t = manager.load_layer_to_pinned_slot(0, target_slot_idx=0)
                pipeline.prefetch_tail_to_head(pinned_t)
                pipeline.tail_prefetched = True
            return o

        layer_mod.register_forward_pre_hook(_pre)
        layer_mod.register_forward_hook(_post)

    print("[4/4] Hooks installed! Ready for generation.")

    # 5. Generation
    prompt = "The capital of France is"
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    generated_ids = input_ids[0].tolist()
    max_new_tokens = 5

    print(f"\nStarting inference for prompt: '{prompt}' ({len(generated_ids)} input tokens)...")
    past_key_values = None

    with torch.no_grad():
        for step in range(max_new_tokens):
            t0 = time.perf_counter()

            # Lookup embedding on CPU, send only 8 KB to GPU
            if past_key_values is None:
                step_tokens = torch.tensor([generated_ids], dtype=torch.long)
            else:
                step_tokens = torch.tensor([[generated_ids[-1]]], dtype=torch.long)

            inputs_embeds = torch.nn.functional.embedding(step_tokens, embed_w_cpu).to(device=device, dtype=dtype)

            # Prime ring slots for start of forward pass
            for s in range(min(num_slots, num_layers)):
                if s == 0 and pipeline.tail_prefetched:
                    pipeline.tail_prefetched = False
                    continue
                pinned_t = manager.load_layer_to_pinned_slot(s, target_slot_idx=s)
                slot = pipeline.get_slot_for_loading(s)
                pipeline.load_tensors_async(slot, pinned_t)

            outputs = model(
                inputs_embeds=inputs_embeds,
                input_ids=None,
                past_key_values=past_key_values,
                use_cache=True
            )

            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

            if torch.isnan(logits).any():
                raise ValueError(f"NaN in logits at step {step}!")

            next_tok = torch.argmax(logits, dim=-1).item()
            generated_ids.append(next_tok)

            torch.cuda.synchronize()
            step_ms = (time.perf_counter() - t0) * 1000.0

            tok_str = tokenizer.decode([next_tok])
            print(f"  Step {step + 1}/{max_new_tokens}: ID={next_tok:6d} ('{tok_str}') | Latency: {step_ms:.1f} ms")

    result_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print("\n" + "=" * 80)
    print(f"Generated text: '{result_text}'")
    print(f"VRAM Allocated: {torch.cuda.memory_allocated(device) / (1024**2):.1f} MB")
    print(f"VRAM Reserved:  {torch.cuda.memory_reserved(device) / (1024**2):.1f} MB")
    print("=" * 80)
    return result_text


if __name__ == "__main__":
    test_ring_inference(num_slots=3, tail_to_head_prefetch=True)
