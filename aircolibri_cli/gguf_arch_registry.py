"""Extended zero-mmap GGUF architecture, config, weight-conversion, and tokenizer registry.

``transformers 5.22.2`` natively ships zero-gguf-package GGUF converters for only three
architectures (``qwen35``, ``qwen35moe``, ``lfm2``). This module extends AirColibri's pure
positioned-read GGUF pipeline (zero ``mmap``, zero external ``gguf`` PyPI package, zero sidecar
files) across 35+ dense, MoE, and SSM/hybrid CausalLM families.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
from typing import Any

import torch
from transformers.core_model_loading import (
    ConversionOps,
    WeightConverter,
    WeightRenaming,
    WeightTransform,
)
from transformers.integrations.gguf.gguf_config_mapping import GGUF_CONFIG_ARCHS, get_gguf_config
from transformers.integrations.gguf.gguf_conversion_mapping import (
    DENSE_DECODER_RENAMINGS,
    GGUF_ARCHS,
    LogNegate,
    SubtractOne,
    TiledToGroupedInputs,
    TiledToGroupedRows,
    Unsqueeze,
)

_SPLIT_GGUF_RE = re.compile(r"^(?P<stem>.+)-(?P<idx>\d{5})-of-(?P<tot>\d{5})\.gguf$", re.IGNORECASE)


def _is_mmproj_gguf(path: Path) -> bool:
    low = path.name.lower()
    return low.startswith("mmproj") or "-mmproj" in low or "_mmproj" in low or ".mmproj" in low


def find_gguf_files(model_dir: str | Path, error_cls: type[Exception] = ValueError) -> list[Path]:
    """Resolves either a single ``.gguf`` file or an ordered split shard set (ignoring ``mmproj``)."""
    d = Path(model_dir)
    all_gguf = sorted(p for p in d.glob("*.gguf") if p.is_file())
    if not all_gguf:
        raise error_cls(f"{d}: no .gguf file found")

    non_mmproj = [p for p in all_gguf if not _is_mmproj_gguf(p)]
    if not non_mmproj:
        raise error_cls(f"{d}: only multimodal projector GGUF ({all_gguf[0].name}) found; missing text GGUF")

    if len(non_mmproj) == 1:
        return non_mmproj

    parsed: list[tuple[str, int, int, Path]] = []
    for p in non_mmproj:
        m = _SPLIT_GGUF_RE.match(p.name)
        if not m:
            names = ", ".join(x.name for x in non_mmproj[:4])
            raise error_cls(f"{d}: expected a single .gguf file or split shard set, found {len(non_mmproj)} ({names})")
        parsed.append((m.group("stem"), int(m.group("idx")), int(m.group("tot")), p))

    stems = {item[0] for item in parsed}
    totals = {item[2] for item in parsed}
    if len(stems) != 1 or len(totals) != 1:
        names = ", ".join(x.name for x in non_mmproj[:4])
        raise error_cls(f"{d}: multiple unrelated .gguf files found ({names})")

    expected_total = next(iter(totals))
    parsed.sort(key=lambda x: x[1])
    indices = [item[1] for item in parsed]
    if expected_total != len(parsed) or indices != list(range(1, expected_total + 1)):
        raise error_cls(
            f"{d}: incomplete split GGUF set (expected shards 1..{expected_total}, found {indices})"
        )
    return [item[3] for item in parsed]


# --------------------------------------------------------------------------------------
# Custom tensor ConversionOps (zero-gguf-package, pure torch)
# --------------------------------------------------------------------------------------


class LlamaRopeReversePermute(ConversionOps):
    """Reverses ``llama.cpp``'s interleaved-to-half-split RoPE row permutation on ``attn_q`` / ``attn_k``."""

    def __init__(self, num_heads: int):
        self.num_heads = max(1, int(num_heads))

    @torch.no_grad
    def convert(
        self, input_dict: dict[str, torch.Tensor], source_patterns: list[str], target_patterns: list[str], **kwargs
    ) -> dict[str, torch.Tensor]:
        tensor = next(iter(input_dict.values()))
        n_h = self.num_heads
        if tensor.ndim == 0 or tensor.shape[0] % (2 * n_h) != 0:
            return {target_patterns[0]: tensor}
        half_d = tensor.shape[0] // n_h // 2
        out = tensor.reshape(n_h, half_d, 2, *tensor.shape[1:]).transpose(1, 2).reshape(tensor.shape).contiguous()
        return {target_patterns[0]: out}


class Transpose2D(ConversionOps):
    """Transposes 2D weights for ``Conv1D`` architectures (such as ``gpt2``)."""

    @torch.no_grad
    def convert(
        self, input_dict: dict[str, torch.Tensor], source_patterns: list[str], target_patterns: list[str], **kwargs
    ) -> dict[str, torch.Tensor]:
        tensor = next(iter(input_dict.values()))
        if tensor.ndim == 2:
            return {target_patterns[0]: tensor.t().contiguous()}
        return {target_patterns[0]: tensor}


class BloomQKVPermute(ConversionOps):
    """Restores Hugging Face BLOOM head-interleaved ``query_key_value`` from GGUF stacked ``[Q; K; V]``."""

    def __init__(self, num_heads: int, hidden_size: int):
        self.num_heads = max(1, int(num_heads))
        self.hidden_size = int(hidden_size)

    @torch.no_grad
    def convert(
        self, input_dict: dict[str, torch.Tensor], source_patterns: list[str], target_patterns: list[str], **kwargs
    ) -> dict[str, torch.Tensor]:
        tensor = next(iter(input_dict.values()))
        n_h = self.num_heads
        head_dim = self.hidden_size // n_h
        q, k, v = tensor.chunk(3, dim=0)
        rest = tensor.shape[1:]
        q = q.reshape(n_h, head_dim, *rest)
        k = k.reshape(n_h, head_dim, *rest)
        v = v.reshape(n_h, head_dim, *rest)
        stacked = torch.stack([q, k, v], dim=1).reshape(tensor.shape).contiguous()
        return {target_patterns[0]: stacked}


# --------------------------------------------------------------------------------------
# Supported GGUF architecture table & config builders
# --------------------------------------------------------------------------------------

EXTENDED_GGUF_ARCH_ALIASES: dict[str, str] = {
    "llama": "llama",
    "mistral": "mistral",
    "mistral3": "mistral",
    "deci": "llama",
    "mixtral": "mixtral",
    "qwen2": "qwen2",
    "qwen2moe": "qwen2_moe",
    "qwen2_moe": "qwen2_moe",
    "qwen3": "qwen3",
    "qwen3moe": "qwen3_moe",
    "qwen3_moe": "qwen3_moe",
    "qwen3next": "qwen3_next",
    "qwen3_next": "qwen3_next",
    "gemma": "gemma",
    "gemma2": "gemma2",
    "gemma3": "gemma3_text",
    "gemma3_text": "gemma3_text",
    "gemma4": "gemma4_text",
    "gemma4_text": "gemma4_text",
    "phi2": "phi",
    "phi": "phi",
    "phi3": "phi3",
    "phimoe": "phimoe",
    "olmo": "olmo",
    "olmo2": "olmo2",
    "olmoe": "olmoe",
    "starcoder2": "starcoder2",
    "stablelm": "stablelm",
    "command-r": "cohere",
    "cohere": "cohere",
    "cohere2": "cohere2",
    "glm4": "glm4",
    "chatglm": "glm4",
    "glm4moe": "glm4_moe",
    "glm4_moe": "glm4_moe",
    "deepseek2": "deepseek_v3",
    "deepseek_v2": "deepseek_v2",
    "deepseek_v3": "deepseek_v3",
    "granite": "granite",
    "granitemoe": "granitemoe",
    "granitehybrid": "granitemoehybrid",
    "granitemoehybrid": "granitemoehybrid",
    "nemotron": "nemotron",
    "minimax-m2": "minimax_m2",
    "minimax_m2": "minimax_m2",
    "gpt_oss": "gpt_oss",
    "gpt-oss": "gpt_oss",
    "exaone4": "exaone4",
    "smollm3": "smollm3",
    "ernie4_5": "ernie4_5",
    "ernie4_5-moe": "ernie4_5_moe",
    "ernie4_5_moe": "ernie4_5_moe",
    "falcon": "falcon",
    "falcon-h1": "falcon_h1",
    "falcon_h1": "falcon_h1",
    "gpt2": "gpt2",
    "bloom": "bloom",
    "gptneox": "gpt_neox",
    "gpt_neox": "gpt_neox",
    "opt": "opt",
    "mamba": "mamba",
    "mamba2": "mamba2",
    "jamba": "jamba",
    "bamba": "bamba",
}


def is_extended_gguf_arch_supported(arch: str) -> bool:
    return bool(arch in GGUF_CONFIG_ARCHS or arch in EXTENDED_GGUF_ARCH_ALIASES)


def _meta_get(meta: dict[str, Any], prefix: str, suffix: str, default: Any = None) -> Any:
    val = meta.get(f"{prefix}.{suffix}")
    if val is not None:
        return val
    # Fallback for aliased prefixes (e.g. mistral checkpoint stored under llama.* or vice versa)
    for alt in ("llama", prefix.replace("-", "_"), prefix.replace("_", "-")):
        if alt != prefix and f"{alt}.{suffix}" in meta:
            return meta[f"{alt}.{suffix}"]
    return default


def _scalar_or_max(val: Any, default: Any = None) -> Any:
    if isinstance(val, (list, tuple)):
        non_zero = [int(x) for x in val if x is not None and int(x) > 0]
        return non_zero[0] if len(set(non_zero)) == 1 else (max(non_zero) if non_zero else default)
    return val if val is not None else default


def _sanitize_tok_id(tok_id: Any, vocab_size: int) -> int | None:
    if tok_id is None or isinstance(tok_id, (list, tuple)):
        return None
    tid = int(tok_id)
    return tid if 0 <= tid < vocab_size else None


def build_gguf_config_dict(
    meta: dict[str, Any],
    tensor_names: tuple[str, ...],
    tensor_shapes: dict[str, tuple[int, ...]] | None = None,
) -> dict[str, Any]:
    """Builds a ``transformers`` ``AutoConfig.for_model(**cfg_dict)`` dictionary from GGUF metadata."""
    arch = str(meta.get("general.architecture") or "")
    if arch in GGUF_CONFIG_ARCHS:
        cfg = dict(get_gguf_config(meta, tensor_names))
        vocab = int(cfg.get("vocab_size") or 0)
        if vocab > 0:
            for tk in ("pad_token_id", "bos_token_id", "eos_token_id"):
                cfg[tk] = _sanitize_tok_id(cfg.get(tk), vocab)
        return cfg

    if arch not in EXTENDED_GGUF_ARCH_ALIASES:
        raise ValueError(f"Unsupported GGUF architecture {arch!r}")

    p = arch
    hf_type = EXTENDED_GGUF_ARCH_ALIASES[arch]
    general_name = str(meta.get("general.name") or "").lower()

    expert_count = int(_scalar_or_max(_meta_get(meta, p, "expert_count"), 0) or 0)
    expert_used = int(_scalar_or_max(_meta_get(meta, p, "expert_used_count"), 0) or 0)
    sliding_window = _scalar_or_max(_meta_get(meta, p, "attention.sliding_window"))
    if sliding_window is None:
        sliding_window = _scalar_or_max(_meta_get(meta, p, "sliding_window"))

    if arch == "llama":
        has_moe_tensors = expert_count > 0 or any(
            ".ffn_gate_exps." in n or ".ffn_gate.0." in n for n in tensor_names
        )
        if has_moe_tensors:
            hf_type = "mixtral"
        elif "mistral" in general_name and sliding_window is not None:
            hf_type = "mistral"

    tok_field = meta.get("tokenizer.ggml.tokens")
    tok_len = len(tok_field) if isinstance(tok_field, (list, tuple)) else int(tok_field or 0)
    emb_rows = 0
    if tensor_shapes and "token_embd.weight" in tensor_shapes:
        emb_rows = int(tensor_shapes["token_embd.weight"][0])
    vocab_size = int(_meta_get(meta, p, "vocab_size") or emb_rows or tok_len or 32000)

    n_layers = int(_meta_get(meta, p, "block_count") or 0)
    mtp_layers = int(_meta_get(meta, p, "nextn_predict_layers") or 0)
    if mtp_layers > 0 and n_layers > mtp_layers:
        n_layers -= mtp_layers
    hidden_size = int(_meta_get(meta, p, "embedding_length") or 0)
    ctx_len = int(_meta_get(meta, p, "context_length") or 4096)
    ff_len = _scalar_or_max(_meta_get(meta, p, "feed_forward_length"))
    n_heads = _scalar_or_max(_meta_get(meta, p, "attention.head_count"))
    n_kv_heads = _scalar_or_max(_meta_get(meta, p, "attention.head_count_kv"), n_heads)
    key_len = _scalar_or_max(_meta_get(meta, p, "attention.key_length"))
    rms_eps = _meta_get(meta, p, "attention.layer_norm_rms_epsilon")
    ln_eps = _meta_get(meta, p, "attention.layer_norm_epsilon")
    rope_theta = _meta_get(meta, p, "rope.freq_base")
    rope_dim = _scalar_or_max(_meta_get(meta, p, "rope.dimension_count"))

    tie_emb = "output.weight" not in tensor_names
    eos_id = _sanitize_tok_id(meta.get("tokenizer.ggml.eos_token_id"), vocab_size)
    bos_id = _sanitize_tok_id(meta.get("tokenizer.ggml.bos_token_id"), vocab_size)
    pad_id = _sanitize_tok_id(meta.get("tokenizer.ggml.padding_token_id"), vocab_size)

    # Architecture-specific non-standard config schemas first
    if hf_type == "gpt2":
        cfg: dict[str, Any] = {
            "model_type": "gpt2",
            "vocab_size": vocab_size,
            "n_positions": ctx_len,
            "n_ctx": ctx_len,
            "n_embd": hidden_size,
            "n_layer": n_layers,
            "n_head": int(n_heads or 12),
            "layer_norm_epsilon": float(ln_eps or 1e-5),
            "tie_word_embeddings": tie_emb,
            "bos_token_id": bos_id,
            "eos_token_id": eos_id,
            "pad_token_id": pad_id,
        }
        if ff_len is not None:
            cfg["n_inner"] = int(ff_len)
        return cfg

    if hf_type == "bloom":
        return {
            "model_type": "bloom",
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "n_layer": n_layers,
            "n_head": int(n_heads or 16),
            "layer_norm_epsilon": float(ln_eps or 1e-5),
            "tie_word_embeddings": tie_emb,
            "bos_token_id": bos_id,
            "eos_token_id": eos_id,
            "pad_token_id": pad_id,
        }

    if hf_type == "mamba":
        inner = int(_meta_get(meta, p, "ssm.inner_size") or (hidden_size * 2))
        state_sz = int(_meta_get(meta, p, "ssm.state_size") or 16)
        conv_k = int(_meta_get(meta, p, "ssm.conv_kernel") or 4)
        dt_rank = int(_meta_get(meta, p, "ssm.time_step_rank") or max(1, (hidden_size + 15) // 16))
        return {
            "model_type": "mamba",
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "num_hidden_layers": n_layers,
            "state_size": state_sz,
            "conv_kernel": conv_k,
            "intermediate_size": inner,
            "expand": max(1, inner // max(1, hidden_size)),
            "time_step_rank": dt_rank,
            "layer_norm_epsilon": float(rms_eps or ln_eps or 1e-5),
            "use_bias": any(n.endswith(".bias") and not n.endswith("conv1d.bias") and not n.endswith("ssm_dt.bias") for n in tensor_names),
            "use_conv_bias": any(n.endswith("ssm_conv1d.bias") for n in tensor_names),
            "tie_word_embeddings": tie_emb,
            "bos_token_id": bos_id,
            "eos_token_id": eos_id,
            "pad_token_id": pad_id,
        }

    if hf_type == "mamba2":
        inner = int(_meta_get(meta, p, "ssm.inner_size") or (hidden_size * 2))
        state_sz = int(_meta_get(meta, p, "ssm.state_size") or 128)
        conv_k = int(_meta_get(meta, p, "ssm.conv_kernel") or 4)
        ssm_heads = int(_meta_get(meta, p, "ssm.time_step_rank") or max(1, inner // 64))
        ssm_groups = int(_meta_get(meta, p, "ssm.group_count") or 1)
        return {
            "model_type": "mamba2",
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "num_hidden_layers": n_layers,
            "state_size": state_sz,
            "conv_kernel": conv_k,
            "num_heads": ssm_heads,
            "head_dim": max(1, inner // max(1, ssm_heads)),
            "n_groups": ssm_groups,
            "expand": max(1, inner // max(1, hidden_size)),
            "rms_norm_eps": float(rms_eps or 1e-5),
            "tie_word_embeddings": tie_emb,
            "bos_token_id": bos_id,
            "eos_token_id": eos_id,
            "pad_token_id": pad_id,
        }

    cfg = {
        "model_type": hf_type,
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "num_hidden_layers": n_layers,
        "max_position_embeddings": ctx_len,
        "tie_word_embeddings": tie_emb,
        "bos_token_id": bos_id,
        "eos_token_id": eos_id,
        "pad_token_id": pad_id,
    }
    if n_heads is not None:
        cfg["num_attention_heads"] = int(n_heads)
    if n_kv_heads is not None and hf_type not in ("opt", "gpt_neox"):
        cfg["num_key_value_heads"] = int(n_kv_heads)
    if ff_len is not None:
        if hf_type == "opt":
            cfg["ffn_dim"] = int(ff_len)
        else:
            cfg["intermediate_size"] = int(ff_len)

    # Epsilon field names vary across transformers configs
    if rms_eps is not None:
        if hf_type in ("nemotron", "lfm2"):
            cfg["norm_eps"] = float(rms_eps)
        else:
            cfg["rms_norm_eps"] = float(rms_eps)
    elif ln_eps is not None:
        if hf_type in ("stablelm", "phi", "gpt_neox"):
            cfg["layer_norm_eps"] = float(ln_eps)
        elif hf_type == "starcoder2":
            cfg["norm_epsilon"] = float(ln_eps)
        elif hf_type == "falcon":
            cfg["layer_norm_epsilon"] = float(ln_eps)
        else:
            cfg["rms_norm_eps"] = float(ln_eps)

    if rope_theta is not None and hf_type != "gemma4_text":
        cfg["rope_theta"] = float(rope_theta)

    # Head dimension & partial rotary handling
    default_hd = hidden_size // max(1, int(n_heads or 1)) if hidden_size and n_heads else None
    if hf_type == "gemma4_text":
        if key_len is not None:
            cfg["global_head_dim"] = int(key_len)
        swa_kl = _scalar_or_max(_meta_get(meta, p, "attention.key_length_swa"))
        if swa_kl is not None:
            cfg["head_dim"] = int(swa_kl)
        elif default_hd is not None and (key_len is None or int(key_len) <= default_hd):
            cfg["head_dim"] = int(key_len or default_hd)
    elif key_len is not None and hf_type in (
        "gemma", "gemma2", "gemma3_text", "qwen3", "qwen3_moe", "qwen3_next",
        "glm4", "glm4_moe", "minimax_m2", "gpt_oss", "olmo2", "exaone4", "smollm3",
    ):
        cfg["head_dim"] = int(key_len)
    elif rope_dim is not None and hf_type in ("llama", "mistral", "mixtral") and default_hd and int(rope_dim) == default_hd:
        cfg["head_dim"] = int(rope_dim)
    elif key_len is not None and default_hd and int(key_len) != default_hd and hf_type in ("llama", "mistral", "cohere", "cohere2"):
        cfg["head_dim"] = int(key_len)

    eff_hd = int(cfg.get("head_dim") or default_hd or 0)
    if rope_dim is not None and eff_hd > 0 and 0 < int(rope_dim) < eff_hd:
        if hf_type == "minimax_m2":
            cfg["rotary_dim"] = int(rope_dim)
        elif hf_type in ("phi", "phi3", "stablelm", "gpt_neox", "nemotron", "glm4", "glm4_moe"):
            cfg["partial_rotary_factor"] = float(rope_dim) / float(eff_hd)

    if sliding_window is not None and hf_type in (
        "mistral", "mixtral", "gemma2", "gemma3_text", "gemma4_text", "phi3", "phimoe", "qwen2", "cohere2", "gpt_oss"
    ):
        cfg["sliding_window"] = int(sliding_window)

    # Bias presence inference directly from GGUF tensor names
    has_q_bias = any(n.endswith(".attn_q.bias") or n.endswith(".attn_qkv.bias") for n in tensor_names)
    has_o_bias = any(n.endswith(".attn_output.bias") for n in tensor_names)
    has_mlp_bias = any(n.endswith(".ffn_down.bias") or n.endswith(".ffn_up.bias") for n in tensor_names)
    if hf_type in ("llama", "qwen3", "qwen3_moe", "gemma", "olmo", "granite", "granitemoe", "smollm3"):
        cfg["attention_bias"] = bool(has_q_bias and has_o_bias)
        if hf_type in ("llama", "granite", "granitemoe", "smollm3"):
            cfg["mlp_bias"] = bool(has_mlp_bias)
    elif hf_type == "qwen2":
        # Qwen2Config always uses QKV bias by default; nothing extra needed
        pass
    elif hf_type in ("phi", "starcoder2", "stablelm"):
        cfg["use_bias"] = bool(has_q_bias or has_mlp_bias)
        if hf_type == "stablelm":
            cfg["use_qkv_bias"] = bool(has_q_bias)
    elif hf_type == "falcon":
        cfg["bias"] = bool(has_q_bias or has_mlp_bias)
        cfg["new_decoder_architecture"] = bool(n_kv_heads is not None and int(n_kv_heads) > 1 and any(".attn_norm_2." in n for n in tensor_names))
        if not cfg["new_decoder_architecture"] and n_kv_heads is not None and int(n_kv_heads) == 1:
            cfg["multi_query"] = True

    # MoE family specifics
    exp_ff = _scalar_or_max(_meta_get(meta, p, "expert_feed_forward_length"))
    shared_ff = _scalar_or_max(_meta_get(meta, p, "expert_shared_feed_forward_length"))
    shared_cnt = int(_scalar_or_max(_meta_get(meta, p, "expert_shared_count"), 0) or 0)

    if hf_type in ("mixtral", "phimoe", "minimax_m2", "gpt_oss", "granitemoe", "granitemoehybrid"):
        if expert_count > 0:
            cfg["num_local_experts"] = expert_count
        if expert_used > 0:
            cfg["num_experts_per_tok"] = expert_used
        if exp_ff is not None and "intermediate_size" not in cfg:
            cfg["intermediate_size"] = int(exp_ff)
        if hf_type == "minimax_m2":
            cfg["use_routing_bias"] = any(".exp_probs_b." in n for n in tensor_names) or True
            gating = _meta_get(meta, p, "expert_gating_func")
            if gating == 2 or gating == "sigmoid":
                cfg["scoring_func"] = "sigmoid"
            elif gating == 1 or gating == "softmax":
                cfg["scoring_func"] = "softmax"
    elif hf_type in ("olmoe", "qwen2_moe", "qwen3_moe", "qwen3_next", "jamba"):
        if expert_count > 0:
            cfg["num_experts"] = expert_count
        if expert_used > 0:
            cfg["num_experts_per_tok"] = expert_used
        if exp_ff is not None and hf_type != "jamba":
            cfg["moe_intermediate_size"] = int(exp_ff)
        if shared_ff is not None:
            cfg["shared_expert_intermediate_size"] = int(shared_ff)
        if hf_type == "qwen3_moe":
            cfg["norm_topk_prob"] = True
    elif hf_type in ("deepseek_v2", "deepseek_v3", "glm4_moe"):
        n_dense = int(_meta_get(meta, p, "leading_dense_block_count") or 0)
        cfg["first_k_dense_replace"] = n_dense
        if expert_count > 0:
            cfg["n_routed_experts"] = expert_count
        if expert_used > 0:
            cfg["num_experts_per_tok"] = expert_used
        if exp_ff is not None:
            cfg["moe_intermediate_size"] = int(exp_ff)
        if shared_cnt > 0:
            cfg["n_shared_experts"] = shared_cnt
        if hf_type in ("deepseek_v2", "deepseek_v3"):
            q_lora = _meta_get(meta, p, "attention.q_lora_rank")
            kv_lora = _meta_get(meta, p, "attention.kv_lora_rank")
            if q_lora is not None:
                cfg["q_lora_rank"] = int(q_lora)
            if kv_lora is not None:
                cfg["kv_lora_rank"] = int(kv_lora)
    elif hf_type == "ernie4_5_moe":
        if expert_count > 0:
            cfg["moe_num_experts"] = expert_count
        if expert_used > 0:
            cfg["moe_k"] = expert_used
        if exp_ff is not None:
            cfg["moe_intermediate_size"] = int(exp_ff)

    if hf_type == "qwen3_next":
        val_heads = int(_meta_get(meta, p, "ssm.time_step_rank") or 32)
        cfg["full_attention_interval"] = int(_meta_get(meta, p, "full_attention_interval") or 4)
        cfg["linear_conv_kernel_dim"] = int(_meta_get(meta, p, "ssm.conv_kernel") or 4)
        cfg["linear_key_head_dim"] = int(_meta_get(meta, p, "ssm.state_size") or 128)
        cfg["linear_num_key_heads"] = int(_meta_get(meta, p, "ssm.group_count") or 16)
        cfg["linear_num_value_heads"] = val_heads
        ssm_in = int(_meta_get(meta, p, "ssm.inner_size") or (val_heads * 128))
        cfg["linear_value_head_dim"] = max(1, ssm_in // max(1, val_heads))

    # Granite scaling multipliers if present
    for gk, tk in (
        ("embedding_scale", "embedding_multiplier"),
        ("residual_scale", "residual_multiplier"),
        ("attention.scale", "attention_multiplier"),
        ("logit_scale", "logits_scaling"),
    ):
        gv = _meta_get(meta, p, gk)
        if gv is not None and hf_type in ("granite", "granitemoe", "granitemoehybrid"):
            cfg[tk] = float(gv)

    return cfg


# --------------------------------------------------------------------------------------
# GGUF -> Canonical Model Layer / Global Weight Mappings
# --------------------------------------------------------------------------------------


def _std_norms() -> list[WeightTransform]:
    return [
        WeightRenaming(r"^output_norm\.", "model.norm."),
        WeightRenaming(r"\.attn_norm\.", ".input_layernorm."),
        WeightRenaming(r"\.ffn_norm\.", ".post_attention_layernorm."),
        WeightRenaming(r"\.attn_q_norm\.", ".self_attn.q_norm."),
        WeightRenaming(r"\.attn_k_norm\.", ".self_attn.k_norm."),
    ]


def _moe_common_renamings(router_target: str = ".mlp.gate.", shared_plural: bool = False) -> list[WeightTransform]:
    sh = "shared_experts" if shared_plural else "shared_expert"
    return [
        WeightRenaming(r"\.ffn_gate_inp\.", router_target),
        WeightRenaming(r"\.ffn_gate_exps\.weight", ".mlp.experts.gate_proj"),
        WeightRenaming(r"\.ffn_up_exps\.weight", ".mlp.experts.up_proj"),
        WeightRenaming(r"\.ffn_down_exps\.weight", ".mlp.experts.down_proj"),
        WeightRenaming(r"\.ffn_gate_projs\.", ".mlp.experts.gate_proj."),
        WeightRenaming(r"\.ffn_up_projs\.", ".mlp.experts.up_proj."),
        WeightRenaming(r"\.ffn_down_projs\.", ".mlp.experts.down_proj."),
        WeightRenaming(r"\.ffn_gate_up_projs\.", ".mlp.experts.gate_up_proj."),
        WeightRenaming(r"\.ffn_gate_inp_shexp\.", ".mlp.shared_expert_gate."),
        WeightRenaming(r"\.ffn_gate_shexp\.", f".mlp.{sh}.gate_proj."),
        WeightRenaming(r"\.ffn_up_shexp\.", f".mlp.{sh}.up_proj."),
        WeightRenaming(r"\.ffn_down_shexp\.", f".mlp.{sh}.down_proj."),
    ]


def get_extended_gguf_mapping(
    arch: str,
    config: Any,
    tensor_names: tuple[str, ...] = (),
) -> list[WeightTransform]:
    """Returns ``WeightRenaming`` + ``WeightConverter`` rules mapping GGUF names to canonical names."""
    if arch in GGUF_ARCHS:
        base = list(deepcopy(GGUF_ARCHS[arch](config)))
        if arch == "lfm2":
            base.insert(0, WeightRenaming(r"^output_norm\.", "model.embedding_norm."))
        return base

    text_cfg = config.get_text_config() if hasattr(config, "get_text_config") else config
    hf_type = str(getattr(text_cfg, "model_type", "") or EXTENDED_GGUF_ARCH_ALIASES.get(arch, arch))

    # 1. GPT-2 (Conv1D transposed 2D weights + position_embd)
    if hf_type == "gpt2":
        renamings: list[WeightTransform] = [
            WeightRenaming(r"^token_embd\.", "model.embed_tokens."),
            WeightRenaming(r"^position_embd\.", "transformer.wpe."),
            WeightRenaming(r"^output_norm\.", "model.norm."),
            WeightRenaming(r"^output\.", "lm_head."),
            WeightRenaming(r"^blk\.", "model.layers."),
            WeightRenaming(r"\.attn_norm\.", ".ln_1."),
            WeightRenaming(r"\.ffn_norm\.", ".ln_2."),
            WeightRenaming(r"\.attn_qkv\.", ".attn.c_attn."),
            WeightRenaming(r"\.attn_output\.", ".attn.c_proj."),
            WeightRenaming(r"\.ffn_up\.", ".mlp.c_fc."),
            WeightRenaming(r"\.ffn_down\.", ".mlp.c_proj."),
        ]
        convs: list[WeightTransform] = [
            WeightConverter(r"\.attn\.c_attn\.weight$", ".attn.c_attn.weight", [Transpose2D()]),
            WeightConverter(r"\.attn\.c_proj\.weight$", ".attn.c_proj.weight", [Transpose2D()]),
            WeightConverter(r"\.mlp\.c_fc\.weight$", ".mlp.c_fc.weight", [Transpose2D()]),
            WeightConverter(r"\.mlp\.c_proj\.weight$", ".mlp.c_proj.weight", [Transpose2D()]),
        ]
        return renamings + convs

    # 2. BLOOM
    if hf_type == "bloom":
        n_h = int(getattr(text_cfg, "n_head", None) or getattr(text_cfg, "num_attention_heads", 16))
        h_sz = int(getattr(text_cfg, "hidden_size", 1024))
        return [
            WeightRenaming(r"^token_embd\.", "model.embed_tokens."),
            WeightRenaming(r"^token_embd_norm\.", "transformer.word_embeddings_layernorm."),
            WeightRenaming(r"^output_norm\.", "model.norm."),
            WeightRenaming(r"^output\.", "lm_head."),
            WeightRenaming(r"^blk\.", "model.layers."),
            WeightRenaming(r"\.attn_norm\.", ".input_layernorm."),
            WeightRenaming(r"\.ffn_norm\.", ".post_attention_layernorm."),
            WeightRenaming(r"\.attn_qkv\.", ".self_attention.query_key_value."),
            WeightRenaming(r"\.attn_output\.", ".self_attention.dense."),
            WeightRenaming(r"\.ffn_up\.", ".mlp.dense_h_to_4h."),
            WeightRenaming(r"\.ffn_down\.", ".mlp.dense_4h_to_h."),
            WeightConverter(
                r"\.self_attention\.query_key_value\.weight$",
                ".self_attention.query_key_value.weight",
                [BloomQKVPermute(n_h, h_sz)],
            ),
            WeightConverter(
                r"\.self_attention\.query_key_value\.bias$",
                ".self_attention.query_key_value.bias",
                [BloomQKVPermute(n_h, h_sz)],
            ),
        ]

    # 3. GPT-NeoX & Falcon
    if hf_type == "gpt_neox":
        return [
            WeightRenaming(r"^token_embd\.", "model.embed_tokens."),
            WeightRenaming(r"^output_norm\.", "model.norm."),
            WeightRenaming(r"^output\.", "lm_head."),
            WeightRenaming(r"^blk\.", "model.layers."),
            WeightRenaming(r"\.attn_norm\.", ".input_layernorm."),
            WeightRenaming(r"\.ffn_norm\.", ".post_attention_layernorm."),
            WeightRenaming(r"\.attn_qkv\.", ".attention.query_key_value."),
            WeightRenaming(r"\.attn_output\.", ".attention.dense."),
            WeightRenaming(r"\.ffn_up\.", ".mlp.dense_h_to_4h."),
            WeightRenaming(r"\.ffn_down\.", ".mlp.dense_4h_to_h."),
        ]

    if hf_type == "falcon":
        return [
            WeightRenaming(r"^token_embd\.", "model.embed_tokens."),
            WeightRenaming(r"^output_norm\.", "model.norm."),
            WeightRenaming(r"^output\.", "lm_head."),
            WeightRenaming(r"^blk\.", "model.layers."),
            WeightRenaming(r"\.attn_norm\.", ".input_layernorm."),
            WeightRenaming(r"\.attn_norm_2\.", ".ln_mlp."),
            WeightRenaming(r"\.ffn_norm\.", ".post_attention_layernorm."),
            WeightRenaming(r"\.attn_qkv\.", ".self_attention.query_key_value."),
            WeightRenaming(r"\.attn_output\.", ".self_attention.dense."),
            WeightRenaming(r"\.ffn_up\.", ".mlp.dense_h_to_4h."),
            WeightRenaming(r"\.ffn_down\.", ".mlp.dense_4h_to_h."),
        ]

    # 4. OPT
    if hf_type == "opt":
        return [
            WeightRenaming(r"^token_embd\.", "model.embed_tokens."),
            WeightRenaming(r"^position_embd\.", "model.decoder.embed_positions."),
            WeightRenaming(r"^output_norm\.", "model.norm."),
            WeightRenaming(r"^output\.", "lm_head."),
            WeightRenaming(r"^blk\.", "model.layers."),
            WeightRenaming(r"\.attn_norm\.", ".self_attn_layer_norm."),
            WeightRenaming(r"\.ffn_norm\.", ".final_layer_norm."),
            WeightRenaming(r"\.attn_q\.", ".self_attn.q_proj."),
            WeightRenaming(r"\.attn_k\.", ".self_attn.k_proj."),
            WeightRenaming(r"\.attn_v\.", ".self_attn.v_proj."),
            WeightRenaming(r"\.attn_output\.", ".self_attn.out_proj."),
            WeightRenaming(r"\.ffn_up\.", ".fc1."),
            WeightRenaming(r"\.ffn_down\.", ".fc2."),
        ]

    # 5. Mamba / Mamba2
    if hf_type in ("mamba", "mamba2"):
        dt_rule = (
            WeightRenaming(r"\.ssm_dt\.bias$", ".mixer.dt_bias")
            if hf_type == "mamba2"
            else WeightRenaming(r"\.ssm_dt\.", ".mixer.dt_proj.")
        )
        return [
            WeightRenaming(r"^token_embd\.", "model.embed_tokens."),
            WeightRenaming(r"^output_norm\.", "model.norm."),
            WeightRenaming(r"^output\.", "lm_head."),
            WeightRenaming(r"^blk\.", "model.layers."),
            WeightRenaming(r"\.attn_norm\.", ".norm."),
            WeightRenaming(r"\.ssm_in\.", ".mixer.in_proj."),
            WeightRenaming(r"\.ssm_conv1d\.", ".mixer.conv1d."),
            WeightRenaming(r"\.ssm_x\.", ".mixer.x_proj."),
            dt_rule,
            WeightRenaming(r"\.ssm_a$", ".mixer.A_log"),
            WeightRenaming(r"\.ssm_d$", ".mixer.D"),
            WeightRenaming(r"\.ssm_norm\.", ".mixer.norm."),
            WeightRenaming(r"\.ssm_out\.", ".mixer.out_proj."),
            WeightConverter(r"\.mixer\.conv1d\.weight$", ".mixer.conv1d.weight", [Unsqueeze(1, expected_ndim=3)]),
            WeightConverter(r"\.mixer\.A_log$", ".mixer.A_log", [LogNegate()]),
        ]

    # 6. Gemma 1 / Gemma 2 / Gemma 3 / Gemma 4 (zero-centered RMSNorms stored as w + 1)
    if hf_type in ("gemma", "gemma2", "gemma3_text", "gemma4_text"):
        has_4_norms = hf_type in ("gemma2", "gemma3_text", "gemma4_text") or any(
            ".post_attention_norm." in n for n in tensor_names
        )
        ffn_norm_target = ".pre_feedforward_layernorm.weight" if has_4_norms else ".post_attention_layernorm.weight"
        norms = [
            WeightConverter(r"^output_norm\.weight$", "model.norm.weight", [SubtractOne()]),
            WeightConverter(r"\.attn_norm\.weight$", ".input_layernorm.weight", [SubtractOne()]),
            WeightConverter(r"\.ffn_norm\.weight$", ffn_norm_target, [SubtractOne()]),
            WeightConverter(r"\.post_attention_norm\.weight$", ".post_attention_layernorm.weight", [SubtractOne()]),
            WeightConverter(r"\.post_ffw_norm\.weight$", ".post_feedforward_layernorm.weight", [SubtractOne()]),
            WeightConverter(r"\.attn_q_norm\.weight$", ".self_attn.q_norm.weight", [SubtractOne()]),
            WeightConverter(r"\.attn_k_norm\.weight$", ".self_attn.k_norm.weight", [SubtractOne()]),
        ]
        return list(DENSE_DECODER_RENAMINGS) + norms

    # 7. Nemotron (zero-centered LayerNorm1P stored as w + 1, squared-ReLU 2-layer MLP)
    if hf_type == "nemotron":
        return list(DENSE_DECODER_RENAMINGS) + [
            WeightRenaming(r"^output_norm\.bias$", "model.norm.bias"),
            WeightRenaming(r"\.attn_norm\.bias$", ".input_layernorm.bias"),
            WeightRenaming(r"\.ffn_norm\.bias$", ".post_attention_layernorm.bias"),
            WeightConverter(r"^output_norm\.weight$", "model.norm.weight", [SubtractOne()]),
            WeightConverter(r"\.attn_norm\.weight$", ".input_layernorm.weight", [SubtractOne()]),
            WeightConverter(r"\.ffn_norm\.weight$", ".post_attention_layernorm.weight", [SubtractOne()]),
        ]

    # 8. Phi-2 (`phi`) & StarCoder2
    if hf_type == "phi":
        return [
            WeightRenaming(r"^token_embd\.", "model.embed_tokens."),
            WeightRenaming(r"^output_norm\.", "model.norm."),
            WeightRenaming(r"^output\.", "lm_head."),
            WeightRenaming(r"^blk\.", "model.layers."),
            WeightRenaming(r"\.attn_norm\.", ".input_layernorm."),
            WeightRenaming(r"\.attn_q\.", ".self_attn.q_proj."),
            WeightRenaming(r"\.attn_k\.", ".self_attn.k_proj."),
            WeightRenaming(r"\.attn_v\.", ".self_attn.v_proj."),
            WeightRenaming(r"\.attn_qkv\.", ".self_attn.qkv_proj."),
            WeightRenaming(r"\.attn_output\.", ".self_attn.dense."),
            WeightRenaming(r"\.ffn_up\.", ".mlp.fc1."),
            WeightRenaming(r"\.ffn_down\.", ".mlp.fc2."),
        ]

    if hf_type == "starcoder2":
        return [
            WeightRenaming(r"^token_embd\.", "model.embed_tokens."),
            WeightRenaming(r"^output_norm\.", "model.norm."),
            WeightRenaming(r"^output\.", "lm_head."),
            WeightRenaming(r"^blk\.", "model.layers."),
            WeightRenaming(r"\.attn_norm\.", ".input_layernorm."),
            WeightRenaming(r"\.ffn_norm\.", ".post_attention_layernorm."),
            WeightRenaming(r"\.attn_q\.", ".self_attn.q_proj."),
            WeightRenaming(r"\.attn_k\.", ".self_attn.k_proj."),
            WeightRenaming(r"\.attn_v\.", ".self_attn.v_proj."),
            WeightRenaming(r"\.attn_output\.", ".self_attn.o_proj."),
            WeightRenaming(r"\.ffn_up\.", ".mlp.c_fc."),
            WeightRenaming(r"\.ffn_down\.", ".mlp.c_proj."),
        ]

    # 9. Phi-3 (`phi3`: fused `qkv_proj` + `gate_up_proj`)
    if hf_type == "phi3":
        return [
            WeightRenaming(r"^token_embd\.", "model.embed_tokens."),
            WeightRenaming(r"^output_norm\.", "model.norm."),
            WeightRenaming(r"^output\.", "lm_head."),
            WeightRenaming(r"^blk\.", "model.layers."),
            WeightRenaming(r"\.attn_norm\.", ".input_layernorm."),
            WeightRenaming(r"\.ffn_norm\.", ".post_attention_layernorm."),
            WeightRenaming(r"\.attn_qkv\.", ".self_attn.qkv_proj."),
            WeightRenaming(r"\.attn_q\.", ".self_attn.q_proj."),
            WeightRenaming(r"\.attn_k\.", ".self_attn.k_proj."),
            WeightRenaming(r"\.attn_v\.", ".self_attn.v_proj."),
            WeightRenaming(r"\.attn_output\.", ".self_attn.o_proj."),
            WeightRenaming(r"\.ffn_up\.", ".mlp.gate_up_proj."),
            WeightRenaming(r"\.ffn_gate\.", ".mlp.gate_proj."),
            WeightRenaming(r"\.ffn_down\.", ".mlp.down_proj."),
        ]

    # 10. GLM-4 (`glm4`: fused `gate_up_proj` + 4 norms)
    if hf_type == "glm4":
        return [
            WeightRenaming(r"^token_embd\.", "model.embed_tokens."),
            WeightRenaming(r"^output_norm\.", "model.norm."),
            WeightRenaming(r"^output\.", "lm_head."),
            WeightRenaming(r"^blk\.", "model.layers."),
            WeightRenaming(r"\.attn_norm\.", ".input_layernorm."),
            WeightRenaming(r"\.ffn_norm\.", ".post_attention_layernorm."),
            WeightRenaming(r"\.post_attention_norm\.", ".post_self_attn_layernorm."),
            WeightRenaming(r"\.post_ffw_norm\.", ".post_mlp_layernorm."),
            WeightRenaming(r"\.attn_q\.", ".self_attn.q_proj."),
            WeightRenaming(r"\.attn_k\.", ".self_attn.k_proj."),
            WeightRenaming(r"\.attn_v\.", ".self_attn.v_proj."),
            WeightRenaming(r"\.attn_output\.", ".self_attn.o_proj."),
            WeightRenaming(r"\.ffn_up\.", ".mlp.gate_up_proj."),
            WeightRenaming(r"\.ffn_down\.", ".mlp.down_proj."),
        ]

    # 11. OLMo2 / EXAONE-4 (post-norm only: post_attention_layernorm + post_feedforward_layernorm)
    if hf_type in ("olmo2", "exaone4"):
        return list(DENSE_DECODER_RENAMINGS) + [
            WeightRenaming(r"^output_norm\.", "model.norm."),
            WeightRenaming(r"\.attn_norm\.", ".input_layernorm."),
            WeightRenaming(r"\.post_attention_norm\.", ".post_attention_layernorm."),
            WeightRenaming(r"\.ffn_norm\.", ".post_attention_layernorm."),
            WeightRenaming(r"\.post_ffw_norm\.", ".post_feedforward_layernorm."),
            WeightRenaming(r"\.attn_q_norm\.", ".self_attn.q_norm."),
            WeightRenaming(r"\.attn_k_norm\.", ".self_attn.k_norm."),
        ]

    # 12. Qwen3-Next (hybrid GatedDeltaNet + MoE)
    if hf_type == "qwen3_next":
        num_k = int(getattr(text_cfg, "linear_num_key_heads", 16))
        num_v = int(getattr(text_cfg, "linear_num_value_heads", 32))
        v_hd = int(getattr(text_cfg, "linear_value_head_dim", 128))
        k_hd = int(getattr(text_cfg, "linear_key_head_dim", 128))
        heads = (num_k, max(1, num_v // max(1, num_k)))
        qk_rows = 2 * k_hd * num_k
        return (
            list(DENSE_DECODER_RENAMINGS)
            + _moe_common_renamings(router_target=".mlp.gate.", shared_plural=False)
            + [
                WeightRenaming(r"\.ssm_in\.", ".linear_attn.in_proj_qkvz."),
                WeightRenaming(r"\.ssm_ba\.", ".linear_attn.in_proj_ba."),
                WeightRenaming(r"\.ssm_conv1d\.", ".linear_attn.conv1d."),
                WeightRenaming(r"\.ssm_norm\.", ".linear_attn.norm."),
                WeightRenaming(r"\.ssm_out\.", ".linear_attn.out_proj."),
                WeightRenaming(r"\.ssm_a$", ".linear_attn.A_log"),
                WeightRenaming(r"\.ssm_dt\.bias$", ".linear_attn.dt_bias"),
                WeightConverter(r"^output_norm\.weight$", "model.norm.weight", [SubtractOne()]),
                WeightConverter(r"\.attn_norm\.weight$", ".input_layernorm.weight", [SubtractOne()]),
                WeightConverter(r"\.post_attention_norm\.weight$", ".post_attention_layernorm.weight", [SubtractOne()]),
                WeightConverter(r"\.ffn_norm\.weight$", ".post_attention_layernorm.weight", [SubtractOne()]),
                WeightConverter(r"\.attn_q_norm\.weight$", ".self_attn.q_norm.weight", [SubtractOne()]),
                WeightConverter(r"\.attn_k_norm\.weight$", ".self_attn.k_norm.weight", [SubtractOne()]),
                WeightConverter("linear_attn.A_log", "linear_attn.A_log", [LogNegate(), TiledToGroupedRows(*heads)]),
                WeightConverter("linear_attn.dt_bias", "linear_attn.dt_bias", [TiledToGroupedRows(*heads)]),
                WeightConverter(
                    "linear_attn.conv1d.weight",
                    "linear_attn.conv1d.weight",
                    [TiledToGroupedRows(*heads, v_hd, offset=qk_rows), Unsqueeze(1, expected_ndim=3)],
                ),
                WeightConverter(
                    "linear_attn.out_proj.weight",
                    "linear_attn.out_proj.weight",
                    [TiledToGroupedInputs(*heads, v_hd)],
                ),
                WeightConverter(
                    "mlp.shared_expert_gate.weight",
                    "mlp.shared_expert_gate.weight",
                    [Unsqueeze(0, expected_ndim=2)],
                ),
            ]
        )

    # 13. Jamba / Bamba / Falcon-H1 (hybrid Mamba + Attention + feed_forward)
    if hf_type in ("jamba", "bamba", "falcon_h1"):
        dt_rule = (
            WeightRenaming(r"\.ssm_dt\.", ".mamba.dt_proj.")
            if hf_type == "jamba"
            else WeightRenaming(r"\.ssm_dt\.bias$", ".mamba.dt_bias")
        )
        return [
            WeightRenaming(r"^token_embd\.", "model.embed_tokens."),
            WeightRenaming(r"^output_norm\.", "model.norm."),
            WeightRenaming(r"^output\.", "lm_head."),
            WeightRenaming(r"^blk\.", "model.layers."),
            WeightRenaming(r"\.attn_norm\.", ".input_layernorm."),
            WeightRenaming(r"\.ffn_norm\.", ".pre_ff_layernorm."),
            WeightRenaming(r"\.attn_q\.", ".self_attn.q_proj."),
            WeightRenaming(r"\.attn_k\.", ".self_attn.k_proj."),
            WeightRenaming(r"\.attn_v\.", ".self_attn.v_proj."),
            WeightRenaming(r"\.attn_output\.", ".self_attn.o_proj."),
            WeightRenaming(r"\.ffn_gate\.", ".feed_forward.gate_proj."),
            WeightRenaming(r"\.ffn_up\.", ".feed_forward.up_proj."),
            WeightRenaming(r"\.ffn_down\.", ".feed_forward.down_proj."),
            WeightRenaming(r"\.ffn_gate_inp\.", ".feed_forward.router."),
            WeightRenaming(r"\.ffn_gate_exps\.weight", ".feed_forward.experts.gate_proj"),
            WeightRenaming(r"\.ffn_up_exps\.weight", ".feed_forward.experts.up_proj"),
            WeightRenaming(r"\.ffn_down_exps\.weight", ".feed_forward.experts.down_proj"),
            WeightRenaming(r"\.ssm_in\.", ".mamba.in_proj."),
            WeightRenaming(r"\.ssm_conv1d\.", ".mamba.conv1d."),
            WeightRenaming(r"\.ssm_x\.", ".mamba.x_proj."),
            dt_rule,
            WeightRenaming(r"\.ssm_a$", ".mamba.A_log"),
            WeightRenaming(r"\.ssm_d$", ".mamba.D"),
            WeightRenaming(r"\.ssm_norm\.", ".mamba.norm."),
            WeightRenaming(r"\.ssm_dt_norm\.", ".mamba.dt_layernorm."),
            WeightRenaming(r"\.ssm_b_norm\.", ".mamba.b_layernorm."),
            WeightRenaming(r"\.ssm_c_norm\.", ".mamba.c_layernorm."),
            WeightRenaming(r"\.ssm_out\.", ".mamba.out_proj."),
            WeightConverter(r"\.mamba\.conv1d\.weight$", ".mamba.conv1d.weight", [Unsqueeze(1, expected_ndim=3)]),
            WeightConverter(r"\.mamba\.A_log$", ".mamba.A_log", [LogNegate()]),
        ]

    # 14. GraniteMoeHybrid / GraniteMoe
    if hf_type in ("granitemoe", "granitemoehybrid"):
        return [
            WeightRenaming(r"^token_embd\.", "model.embed_tokens."),
            WeightRenaming(r"^output_norm\.", "model.norm."),
            WeightRenaming(r"^output\.", "lm_head."),
            WeightRenaming(r"^blk\.", "model.layers."),
            WeightRenaming(r"\.attn_norm\.", ".input_layernorm."),
            WeightRenaming(r"\.ffn_norm\.", ".post_attention_layernorm."),
            WeightRenaming(r"\.attn_q\.", ".self_attn.q_proj."),
            WeightRenaming(r"\.attn_k\.", ".self_attn.k_proj."),
            WeightRenaming(r"\.attn_v\.", ".self_attn.v_proj."),
            WeightRenaming(r"\.attn_output\.", ".self_attn.o_proj."),
            WeightRenaming(r"\.ffn_gate_inp\.", ".block_sparse_moe.router."),
            WeightRenaming(r"\.ffn_gate_exps\.weight", ".block_sparse_moe.experts.gate_proj"),
            WeightRenaming(r"\.ffn_up_exps\.weight", ".block_sparse_moe.experts.up_proj"),
            WeightRenaming(r"\.ffn_down_exps\.weight", ".block_sparse_moe.experts.down_proj"),
            WeightRenaming(r"\.ffn_up_shexp\.", ".shared_mlp.input_linear."),
            WeightRenaming(r"\.ffn_down_shexp\.", ".shared_mlp.output_linear."),
            WeightRenaming(r"\.ssm_in\.", ".mamba.in_proj."),
            WeightRenaming(r"\.ssm_conv1d\.", ".mamba.conv1d."),
            WeightRenaming(r"\.ssm_dt\.bias$", ".mamba.dt_bias"),
            WeightRenaming(r"\.ssm_a$", ".mamba.A_log"),
            WeightRenaming(r"\.ssm_d$", ".mamba.D"),
            WeightRenaming(r"\.ssm_norm\.", ".mamba.norm."),
            WeightRenaming(r"\.ssm_out\.", ".mamba.out_proj."),
            WeightConverter(r"\.mamba\.conv1d\.weight$", ".mamba.conv1d.weight", [Unsqueeze(1, expected_ndim=3)]),
            WeightConverter(r"\.mamba\.A_log$", ".mamba.A_log", [LogNegate()]),
        ]

    # 15. Standard Dense + MoE Decoder Families (Llama, Mistral, Mixtral, Qwen2/3, OLMo/OLMoE,
    #     PhiMoE, DeepSeek-V2/V3, GLM4-MoE, MiniMax-M2, GPT-OSS, Cohere, Ernie4.5, SmolLM3, StableLM)
    router_target = ".mlp.router." if hf_type in ("phimoe", "gpt_oss") else ".mlp.gate."
    shared_plural = hf_type in ("deepseek_v2", "deepseek_v3", "glm4_moe")
    rules: list[WeightTransform] = (
        list(DENSE_DECODER_RENAMINGS)
        + _std_norms()
        + _moe_common_renamings(router_target=router_target, shared_plural=shared_plural)
        + [
            # Per-expert 2D GGUF fallback (e.g. legacy Mixtral blk.0.ffn_gate.0.weight)
            WeightRenaming(r"\.ffn_gate\.(\d+)\.", r".mlp.experts.\1.gate_proj."),
            WeightRenaming(r"\.ffn_up\.(\d+)\.", r".mlp.experts.\1.up_proj."),
            WeightRenaming(r"\.ffn_down\.(\d+)\.", r".mlp.experts.\1.down_proj."),
            # MLA (DeepSeek-V2 / V3) projections
            WeightRenaming(r"\.attn_q_a\.", ".self_attn.q_a_proj."),
            WeightRenaming(r"\.attn_q_a_norm\.", ".self_attn.q_a_layernorm."),
            WeightRenaming(r"\.attn_q_b\.", ".self_attn.q_b_proj."),
            WeightRenaming(r"\.attn_kv_a_mqa\.", ".self_attn.kv_a_proj_with_mqa."),
            WeightRenaming(r"\.attn_kv_a_norm\.", ".self_attn.kv_a_layernorm."),
            WeightRenaming(r"\.attn_kv_b\.", ".self_attn.kv_b_proj."),
            # GPT-OSS attention sinks
            WeightRenaming(r"\.attn_sinks\.weight$", ".self_attn.sinks"),
            WeightRenaming(r"\.attn_sinks$", ".self_attn.sinks"),
        ]
    )

    if hf_type == "minimax_m2":
        rules.append(WeightRenaming(r"\.exp_probs_b\.bias$", ".mlp.e_score_correction_bias"))
    elif hf_type == "ernie4_5_moe":
        rules.append(WeightRenaming(r"\.exp_probs_b\.bias$", ".mlp.gate.moe_statics.e_score_correction_bias"))
    else:
        rules.append(WeightRenaming(r"\.exp_probs_b\.bias$", ".mlp.gate.e_score_correction_bias"))

    if hf_type == "qwen2_moe":
        rules.append(
            WeightConverter(
                "mlp.shared_expert_gate.weight",
                "mlp.shared_expert_gate.weight",
                [Unsqueeze(0, expected_ndim=2)],
            )
        )

    # Reverse llama.cpp's interleaved RoPE Q/K permutation when arch == "llama" (Llama / Mistral / Mixtral)
    if arch == "llama":
        n_q = int(getattr(text_cfg, "num_attention_heads", 0) or 0)
        n_kv = int(getattr(text_cfg, "num_key_value_heads", None) or n_q)
        if n_q > 0:
            rules.extend(
                [
                    WeightConverter(
                        r"\.self_attn\.q_proj\.weight$",
                        ".self_attn.q_proj.weight",
                        [LlamaRopeReversePermute(n_q)],
                    ),
                    WeightConverter(
                        r"\.self_attn\.q_proj\.bias$",
                        ".self_attn.q_proj.bias",
                        [LlamaRopeReversePermute(n_q)],
                    ),
                    WeightConverter(
                        r"\.self_attn\.k_proj\.weight$",
                        ".self_attn.k_proj.weight",
                        [LlamaRopeReversePermute(n_kv)],
                    ),
                    WeightConverter(
                        r"\.self_attn\.k_proj\.bias$",
                        ".self_attn.k_proj.bias",
                        [LlamaRopeReversePermute(n_kv)],
                    ),
                ]
            )

    return rules


# --------------------------------------------------------------------------------------
# Zero-mmap GGUF Tokenizer Builder
# --------------------------------------------------------------------------------------

_GPT2_SPLIT_REGEX = (
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)
_QWEN_SPLIT_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
_LLAMA3_SPLIT_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

_EXTRA_PRE_TOKENIZER_SPLITS: dict[str, str] = {
    "default": _GPT2_SPLIT_REGEX,
    "gpt-2": _GPT2_SPLIT_REGEX,
    "gpt2": _GPT2_SPLIT_REGEX,
    "roberta-bpe": _GPT2_SPLIT_REGEX,
    "olmo": _GPT2_SPLIT_REGEX,
    "smollm": _GPT2_SPLIT_REGEX,
    "starcoder": r"\p{N}|'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
    "qwen3": _QWEN_SPLIT_REGEX,
    "qwen3moe": _QWEN_SPLIT_REGEX,
    "qwen3next": _QWEN_SPLIT_REGEX,
    "glm4": _LLAMA3_SPLIT_REGEX,
    "chatglm-bpe": _LLAMA3_SPLIT_REGEX,
    "command-r": _LLAMA3_SPLIT_REGEX,
    "exaone": _GPT2_SPLIT_REGEX,
    "exaone4": _GPT2_SPLIT_REGEX,
    "deepseek-llm": r"[\r\n]|[\p{L}\p{M}]+|[\p{N}]|[^\s\p{L}\p{M}\p{N}]+|\s+(?!\S)|\s+",
    "deepseek-coder": r"[\r\n]|[\p{L}\p{M}]+|[\p{N}]|\p{P}|[^\s\p{L}\p{M}\p{N}\p{P}]+|\s+(?!\S)|\s+",
    "deepseek-v3": (
        r"\p{N}{1,3}"
        r"|[一-龥぀-ゟ゠-ヿ]+"
        r"|[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    ),
    "ernie4_5": _LLAMA3_SPLIT_REGEX,
    "granite": _GPT2_SPLIT_REGEX,
    "nemotron": _LLAMA3_SPLIT_REGEX,
    "gpt-oss": _LLAMA3_SPLIT_REGEX,
    "gpt_oss": _LLAMA3_SPLIT_REGEX,
}


def _ensure_pre_tokenizer_splits_registered() -> None:
    from transformers.integrations.gguf.gguf_tokenizer_mapping import GGUF_PRE_TOKENIZER_SPLITS

    for k, v in _EXTRA_PRE_TOKENIZER_SPLITS.items():
        if k not in GGUF_PRE_TOKENIZER_SPLITS or isinstance(GGUF_PRE_TOKENIZER_SPLITS[k], list):
            GGUF_PRE_TOKENIZER_SPLITS[k] = v


def build_extended_gguf_tokenizer(arch: str, meta: dict[str, Any], cfg: Any):
    """Instantiates a fast tokenizer from GGUF header metadata without calling ``mmap``."""
    from transformers import PreTrainedTokenizerFast
    from transformers.integrations.gguf import GGUF_TOKENIZER_MAPPING, convert_gguf_tokenizer
    from transformers.integrations.gguf.gguf_tokenizer_mapping import (
        GGUF_PRE_TOKENIZER_SPLITS,
        _SPECIAL_TOKENS,
    )
    from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING

    _ensure_pre_tokenizer_splits_registered()

    grouped = {
        group: {name: meta[f"tokenizer.{key}"] for key, name in renames.items() if f"tokenizer.{key}" in meta}
        for group, renames in GGUF_TOKENIZER_MAPPING.items()
    }
    tok_dict = dict(grouped["tokenizer"])
    tok_cfg = dict(grouped["tokenizer_config"])
    tokens = list(tok_dict.get("tokens") or [])
    for name, key in _SPECIAL_TOKENS.items():
        tid = meta.get(f"tokenizer.ggml.{key}")
        tok_cfg[name] = tokens[tid] if (tid is not None and 0 <= int(tid) < len(tokens)) else None

    tok_type = tok_dict.get("tokenizer_type")
    if tok_type == "gpt2" and "merges" not in tok_dict:
        tok_dict["merges"] = []
    if tok_type in ("llama", "spm") and "scores" not in tok_dict:
        tok_dict["scores"] = [0.0] * len(tokens)
    pre_type = tok_dict.get("pre_tokenizer_type")
    # GGUF's "default" also occurs on SentencePiece checkpoints. The regex
    # registry installs a ByteLevel pre-tokenizer and is only valid for BPE
    # byte vocabularies; applying it to metaspace tokens corrupts UTF-8.
    if tok_type != "gpt2":
        tok_dict.pop("pre_tokenizer_type", None)
    elif pre_type and pre_type not in GGUF_PRE_TOKENIZER_SPLITS:
        tok_dict["pre_tokenizer_type"] = "default"

    fast_tok, add_kw = convert_gguf_tokenizer(arch, tok_dict)
    mapped = TOKENIZER_MAPPING.get(type(cfg), None)
    tok_cls = (mapped[1] or mapped[0]) if isinstance(mapped, (tuple, list)) else mapped
    if tok_cls is not None:
        try:
            return tok_cls(tokenizer_object=fast_tok, **tok_cfg, **add_kw)
        except Exception:
            pass
    return PreTrainedTokenizerFast(tokenizer_object=fast_tok, **tok_cfg, **add_kw)
