"""AirColibri runtime: W4A16 + NVRTC kernels + native MTP speculative decoding + budgeted VRAM.

`AirColibriUnifiedEngine` combines:
- NVRTC CUDA kernels, including single-weight-read multi-row kernels (`M in {1,2,3,4}`)
- Native MTP head (`ColibriNativeMTPHead`, ~130 MiB Q4)
- `ColibriDraftVocab`, `ColibriSpecGate`, `colibri_spec_lookup`, `HybridSpecSnapshotManager`
- VRAM modes `normal` (almost all free VRAM, with a safety reserve) and `turbo` (all of it):
  all layers resident when they fit, otherwise resident layers + an equal-slot streaming ring
- Chunked prefill that projects only the last position onto the vocabulary
- Cooperative cancellation and guaranteed VRAM cleanup (`try/finally`) for every generation
- `ColibriKVPrefixSession`: multi-turn prefix reuse with pinned host-RAM parking
- Colibri containers and standard Hugging Face shards (see `model_source`)
- Optional `ZeroProbSubspaceVocabHead` (`--fast-vocab`, off by default)
"""
from __future__ import annotations

import copy
import ctypes
from dataclasses import asdict, dataclass
import gc
import os
from pathlib import Path
import sys
import time
import traceback
import types
from typing import Any, Callable, Iterator

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def find_aircolibri_root(explicit_root: str | Path | None = None) -> Path:
    """Returns the project root (folder containing `src/aircolibri_integrated.py`)."""
    candidates: list[Path] = []
    if explicit_root:
        candidates.append(Path(explicit_root))
    env_root = os.environ.get("AIRCOLIBRI_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    candidates.append(PROJECT_ROOT)
    for cand in candidates:
        if (cand / "src" / "aircolibri_integrated.py").exists():
            resolved = cand.resolve()
            if str(resolved) not in sys.path:
                sys.path.insert(0, str(resolved))
            return resolved
    raise FileNotFoundError(
        "Could not locate the AirColibri project root (expected `src/aircolibri_integrated.py`). "
        "Pass `--aircolibri-root <path>` or set `AIRCOLIBRI_ROOT`."
    )


def checkpoint_manifest_path(root: Path) -> Path:
    return root / "data" / "trusted-checkpoint-manifest.json"


DEFAULT_DRAFT_SEED_CORPUS = (
    "AirColibri combines 4-bit group-64 quantization, NVRTC CUDA kernel fusion, "
    "Multi-Token Prediction (MTP), GatedDeltaNet recurrent state snapshots, and "
    "PCIe triple-ring asynchronous DMA prefetching.\n"
    "В физике шин памяти видеокарты главная задержка при генерации по одному токену "
    "возникает из-за повторного чтения весов слоёв из GDDR6. Спекулятивная проверка "
    "нескольких кандидатов за один проход делит чтение весов на 2-3 токена без потери точности.\n"
    "def verify_ring_slot_lease(slot_id: int, layer_idx: int) -> bool:\n"
    "    return slot_id >= 0 and layer_idx >= 0\n"
)


class ColibriDraftVocab:
    """Port of `upstream_colibri/c/draft_vocab.h` scored via `indexed_vocab_gemv_q4_f32`."""

    NEAR_NUM = 3
    NEAR_DEN = 4
    HALVE_EVERY = 1024

    def __init__(self, vocab_size: int, top_k: int = 16000, device: torch.device = torch.device("cuda")):
        self.vocab_size = int(vocab_size)
        self.top_k = min(int(top_k), self.vocab_size)
        self.device = device
        self.counts = torch.zeros(self.vocab_size, dtype=torch.float32)
        self.in_conversation = torch.zeros(self.vocab_size, dtype=torch.bool)
        self.in_list = torch.zeros(self.vocab_size, dtype=torch.bool)
        self.counted = 0
        self.gpu_list: torch.Tensor | None = None
        self.stats = {"picks": 0, "picks_inside": 0, "appended": 0, "rebuilds": 0, "halvings": 0}

    def seed_from_corpus_tokens(self, token_ids: list[int]) -> None:
        for tid in token_ids:
            if 0 <= tid < self.vocab_size:
                self.counts[tid] += 1.0

    def _count_one(self, tok: int) -> bool:
        self.counts[tok] += 1.0
        self.counted += 1
        if self.counted >= self.HALVE_EVERY:
            self.counts.mul_(0.5)
            self.counted = 0
            self.stats["halvings"] += 1
            return True
        return False

    def rebuild(self) -> None:
        pos_mask = self.counts > 0
        n_pos = int(pos_mask.sum().item())
        if n_pos >= self.top_k:
            kth_val = float(torch.topk(self.counts, k=self.top_k).values[-1].item())
            cutoff = (self.NEAR_NUM / self.NEAR_DEN) * kth_val
            selected = (self.counts >= cutoff) | self.in_conversation
        else:
            selected = pos_mask | self.in_conversation
            rem = self.top_k - int(selected.sum().item())
            if rem > 0:
                not_sel = (~selected).nonzero(as_tuple=False).squeeze(-1)[:rem]
                selected[not_sel] = True

        self.in_list = selected
        idx_cpu = selected.nonzero(as_tuple=False).squeeze(-1).to(torch.int32)
        self.gpu_list = idx_cpu.to(self.device, non_blocking=True).unsqueeze(0).contiguous()
        self.stats["rebuilds"] += 1

    def begin_request(self, prompt_ids: list[int], cached_prompt_tokens: int = 0) -> None:
        self.in_conversation.zero_()
        for tid in prompt_ids:
            if 0 <= tid < self.vocab_size:
                self.in_conversation[tid] = True
        for tid in prompt_ids[cached_prompt_tokens:]:
            if 0 <= tid < self.vocab_size:
                self._count_one(tid)
        self.rebuild()

    def observe_picked_token(self, tok: int) -> None:
        if not (0 <= tok < self.vocab_size):
            return
        self.stats["picks"] += 1
        was_inside = bool(self.in_list[tok].item())
        if was_inside:
            self.stats["picks_inside"] += 1
        self.in_conversation[tok] = True
        halved = self._count_one(tok)
        if halved:
            self.rebuild()
        elif not was_inside:
            self.in_list[tok] = True
            new_id = torch.tensor([[tok]], dtype=torch.int32, device=self.device)
            self.gpu_list = torch.cat([self.gpu_list, new_id], dim=1).contiguous()
            self.stats["appended"] += 1


def colibri_spec_lookup(history: list[int], max_drafts: int = 2, max_n: int = 5, min_n: int = 2) -> list[int]:
    """Port of `spec_lookup` from `upstream_colibri/c/spec_draft.h`."""
    H = len(history)
    if H < min_n + 1 or max_drafts <= 0:
        return []
    search_start = max(0, H - 1024)
    for n in range(min(max_n, H - 1), min_n - 1, -1):
        pat = history[H - n :]
        last_tok = pat[-1]
        for pos in range(H - n - 1, search_start + n - 2, -1):
            if history[pos] == last_tok and history[pos - n + 1 : pos + 1] == pat:
                nxt_start = pos + 1
                nxt_end = min(H - 1, nxt_start + max_drafts)
                if nxt_end > nxt_start:
                    return history[nxt_start:nxt_end]
    return []


class ColibriSpecGate:
    """Port of confidence-scheduled `SpecGate` from `upstream_colibri/c/spec_draft.h`."""

    SRC_MTP = 0
    SRC_LOOKUP = 1
    PBINS = 10

    def __init__(self, max_drafts: int = 2, force_off: bool = False):
        self.max_drafts = max_drafts
        self.off = force_off
        self.acc = [
            [0.80, 0.58, 0.45, 0.40],
            [0.65, 0.52, 0.45, 0.40],
        ]
        self.acc_w = [[2.0] * 4, [2.0] * 4]
        self.t = [0.0] * 6
        self.t_w = [0.0] * 6
        self.rho = 0.12
        self.rho_w = 1.0
        self.d = [0.0, 0.0]
        self.d_w = [0.0, 0.0]
        self.pb = [[(b + 0.5) / self.PBINS for b in range(self.PBINS)] for _ in range(2)]
        self.pb_w = [[2.0] * self.PBINS for _ in range(2)]
        self.prop = [[0, 0, 0, 0], [0, 0, 0, 0]]
        self.hit = [[0, 0, 0, 0], [0, 0, 0, 0]]
        self.verifies = [0, 0]
        self.declined = [0, 0]

    @staticmethod
    def _ema(val: float, weight: float, sample: float, memory: float) -> tuple[float, float]:
        if weight < memory:
            weight += 1.0
        val += (sample - val) / weight
        return val, weight

    def record_forward(self, rows: int, seconds: float) -> None:
        if 1 <= rows < len(self.t) and seconds > 0.0:
            self.t[rows], self.t_w[rows] = self._ema(self.t[rows], self.t_w[rows], seconds, 8.0)
            if rows > 1 and self.t_w[1] > 0.0 and self.t[1] > 0.0:
                r = max(0.0, min(4.0, (seconds / self.t[1] - 1.0) / (rows - 1)))
                self.rho, self.rho_w = self._ema(self.rho, self.rho_w, r, 8.0)

    def record_draft_cost(self, src: int, seconds: float) -> None:
        self.d[src], self.d_w[src] = self._ema(self.d[src], self.d_w[src], seconds, 8.0)

    def rows_cost(self, rows: int) -> float:
        if rows <= 1:
            return 1.0
        if rows < len(self.t) and self.t_w[rows] > 0.0 and self.t_w[1] > 0.0 and self.t[1] > 0.0:
            return self.t[rows] / self.t[1]
        return 1.0 + (rows - 1) * self.rho

    def _pbin(self, p: float) -> int:
        b = int(p * self.PBINS) if p >= 0.0 else 0
        return max(0, min(self.PBINS - 1, b))

    def calibrated_p(self, src: int, p_est: float, pos_j: int) -> float:
        bin_rate = self.pb[src][self._pbin(p_est)]
        return 0.5 * bin_rate + 0.5 * max(p_est, self.acc[src][pos_j] * 0.85)

    def should_draft_second_mtp(self, p1: float) -> bool:
        if self.max_drafts < 2:
            return False
        if self.off:
            return p1 >= 0.55
        a1 = self.calibrated_p(self.SRC_MTP, p1, 0)
        if a1 < 0.55:
            return False
        a2 = self.acc[self.SRC_MTP][1]
        v1 = (1.0 + a1) / self.rows_cost(2)
        extra_d = (self.d[self.SRC_MTP] / self.t[1]) if (self.d_w[self.SRC_MTP] > 0 and self.t[1] > 0) else 0.03
        v2 = (1.0 + a1 + a1 * a2) / (self.rows_cost(3) + extra_d)
        return v2 > v1 and v2 > 1.02

    def pick_depth(self, src: int, probs: list[float]) -> int:
        k_avail = min(len(probs), self.max_drafts)
        if k_avail == 0:
            return 0
        if self.off:
            return k_avail
        best_k = 0
        best_val = 1.0
        e = 1.0
        prod = 1.0
        for j in range(k_avail):
            aj = self.calibrated_p(src, probs[j], j)
            prod *= aj
            e += prod
            val = e / self.rows_cost(j + 2)
            if val > best_val:
                best_val = val
                best_k = j + 1
        if best_k == 0:
            self.declined[src] += 1
        return best_k

    def record_result(self, src: int, judged: int, accepted: int, probs: list[float] | None = None) -> None:
        if judged < 1:
            return
        self.verifies[src] += 1
        for j in range(min(judged, 4)):
            self.prop[src][j] += 1
            ok = 1.0 if j < accepted else 0.0
            if j < accepted:
                self.hit[src][j] += 1
            self.acc[src][j], self.acc_w[src][j] = self._ema(self.acc[src][j], self.acc_w[src][j], ok, 16.0)
            if probs is not None and j < len(probs):
                b = self._pbin(probs[j])
                self.pb[src][b], self.pb_w[src][b] = self._ema(self.pb[src][b], self.pb_w[src][b], ok, 64.0)

    def summary(self) -> dict[str, Any]:
        mtp_prop = sum(self.prop[0])
        mtp_hit = sum(self.hit[0])
        lkp_prop = sum(self.prop[1])
        lkp_hit = sum(self.hit[1])
        tot_prop = mtp_prop + lkp_prop
        tot_hit = mtp_hit + lkp_hit
        return {
            "mtp_verifies": self.verifies[0],
            "mtp_proposed": mtp_prop,
            "mtp_accepted": mtp_hit,
            "mtp_accept_rate_pct": round(100.0 * mtp_hit / max(1, mtp_prop), 2),
            "lookup_verifies": self.verifies[1],
            "lookup_proposed": lkp_prop,
            "lookup_accepted": lkp_hit,
            "lookup_accept_rate_pct": round(100.0 * lkp_hit / max(1, lkp_prop), 2),
            "overall_accept_rate_pct": round(100.0 * tot_hit / max(1, tot_prop), 2),
            "declined_low_confidence": sum(self.declined),
            "measured_rho_extra_row_cost": round(self.rho, 4),
        }


class HybridSpecSnapshotManager:
    """GPU-resident snapshot slots for Qwen3.5's 24 GatedDeltaNet + 8 Full-Attention layers."""

    def __init__(self, config, device: torch.device, max_drafts: int = 3):
        self.device = device
        self.max_slots = max_drafts
        self.linear_layers = [i for i, lt in enumerate(config.layer_types) if lt == "linear_attention"]
        self.full_layers = [i for i, lt in enumerate(config.layer_types) if lt == "full_attention"]
        self.layer_to_lin_slot = {l_idx: pos for pos, l_idx in enumerate(self.linear_layers)}
        n_lin = len(self.linear_layers)
        conv_dim = (
            config.linear_key_head_dim * config.linear_num_key_heads * 2
            + config.linear_value_head_dim * config.linear_num_value_heads
        )
        conv_k = config.linear_conv_kernel_dim
        self.snap_conv = torch.empty((n_lin, self.max_slots, 1, conv_dim, conv_k), dtype=torch.bfloat16, device=device)
        self.snap_rec = torch.empty(
            (n_lin, self.max_slots, 1, config.linear_num_value_heads, config.linear_key_head_dim, config.linear_value_head_dim),
            dtype=torch.float32,
            device=device,
        )
        self.active_capture = False
        self.rollbacks = 0

    def save_row(self, layer_idx: int, row_r: int, conv_state: torch.Tensor, rec_state: torch.Tensor) -> None:
        if not self.active_capture or row_r >= self.max_slots:
            return
        lin_i = self.layer_to_lin_slot[layer_idx]
        c_len = conv_state.shape[-1]
        self.snap_conv[lin_i, row_r, :, :, :c_len].copy_(conv_state)
        self.snap_rec[lin_i, row_r].copy_(rec_state)

    def rollback_to_accepted(self, past_key_values: Any, accepted_drafts: int, total_drafts: int) -> None:
        rejected = total_drafts - accepted_drafts
        if rejected <= 0:
            return
        for l_idx in self.linear_layers:
            lin_i = self.layer_to_lin_slot[l_idx]
            cs = past_key_values.get_conv_state(l_idx, state_idx=0)
            rs = past_key_values.get_recurrent_state(l_idx, state_idx=0)
            c_len = cs.shape[-1]
            cs.copy_(self.snap_conv[lin_i, accepted_drafts, :, :, :c_len])
            rs.copy_(self.snap_rec[lin_i, accepted_drafts])
        for l_idx in self.full_layers:
            past_key_values.layers[l_idx].crop(-rejected)
        self.rollbacks += 1


_DECODE_NO_MASK = {"full_attention": None, "linear_attention": None}


def fused_deltanet_multirow(
    gk: Any,
    lat_mod: Any,
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    recurrent_state: torch.Tensor,
    snap_mgr: HybridSpecSnapshotManager | None = None,
) -> torch.Tensor:
    from .kernels import get_mr_kernels

    B, S, _ = hidden_states.shape
    if S == 1 and B <= 8:
        return gk.fused_deltanet_full_step(lat_mod, hidden_states, conv_state, recurrent_state)

    assert B == 1 and 2 <= S <= 4
    mrk = get_mr_kernels(gk.device)
    x_c = hidden_states.reshape(S, 4096)
    if not x_c.is_contiguous():
        x_c = x_c.contiguous()
    stream_ptr = ctypes.c_void_p(torch.cuda.current_stream(gk.device).cuda_stream)

    c_x = ctypes.c_uint64(x_c.data_ptr())
    c_qkv_q = ctypes.c_uint64(lat_mod.in_proj_qkv.qweight.data_ptr())
    c_qkv_s = ctypes.c_uint64(lat_mod.in_proj_qkv.scales.data_ptr())
    c_qkv_z = ctypes.c_uint64(lat_mod.in_proj_qkv.zeros.data_ptr())
    c_z_q = ctypes.c_uint64(lat_mod.in_proj_z.qweight.data_ptr())
    c_z_s = ctypes.c_uint64(lat_mod.in_proj_z.scales.data_ptr())
    c_z_z = ctypes.c_uint64(lat_mod.in_proj_z.zeros.data_ptr())
    c_b_q = ctypes.c_uint64(lat_mod.in_proj_b.qweight.data_ptr())
    c_b_s = ctypes.c_uint64(lat_mod.in_proj_b.scales.data_ptr())
    c_b_z = ctypes.c_uint64(lat_mod.in_proj_b.zeros.data_ptr())
    c_a_q = ctypes.c_uint64(lat_mod.in_proj_a.qweight.data_ptr())
    c_a_s = ctypes.c_uint64(lat_mod.in_proj_a.scales.data_ptr())
    c_a_z = ctypes.c_uint64(lat_mod.in_proj_a.zeros.data_ptr())

    c_oqkv = ctypes.c_uint64(gk.ws_qkv.data_ptr())
    c_oz = ctypes.c_uint64(gk.ws_z.data_ptr())
    c_ob = ctypes.c_uint64(gk.ws_b.data_ptr())
    c_oa = ctypes.c_uint64(gk.ws_a.data_ptr())

    p_proj = (ctypes.c_void_p * 17)(
        ctypes.cast(ctypes.byref(c_x), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_qkv_q), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_qkv_s), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_qkv_z), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_z_q), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_z_s), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_z_z), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_b_q), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_b_s), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_b_z), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_a_q), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_a_s), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_a_z), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_oqkv), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_oz), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_ob), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_oa), ctypes.c_void_p),
    )
    rc0 = mrk.nvcuda.cuLaunchKernel(mrk.qkv_fns[S], 1544, 1, 1, 256, 1, 1, S * 8448, stream_ptr, p_proj, None)
    if rc0 != 0:
        raise RuntimeError(f"multirow_qkvzba_m{S} failed: {rc0}")

    w2d = lat_mod.conv1d.weight.squeeze(1)
    if not w2d.is_contiguous():
        w2d = w2d.contiguous()
    c_st = ctypes.c_uint64(conv_state.data_ptr())
    c_w = ctypes.c_uint64(w2d.data_ptr())
    c_slen = ctypes.c_int(int(conv_state.shape[-1]))
    c_alog = ctypes.c_uint64(lat_mod.A_log.data_ptr())
    c_dtb = ctypes.c_uint64(lat_mod.dt_bias.data_ptr())
    c_nw = ctypes.c_uint64(lat_mod.norm.weight.data_ptr())
    c_rec = ctypes.c_uint64(recurrent_state.data_ptr())
    c_abf = ctypes.c_int(1 if lat_mod.A_log.dtype == torch.bfloat16 else 0)
    c_dbf = ctypes.c_int(1 if lat_mod.dt_bias.dtype == torch.bfloat16 else 0)

    for r in range(S):
        c_rqkv = ctypes.c_uint64(gk.ws_qkv[r].data_ptr())
        c_rqkvc = ctypes.c_uint64(gk.ws_qkv_conv[r].data_ptr())
        c_rz = ctypes.c_uint64(gk.ws_z[r].data_ptr())
        c_rb = ctypes.c_uint64(gk.ws_b[r].data_ptr())
        c_ra = ctypes.c_uint64(gk.ws_a[r].data_ptr())
        c_rcore = ctypes.c_uint64(gk.ws_core[r].data_ptr())

        p_conv = (ctypes.c_void_p * 5)(
            ctypes.cast(ctypes.byref(c_rqkv), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_st), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_w), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_rqkvc), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_slen), ctypes.c_void_p),
        )
        rc1 = gk.nvcuda.cuLaunchKernel(
            gk.fns[b"fused_causal_conv1d_silu_batched"], 32, 1, 1, 256, 1, 1, 0, stream_ptr, p_conv, None
        )
        if rc1 != 0:
            raise RuntimeError(f"conv row {r} failed: {rc1}")

        p_rec = (ctypes.c_void_p * 11)(
            ctypes.cast(ctypes.byref(c_rqkvc), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_rz), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_rb), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_ra), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_alog), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_dtb), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_nw), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_rec), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_rcore), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_abf), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_dbf), ctypes.c_void_p),
        )
        rc2 = gk.nvcuda.cuLaunchKernel(
            gk.fns[b"fused_gated_delta_recurrent_batched"], 32, 1, 1, 128, 1, 1, 0, stream_ptr, p_rec, None
        )
        if rc2 != 0:
            raise RuntimeError(f"delta row {r} failed: {rc2}")

        if snap_mgr is not None and r < S - 1:
            snap_mgr.save_row(lat_mod.layer_idx, r, conv_state, recurrent_state)

    return lat_mod.out_proj(gk.ws_core[:S].view(1, S, 4096))


def forward_step_all_logits(
    engine: Any,
    step_tokens: torch.Tensor,
    past_key_values: Any = None,
    prefetch_next: bool = True,
    logits_to_keep: int = 0,
) -> tuple[torch.Tensor, Any]:
    """One backbone pass. ``logits_to_keep=1`` projects only the last position onto the vocabulary.

    Prefill must always pass ``logits_to_keep=1``: projecting every prompt position materialises
    ``[1, S, 248320]`` logits (about 0.5 MiB per token in BF16, 1-2 GiB for a ForgeMind prompt).
    """
    extra = {"logits_to_keep": int(logits_to_keep)} if logits_to_keep else {}
    if hasattr(engine, "_anchor_ring_prepare"):
        return engine._anchor_ring_forward(
            step_tokens, past_key_values=past_key_values, prefetch_next=prefetch_next, **extra
        )

    engine._prefetch_next_step = prefetch_next
    num_layers = engine.config.num_hidden_layers
    inputs_embeds = engine._embed_step_tokens(step_tokens)
    if engine.num_slots > 1 or engine.pipeline_mode != "sequential":
        first_s = engine.vram_resident_layers
        for s in range(first_s, min(first_s + engine.num_slots, num_layers)):
            if s == first_s and engine.tail_prefetched:
                engine.tail_prefetched = False
                continue
            pinned_t = engine.weight_manager.load_layer_to_pinned_slot(s, target_slot_idx=s % engine.num_slots)
            slot = engine.pipeline.get_slot_for_loading(s)
            engine.pipeline.load_tensors_async(slot, pinned_t, pinned_pool=engine.weight_manager.pinned_pool)

    B, S = step_tokens.shape
    mask_arg = (
        _DECODE_NO_MASK
        if (B == 1 and 1 <= S <= 4 and past_key_values is not None and past_key_values.get_seq_length() > 0)
        else None
    )
    outputs = engine.model(
        inputs_embeds=inputs_embeds,
        input_ids=None,
        attention_mask=mask_arg,
        past_key_values=past_key_values,
        use_cache=True,
        **extra,
    )
    return outputs.logits, outputs.past_key_values


def install_colibri_multirow_unifications(
    engine: Any,
    gk: Any,
    snap_mgr: HybridSpecSnapshotManager | None = None,
    enable_vocab_pruner: bool = False,
    captured: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        ALL_ATTENTION_FUNCTIONS,
        apply_rotary_pos_emb,
        eager_attention_forward,
    )
    from src.quantization_q4 import ColibriQ4Linear
    from .kernels import ZeroProbSubspaceVocabHead, get_mr_kernels

    mrk = get_mr_kernels(gk.device)
    if captured is None:
        captured = {}

    if not getattr(engine, "_colibri_mr_installed", False):
        for mod in engine.model.modules():
            if type(mod).__name__ == "Qwen3_5RMSNorm":
                eps_val = float(mod.eps)

                def make_rn_fwd(e_v):
                    def _rnf(self, x):
                        if x.is_cuda and x.dtype == torch.bfloat16 and self.weight.dtype == torch.bfloat16:
                            return gk.rmsnorm_zero_centered(x, self.weight, e_v)
                        out = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + e_v)
                        return (out * (1.0 + self.weight.float())).type_as(x)

                    return _rnf

                mod.forward = types.MethodType(make_rn_fwd(eps_val), mod)

        for submod in engine.model.model.layers.modules():
            if isinstance(submod, ColibriQ4Linear):
                orig_qf = submod.forward

                def _mr_qfwd(x, _m=submod, _of=orig_qf):
                    M = x.numel() // x.shape[-1]
                    if x.is_cuda and x.dtype == torch.bfloat16 and _m.qweight.is_cuda and 1 <= M <= 4:
                        return mrk.linear_q4(x, _m)
                    return _of(x)

                submod.forward = _mr_qfwd

        orig_final_norm = engine.model.model.norm.forward

        def _capturing_final_norm(x):
            out = orig_final_norm(x)
            captured["post_norm"] = out
            return out

        engine.model.model.norm.forward = _capturing_final_norm

        for layer in engine.model.model.layers:
            mlp = layer.mlp
            orig_mlp_fwd = mlp.forward

            def make_mlp_fwd(orig_f):
                def _mfwd(self, x):
                    if x.is_cuda and x.dtype == torch.bfloat16 and hasattr(self.gate_proj, "qweight"):
                        M = x.numel() // x.shape[-1]
                        if x.ndim == 3 and x.shape[1] == 1 and x.shape[0] <= 8:
                            act = gk.fused_mlp_gate_up_swiglu(x, self.gate_proj, self.up_proj)
                            return self.down_proj(act)
                        if 2 <= M <= 4:
                            act = mrk.gate_up_swiglu(x, self.gate_proj, self.up_proj)
                            return self.down_proj(act)
                    return orig_f(x)

                return _mfwd

            mlp.forward = types.MethodType(make_mlp_fwd(orig_mlp_fwd), mlp)

            lat = getattr(layer, "linear_attn", None)
            if lat is not None:
                orig_lat_fwd = lat.forward

                def make_lat_fwd(orig_lf):
                    def _lfwd(self, hidden_states, cache_params=None, attention_mask=None, **kw):
                        B, S, _ = hidden_states.shape
                        if (
                            hidden_states.is_cuda
                            and ((S == 1 and B <= 8) or (B == 1 and 2 <= S <= 4))
                            and attention_mask is None
                            and cache_params is not None
                            and not cache_params.is_recording_past(self.layer_idx)
                            and hasattr(self.in_proj_qkv, "qweight")
                        ):
                            cs = cache_params.get_conv_state(self.layer_idx, state_idx=0)
                            rs = cache_params.get_recurrent_state(self.layer_idx, state_idx=0)
                            if (
                                cs is not None
                                and rs is not None
                                and cs.is_contiguous()
                                and rs.is_contiguous()
                                and rs.dtype == torch.float32
                            ):
                                return fused_deltanet_multirow(gk, self, hidden_states, cs, rs, snap_mgr=snap_mgr)
                        return orig_lf(hidden_states, cache_params=cache_params, attention_mask=attention_mask, **kw)

                    return _lfwd

                lat.forward = types.MethodType(make_lat_fwd(orig_lat_fwd), lat)

            s_attn = getattr(layer, "self_attn", None)
            if s_attn is not None:
                orig_sa_fwd = s_attn.forward

                def make_sa_fwd(orig_sf):
                    def _safwd(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
                        B, S, _ = hidden_states.shape
                        if (
                            B == 1
                            and 2 <= S <= 4
                            and past_key_values is not None
                            and past_key_values.get_seq_length() > 0
                        ):
                            input_shape = hidden_states.shape[:-1]
                            hidden_shape = (*input_shape, -1, self.head_dim)
                            query_states, gate = torch.chunk(
                                self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
                            )
                            gate = gate.reshape(*input_shape, -1)
                            query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
                            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
                            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                            cos, sin = position_embeddings
                            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
                            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
                            attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                                self.config._attn_implementation, eager_attention_forward
                            )
                            base_kv = key_states.shape[2] - S
                            row_outs = []
                            for r in range(S):
                                ao_r, _ = attention_interface(
                                    self,
                                    query_states[:, :, r : r + 1, :],
                                    key_states[:, :, : base_kv + r + 1, :],
                                    value_states[:, :, : base_kv + r + 1, :],
                                    None,
                                    dropout=0.0,
                                    scaling=self.scaling,
                                    **kwargs,
                                )
                                row_outs.append(ao_r)
                            attn_output = torch.cat(row_outs, dim=1).reshape(*input_shape, -1).contiguous()
                            attn_output = attn_output * torch.sigmoid(gate)
                            return self.o_proj(attn_output), None
                        return orig_sf(
                            hidden_states, position_embeddings, attention_mask, past_key_values=past_key_values, **kwargs
                        )

                    return _safwd

                s_attn.forward = types.MethodType(make_sa_fwd(orig_sa_fwd), s_attn)

        engine._colibri_mr_installed = True

    raw_q4_head = engine.model.lm_head.q4_head if hasattr(engine.model.lm_head, "q4_head") else engine.model.lm_head
    if hasattr(raw_q4_head, "qweight") and raw_q4_head.qweight.is_cuda:
        if enable_vocab_pruner:
            pin_set = set(getattr(engine.tokenizer, "all_special_ids", []) or [])
            for tok_str in ("<think>", "</think>", "<|im_end|>", "<|im_start|>", "<|endoftext|>", "\n", "\n\n"):
                for tid in engine.tokenizer.encode(tok_str, add_special_tokens=False):
                    pin_set.add(int(tid))
            pruner = ZeroProbSubspaceVocabHead(
                raw_q4_head,
                gk,
                norm_weight=engine.model.model.norm.weight,
                zero_centered_norm=True,
                rank_r=512,
                survivors=12288,
                pinned_token_ids=sorted(pin_set),
            )
            engine.model.lm_head = pruner
        else:
            orig_h_fwd = raw_q4_head.forward

            def _phfwd(x, _q=raw_q4_head, _of=orig_h_fwd):
                if x.is_cuda and x.dtype == torch.bfloat16 and x.ndim == 3:
                    M = x.numel() // x.shape[-1]
                    if x.shape[1] == 1 and x.shape[0] <= 8:
                        return gk.persistent_gemv_q4(x, _q)
                    if 2 <= M <= 4:
                        return mrk.linear_q4(x, _q)
                return _of(x)

            raw_q4_head.forward = _phfwd
            engine.model.lm_head = raw_q4_head

    return captured


class ColibriNativeMTPHead:
    """Loads `model-mtp.safetensors` as Q4 (130.5 MiB VRAM) and scores over `ColibriDraftVocab`."""

    def __init__(self, engine: Any, gk: Any, draft_vocab: ColibriDraftVocab):
        from safetensors import safe_open
        from transformers.cache_utils import DynamicCache
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5RMSNorm
        from src.aircolibri_integrated import set_module_tensor_to_device
        from src.colibri_nvrtc import can_use_fused_dense_q4, fused_dense_linear_q4
        from src.quantization_q4 import ColibriQ4Linear, quantize_tensor_q4
        from .kernels import get_mr_kernels

        self._DynamicCache = DynamicCache
        self.engine = engine
        self.gk = gk
        self.draft_vocab = draft_vocab
        self.device = engine.device
        self.dtype = engine.dtype
        cfg = engine.config

        mtp_cfg = copy.deepcopy(cfg)
        mtp_cfg.num_hidden_layers = 1
        mtp_cfg.layer_types = ["full_attention"]
        mtp_cfg._attn_implementation = getattr(cfg, "_attn_implementation", "sdpa")
        self.mtp_cfg = mtp_cfg

        mtp_raw = {}
        source = getattr(engine, "model_source", None)
        if source is not None:
            for k, (path, stored) in source.mtp_entries().items():
                with safe_open(path, framework="pt", device="cpu") as f:
                    mtp_raw[k] = f.get_tensor(stored)
        else:
            with safe_open(str(Path(engine.model_dir) / "model-mtp.safetensors"), framework="pt", device="cpu") as f:
                for k in f.keys():
                    mtp_raw[k] = f.get_tensor(k)

        layer = Qwen3_5DecoderLayer(mtp_cfg, layer_idx=0).to(dtype=self.dtype)
        for parent in (layer.self_attn, layer.mlp):
            for ch_name, ch in list(parent.named_children()):
                if isinstance(ch, torch.nn.Linear):
                    setattr(
                        parent,
                        ch_name,
                        ColibriQ4Linear(
                            out_features=ch.out_features,
                            in_features=ch.in_features,
                            has_bias=(ch.bias is not None),
                            group_size=64,
                            compute_dtype=self.dtype,
                            scratch_provider=engine._get_q4_scratch,
                        ),
                    )
        self.fc_q4 = ColibriQ4Linear(
            out_features=cfg.hidden_size,
            in_features=cfg.hidden_size * 2,
            has_bias=False,
            scratch_provider=engine._get_q4_scratch,
            group_size=64,
            compute_dtype=self.dtype,
        )
        p_fc, s_fc, z_fc = quantize_tensor_q4(
            mtp_raw.pop("mtp.fc.weight"),
            group_size=64,
            compute_dtype=self.dtype,
            quant_device=self.device,
            target_device=self.device,
        )
        self.fc_q4.bind_q4_tensors(p_fc, s_fc, z_fc)

        for mod_name, submod in layer.named_modules():
            if isinstance(submod, ColibriQ4Linear):
                w_cpu = mtp_raw.pop(f"mtp.layers.0.{mod_name}.weight")
                pq, sq, zq = quantize_tensor_q4(
                    w_cpu,
                    group_size=64,
                    compute_dtype=self.dtype,
                    quant_device=self.device,
                    target_device=self.device,
                )
                submod.bind_q4_tensors(pq, sq, zq)

        for p_name, _ in list(layer.named_parameters()):
            w_cpu = mtp_raw.pop(f"mtp.layers.0.{p_name}")
            set_module_tensor_to_device(
                layer, p_name, self.device, value=w_cpu.to(device=self.device, dtype=self.dtype), clear_cache=False
            )

        self.layer = layer.eval()
        self.enorm = Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(device=self.device, dtype=self.dtype)
        self.hnorm = Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(device=self.device, dtype=self.dtype)
        self.head_norm = Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(device=self.device, dtype=self.dtype)
        self.enorm.weight.data.copy_(mtp_raw.pop("mtp.pre_fc_norm_embedding.weight").to(device=self.device, dtype=self.dtype))
        self.hnorm.weight.data.copy_(mtp_raw.pop("mtp.pre_fc_norm_hidden.weight").to(device=self.device, dtype=self.dtype))
        self.head_norm.weight.data.copy_(mtp_raw.pop("mtp.norm.weight").to(device=self.device, dtype=self.dtype))
        assert len(mtp_raw) == 0, f"Unconsumed MTP tensors: {list(mtp_raw.keys())}"

        mrk = get_mr_kernels(self.device)
        for mod in [self.fc_q4, self.enorm, self.hnorm, self.head_norm, *self.layer.modules()]:
            if type(mod).__name__ == "Qwen3_5RMSNorm":
                eps_v = float(mod.eps)

                def make_rn(ev):
                    def _rf(s_mod, x):
                        if x.is_cuda and x.dtype == torch.bfloat16:
                            return gk.rmsnorm_zero_centered(x, s_mod.weight, ev)
                        out = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + ev)
                        return (out * (1.0 + s_mod.weight.float())).type_as(x)

                    return _rf

                mod.forward = types.MethodType(make_rn(eps_v), mod)
            elif isinstance(mod, ColibriQ4Linear):
                orig_f = mod.forward

                def _qfwd(x, _m=mod, _of=orig_f):
                    if x.is_cuda and x.dtype == torch.bfloat16:
                        M = x.numel() // x.shape[-1]
                        if M == 1 and can_use_fused_dense_q4(x, _m.qweight, _m.scales, _m.zeros, _m.group_size):
                            return fused_dense_linear_q4(
                                x, _m.qweight, _m.scales, _m.zeros, bias=_m._buffers["bias"], group_size=_m.group_size
                            )
                        if 2 <= M <= 4:
                            return mrk.linear_q4(x, _m)
                    return _of(x)

                mod.forward = _qfwd

        orig_mtp_mlp = self.layer.mlp.forward

        def _mtp_mlp_fwd(s_mlp, x):
            if x.is_cuda and x.dtype == torch.bfloat16:
                M = x.numel() // x.shape[-1]
                if x.ndim == 3 and x.shape[1] == 1 and x.shape[0] == 1:
                    act = gk.fused_mlp_gate_up_swiglu(x, s_mlp.gate_proj, s_mlp.up_proj)
                    return s_mlp.down_proj(act)
                if 2 <= M <= 4:
                    act = mrk.gate_up_swiglu(x, s_mlp.gate_proj, s_mlp.up_proj)
                    return s_mlp.down_proj(act)
            return orig_mtp_mlp(x)

        self.layer.mlp.forward = types.MethodType(_mtp_mlp_fwd, self.layer.mlp)
        self._ColibriQ4Linear = ColibriQ4Linear
        self.kv_cache = DynamicCache(config=self.mtp_cfg)
        self.committed_len = 0
        self.mtp_step_times_ms: list[float] = []

    def reset_cache(self) -> None:
        self.kv_cache = self._DynamicCache(config=self.mtp_cfg)
        self.committed_len = 0
        self.mtp_step_times_ms.clear()

    def q4_vram_bytes(self) -> int:
        total = 0
        for m in [self.fc_q4, *self.layer.modules(), self.enorm, self.hnorm, self.head_norm]:
            if isinstance(m, self._ColibriQ4Linear):
                total += m.qweight.nbytes + m.scales.nbytes + m.zeros.nbytes
        for p in [*self.layer.parameters(), *self.enorm.parameters(), *self.hnorm.parameters(), *self.head_norm.parameters()]:
            total += p.numel() * p.element_size()
        return total

    def _raw_lm_head(self):
        h = self.engine.model.lm_head
        return h.q4_head if hasattr(h, "q4_head") else h

    def _score_draft(self, normed_last_1x1xd: torch.Tensor) -> tuple[int, float]:
        x2d = normed_last_1x1xd.reshape(1, -1)
        cand = self.draft_vocab.gpu_list
        scores = self.gk.indexed_vocab_scores(x2d, cand, self._raw_lm_head())[0]
        top_vals, top_pos = torch.topk(scores, k=min(64, scores.numel()), dim=-1)
        probs_top = torch.softmax(top_vals, dim=-1)
        best_tok = cand[0, top_pos[0]]
        tok_f, conf = torch.stack([best_tok.to(torch.float32), probs_top[0]]).tolist()
        return int(tok_f), float(conf)

    def _forward_pairs(self, h_post_seq: torch.Tensor, next_token_ids: list[int], start_pos: int) -> torch.Tensor:
        L = len(next_token_ids)
        tok_t = torch.tensor([next_token_ids], dtype=torch.long)
        emb = self.engine._embed_step_tokens(tok_t)
        en = self.enorm(emb)
        hn = self.hnorm(h_post_seq)
        x = self.fc_q4(torch.cat([en, hn], dim=-1))

        pos_1d = torch.arange(start_pos, start_pos + L, device=self.device, dtype=torch.long)
        pos_3d = pos_1d.view(1, 1, L).expand(3, 1, L)
        pos_emb = self.engine.model.model.rotary_emb(x, pos_3d)

        attn_mask = None
        if L > 1:
            total_kv = start_pos + L
            mask = torch.zeros((L, total_kv), device=self.device, dtype=self.dtype)
            causal_part = torch.triu(torch.full((L, L), float("-inf"), device=self.device, dtype=self.dtype), diagonal=1)
            mask[:, start_pos:] = causal_part
            attn_mask = mask.view(1, 1, L, total_kv)

        out = self.layer(
            x,
            attention_mask=attn_mask,
            position_ids=pos_1d.unsqueeze(0),
            past_key_values=self.kv_cache,
            use_cache=True,
            position_embeddings=pos_emb,
        )
        if isinstance(out, tuple):
            out = out[0]
        return self.head_norm(out)

    def commit_and_draft_first(self, h_post_seq: torch.Tensor, next_token_ids: list[int]) -> tuple[int, float, torch.Tensor]:
        t0 = time.perf_counter()
        cur_kv = self.kv_cache.get_seq_length()
        if cur_kv > self.committed_len:
            self.kv_cache.layers[0].crop(-(cur_kv - self.committed_len))

        L = len(next_token_ids)
        normed = self._forward_pairs(h_post_seq, next_token_ids, start_pos=self.committed_len)
        self.committed_len += L
        last_normed = normed[:, -1:, :]
        d1, p1 = self._score_draft(last_normed)
        if L <= 4:
            self.mtp_step_times_ms.append((time.perf_counter() - t0) * 1000.0)
        return d1, p1, last_normed

    def draft_more(self, prev_mtp_normed: torch.Tensor, prev_draft_tok: int) -> tuple[int, float, torch.Tensor]:
        t0 = time.perf_counter()
        cur_kv = self.kv_cache.get_seq_length()
        normed = self._forward_pairs(prev_mtp_normed, [prev_draft_tok], start_pos=cur_kv)
        last_normed = normed[:, -1:, :]
        d2, p2 = self._score_draft(last_normed)
        self.mtp_step_times_ms.append((time.perf_counter() - t0) * 1000.0)
        return d2, p2, last_normed


def install_colibri_anchor_ring_schedule(engine: Any, resident_set: set[int]) -> dict[str, Any]:
    """Installs `.coli_analysis.json` / `PriorityRanker` non-contiguous Anchor Residency + 3-slot queue."""
    from src.aircolibri_integrated import set_module_tensor_to_device

    for h in engine._hook_handles:
        h.remove()
    engine._hook_handles.clear()

    for r_idx in sorted(resident_set):
        engine._load_resident_layer(r_idx)

    num_layers = engine.config.num_hidden_layers
    stream_layers = [i for i in range(num_layers) if i not in resident_set]
    slot_queues: list[list[int]] = [[] for _ in range(engine.num_slots)]
    next_in_slot_queue: dict[int, int | None] = {}
    for s_idx in stream_layers:
        slot_queues[s_idx % engine.num_slots].append(s_idx)
    for q in slot_queues:
        for pos, l_id in enumerate(q):
            next_in_slot_queue[l_id] = q[pos + 1] if (pos + 1 < len(q)) else None

    tail_prefetched_slot = [False] * engine.num_slots
    layer_q4_bindings = []
    layer_param_names = []
    for layer_mod in engine.model.model.layers:
        qb, rp = engine._collect_layer_bindings(layer_mod)
        layer_q4_bindings.append(qb)
        layer_param_names.append(rp)

    engine.active_slot_per_layer = [None] * num_layers

    for idx, layer_module in enumerate(engine.model.model.layers):
        layer_module._air_layer_idx = idx

        def _pre_hook(module, _args):
            i = module._air_layer_idx
            if i in resident_set:
                return
            slot = engine.pipeline.acquire_slot_for_compute(i)
            engine.active_slot_per_layer[i] = slot
            for submod, q_k, s_k, z_k in layer_q4_bindings[i]:
                submod.bind_q4_tensors(slot.tensors[q_k], slot.tensors[s_k], slot.tensors[z_k])
            for k in layer_param_names[i]:
                if k in slot.tensors:
                    set_module_tensor_to_device(module, k, engine.device, value=slot.tensors[k], clear_cache=False)

        def _post_hook(module, _args, output):
            i = module._air_layer_idx
            if i in resident_set:
                return output
            slot = engine.active_slot_per_layer[i]
            engine.pipeline.release_slot_after_compute(slot)
            engine.active_slot_per_layer[i] = None

            sid = slot.slot_id
            nxt_l = next_in_slot_queue[i]
            if nxt_l is not None:
                pinned_t = engine.weight_manager.load_layer_to_pinned_slot(nxt_l, target_slot_idx=sid)
                next_slot = engine.pipeline.get_slot_for_loading(nxt_l)
                engine.pipeline.load_tensors_async(next_slot, pinned_t, pinned_pool=engine.weight_manager.pinned_pool)
            elif engine.tail_to_head_prefetch and engine._prefetch_next_step and slot_queues[sid]:
                first_l = slot_queues[sid][0]
                pinned_t = engine.weight_manager.load_layer_to_pinned_slot(first_l, target_slot_idx=sid)
                next_slot = engine.pipeline.get_slot_for_loading(first_l)
                engine.pipeline.load_tensors_async(next_slot, pinned_t, pinned_pool=engine.weight_manager.pinned_pool)
                tail_prefetched_slot[sid] = True
            return output

        engine._hook_handles.append(layer_module.register_forward_pre_hook(_pre_hook))
        engine._hook_handles.append(layer_module.register_forward_hook(_post_hook))

    def _anchor_ring_forward(
        step_tokens: torch.Tensor, past_key_values: Any = None, prefetch_next: bool = True, **model_kwargs
    ):
        engine._prefetch_next_step = prefetch_next
        inputs_embeds = engine._embed_step_tokens(step_tokens)
        for sid, q in enumerate(slot_queues):
            if not q:
                continue
            if tail_prefetched_slot[sid]:
                tail_prefetched_slot[sid] = False
                continue
            first_l = q[0]
            pinned_t = engine.weight_manager.load_layer_to_pinned_slot(first_l, target_slot_idx=sid)
            slot = engine.pipeline.get_slot_for_loading(first_l)
            engine.pipeline.load_tensors_async(slot, pinned_t, pinned_pool=engine.weight_manager.pinned_pool)
        B, S = step_tokens.shape
        mask_arg = (
            _DECODE_NO_MASK
            if (B == 1 and 1 <= S <= 4 and past_key_values is not None and past_key_values.get_seq_length() > 0)
            else None
        )
        outputs = engine.model(
            inputs_embeds=inputs_embeds,
            input_ids=None,
            attention_mask=mask_arg,
            past_key_values=past_key_values,
            use_cache=True,
            **model_kwargs,
        )
        return outputs.logits, outputs.past_key_values

    engine._anchor_ring_prepare = True
    engine._anchor_ring_forward = _anchor_ring_forward
    return {"resident_layers": sorted(resident_set), "streamed_layers": stream_layers, "slot_queues": slot_queues}


class ColibriKVPrefixSession:
    """Multi-turn KV + GatedDeltaNet + MTP session ledger with pinned Host-RAM parking (`0 MiB` idle VRAM)."""

    def __init__(self):
        self.fed: list[int] = []
        self.tainted: bool = False
        self.parked_backbone: dict[int, dict[str, torch.Tensor]] | None = None
        self.parked_mtp: dict[str, torch.Tensor] | None = None
        self.parked_mtp_committed_len: int = 0
        self.config = None
        self.park_ms: float = 0.0
        self.parked_bytes: int = 0
        self._pending_bridge_h: torch.Tensor | None = None

    def reset(self) -> None:
        self.fed = []
        self.tainted = False
        self.parked_backbone = None
        self.parked_mtp = None
        self.parked_mtp_committed_len = 0
        self.parked_bytes = 0
        self._pending_bridge_h = None

    def record_turn_end(
        self,
        config: Any,
        all_ids_full: list[int],
        past_key_values: Any,
        last_fed_h_post: torch.Tensor | None,
        engine: Any,
        mtp_head: ColibriNativeMTPHead | None,
        captured: dict[str, torch.Tensor],
    ) -> float:
        self.config = config
        all_ids = list(all_ids_full)
        past_len = int(past_key_values.get_seq_length())
        if past_len < len(all_ids):
            unfed = all_ids[past_len:]
            with torch.no_grad():
                st = torch.tensor([unfed], dtype=torch.long)
                _, past_key_values = forward_step_all_logits(
                    engine, st, past_key_values=past_key_values, prefetch_next=False
                )
                last_h = captured["post_norm"][:, -1:, :].clone()
        else:
            last_h = last_fed_h_post

        self.fed = all_ids
        self._pending_bridge_h = last_h.cpu().pin_memory() if last_h is not None else None

        t0 = time.perf_counter()
        self.parked_backbone = {}
        total_bytes = 0
        for l_idx, l_cache in enumerate(past_key_values.layers):
            entry: dict[str, torch.Tensor] = {}
            if getattr(l_cache, "keys", None) is not None and l_cache.keys.numel() > 0:
                entry["keys"] = l_cache.keys.detach().to("cpu", non_blocking=False).pin_memory()
                entry["values"] = l_cache.values.detach().to("cpu", non_blocking=False).pin_memory()
                total_bytes += entry["keys"].nbytes + entry["values"].nbytes
            if hasattr(l_cache, "conv_states") and l_cache.is_conv_states_initialized.get(0, False):
                entry["conv"] = l_cache.conv_states[0].detach().to("cpu", non_blocking=False).pin_memory()
                entry["rec"] = l_cache.recurrent_states[0].detach().to("cpu", non_blocking=False).pin_memory()
                total_bytes += entry["conv"].nbytes + entry["rec"].nbytes
            self.parked_backbone[l_idx] = entry

        if mtp_head is not None and mtp_head.kv_cache.get_seq_length() > 0:
            mtp_l0 = mtp_head.kv_cache.layers[0]
            self.parked_mtp = {
                "keys": mtp_l0.keys.detach().to("cpu", non_blocking=False).pin_memory(),
                "values": mtp_l0.values.detach().to("cpu", non_blocking=False).pin_memory(),
            }
            total_bytes += self.parked_mtp["keys"].nbytes + self.parked_mtp["values"].nbytes
            self.parked_mtp_committed_len = mtp_head.committed_len
            mtp_head.reset_cache()
        else:
            self.parked_mtp = None
            self.parked_mtp_committed_len = 0

        del past_key_values
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self.park_ms = (time.perf_counter() - t0) * 1000.0
        self.parked_bytes = total_bytes
        return self.park_ms

    def kv_prefix_reuse(self, new_prompt_ids: list[int]) -> int:
        L = len(self.fed)
        if self.tainted or L <= 0 or L >= len(new_prompt_ids) or self.parked_backbone is None:
            return 0
        if new_prompt_ids[:L] != self.fed:
            return 0
        return L

    def unpark_to_gpu(
        self,
        device: torch.device,
        mtp_head: ColibriNativeMTPHead | None,
        first_new_token: int,
    ) -> tuple[Any, float]:
        from transformers.cache_utils import DynamicCache

        t0 = time.perf_counter()
        past = DynamicCache(config=self.config)
        for l_idx, entry in self.parked_backbone.items():
            l_cache = past.layers[l_idx]
            if "keys" in entry:
                k_gpu = entry["keys"].to(device, non_blocking=True)
                v_gpu = entry["values"].to(device, non_blocking=True)
                l_cache.update(k_gpu, v_gpu)
            if "conv" in entry:
                c_gpu = entry["conv"].to(device, non_blocking=True)
                r_gpu = entry["rec"].to(device, non_blocking=True)
                l_cache.lazy_initialization(conv_states=c_gpu, state_idx=0, conv_kernel_size=c_gpu.shape[-1])
                l_cache.conv_states[0].copy_(c_gpu)
                l_cache.has_previous_state[0] = True
                l_cache.update_recurrent_state(r_gpu, state_idx=0)

        if mtp_head is not None and self.parked_mtp is not None and self._pending_bridge_h is not None:
            mtp_head.reset_cache()
            mk = self.parked_mtp["keys"].to(device, non_blocking=True)
            mv = self.parked_mtp["values"].to(device, non_blocking=True)
            mtp_head.kv_cache.layers[0].update(mk, mv)
            mtp_head.committed_len = self.parked_mtp_committed_len
            with torch.no_grad():
                h_bridge = self._pending_bridge_h.to(device, non_blocking=True)
                mtp_head._forward_pairs(h_bridge, [first_new_token], start_pos=mtp_head.committed_len)
                mtp_head.committed_len += 1

        torch.cuda.synchronize()
        return past, (time.perf_counter() - t0) * 1000.0


class _IncrementalThinkSplitter:
    """UTF-8 & tag-boundary-safe splitter emitting `("thinking"|"answer", text_delta)`."""

    CLOSE_TAG = "</think>"

    def __init__(self, start_in_thinking: bool):
        self.phase = "thinking" if start_in_thinking else "answer"
        self.buf = ""
        self._strip_leading_newlines = False

    def push(self, new_text: str, final: bool = False) -> list[tuple[str, str]]:
        if not new_text and not final:
            return []
        self.buf += new_text
        out: list[tuple[str, str]] = []

        if self.phase == "thinking":
            idx = self.buf.find(self.CLOSE_TAG)
            if idx != -1:
                before = self.buf[:idx]
                if before:
                    out.append(("thinking", before))
                self.buf = self.buf[idx + len(self.CLOSE_TAG) :]
                self.phase = "answer"
                self._strip_leading_newlines = True
            else:
                if final:
                    if self.buf:
                        out.append(("thinking", self.buf))
                        self.buf = ""
                    return out
                # Hold back up to 7 chars if suffix matches a prefix of `</think>`
                hold = 0
                max_k = min(len(self.CLOSE_TAG) - 1, len(self.buf))
                for k in range(max_k, 0, -1):
                    if self.buf.endswith(self.CLOSE_TAG[:k]):
                        hold = k
                        break
                emit_n = len(self.buf) - hold
                if emit_n > 0:
                    out.append(("thinking", self.buf[:emit_n]))
                    self.buf = self.buf[emit_n:]
                return out

        if self.phase == "answer" and self.buf:
            if self._strip_leading_newlines:
                stripped = self.buf.lstrip("\r\n")
                if not stripped and not final:
                    self.buf = ""
                    return out
                self.buf = stripped
                self._strip_leading_newlines = False
            if self.buf:
                out.append(("answer", self.buf))
                self.buf = ""
        return out


MiB = 1024 ** 2
GiB = 1024 ** 3
VRAM_MODES = ("normal", "turbo")


class GenerationCancelled(RuntimeError):
    """Raised inside a generation when its cancel check fires (client gone, stop request, preemption)."""

    def __init__(self, phase: str, reason: str = "cancelled"):
        super().__init__(f"generation cancelled during {phase}: {reason}")
        self.phase = phase
        self.reason = reason


def _align(value: int, alignment: int = 2 * MiB) -> int:
    return ((int(value) + alignment - 1) // alignment) * alignment


def q4_bytes_for_shape(shape: tuple[int, ...], group_size: int = 64) -> int:
    """VRAM of one W4A16 matrix: packed nibbles + BF16 scales and zeros per group."""
    out_f, in_f = int(shape[0]), int(shape[1])
    return out_f * in_f // 2 + 2 * out_f * (in_f // group_size) * 2


def estimate_layer_vram_bytes(layer_shapes: dict[str, tuple[int, ...]], group_size: int = 64) -> int:
    """Resident VRAM of one decoder layer in the engine's Q4 format (from tensor shapes only)."""
    from src.quantization_q4 import is_q4_quantizable

    total = 0
    for name, shape in layer_shapes.items():
        short = name.split(".", 3)[-1] if name.startswith("model.layers.") else name
        if is_q4_quantizable(short, tuple(shape), group_size=group_size):
            total += q4_bytes_for_shape(shape, group_size)
        else:
            n = 1
            for s in shape:
                n *= int(s)
            total += 2 * n
    return total


@dataclass
class VRAMPlan:
    """How the engine spends the VRAM budget for one run."""

    mode: str
    total_bytes: int
    free_bytes: int
    reserve_bytes: int
    budget_bytes: int
    layers_total: int
    resident_layers: int
    ring_slots: int
    slot_bytes: int
    full_residency: bool
    enable_mtp: bool
    prefill_chunk_tokens: int
    max_context_tokens: int
    weights_bytes: int

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("total_bytes", "free_bytes", "reserve_bytes", "budget_bytes", "slot_bytes", "weights_bytes"):
            d[k.replace("_bytes", "_mib")] = round(d.pop(k) / MiB, 1)
        return d


def plan_vram(
    mode: str,
    total_bytes: int,
    free_bytes: int,
    layer_bytes: list[int],
    lm_head_bytes: int,
    mtp_bytes: int,
    kv_bytes_per_token: int,
    recurrent_state_bytes: int,
    hidden_size: int,
    intermediate_size: int,
    max_context_tokens: int | None = None,
    prefill_chunk_tokens: int | None = None,
    ring_slots: int = 3,
) -> VRAMPlan:
    """Splits the *available* VRAM between weights and a guaranteed working reserve.

    ``normal`` takes almost all free VRAM but keeps a safety margin for the OS/WDDM, the KV and
    GatedDeltaNet states, the chunked prefill workspace and CUDA library workspaces, so a long
    prompt cannot run the card out of memory. ``turbo`` keeps only a minimal margin.

    When the weights do not fit, the remaining budget is cut into ``ring_slots`` equal slots that
    rotate roles (computing -> clean -> receiving the next weights) so the next layer is already
    in VRAM when the current one finishes; as many layers as fit stay resident.
    """
    if mode not in VRAM_MODES:
        raise ValueError(
            f"vram_mode must be one of {VRAM_MODES} (got {mode!r}); fixed `4gb`/`8gb` profiles were removed"
        )
    low_vram = total_bytes <= 5 * GiB
    chunk = int(prefill_chunk_tokens or (256 if low_vram else 512))
    max_ctx = int(max_context_tokens or (4096 if low_vram else 8192))
    context_bytes = kv_bytes_per_token * max_ctx + 2 * recurrent_state_bytes
    prefill_bytes = _align(int(chunk * (4 * hidden_size + 3 * intermediate_size) * 2 * 1.5))
    logits_bytes = 4 * MiB
    if mode == "normal":
        guard = _align(max(160 * MiB, int(0.045 * total_bytes))) if low_vram else _align(max(384 * MiB, int(0.055 * total_bytes)))
        slack = 256 * MiB
    else:
        guard = 84 * MiB if low_vram else 100 * MiB
        slack = 128 * MiB
    reserve = guard + slack + _align(context_bytes) + prefill_bytes + logits_bytes
    budget = max(0, int(free_bytes) - reserve)

    n = len(layer_bytes)
    weights_all = sum(layer_bytes) + lm_head_bytes
    if budget >= weights_all + mtp_bytes:
        return VRAMPlan(mode, total_bytes, free_bytes, reserve, budget, n, n, 0, 0, True, mtp_bytes > 0,
                        chunk, max_ctx, weights_all + mtp_bytes)

    slot = max(layer_bytes) if layer_bytes else 0
    ring_slots = max(3, min(4, int(ring_slots)))
    base = lm_head_bytes + ring_slots * slot
    if budget < base:
        raise RuntimeError(
            f"Not enough free VRAM for mode={mode}: budget {budget / MiB:.0f} MiB < "
            f"{base / MiB:.0f} MiB (LM head + {ring_slots} ring slots). Close other GPU programs or use --vram-mode turbo."
        )
    enable_mtp = mtp_bytes > 0 and budget - base - mtp_bytes >= min(layer_bytes or [0]) * 4
    room = budget - base - (mtp_bytes if enable_mtp else 0)
    resident = 0
    used = 0
    for b in sorted(layer_bytes):
        if used + b > room:
            break
        used += b
        resident += 1
    resident = min(resident, n - 1)
    weights = lm_head_bytes + used + ring_slots * slot + (mtp_bytes if enable_mtp else 0)
    return VRAMPlan(mode, total_bytes, free_bytes, reserve, budget, n, resident, ring_slots, slot, False,
                    enable_mtp, chunk, max_ctx, weights)


def _model_shape_inventory(model_dir: Path, source: Any, num_layers: int) -> tuple[list[dict[str, tuple]], dict[str, tuple], dict[str, tuple]]:
    """Reads tensor shapes (headers only) of every layer, the globals and the MTP head."""
    from .model_source import read_safetensors_header

    layers: list[dict[str, tuple]] = []
    if source is not None:
        for i in range(num_layers):
            layers.append({k: source.tensor_shape(e) for k, e in source.layer_entries(i).items()})
        globals_ = {k: source.tensor_shape(e) for k, e in source.globals_entries().items()}
        mtp = {k: source.tensor_shape(e) for k, e in source.mtp_entries().items()}
        return layers, globals_, mtp

    def _shapes(path: Path) -> dict[str, tuple]:
        if not path.is_file():
            return {}
        hdr, _ = read_safetensors_header(path)
        return {k: tuple(v["shape"]) for k, v in hdr.items() if k != "__metadata__"}

    for i in range(num_layers):
        layers.append(_shapes(model_dir / f"model-{i:05d}.safetensors"))
    return layers, _shapes(model_dir / "model-globals.safetensors"), _shapes(model_dir / "model-mtp.safetensors")


def choose_resident_layers(count: int, num_layers: int, priority_summary: dict[str, list[int]] | None) -> set[int]:
    """Picks which layers stay resident when the model is streamed: Colibrì priority ranks first."""
    order: list[int] = []
    if priority_summary:
        for rank in ("rank_1_critical", "rank_2_medium", "rank_3_standard"):
            for i in priority_summary.get(rank, []):
                if 0 <= int(i) < num_layers and int(i) not in order:
                    order.append(int(i))
    for i in range(num_layers):
        if i not in order:
            order.append(i)
    return set(order[: max(0, min(count, num_layers))])


class AirColibriUnifiedEngine:
    """High-level self-contained unified engine for CLI `ask`, `chat`, and LAN `serve`."""

    def __init__(
        self,
        model_dir: str | Path | None = None,
        aircolibri_root: str | Path | None = None,
        vram_mode: str = "normal",
        enable_mtp: bool = True,
        enable_vocab_pruner: bool = False,
        max_drafts: int = 2,
        verbose: bool = True,
        max_context_tokens: int | None = None,
        prefill_chunk_tokens: int | None = None,
        allow_download: bool = False,
    ):
        self.root = find_aircolibri_root(aircolibri_root)
        from src.aircolibri_engine import PriorityRanker
        from src.aircolibri_integrated import AirColibriModel
        from src.catalog_bridge import ExistingCrossLayerCatalog
        from src.existing_hooks import ExistingForwardHookAdapter
        from src.lease_ack_bridge import ExistingRingAcknowledgementAdapter
        from src.upstream_bridge import NativeTierPolicy
        from .kernels import GrandUnifiedKernels
        from .model_source import (
            PreadSafetensorsReader,
            open_model_source,
            raw_text_config_dict,
            resolve_model_dir,
        )

        if vram_mode not in VRAM_MODES:
            raise ValueError(
                f"vram_mode must be one of {VRAM_MODES} (got {vram_mode!r}); fixed `4gb`/`8gb` profiles were removed"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required to run AirColibriUnifiedEngine.")

        self.engine = None
        self.pread_reader = None
        self.native_policy = None
        self.device = torch.device("cuda:0")
        self.gpu_name = torch.cuda.get_device_name(self.device)
        self.model_dir = resolve_model_dir(model_dir, self.root, allow_download=allow_download)
        self.model_id = self.model_dir.name
        self.model_source = open_model_source(self.model_dir)
        self.model_format = "hf" if self.model_source is not None else "colibri"
        cfg = raw_text_config_dict(self.model_dir)
        num_layers = int(cfg["num_hidden_layers"])

        free_b, total_b = torch.cuda.mem_get_info(self.device)
        self.total_vram_gib = round(total_b / GiB, 2)
        layers_sh, globals_sh, mtp_sh = _model_shape_inventory(self.model_dir, self.model_source, num_layers)
        layer_bytes = [estimate_layer_vram_bytes(sh) for sh in layers_sh]
        vocab, hidden = globals_sh.get("lm_head.weight", (cfg["vocab_size"], cfg["hidden_size"]))
        lm_head_bytes = q4_bytes_for_shape((vocab, hidden))
        mtp_bytes = int(estimate_layer_vram_bytes({k: v for k, v in mtp_sh.items() if k.startswith("mtp.layers.")}))
        if "mtp.fc.weight" in mtp_sh:
            mtp_bytes += q4_bytes_for_shape(mtp_sh["mtp.fc.weight"])
        has_mtp = "mtp.fc.weight" in mtp_sh
        full_attn = sum(1 for t in cfg.get("layer_types", []) if t == "full_attention")
        kv_per_token = full_attn * 2 * int(cfg.get("num_key_value_heads", 4)) * int(cfg.get("head_dim", 256)) * 2
        lin_layers = num_layers - full_attn
        rec_state = lin_layers * (
            int(cfg.get("linear_num_value_heads", 32)) * int(cfg.get("linear_key_head_dim", 128))
            * int(cfg.get("linear_value_head_dim", 128)) * 4
            + 2 * int(cfg.get("linear_num_key_heads", 16)) * int(cfg.get("linear_key_head_dim", 128)) * 4 * 2
        )
        self.plan = plan_vram(
            vram_mode,
            total_bytes=total_b,
            free_bytes=free_b,
            layer_bytes=layer_bytes,
            lm_head_bytes=lm_head_bytes,
            mtp_bytes=mtp_bytes if (enable_mtp and has_mtp) else 0,
            kv_bytes_per_token=kv_per_token,
            recurrent_state_bytes=rec_state,
            hidden_size=int(cfg["hidden_size"]),
            intermediate_size=int(cfg["intermediate_size"]),
            max_context_tokens=max_context_tokens,
            prefill_chunk_tokens=prefill_chunk_tokens,
        )
        self.vram_mode = vram_mode
        self.prefill_chunk_tokens = self.plan.prefill_chunk_tokens

        q4_cache = Path(os.environ.get("AIRCOLIBRI_CACHE_DIR") or (self.root / "data" / "q4-g64-cache"))
        if self.model_source is not None:
            use_cache_dir, use_manifest = str(q4_cache), None
        else:
            manifest = checkpoint_manifest_path(self.root)
            have = q4_cache.exists() and manifest.exists()
            use_cache_dir = str(q4_cache) if have else None
            use_manifest = str(manifest) if have else None

        dll_path = self.root / "data" / "native" / "colibri_tier.dll"
        self.has_native_dll = dll_path.exists() and os.name == "nt"
        self.pread_reader = PreadSafetensorsReader(max_handles=8)

        t_boot0 = time.perf_counter()
        if verbose:
            p = self.plan
            layout = (
                f"all {p.layers_total} layers resident"
                if p.full_residency
                else f"{p.resident_layers}/{p.layers_total} resident + {p.ring_slots}-slot ring ({p.slot_bytes / MiB:.0f} MiB each)"
            )
            print(
                f"[AirColibri] GPU: {self.gpu_name} ({self.total_vram_gib} GiB, free {free_b / GiB:.2f} GiB) | "
                f"Mode: {vram_mode} | Budget: {p.budget_bytes / GiB:.2f} GiB, reserve {p.reserve_bytes / GiB:.2f} GiB | "
                f"{layout} | Model: {self.model_id} ({self.model_format})",
                flush=True,
            )

        self.gk = GrandUnifiedKernels(self.device)
        common = dict(
            pipeline_mode="triple_ring_prefetch",
            weight_format="q4",
            quantize_lm_head=True,
            q4_group_size=64,
            q4_cache_dir=use_cache_dir,
            checkpoint_manifest=use_manifest,
            model_source=self.model_source,
            weight_reader=self.pread_reader,
        )
        if self.plan.full_residency:
            self.engine = AirColibriModel(str(self.model_dir), vram_resident_layers=num_layers, **common)
            self.anchor_schedule = {"resident_layers": list(range(num_layers)), "streamed_layers": []}
        else:
            prio = None
            if self.model_source is None:
                try:
                    prio = PriorityRanker(str(self.model_dir)).get_priority_summary()
                except Exception:
                    prio = None
            resident_set = choose_resident_layers(self.plan.resident_layers, num_layers, prio)
            max_ram_layers = max(4, min(num_layers, 16))
            if self.has_native_dll:
                try:
                    self.native_policy = NativeTierPolicy(
                        str(dll_path), num_layers=num_layers, capacity=max_ram_layers, mode="apply", decay_interval=16
                    )
                except Exception:
                    self.native_policy = None
            self.engine = AirColibriModel(
                str(self.model_dir),
                vram_resident_layers=0,
                max_ram_layers=max_ram_layers,
                cache_policy="lru",
                num_slots=self.plan.ring_slots,
                control_mode="off",
                native_policy=self.native_policy,
                **common,
            )
            self.engine.tail_to_head_prefetch = True
            self.anchor_schedule = install_colibri_anchor_ring_schedule(self.engine, resident_set)
            if self.model_source is None:
                catalog = ExistingCrossLayerCatalog(str(self.model_dir))
                self.engine.trace_adapter = ExistingForwardHookAdapter(self.engine, catalog).install()
                self.engine.ack_adapter = ExistingRingAcknowledgementAdapter(self.engine.pipeline).install()

        self.max_drafts = max(0, min(3, int(max_drafts)))
        self.snap_mgr = HybridSpecSnapshotManager(self.engine.config, self.device, max_drafts=max(2, self.max_drafts))
        self.captured = install_colibri_multirow_unifications(
            self.engine,
            self.gk,
            snap_mgr=self.snap_mgr,
            enable_vocab_pruner=enable_vocab_pruner,
        )
        self.enable_vocab_pruner = enable_vocab_pruner

        self.mtp_enabled = bool(enable_mtp and has_mtp and self.plan.enable_mtp and self.max_drafts >= 1)
        if self.mtp_enabled:
            self.draft_vocab = ColibriDraftVocab(self.engine.config.vocab_size, top_k=16000, device=self.device)
            seed_ids = self.engine.tokenizer(DEFAULT_DRAFT_SEED_CORPUS, add_special_tokens=False).input_ids
            self.draft_vocab.seed_from_corpus_tokens(seed_ids)
            self.mtp_head: ColibriNativeMTPHead | None = ColibriNativeMTPHead(self.engine, self.gk, self.draft_vocab)
            self.mtp_vram_mib = round(self.mtp_head.q4_vram_bytes() / MiB, 2)
        else:
            self.draft_vocab = None
            self.mtp_head = None
            self.mtp_vram_mib = 0.0

        self.session = ColibriKVPrefixSession()
        self._session_messages: list[dict[str, str]] = []
        self.last_metrics: dict[str, Any] = {}
        self.generations_started = 0
        self.generations_cancelled = 0

        # Short CUDA kernel warmup (S=1 & S=2) so the first user prompt runs at full speed.
        warm_ids = self.engine.tokenizer("Привет!", return_tensors="pt").input_ids[0].tolist()
        for _ in self._stream_token_ids(warm_ids, max_new_tokens=4, stop_on_eos=True, cached_prefix_len=0, past_kv=None):
            pass
        self._release_transient_state()

        self.boot_seconds = round(time.perf_counter() - t_boot0, 2)
        self.resident_vram_mib = round(torch.cuda.memory_allocated(self.device) / MiB, 2)
        if verbose:
            print(
                f"[AirColibri] Ready in {self.boot_seconds}s | Resident VRAM: {self.resident_vram_mib} MiB "
                f"(MTP Q4: {self.mtp_vram_mib} MiB, FastVocab: {self.enable_vocab_pruner})",
                flush=True,
            )

    @property
    def tokenizer(self):
        return self.engine.tokenizer

    # ------------------------------------------------------------------ state management

    def _release_transient_state(self) -> None:
        """Drops per-generation tensors (MTP cache, captured hidden states, snapshots) and returns VRAM."""
        if self.mtp_head is not None:
            self.mtp_head.reset_cache()
        self.snap_mgr.active_capture = False
        self.captured.pop("post_norm", None)
        gc.collect()
        torch.cuda.empty_cache()

    def _abort_cleanup(self) -> None:
        """Restores a clean engine after a cancelled or failed generation (no VRAM may stay behind)."""
        self.session.reset()
        self._session_messages.clear()
        eng = self.engine
        if eng is not None and getattr(eng, "active_slot_per_layer", None):
            for i, slot in enumerate(eng.active_slot_per_layer):
                if slot is not None:
                    try:
                        eng.pipeline.release_slot_after_compute(slot)
                    except Exception:
                        pass
                    eng.active_slot_per_layer[i] = None
        self._release_transient_state()

    def reset_session(self) -> None:
        self.session.reset()
        self._session_messages.clear()
        self._release_transient_state()

    def status_snapshot(self) -> dict[str, Any]:
        alloc_mib = round(torch.cuda.memory_allocated(self.device) / MiB, 2)
        return {
            "model_id": self.model_id,
            "model_dir": str(self.model_dir),
            "model_format": self.model_format,
            "gpu_name": self.gpu_name,
            "total_vram_gib": self.total_vram_gib,
            "vram_mode": self.vram_mode,
            "vram_plan": self.plan.as_dict(),
            "allocated_vram_mib": alloc_mib,
            "mtp_enabled": self.mtp_enabled,
            "mtp_vram_mib": self.mtp_vram_mib,
            "vocab_pruner_enabled": self.enable_vocab_pruner,
            "parked_session_tokens": len(self.session.fed),
            "parked_session_ram_mib": round(self.session.parked_bytes / MiB, 2),
            "generations_started": self.generations_started,
            "generations_cancelled": self.generations_cancelled,
            "last_metrics": self.last_metrics,
        }

    # ------------------------------------------------------------------ prompt preparation

    def _prepare_prompt_and_cache(
        self,
        messages: list[dict[str, str]],
        enable_thinking: bool,
        use_session_cache: bool,
    ) -> tuple[list[int], int, Any, float]:
        eos_id = self.tokenizer.eos_token_id
        norm_msgs = [{"role": str(m["role"]), "content": str(m.get("content", ""))} for m in messages]

        if (
            use_session_cache
            and len(norm_msgs) >= 3
            and norm_msgs[-1]["role"] == "user"
            and norm_msgs[:-1] == self._session_messages
            and len(self.session.fed) > 0
            and self.session.parked_backbone is not None
        ):
            lead = "" if self.session.fed[-1] == eos_id else "<|im_end|>"
            think_tail = "<think>\n" if enable_thinking else "<think>\n\n</think>\n\n"
            suffix_str = (
                f"{lead}\n<|im_start|>user\n{norm_msgs[-1]['content']}<|im_end|>\n"
                f"<|im_start|>assistant\n{think_tail}"
            )
            suffix_ids = self.tokenizer(suffix_str, add_special_tokens=False).input_ids
            full_ids = self.session.fed + suffix_ids
            reused = self.session.kv_prefix_reuse(full_ids)
            if reused > 0:
                past_kv, unpark_ms = self.session.unpark_to_gpu(
                    self.device, self.mtp_head, first_new_token=full_ids[reused]
                )
                return full_ids, reused, past_kv, unpark_ms

        if use_session_cache:
            self.session.reset()
        prompt_str = self.tokenizer.apply_chat_template(
            norm_msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        prompt_ids = self.tokenizer(prompt_str, return_tensors="pt").input_ids[0].tolist()
        return prompt_ids, 0, None, 0.0

    # ------------------------------------------------------------------ generation core

    @staticmethod
    def _make_cancel_checker(cancel_check: Callable[[], bool] | None) -> Callable[[str], None]:
        def _check(phase: str) -> None:
            if cancel_check is not None and cancel_check():
                raise GenerationCancelled(phase)

        return _check

    def _prefill(
        self,
        tail_ids: list[int],
        past_kv: Any,
        use_mtp: bool,
        check: Callable[[str], None],
        prefetch_next: bool,
    ) -> tuple[int, Any, torch.Tensor, tuple[int, float, torch.Tensor] | None]:
        """Feeds the prompt tail in chunks; only the last position is projected to the vocabulary.

        Returns ``(first_token, past, last_hidden[1,1,H], mtp_draft or None)``. The MTP head is
        committed chunk by chunk so its attention never sees the whole prompt at once either.
        """
        chunk = max(1, int(self.prefill_chunk_tokens))
        n = len(tail_ids)
        past = past_kv
        picked = -1
        last_h = None
        mtp_draft = None
        for a in range(0, n, chunk):
            check("prefill")
            b = min(n, a + chunk)
            is_last = b == n
            lg, past = forward_step_all_logits(
                self.engine,
                torch.tensor([tail_ids[a:b]], dtype=torch.long),
                past_key_values=past,
                prefetch_next=prefetch_next if is_last else True,
                logits_to_keep=1,
            )
            h_chunk = self.captured["post_norm"]
            if is_last:
                picked = int(lg[0, -1].argmax(dim=-1).item())
                last_h = h_chunk[:, -1:, :].clone()
            del lg
            if use_mtp:
                next_toks = tail_ids[a + 1 : b + 1] if not is_last else tail_ids[a + 1 :] + [picked]
                mtp_draft = self.mtp_head.commit_and_draft_first(h_chunk, next_toks)
            del h_chunk
            self.captured.pop("post_norm", None)
        return picked, past, last_h, mtp_draft

    def _stream_token_ids(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        stop_on_eos: bool,
        cached_prefix_len: int,
        past_kv: Any,
        cancel_check: Callable[[], bool] | None = None,
    ) -> Iterator[dict[str, Any]]:
        eos_id = self.tokenizer.eos_token_id
        check = self._make_cancel_checker(cancel_check)
        all_ids = list(prompt_ids)
        tail_ids = prompt_ids[cached_prefix_len:]
        assert len(tail_ids) >= 1
        past = None

        if not self.mtp_enabled or self.mtp_head is None or self.draft_vocab is None:
            with torch.no_grad():
                torch.cuda.reset_peak_memory_stats(self.device)
                torch.cuda.synchronize()
                t_pre0 = time.perf_counter()
                picked, past, last_fed_h, _ = self._prefill(
                    tail_ids, past_kv, use_mtp=False, check=check, prefetch_next=max_new_tokens > 1
                )
                all_ids.append(picked)
                torch.cuda.synchronize()
                prefill_ms = (time.perf_counter() - t_pre0) * 1000.0
                yield {"kind": "tokens", "new_ids": [picked]}

                generated_count = 1
                passes = 0
                torch.cuda.synchronize()
                t_dec0 = time.perf_counter()
                while generated_count < max_new_tokens:
                    if stop_on_eos and picked == eos_id:
                        break
                    check("decode")
                    lg, past = forward_step_all_logits(
                        self.engine,
                        torch.tensor([[picked]], dtype=torch.long),
                        past_key_values=past,
                        prefetch_next=(generated_count + 1 < max_new_tokens),
                    )
                    last_fed_h = self.captured["post_norm"][:, -1:, :].clone()
                    picked = int(lg[0, -1].argmax(dim=-1).item())
                    del lg
                    all_ids.append(picked)
                    generated_count += 1
                    passes += 1
                    yield {"kind": "tokens", "new_ids": [picked]}
                torch.cuda.synchronize()
                dec_s = time.perf_counter() - t_dec0

            yield {
                "kind": "final_state",
                "all_ids": all_ids,
                "past": past,
                "last_fed_h": last_fed_h,
                "prefill_ms": prefill_ms,
                "decode_s": dec_s,
                "decode_tokens": max(0, generated_count - 1),
                "backbone_passes": passes,
                "spec_gate": {"overall_accept_rate_pct": 0.0, "mtp_accept_rate_pct": 0.0},
            }
            return

        gate = ColibriSpecGate(max_drafts=self.max_drafts, force_off=False)
        self.draft_vocab.begin_request(prompt_ids, cached_prompt_tokens=cached_prefix_len)
        if cached_prefix_len == 0:
            self.mtp_head.reset_cache()

        with torch.no_grad():
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize()
            t_pre0 = time.perf_counter()
            picked, past, last_fed_h_post, mtp_draft = self._prefill(
                tail_ids, past_kv, use_mtp=True, check=check, prefetch_next=True
            )
            d1, p1, last_mtp_norm = mtp_draft
            all_ids.append(picked)
            self.draft_vocab.observe_picked_token(picked)
            torch.cuda.synchronize()
            prefill_ms = (time.perf_counter() - t_pre0) * 1000.0
            yield {"kind": "tokens", "new_ids": [picked]}

            generated_count = 1
            backbone_passes = 0
            torch.cuda.synchronize()
            t_dec0 = time.perf_counter()

            while generated_count < max_new_tokens:
                if stop_on_eos and picked == eos_id:
                    break
                check("decode")
                rem = max_new_tokens - generated_count

                lkp = colibri_spec_lookup(all_ids, max_drafts=min(self.max_drafts, rem))
                if len(lkp) >= 2 or (len(lkp) == 1 and p1 < 0.45):
                    src = ColibriSpecGate.SRC_LOOKUP
                    cand_drafts = lkp[: min(self.max_drafts, rem)]
                    cand_probs = [0.75] * len(cand_drafts)
                else:
                    src = ColibriSpecGate.SRC_MTP
                    cand_drafts = [d1]
                    cand_probs = [p1]
                    if rem >= 2 and gate.should_draft_second_mtp(p1):
                        t_m2 = time.perf_counter()
                        d2, p2, _ = self.mtp_head.draft_more(last_mtp_norm, d1)
                        gate.record_draft_cost(src, time.perf_counter() - t_m2)
                        cand_drafts.append(d2)
                        cand_probs.append(p2)

                k_use = min(rem - 1, gate.pick_depth(src, cand_probs)) if rem > 1 else 0
                drafts = cand_drafts[:k_use]
                probs_used = cand_probs[:k_use]

                batch_tokens = [picked] + drafts
                step_t = torch.tensor([batch_tokens], dtype=torch.long)
                self.snap_mgr.active_capture = k_use > 0
                t_f0 = time.perf_counter()
                try:
                    logits_b, past = forward_step_all_logits(
                        self.engine,
                        step_t,
                        past_key_values=past,
                        prefetch_next=(generated_count + 1 < max_new_tokens),
                    )
                finally:
                    self.snap_mgr.active_capture = False
                h_post_b = self.captured["post_norm"]

                row_preds = logits_b[0].argmax(dim=-1).tolist()
                del logits_b
                gate.record_forward(1 + k_use, time.perf_counter() - t_f0)
                backbone_passes += 1

                step_emitted: list[int] = []
                accepted = 0
                hit_eos = False
                for j in range(k_use):
                    if drafts[j] == row_preds[j]:
                        accepted += 1
                        all_ids.append(drafts[j])
                        step_emitted.append(drafts[j])
                        self.draft_vocab.observe_picked_token(drafts[j])
                        generated_count += 1
                        if stop_on_eos and drafts[j] == eos_id:
                            hit_eos = True
                            break
                    else:
                        break

                judged = accepted if (accepted == k_use or hit_eos) else (accepted + 1)
                if k_use > 0:
                    gate.record_result(src, judged=judged, accepted=accepted, probs=probs_used)

                self.snap_mgr.rollback_to_accepted(past, accepted_drafts=accepted, total_drafts=k_use)

                if hit_eos or generated_count >= max_new_tokens:
                    if accepted > 0:
                        self.mtp_head.commit_and_draft_first(h_post_b[:, :accepted, :].clone(), drafts[:accepted])
                    last_fed_h_post = h_post_b[:, accepted : accepted + 1, :].clone()
                    picked = all_ids[-1]
                    if step_emitted:
                        yield {"kind": "tokens", "new_ids": step_emitted}
                    break

                bonus_tok = row_preds[accepted]
                all_ids.append(bonus_tok)
                step_emitted.append(bonus_tok)
                self.draft_vocab.observe_picked_token(bonus_tok)
                generated_count += 1
                picked = bonus_tok
                last_fed_h_post = None
                yield {"kind": "tokens", "new_ids": step_emitted}

                mtp_tokens = drafts[:accepted] + [bonus_tok]
                if stop_on_eos and bonus_tok == eos_id:
                    self.mtp_head.commit_and_draft_first(h_post_b[:, : accepted + 1, :].clone(), mtp_tokens)
                    break

                if generated_count < max_new_tokens:
                    d1, p1, last_mtp_norm = self.mtp_head.commit_and_draft_first(
                        h_post_b[:, : accepted + 1, :].clone(), mtp_tokens
                    )
                else:
                    self.mtp_head.commit_and_draft_first(h_post_b[:, : accepted + 1, :].clone(), mtp_tokens)
                del h_post_b

            torch.cuda.synchronize()
            dec_s = time.perf_counter() - t_dec0

        yield {
            "kind": "final_state",
            "all_ids": all_ids,
            "past": past,
            "last_fed_h": last_fed_h_post,
            "prefill_ms": prefill_ms,
            "decode_s": dec_s,
            "decode_tokens": max(0, generated_count - 1),
            "backbone_passes": backbone_passes,
            "spec_gate": gate.summary(),
        }

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 1024,
        enable_thinking: bool = True,
        use_session_cache: bool = False,
        stop_on_eos: bool = True,
        cancel_check: Callable[[], bool] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Streams UTF-8-safe `thinking` and `answer` deltas, followed by a `done` telemetry packet.

        ``cancel_check`` is polled before every prefill chunk and every decode step; when it returns
        True the generation stops with ``GenerationCancelled``. On cancellation, errors and early
        close (``generator.close()``) every per-request tensor is released in ``finally``.
        """
        eos_id = self.tokenizer.eos_token_id
        check = self._make_cancel_checker(cancel_check)
        self.generations_started += 1
        completed = False
        inner = None
        final_info: dict[str, Any] = {}
        try:
            check("queue")
            prompt_ids, cached_len, past_kv, unpark_ms = self._prepare_prompt_and_cache(
                messages, enable_thinking=enable_thinking, use_session_cache=use_session_cache
            )

            splitter = _IncrementalThinkSplitter(start_in_thinking=enable_thinking)
            clean_gen_ids: list[int] = []
            raw_gen_ids: list[int] = []
            committed_tok_idx = 0
            thinking_acc: list[str] = []
            answer_acc: list[str] = []

            inner = self._stream_token_ids(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                stop_on_eos=stop_on_eos,
                cached_prefix_len=cached_len,
                past_kv=past_kv,
                cancel_check=cancel_check,
            )
            del past_kv
            for ev in inner:
                if ev["kind"] == "tokens":
                    new_ids = ev["new_ids"]
                    raw_gen_ids.extend(new_ids)
                    for tid in new_ids:
                        if tid != eos_id:
                            clean_gen_ids.append(tid)
                    if len(clean_gen_ids) > committed_tok_idx:
                        if committed_tok_idx >= 4:
                            ctx_ids = clean_gen_ids[committed_tok_idx - 4 : committed_tok_idx]
                            win_ids = clean_gen_ids[committed_tok_idx - 4 :]
                            ctx_s = self.tokenizer.decode(ctx_ids, skip_special_tokens=False)
                            win_s = self.tokenizer.decode(win_ids, skip_special_tokens=False)
                            delta_cand = win_s[len(ctx_s) :] if win_s.startswith(ctx_s) else ""
                        else:
                            prev_s = (
                                self.tokenizer.decode(clean_gen_ids[:committed_tok_idx], skip_special_tokens=False)
                                if committed_tok_idx > 0
                                else ""
                            )
                            win_s = self.tokenizer.decode(clean_gen_ids, skip_special_tokens=False)
                            delta_cand = win_s[len(prev_s) :] if win_s.startswith(prev_s) else ""

                        if delta_cand and not delta_cand.endswith("�"):
                            committed_tok_idx = len(clean_gen_ids)
                            for phase, chunk in splitter.push(delta_cand, final=False):
                                if phase == "thinking":
                                    thinking_acc.append(chunk)
                                else:
                                    answer_acc.append(chunk)
                                yield {"type": "delta", "phase": phase, "text": chunk}
                elif ev["kind"] == "final_state":
                    final_info = ev
            inner = None

            # Flush any remaining uncommitted tokens & buffered tail characters
            rem_raw = (
                self.tokenizer.decode(clean_gen_ids[committed_tok_idx:], skip_special_tokens=False)
                if len(clean_gen_ids) > committed_tok_idx
                else ""
            )
            for phase, chunk in splitter.push(rem_raw, final=True):
                if phase == "thinking":
                    thinking_acc.append(chunk)
                else:
                    answer_acc.append(chunk)
                yield {"type": "delta", "phase": phase, "text": chunk}

            peak_vram_mib = round(torch.cuda.max_memory_allocated(self.device) / MiB, 2)
            dec_tokens = int(final_info.get("decode_tokens", 0))
            dec_s = float(final_info.get("decode_s", 0.0))
            tok_per_s = round(dec_tokens / max(1e-6, dec_s), 2) if dec_tokens > 0 else 0.0
            passes = max(1, int(final_info.get("backbone_passes", 1)))
            answer_text = "".join(answer_acc)
            thinking_text = "".join(thinking_acc)

            park_ms = 0.0
            if use_session_cache and final_info.get("past") is not None:
                park_ms = self.session.record_turn_end(
                    config=self.engine.config,
                    all_ids_full=final_info["all_ids"],
                    past_key_values=final_info["past"],
                    last_fed_h_post=final_info.get("last_fed_h"),
                    engine=self.engine,
                    mtp_head=self.mtp_head,
                    captured=self.captured,
                )
                self._session_messages = [
                    *[{"role": str(m["role"]), "content": str(m.get("content", ""))} for m in messages],
                    {"role": "assistant", "content": answer_text},
                ]
            final_info.pop("past", None)
            final_info.pop("last_fed_h", None)
            if not use_session_cache:
                self._release_transient_state()
            else:
                gc.collect()
                torch.cuda.empty_cache()

            stopped_by_eos = bool(raw_gen_ids and raw_gen_ids[-1] == eos_id)
            metrics = {
                "prompt_tokens": len(prompt_ids),
                "cached_prefix_tokens": cached_len,
                "generated_tokens": len(raw_gen_ids),
                "stopped_by_eos": stopped_by_eos,
                "finish_reason": "stop" if stopped_by_eos else "length",
                "unpark_ms": round(unpark_ms, 2),
                "prefill_ms": round(float(final_info.get("prefill_ms", 0.0)), 2),
                "prefill_chunk_tokens": self.prefill_chunk_tokens,
                "decode_s": round(dec_s, 3),
                "decode_tok_per_s": tok_per_s,
                "avg_tokens_per_backbone_pass": round(dec_tokens / passes, 3) if dec_tokens > 0 else 1.0,
                "peak_vram_mib": peak_vram_mib,
                "park_to_ram_ms": round(park_ms, 2),
                "parked_session_ram_mib": round(self.session.parked_bytes / MiB, 2),
                "spec_gate": final_info.get("spec_gate", {}),
            }
            self.last_metrics = metrics
            completed = True
            yield {
                "type": "done",
                "thinking_text": thinking_text,
                "answer_text": answer_text,
                "metrics": metrics,
            }
        except BaseException as exc:
            if isinstance(exc, GenerationCancelled):
                self.generations_cancelled += 1
            # The traceback keeps the finished generator frames alive, and with them the partial
            # KV cache / GatedDeltaNet states of the aborted request. Drop those frame locals now.
            traceback.clear_frames(exc.__traceback__)
            raise
        finally:
            if inner is not None:
                inner.close()
                inner = None
            final_info.clear()
            if not completed:
                self._abort_cleanup()

    def close(self) -> None:
        if getattr(self, "session", None) is not None:
            self.reset_session()
        if self.pread_reader is not None:
            try:
                self.pread_reader.close()
            except Exception:
                pass
            self.pread_reader = None
        if getattr(self, "engine", None) is not None:
            try:
                self.engine.close()
            except Exception:
                pass
            self.engine = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> "AirColibriUnifiedEngine":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
