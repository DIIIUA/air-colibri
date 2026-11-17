"""Experimental Thought Buffer & Initial-Probability Anchor (`ColibriThoughtBuffer`).

Retains a task-derived probability field and state across generation. The
mechanisms below do not by themselves establish improved reasoning quality:
1. **Initial Semantic Probability Field (`Phi_0`)**: Captures the contrastive log-probability
   field `log P(w | Task) - log P(w | Neutral)` across start-of-thought and mid-clause probes
   at `t = 0` so syntactic starter tokens cancel out (`~0`) while task-grounded concepts,
   constraints, entities, and deductions retain positive probability weight throughout decode.
2. **Entropy-Gated Participation on Syntactic Support**: At step `t`, `Phi_buf` participates
   only inside the grammatically valid candidate set `z_t >= max(z_t) - syntax_margin` and
   scales with current step entropy `H(P_t)` — leaving deterministic syntax/math untouched
   while steering uncertain semantic forks back toward the initial meaning.
3. **Un-corrupted Layer State Snapshot (`KV_0, Conv_0, Rec_0`) & Non-Decaying `S_0` Anchor**:
   Snapshots Full-Attention KV lengths (`0 MiB` copy via `crop`) and GatedDeltaNet recurrent
   matrices (`S_0`), keeping an un-erased `S_0` readout channel alive (`S_t <- (1 - lam) S_t + lam S_0`)
   and enabling millisecond rollback of noisy intermediate reasoning waves (`ThoughtWaveRunner`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import contextmanager, nullcontext
import copy
import hashlib
import time
from typing import Any

import torch


@contextmanager
def _auxiliary_forwards(u_eng: Any):
    """Isolate probe routing and UI capture state from the committed task.

    This does not roll back cache tensors: probes that borrow a task cache must
    also restore their own snapshot in a finally block.
    """
    engine = u_eng.engine
    warmup = getattr(engine, "probability_warmup", None)
    missing = object()
    fields = ("_last_prompt_opened_think", "_last_prompt_chatml_think")
    metadata = {key: getattr(u_eng, key, missing) for key in fields}
    prefetch = getattr(engine, "_prefetch_next_step", missing)
    captured = getattr(u_eng, "captured", None)
    saved = dict(captured) if captured is not None else None
    try:
        with warmup.auxiliary_forwards() if warmup is not None else nullcontext():
            yield
    finally:
        if saved is not None:
            captured.clear()
            captured.update(saved)
        for key, value in metadata.items():
            if value is missing:
                u_eng.__dict__.pop(key, None)
            else:
                setattr(u_eng, key, value)
        if prefetch is not missing:
            engine._prefetch_next_step = prefetch


@dataclass
class CacheStateSnapshot:
    """Lightweight GPU snapshot of Full-Attention KV lengths + hybrid recurrent/conv states."""

    kv_lengths: dict[int, int] = field(default_factory=dict)
    conv_states: dict[int, torch.Tensor] = field(default_factory=dict)
    rec_states: dict[int, torch.Tensor] = field(default_factory=dict)
    layer_states: list[dict[str, Any]] = field(default_factory=list)
    append_only_kv: set[int] = field(default_factory=set)
    vram_bytes: int = 0


def capture_cache_snapshot(past_key_values: Any) -> CacheStateSnapshot:
    """Preserve window contents, positions, initialization flags and linear states.

    Only the exact DynamicLayer implementation is known to append by replacing
    tensors; retaining its old KV references needs no copy. Sliding and recurrent
    layers need independent copies, including metadata, even when shapes do not
    change. Unknown top-level state formats fail before a probe can mutate them.
    """
    from transformers.cache_utils import DynamicLayer

    snap = CacheStateSnapshot()
    if past_key_values is None:
        return snap
    if not hasattr(past_key_values, "layers"):
        raise NotImplementedError("Thought-buffer probes require a supported layered cache; this state format cannot be rolled back")

    def tensor_bytes(value):
        if isinstance(value, torch.Tensor):
            return value.nbytes
        if isinstance(value, dict):
            return sum(tensor_bytes(v) for v in value.values())
        if isinstance(value, (tuple, list)):
            return sum(tensor_bytes(v) for v in value)
        return 0

    total_b = 0
    for l_idx, l_cache in enumerate(past_key_values.layers):
        if not (hasattr(l_cache, "keys") or hasattr(l_cache, "conv_states")):
            raise NotImplementedError(f"Unsupported thought-buffer cache layer: {type(l_cache).__name__}")
        keys = getattr(l_cache, "keys", None)
        if keys is not None and keys.numel() > 0:
            snap.kv_lengths[l_idx] = int(keys.shape[-2])
        memo = {}
        if type(l_cache) is DynamicLayer:
            snap.append_only_kv.add(l_idx)
            memo = {id(t): t for t in (keys, getattr(l_cache, "values", None)) if t is not None}
        state = copy.deepcopy(vars(l_cache), memo)
        snap.layer_states.append(state)
        total_b += tensor_bytes({k: v for k, v in state.items()
                                 if l_idx not in snap.append_only_kv or k not in ("keys", "values")})
        for name, dest in (("conv_states", snap.conv_states), ("recurrent_states", snap.rec_states)):
            states = state.get(name)
            value = states.get(0) if isinstance(states, dict) else (states[0] if states else None)
            if value is not None:
                dest[l_idx] = value
    snap.vram_bytes = total_b
    return snap


def restore_cache_snapshot(past_key_values: Any, snap: CacheStateSnapshot) -> None:
    """Restore complete layer state, including saturated sliding windows."""
    if past_key_values is None:
        return
    if not hasattr(past_key_values, "layers") or len(past_key_values.layers) < len(snap.layer_states):
        raise NotImplementedError("Cannot restore this thought-buffer state format")
    for l_idx, state in enumerate(snap.layer_states):
        memo = {}
        if l_idx in snap.append_only_kv:
            memo = {id(state[k]): state[k] for k in ("keys", "values") if state.get(k) is not None}
        restored = copy.deepcopy(state, memo)
        vars(past_key_values.layers[l_idx]).clear()
        vars(past_key_values.layers[l_idx]).update(restored)
    del past_key_values.layers[len(snap.layer_states):]


def blend_recurrent_anchor(past_key_values: Any, snap: CacheStateSnapshot, lam: float) -> None:
    """Restores non-decaying initial state participation `S_t <- (1 - lam) * S_t + lam * S_0` in GatedDeltaNet."""
    if lam <= 0.0 or past_key_values is None or not snap.rec_states:
        return
    w = float(min(max(lam, 0.0), 1.0))
    for l_idx, r0 in snap.rec_states.items():
        l_cache = past_key_values.layers[l_idx]
        l_cache.recurrent_states[0].lerp_(r0, w)


def _control_token_ids(tokenizer: Any) -> list[int]:
    ids: set[int] = set()
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int):
        ids.add(eos)
    elif isinstance(eos, (list, tuple, set)):
        ids.update(int(x) for x in eos if x is not None)
    for attr in ("bos_token_id", "pad_token_id"):
        v = getattr(tokenizer, attr, None)
        if isinstance(v, int):
            ids.add(v)
    for tok_str in ("<think>", "</think>", "<|im_start|>", "<|im_end|>", "<|endoftext|>"):
        try:
            enc = tokenizer.encode(tok_str, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(int(enc[0]))
        except Exception:
            pass
    return sorted(ids)


@dataclass
class AnchorStepTelemetry:
    """Diagnostic counters proving initial-probability participation and drift prevention."""

    total_rows_scored: int = 0
    fork_rows_gated: int = 0
    argmax_corrections: int = 0
    recurrent_reanchors: int = 0
    snapshot_setup_ms: float = 0.0
    entropy_sum: float = 0.0
    initial_rows_active: int = 0
    initial_logit_l1: float = 0.0
    phi0_chosen_early: list[float] = field(default_factory=list)
    phi0_chosen_late: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        n = max(1, self.total_rows_scored)
        e_early = sum(self.phi0_chosen_early) / max(1, len(self.phi0_chosen_early))
        e_late = sum(self.phi0_chosen_late) / max(1, len(self.phi0_chosen_late))
        return {
            "total_rows_scored": self.total_rows_scored,
            "fork_rows_gated": self.fork_rows_gated,
            "argmax_corrections": self.argmax_corrections,
            "recurrent_reanchors": self.recurrent_reanchors,
            "snapshot_setup_ms": round(self.snapshot_setup_ms, 2),
            "mean_step_entropy": round(self.entropy_sum / n, 4),
            "initial_rows_active": self.initial_rows_active,
            "initial_logit_l1": round(self.initial_logit_l1, 6),
            "mean_phi0_early": round(e_early, 4),
            "mean_phi0_late": round(e_late, 4),
        }


class InitialProbabilityAnchor:
    """Holds `(S_0, KV_0)` and contrastive initial log-probabilities `Phi_0` across generation."""

    def __init__(
        self,
        *,
        alpha: float = 0.42,
        syntax_margin: float = 5.5,
        entropy_low: float = 0.25,
        entropy_high: float = 1.85,
        rec_anchor_lambda: float = 0.06,
        rec_reanchor_every: int = 16,
        phi_clip_neg: float = -2.0,
        phi_clip_pos: float = 4.0,
        use_mid_clause_probe: bool = True,
        initial_weight: float = 0.75,
    ):
        self.alpha = float(alpha)
        self.syntax_margin = float(syntax_margin)
        self.entropy_low = float(entropy_low)
        self.entropy_high = float(entropy_high)
        self.rec_anchor_lambda = float(rec_anchor_lambda)
        self.rec_reanchor_every = max(1, int(rec_reanchor_every))
        self.phi_clip_neg = float(phi_clip_neg)
        self.phi_clip_pos = float(phi_clip_pos)
        self.use_mid_clause_probe = bool(use_mid_clause_probe)
        if not 0.0 < initial_weight <= 1.0:
            raise ValueError("initial_weight must retain a nonzero initial channel")
        self.initial_weight = float(initial_weight)

        self.snap0: CacheStateSnapshot | None = None
        self.phi0: torch.Tensor | None = None
        self.phi_buf: torch.Tensor | None = None
        self.telemetry = AnchorStepTelemetry()
        self._steps_since_reanchor = 0

    def initial_fingerprint(self) -> str | None:
        if self.phi0 is None:
            return None
        data = self.phi0.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def _ensure_neutral_baseline(u_eng: Any) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        with _auxiliary_forwards(u_eng):
            return InitialProbabilityAnchor._build_neutral_baseline(u_eng)

    @staticmethod
    def _build_neutral_baseline(u_eng: Any) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        cached = getattr(u_eng, "_neutral_logprob_cache", None)
        if cached is not None:
            return cached

        from .unified_runtime import forward_step_all_logits

        tok = u_eng.tokenizer
        probe_ids = tok.encode("\nИтог: ", add_special_tokens=False) or [tok.eos_token_id or 0]
        neutral_msgs = [{"role": "user", "content": "."}]
        n_ids, _, _, _ = u_eng._prepare_prompt_and_cache(
            neutral_msgs, enable_thinking=False, use_session_cache=False
        )
        with torch.no_grad():
            if u_eng.fast_qwen:
                lg0, past_n = forward_step_all_logits(
                    u_eng.engine,
                    torch.tensor([n_ids], dtype=torch.long),
                    past_key_values=None,
                    prefetch_next=False,
                    logits_to_keep=1,
                )
                lp0 = torch.log_softmax(lg0[0, -1].float(), dim=-1)
                lg1, past_n = forward_step_all_logits(
                    u_eng.engine,
                    torch.tensor([probe_ids], dtype=torch.long),
                    past_key_values=past_n,
                    prefetch_next=False,
                    logits_to_keep=1,
                )
                lp1 = torch.log_softmax(lg1[0, -1].float(), dim=-1)
                u_eng.captured.pop("post_norm", None)
            else:
                lg0, past_n = u_eng.engine._forward_step(
                    torch.tensor([n_ids], dtype=torch.long),
                    past_key_values=None,
                    prefetch_next=False,
                    cache_position=torch.arange(0, len(n_ids), dtype=torch.long),
                )
                lp0 = torch.log_softmax(lg0[0].float(), dim=-1)
                lg1, past_n = u_eng.engine._forward_step(
                    torch.tensor([probe_ids], dtype=torch.long),
                    past_key_values=past_n,
                    prefetch_next=False,
                    cache_position=torch.arange(len(n_ids), len(n_ids) + len(probe_ids), dtype=torch.long),
                )
                lp1 = torch.log_softmax(lg1[0].float(), dim=-1)
            del past_n
        out = (lp0.to(torch.bfloat16), lp1.to(torch.bfloat16), probe_ids)
        u_eng._neutral_logprob_cache = out
        return out

    def on_prefill_end(self, u_eng: Any, past_key_values: Any, logits_last: torch.Tensor) -> torch.Tensor:
        """Establish Phi_0 once; later input updates only the working field."""
        with _auxiliary_forwards(u_eng):
            return self._on_prefill_end(u_eng, past_key_values, logits_last)

    def _on_prefill_end(self, u_eng: Any, past_key_values: Any, logits_last: torch.Tensor) -> torch.Tensor:
        from .unified_runtime import _cache_seq_len, forward_step_all_logits

        t0 = time.perf_counter()
        lp_n0, lp_n1, probe_ids = self._ensure_neutral_baseline(u_eng)
        probe_snapshot = capture_cache_snapshot(past_key_values)

        z0 = logits_last[0, -1].float() if logits_last.ndim == 3 else logits_last[0].float()
        lp_t0 = torch.log_softmax(z0, dim=-1)
        pmi = lp_t0 - lp_n0.to(device=z0.device, dtype=torch.float32)

        if self.use_mid_clause_probe and past_key_values is not None and hasattr(past_key_values, "layers"):
            cur_len = _cache_seq_len(past_key_values)
            try:
                if u_eng.fast_qwen:
                    lg_p, _ = forward_step_all_logits(
                        u_eng.engine,
                        torch.tensor([probe_ids], dtype=torch.long),
                        past_key_values=past_key_values,
                        prefetch_next=True,
                        logits_to_keep=1,
                    )
                    z_p = lg_p[0, -1].float()
                else:
                    lg_p, _ = u_eng.engine._forward_step(
                        torch.tensor([probe_ids], dtype=torch.long),
                        past_key_values=past_key_values,
                        prefetch_next=True,
                        cache_position=torch.arange(cur_len, cur_len + len(probe_ids), dtype=torch.long),
                    )
                    z_p = lg_p[0].float()
            finally:
                restore_cache_snapshot(past_key_values, probe_snapshot)
            lp_tp = torch.log_softmax(z_p, dim=-1)
            pmi_p = lp_tp - lp_n1.to(device=z0.device, dtype=torch.float32)
            pmi = torch.maximum(pmi, pmi_p)

        ctrl_ids = _control_token_ids(u_eng.tokenizer)
        if ctrl_ids:
            valid_ctrl = [i for i in ctrl_ids if 0 <= i < pmi.numel()]
            if valid_ctrl:
                pmi[torch.tensor(valid_ctrl, device=pmi.device, dtype=torch.long)] = 0.0

        pmi = pmi.clamp(self.phi_clip_neg, self.phi_clip_pos).to(torch.bfloat16)
        if self.snap0 is None:
            self.snap0 = probe_snapshot
        if self.phi0 is None:
            self.phi0 = pmi.clone()
            self.phi_buf = pmi.clone()
        else:
            self.phi0 = self.phi0.to(device=pmi.device)
            if self.phi_buf is not None:
                self.phi_buf = self.phi_buf.to(device=pmi.device)
            self._merge_working_field(pmi, weight=0.35)
        self._steps_since_reanchor = 0
        if torch.cuda.is_available() and z0.is_cuda:
            torch.cuda.synchronize(z0.device)
        self.telemetry.snapshot_setup_ms = (time.perf_counter() - t0) * 1000.0
        return self.transform_logits(logits_last, past_key_values, step_index=0)

    def layer_conclusion_into_buffer(self, step_logits_last: torch.Tensor, lp_neutral: torch.Tensor, weight: float = 0.35) -> None:
        """Layers a verified semantic boundary distribution onto `Phi_buf` without erasing `Phi_0`."""
        if self.phi0 is None:
            return
        z = step_logits_last[0, -1].float() if step_logits_last.ndim == 3 else step_logits_last[0].float()
        pmi_step = (torch.log_softmax(z, dim=-1) - lp_neutral.to(z.device, torch.float32)).clamp(
            self.phi_clip_neg, self.phi_clip_pos
        )
        self._merge_working_field(pmi_step, weight)

    def _merge_working_field(self, field: torch.Tensor, weight: float) -> None:
        base = self.phi_buf.float() if self.phi_buf is not None else self.phi0.float()
        combined = (base + float(weight) * field.float()).clamp(self.phi_clip_neg, self.phi_clip_pos)
        # Never allow `Phi_0` positive support to be erased by intermediate drift
        combined = torch.where(self.phi0.float() > 0.5, torch.maximum(combined, 0.75 * self.phi0.float()), combined)
        self.phi_buf = combined.to(torch.bfloat16)

    def transform_logits(self, logits: torch.Tensor, past_key_values: Any, step_index: int) -> torch.Tensor:
        """Applies entropy-gated `Phi_buf` participation on syntactically admissible candidates + `S_0` retention."""
        if self.phi_buf is None or self.alpha <= 0.0:
            return logits

        is_3d = logits.ndim == 3
        z2d = logits[0].float() if is_3d else logits.float()  # [M, V]
        M, V = z2d.shape
        initial = self.phi0[:V].float() if self.phi0 is not None else self.phi_buf[:V].float()
        # A fixed original channel survives even if the working field is replaced.
        phi = self.initial_weight * initial + (1.0 - self.initial_weight) * self.phi_buf[:V].float()

        probs = torch.softmax(z2d, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1, keepdim=True)  # [M, 1]
        gate = ((entropy - self.entropy_low) / max(1e-6, self.entropy_high - self.entropy_low)).clamp(0.0, 1.0)

        z_max = z2d.max(dim=-1, keepdim=True).values
        syntax_mask = (z2d >= (z_max - self.syntax_margin)).to(torch.float32)

        raw_top = z2d.argmax(dim=-1)
        anchored = z2d + (self.alpha * gate) * (syntax_mask * phi.unsqueeze(0))
        new_top = anchored.argmax(dim=-1)
        initial_delta = (self.alpha * self.initial_weight * gate) * (syntax_mask * initial.unsqueeze(0))
        self.telemetry.initial_rows_active += int((initial_delta.abs().sum(-1) > 0).sum().item())
        self.telemetry.initial_logit_l1 += float(initial_delta.abs().sum().item())

        self.telemetry.total_rows_scored += int(M)
        self.telemetry.fork_rows_gated += int((gate.squeeze(-1) > 0.0).sum().item())
        self.telemetry.argmax_corrections += int((raw_top != new_top).sum().item())
        self.telemetry.entropy_sum += float(entropy.sum().item())

        if self.phi0 is not None:
            chosen_phi = float(self.phi0[int(new_top[0].item())].item())
            if step_index < 32:
                self.telemetry.phi0_chosen_early.append(chosen_phi)
            else:
                self.telemetry.phi0_chosen_late.append(chosen_phi)

        out = anchored.to(dtype=logits.dtype)
        return out.unsqueeze(0) if is_3d else out

    def on_step_committed(self, past_key_values: Any, committed_tokens: int = 1) -> None:
        """Maintains non-decaying `S_0` participation on committed state after any speculative rollback."""
        if self.snap0 is not None and self.snap0.rec_states and self.rec_anchor_lambda > 0.0 and committed_tokens > 0:
            self._steps_since_reanchor += int(committed_tokens)
            if self._steps_since_reanchor >= self.rec_reanchor_every:
                blend_recurrent_anchor(past_key_values, self.snap0, self.rec_anchor_lambda)
                self._steps_since_reanchor = 0
                self.telemetry.recurrent_reanchors += 1

    def score_text_alignment(self, tokenizer: Any, text: str) -> float:
        """Lexical prior-support heuristic; this is not a correctness verifier."""
        if self.phi0 is None or not text.strip():
            return 0.0
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            return 0.0
        idx = torch.tensor([i for i in ids if 0 <= i < self.phi0.numel()], device=self.phi0.device, dtype=torch.long)
        if idx.numel() == 0:
            return 0.0
        vals = self.phi0.float().index_select(0, idx)
        # Top-half mean captures semantic content tokens without diluting on punctuation/prepositions
        k = max(1, idx.numel() // 2)
        return float(torch.topk(vals, k=k).values.mean().item())


class ColibriThoughtBuffer:
    """Two-tier reasoning controller:

    1. `ask_anchored(...)`: Runs standard single-pass generation with continuous `InitialProbabilityAnchor`
       (`Phi_0` + non-decaying `S_0` recurrent memory) so initial probabilities never wash out.
    2. `ask_wave_buffered(...)`: Assembles a compact, verified Semantic Buffer (`Буфер смысла`) from `(KV_0, S_0)`,
       discards raw intermediate draft clutter via GPU snapshot rollback, checks drift against `Phi_0`,
       and generates the final response directly from `[Initial Task + Clean Thought Buffer]`.
    """

    def __init__(self, unified_engine: Any):
        self.u_eng = unified_engine

    def ask_anchored(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int = 512,
        enable_thinking: bool = False,
        alpha: float = 0.42,
        rec_anchor_lambda: float = 0.06,
        rec_reanchor_every: int = 16,
        initial_phi0: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        anchor = InitialProbabilityAnchor(
            alpha=alpha,
            rec_anchor_lambda=rec_anchor_lambda,
            rec_reanchor_every=rec_reanchor_every,
        )
        if initial_phi0 is not None:
            anchor.phi0 = initial_phi0.detach().clone()
            anchor.phi_buf = anchor.phi0.clone()
        prev = getattr(self.u_eng, "_active_thought_anchor", None)
        self.u_eng._active_thought_anchor = anchor
        t0 = time.perf_counter()
        thinking_parts: list[str] = []
        answer_parts: list[str] = []
        metrics: dict[str, Any] = {}
        try:
            for ev in self.u_eng.stream_chat(
                messages,
                max_new_tokens=max_new_tokens,
                enable_thinking=enable_thinking,
                use_session_cache=False,
            ):
                etype = ev.get("type")
                if etype == "delta":
                    if ev.get("phase") == "thinking":
                        thinking_parts.append(ev.get("text", ""))
                    else:
                        answer_parts.append(ev.get("text", ""))
                elif etype == "done":
                    thinking_parts = [str(ev.get("thinking_text", ""))]
                    answer_parts = [str(ev.get("answer_text", ""))]
                    metrics = dict(ev.get("metrics") or {})
        finally:
            self.u_eng._active_thought_anchor = prev

        answer_txt = "".join(answer_parts).strip()
        thinking_txt = "".join(thinking_parts).strip()
        return {
            "answer": answer_txt,
            "thinking": thinking_txt,
            "wall_s": round(time.perf_counter() - t0, 3),
            "metrics": metrics,
            "anchor_telemetry": anchor.telemetry.summary(),
            "initial_phi0_sha256": anchor.initial_fingerprint(),
            "phi0_alignment": round(anchor.score_text_alignment(self.u_eng.tokenizer, answer_txt), 4),
        }

    def ask_wave_buffered(
        self,
        messages: list[dict[str, str]],
        *,
        buffer_tokens: int = 260,
        answer_tokens: int = 384,
        alpha_wave: float = 0.38,
        alpha_final: float = 0.48,
        rec_anchor_lambda: float = 0.08,
        min_buffer_alignment: float = 0.25,
    ) -> dict[str, Any]:
        """Builds a compact Meaning Buffer from `t = 0`, verifies alignment against `Phi_0` (rolling back
        and re-anchoring if mutated draft drifts), discards draft noise, and answers from the clean buffer.
        """
        t_start = time.perf_counter()
        norm_msgs = [{"role": str(m["role"]), "content": str(m.get("content", ""))} for m in messages]

        # Capture the original task BEFORE any draft. Every wave, retry and final
        # synthesis receives this same initial field rather than its own replacement.
        orig_anchor = InitialProbabilityAnchor(alpha=alpha_final, rec_anchor_lambda=rec_anchor_lambda,
                                              rec_reanchor_every=12)
        from .unified_runtime import forward_step_all_logits
        with _auxiliary_forwards(self.u_eng), torch.no_grad():
            orig_ids, _, _, _ = self.u_eng._prepare_prompt_and_cache(
                norm_msgs, enable_thinking=False, use_session_cache=False
            )
            if self.u_eng.fast_qwen:
                lg_orig, past_orig = forward_step_all_logits(
                    self.u_eng.engine, torch.tensor([orig_ids], dtype=torch.long),
                    past_key_values=None, prefetch_next=False, logits_to_keep=1,
                )
            else:
                lg_orig, past_orig = self.u_eng.engine._forward_step(
                    torch.tensor([orig_ids], dtype=torch.long), past_key_values=None,
                    prefetch_next=False, cache_position=torch.arange(0, len(orig_ids), dtype=torch.long),
                )
            orig_anchor.on_prefill_end(self.u_eng, past_orig, lg_orig)
            saved_orig_phi0 = orig_anchor.phi0.clone()
            del past_orig
            self.u_eng._release_transient_state()

        # Wave 1: Assemble compact Delta-Only Meaning Buffer (never re-copying prompt text already held in Phi_0).
        wave_prompt = (
            norm_msgs[-1]["content"]
            + "\n\n---\n"
            "[СИСТЕМНЫЙ БУФЕР СМЫСЛА — НЕ переписывай условие и правила (они уже в памяти) и НЕ пиши эссе! "
            "Запиши ТОЛЬКО сжатую цепочку вычисленных переходов/дельт по одной короткой строке "
            "(где текущее значение расходится с требуемым порогом/предусловием -> какой шаг/значение получается) "
            "до самого последнего шага включительно:]"
        )
        wave_msgs = norm_msgs[:-1] + [{"role": "user", "content": wave_prompt}]
        w1 = self.ask_anchored(
            wave_msgs,
            max_new_tokens=buffer_tokens,
            enable_thinking=False,
            alpha=alpha_wave,
            rec_anchor_lambda=rec_anchor_lambda,
            initial_phi0=saved_orig_phi0,
        )
        buf_text = w1["answer"]
        buf_align = w1["phi0_alignment"]
        retried_wave = False

        # Verification against initial field ("то, что меняется, не всегда правильно"):
        if buf_align < min_buffer_alignment:
            retried_wave = True
            w1_retry = self.ask_anchored(
                wave_msgs,
                max_new_tokens=buffer_tokens,
                enable_thinking=False,
                alpha=min(0.75, alpha_wave + 0.25),
                rec_anchor_lambda=min(0.16, rec_anchor_lambda * 1.8),
                initial_phi0=saved_orig_phi0,
            )
            if w1_retry["phi0_alignment"] >= buf_align:
                w1 = w1_retry
                buf_text = w1["answer"]
                buf_align = w1["phi0_alignment"]

        # Wave 2: Synthesize final response from [Original Task + Verified Meaning Buffer] with Phi_0 + S_0 active.
        synth_msgs = norm_msgs + [
            {"role": "assistant", "content": f"[Буфер смысла]\n{buf_text}"},
            {
                "role": "user",
                "content": (
                    "Опираясь строго на исходные условия и проверенный буфер смысла выше "
                    "(без лишней воды и без пропуска ни одного условия), дай точный итоговый ответ в требуемом формате."
                ),
            },
        ]
        # Run synthesis where `on_prefill_end` layers the buffer's distribution ON TOP OF `saved_orig_phi0`!
        lp_n0, _, _ = InitialProbabilityAnchor._ensure_neutral_baseline(self.u_eng)
        synth_anchor = InitialProbabilityAnchor(
            alpha=alpha_final,
            rec_anchor_lambda=rec_anchor_lambda,
            rec_reanchor_every=12,
        )

        def _merged_prefill(u_e: Any, past_kv: Any, lg_last: torch.Tensor) -> torch.Tensor:
            t_s0 = time.perf_counter()
            synth_anchor.snap0 = capture_cache_snapshot(past_kv)
            synth_anchor.phi0 = saved_orig_phi0.to(device=lg_last.device)
            synth_anchor.phi_buf = saved_orig_phi0.to(device=lg_last.device).clone()
            synth_anchor.layer_conclusion_into_buffer(lg_last, lp_n0, weight=0.40)
            synth_anchor._steps_since_reanchor = 0
            synth_anchor.telemetry.snapshot_setup_ms = (time.perf_counter() - t_s0) * 1000.0
            return synth_anchor.transform_logits(lg_last, past_kv, step_index=0)

        synth_anchor.on_prefill_end = _merged_prefill  # type: ignore[method-assign]

        prev = getattr(self.u_eng, "_active_thought_anchor", None)
        self.u_eng._active_thought_anchor = synth_anchor
        ans_parts: list[str] = []
        final_metrics: dict[str, Any] = {}
        try:
            for ev in self.u_eng.stream_chat(
                synth_msgs,
                max_new_tokens=answer_tokens,
                enable_thinking=False,
                use_session_cache=False,
            ):
                etype = ev.get("type")
                if etype == "delta" and ev.get("phase") != "thinking":
                    ans_parts.append(ev.get("text", ""))
                elif etype == "done":
                    ans_parts = [str(ev.get("answer_text", ""))]
                    final_metrics = dict(ev.get("metrics") or {})
        finally:
            self.u_eng._active_thought_anchor = prev

        final_answer = "".join(ans_parts).strip()
        return {
            "thought_buffer": buf_text,
            "buffer_phi0_alignment": buf_align,
            "retried_drifted_wave": retried_wave,
            "answer": final_answer,
            "wall_s": round(time.perf_counter() - t_start, 3),
            "wave1_telemetry": w1["anchor_telemetry"],
            "synth_telemetry": synth_anchor.telemetry.summary(),
            "initial_phi0_sha256": orig_anchor.initial_fingerprint(),
            "wave1_initial_phi0_sha256": w1["initial_phi0_sha256"],
            "synth_initial_phi0_sha256": synth_anchor.initial_fingerprint(),
            "answer_phi0_alignment": round(
                synth_anchor.score_text_alignment(self.u_eng.tokenizer, final_answer), 4
            ),
            "metrics": final_metrics,
        }
