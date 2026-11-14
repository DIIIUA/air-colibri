"""AirColibri command-line interface.

Commands:
  doctor    Check GPU, VRAM plan, model files, Q4 cache and LAN addresses
  ask       Stream a single prompt (`<think>...</think>` -> answer -> `<|im_end|>`)
  chat      Interactive multi-turn chat (KV-prefix reuse + pinned host-RAM parking)
  serve     Web chat + OpenAI-compatible API (TCP :8787) and UDP LAN beacon (:8788)
  discover  Find AirColibri nodes on the local network

Exit codes: 0 OK, 1 runtime error, 2 usage error, 3 model not found / unsupported.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import secrets
import sys

# Ensure UTF-8 stdout/stderr on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from aircolibri_cli.lan_server import (
    DEFAULT_BEACON_PORT,
    DEFAULT_HTTP_PORT,
    DEFAULT_LOG_FILE,
    DEFAULT_MAX_REQUEST_BYTES,
    AirColibriLANServer,
    detect_lan_ips,
    discover_lan_servers,
    format_connection_passport,
)
from aircolibri_cli.model_source import (
    DEFAULT_MODEL_ID,
    ModelNotFoundError,
    UnsupportedModelError,
    detect_format,
    raw_text_config_dict,
    resolve_model_dir,
)
from aircolibri_cli.unified_runtime import (
    VRAM_MODES,
    AirColibriUnifiedEngine,
    GenerationCancelled,
    checkpoint_manifest_path,
    find_aircolibri_root,
)

EXIT_OK, EXIT_RUNTIME, EXIT_USAGE, EXIT_MODEL = 0, 1, 2, 3


def _add_engine_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--model",
        "--model-dir",
        dest="model",
        type=str,
        default=None,
        help=f"Model folder (Colibri container or Hugging Face shards) or a Hugging Face repo id such as "
        f"{DEFAULT_MODEL_ID} (downloaded once). Default: a supported folder in <root>/models",
    )
    p.add_argument("--aircolibri-root", type=str, default=None, help="Path to the AirColibri program folder")
    p.add_argument(
        "--vram-mode",
        choices=list(VRAM_MODES),
        default="normal",
        help="normal: almost all free VRAM with a safety reserve (default); turbo: all free VRAM",
    )
    p.add_argument("--max-context", type=int, default=None, help="Context tokens to reserve VRAM for (default 8192, 4096 on <=5 GB cards)")
    p.add_argument("--prefill-chunk", type=int, default=None, help="Prompt tokens per prefill chunk (default 512, 256 on <=5 GB cards)")
    p.add_argument("--no-mtp", action="store_true", help="Disable the speculative MTP head")
    p.add_argument("--max-drafts", type=int, default=2, help="Max speculative draft depth (1..3, default: 2)")
    p.add_argument(
        "--fast-vocab",
        action="store_true",
        help="Experimental vocabulary pruner (off by default: it drops rare tokens on long outputs)",
    )


def _engine_from_args(args: argparse.Namespace) -> AirColibriUnifiedEngine:
    explicit_repo = bool(args.model) and not Path(args.model).exists()
    return AirColibriUnifiedEngine(
        model_dir=args.model,
        aircolibri_root=args.aircolibri_root,
        vram_mode=args.vram_mode,
        enable_mtp=not args.no_mtp,
        enable_vocab_pruner=args.fast_vocab,
        max_drafts=args.max_drafts,
        max_context_tokens=args.max_context,
        prefill_chunk_tokens=args.prefill_chunk,
        allow_download=explicit_repo,
    )


def cmd_doctor(args: argparse.Namespace) -> int:
    import torch

    from aircolibri_cli.unified_runtime import (
        _model_shape_inventory,
        estimate_layer_vram_bytes,
        plan_vram,
        q4_bytes_for_shape,
    )

    root = find_aircolibri_root(args.aircolibri_root)
    lan_ip, all_ips = detect_lan_ips()
    cuda_ok = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "NONE"
    report: dict = {
        "cuda_available": cuda_ok,
        "gpu_name": gpu_name,
        "aircolibri_root": str(root),
        "primary_lan_ip": lan_ip,
        "all_lan_ips": all_ips,
    }
    model_dir = None
    model_error = None
    try:
        model_dir = resolve_model_dir(args.model, root, allow_download=False)
    except (ModelNotFoundError, UnsupportedModelError, OSError) as exc:
        model_error = str(exc)
    report["model_dir"] = str(model_dir) if model_dir else None
    report["model_error"] = model_error

    plan = None
    if model_dir is not None:
        fmt = detect_format(model_dir)
        report["model_format"] = fmt
        if fmt == "colibri":
            report["q4_cache_ready"] = (root / "data" / "q4-g64-cache").exists() and checkpoint_manifest_path(root).exists()
        if cuda_ok:
            from aircolibri_cli.model_source import open_model_source

            cfg = raw_text_config_dict(model_dir)
            n = int(cfg["num_hidden_layers"])
            layers_sh, globals_sh, mtp_sh = _model_shape_inventory(model_dir, open_model_source(model_dir), n)
            vocab, hidden = globals_sh.get("lm_head.weight", (cfg["vocab_size"], cfg["hidden_size"]))
            mtp_b = estimate_layer_vram_bytes({k: v for k, v in mtp_sh.items() if k.startswith("mtp.layers.")})
            if "mtp.fc.weight" in mtp_sh:
                mtp_b += q4_bytes_for_shape(mtp_sh["mtp.fc.weight"])
            report["mtp_weights_present"] = "mtp.fc.weight" in mtp_sh
            free_b, total_b = torch.cuda.mem_get_info(0)
            full_attn = sum(1 for t in cfg.get("layer_types", []) if t == "full_attention")
            try:
                plan = plan_vram(
                    args.vram_mode, total_b, free_b, [estimate_layer_vram_bytes(s) for s in layers_sh],
                    q4_bytes_for_shape((vocab, hidden)), mtp_b,
                    full_attn * 2 * int(cfg.get("num_key_value_heads", 4)) * int(cfg.get("head_dim", 256)) * 2,
                    (n - full_attn) * 32 * 128 * 128 * 4, int(cfg["hidden_size"]), int(cfg["intermediate_size"]),
                )
                report["vram_plan"] = plan.as_dict()
            except RuntimeError as exc:
                report["vram_plan_error"] = str(exc)

    print("=" * 76)
    print("  AIRCOLIBRI DOCTOR — ENVIRONMENT & LAN READINESS")
    print("=" * 76)
    print(f"  CUDA / GPU         : {'OK' if cuda_ok else 'MISSING'} — {gpu_name}")
    print(f"  Program folder     : {root}")
    print(f"  Model              : {model_dir or 'NOT FOUND'} ({report.get('model_format', '-')})")
    if model_error:
        print(f"                       {model_error}")
    if "mtp_weights_present" in report:
        print(f"  MTP head           : {'present' if report['mtp_weights_present'] else 'not in checkpoint (MTP off)'}")
    if plan is not None:
        layout = "all layers resident" if plan.full_residency else (
            f"{plan.resident_layers}/{plan.layers_total} resident + {plan.ring_slots}-slot ring"
        )
        print(
            f"  VRAM plan ({plan.mode:6s}): total {plan.total_bytes / 2**30:.2f} GiB, free {plan.free_bytes / 2**30:.2f} GiB, "
            f"reserve {plan.reserve_bytes / 2**30:.2f} GiB -> {layout}, MTP {'on' if plan.enable_mtp else 'off'}"
        )
    elif "vram_plan_error" in report:
        print(f"  VRAM plan          : {report['vram_plan_error']}")
    print(f"  Detected LAN IPv4  : {lan_ip} (all: {', '.join(all_ips)})")
    print("=" * 76)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if model_dir is None:
        return EXIT_MODEL
    return EXIT_OK if cuda_ok and "vram_plan_error" not in report else EXIT_RUNTIME


def _stream_to_console(engine: AirColibriUnifiedEngine, messages: list[dict[str, str]], max_tokens: int, thinking: bool, use_session: bool) -> dict:
    in_think_printed = False
    in_ans_printed = False
    done_packet: dict = {}

    gen = engine.stream_chat(
        messages,
        max_new_tokens=max_tokens,
        enable_thinking=thinking,
        use_session_cache=use_session,
        stop_on_eos=True,
    )
    try:
        for ev in gen:
            if ev["type"] == "delta":
                if ev["phase"] == "thinking":
                    if not in_think_printed:
                        print("\n[Размышление <think>]\n", end="", flush=True)
                        in_think_printed = True
                    print(ev["text"], end="", flush=True)
                else:
                    if in_think_printed and not in_ans_printed:
                        print("\n\n[/think -> Ответ]\n", end="", flush=True)
                    elif not in_ans_printed:
                        print("\n[Ответ]\n", end="", flush=True)
                    in_ans_printed = True
                    print(ev["text"], end="", flush=True)
            elif ev["type"] == "done":
                done_packet = ev
    except KeyboardInterrupt:
        gen.close()
        print("\n\n[Остановлено]", flush=True)
        return {}
    finally:
        gen.close()

    m = done_packet.get("metrics", {})
    sg = m.get("spec_gate", {})
    acc_pct = sg.get("overall_accept_rate_pct", 0.0)
    reuse_str = f" | Prefix reused: {m.get('cached_prefix_tokens', 0)} tok" if m.get("cached_prefix_tokens") else ""
    park_str = f" | Parked RAM: {m.get('parked_session_ram_mib', 0.0)} MiB" if use_session else ""
    print(
        f"\n\n--- [ one answer: {m.get('decode_tok_per_s', 0.0):.2f} tok/s | "
        f"{m.get('generated_tokens', 0)} tok (EOS={m.get('stopped_by_eos')}) | "
        f"Prefill: {m.get('prefill_ms', 0.0):.1f} ms{reuse_str} | "
        f"MTP accept: {acc_pct}% ({m.get('avg_tokens_per_backbone_pass', 1.0):.2f} tok/pass) | "
        f"Peak VRAM: {m.get('peak_vram_mib', 0.0):.1f} MiB{park_str} ] ---",
        flush=True,
    )
    return done_packet


def cmd_ask(args: argparse.Namespace) -> int:
    prompt_text = " ".join(args.prompt).strip()
    if not prompt_text:
        print("Error: please provide a prompt, e.g. `aircolibri ask \"Привет!\"`", file=sys.stderr)
        return EXIT_USAGE

    messages: list[dict[str, str]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": prompt_text})

    with _engine_from_args(args) as eng:
        done_pkt = _stream_to_console(
            eng,
            messages=messages,
            max_tokens=args.max_tokens,
            thinking=not args.no_think,
            use_session=False,
        )
        if args.json_out and done_pkt:
            Path(args.json_out).write_text(json.dumps(done_pkt, ensure_ascii=False, indent=2), encoding="utf-8")
    return EXIT_OK


def cmd_chat(args: argparse.Namespace) -> int:
    thinking = not args.no_think
    messages: list[dict[str, str]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    with _engine_from_args(args) as eng:
        print(
            "\n[AirColibri Interactive Chat] Commands: `/reset`, `/think on|off`, `/status`, `/exit`. "
            "Ctrl+C stops the current answer.\n",
            flush=True,
        )
        while True:
            try:
                user_in = input("\nВы > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nВыход.", flush=True)
                break
            if not user_in:
                continue
            if user_in.lower() in ("/exit", "/quit", "exit", "quit"):
                break
            if user_in.lower() == "/reset":
                eng.reset_session()
                messages = [{"role": "system", "content": args.system}] if args.system else []
                print("[Сессия и KV-кэш очищены]", flush=True)
                continue
            if user_in.lower().startswith("/think"):
                parts = user_in.split()
                thinking = (parts[1].lower() == "on") if len(parts) > 1 else (not thinking)
                print(f"[Режим <think>: {'ON' if thinking else 'OFF'}]", flush=True)
                continue
            if user_in.lower() == "/status":
                print(json.dumps(eng.status_snapshot(), ensure_ascii=False, indent=2), flush=True)
                continue

            messages.append({"role": "user", "content": user_in})
            done_pkt = _stream_to_console(
                eng,
                messages=messages,
                max_tokens=args.max_tokens,
                thinking=thinking,
                use_session=True,
            )
            if done_pkt:
                messages.append({"role": "assistant", "content": done_pkt.get("answer_text", "")})
            else:
                messages.pop()
    return EXIT_OK


def cmd_serve(args: argparse.Namespace) -> int:
    api_key = args.api_key
    if not api_key:
        api_key = f"colibri-{secrets.token_hex(8)}" if args.require_auth else "colibri-local"
    root = find_aircolibri_root(args.aircolibri_root)
    log_file = None if args.no_log_file else (args.log_file or str(root / DEFAULT_LOG_FILE))

    with _engine_from_args(args) as eng:
        srv = AirColibriLANServer(
            engine=eng,
            host=args.host,
            http_port=args.port,
            beacon_port=args.beacon_port,
            api_key=api_key,
            require_auth=args.require_auth,
            enable_beacon=not args.no_beacon,
            max_request_bytes=args.max_request_bytes,
            log_file=log_file,
            preempt=not args.no_preempt,
        )
        srv.serve_forever()
    return EXIT_OK


def cmd_discover(args: argparse.Namespace) -> int:
    print(f"[AirColibri] Scanning LAN via UDP port {args.beacon_port} ({args.timeout:.1f}s)...", flush=True)
    nodes = discover_lan_servers(timeout_s=args.timeout, beacon_port=args.beacon_port)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(nodes, ensure_ascii=False, indent=2), encoding="utf-8")

    if not nodes:
        print("No active AirColibri LAN servers responded on this subnet.", flush=True)
        return EXIT_RUNTIME

    print(f"Found {len(nodes)} active AirColibri node(s) on the LAN:")
    for n in nodes:
        print(format_connection_passport(n))
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aircolibri",
        description="AirColibri: local Qwen3.5 inference (W4A16 + NVRTC + MTP) with a LAN OpenAI-compatible server",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_doc = sub.add_parser("doctor", help="Check GPU, VRAM plan, model files, and LAN interfaces")
    p_doc.add_argument("--model", "--model-dir", dest="model", type=str, default=None)
    p_doc.add_argument("--aircolibri-root", type=str, default=None)
    p_doc.add_argument("--vram-mode", choices=list(VRAM_MODES), default="normal")
    p_doc.add_argument("--json-out", type=str, default=None)
    p_doc.set_defaults(func=cmd_doctor)

    p_ask = sub.add_parser("ask", help="Stream a single prompt (`<think>...</think>` -> `<|im_end|>`)")
    _add_engine_args(p_ask)
    p_ask.add_argument("--max-tokens", type=int, default=1024, help="Max new tokens (default: 1024)")
    p_ask.add_argument("--no-think", action="store_true", help="Disable `<think>` reasoning block")
    p_ask.add_argument("--system", type=str, default=None, help="Optional system prompt")
    p_ask.add_argument("--json-out", type=str, default=None, help="Save full response + telemetry JSON")
    p_ask.add_argument("prompt", nargs="+", help="Prompt text")
    p_ask.set_defaults(func=cmd_ask)

    p_chat = sub.add_parser("chat", help="Interactive terminal chat with `kv_prefix` reuse & RAM parking")
    _add_engine_args(p_chat)
    p_chat.add_argument("--max-tokens", type=int, default=1024)
    p_chat.add_argument("--no-think", action="store_true")
    p_chat.add_argument("--system", type=str, default=None)
    p_chat.set_defaults(func=cmd_chat)

    p_srv = sub.add_parser("serve", help="Start LAN HTTP/SSE server + UDP LAN self-broadcaster")
    _add_engine_args(p_srv)
    p_srv.add_argument("--host", type=str, default="0.0.0.0", help="Bind address (default: 0.0.0.0 for LAN)")
    p_srv.add_argument("--port", type=int, default=DEFAULT_HTTP_PORT, help=f"HTTP/SSE port (default: {DEFAULT_HTTP_PORT})")
    p_srv.add_argument(
        "--beacon-port", type=int, default=DEFAULT_BEACON_PORT, help=f"UDP LAN beacon port (default: {DEFAULT_BEACON_PORT})"
    )
    p_srv.add_argument("--api-key", type=str, default=None, help="API key shown in the Connection Passport")
    p_srv.add_argument("--require-auth", action="store_true", help="Enforce `Authorization: Bearer <api-key>`")
    p_srv.add_argument("--no-beacon", action="store_true", help="Disable UDP LAN broadcast beacon")
    p_srv.add_argument(
        "--max-request-bytes", type=int, default=DEFAULT_MAX_REQUEST_BYTES,
        help=f"Reject larger request bodies with 413 (default: {DEFAULT_MAX_REQUEST_BYTES} = 8 MiB)",
    )
    p_srv.add_argument("--log-file", type=str, default=None, help=f"Server log path (default: <root>/{DEFAULT_LOG_FILE})")
    p_srv.add_argument("--no-log-file", action="store_true", help="Do not write a server log file")
    p_srv.add_argument(
        "--no-preempt", action="store_true",
        help="Queue concurrent requests instead of cancelling the running generation",
    )
    p_srv.set_defaults(func=cmd_serve)

    p_disc = sub.add_parser("discover", help="Discover AirColibri servers broadcasting on the LAN")
    p_disc.add_argument("--beacon-port", type=int, default=DEFAULT_BEACON_PORT)
    p_disc.add_argument("--timeout", type=float, default=2.0, help="Scan duration in seconds (default: 2.0)")
    p_disc.add_argument("--json-out", type=str, default=None)
    p_disc.set_defaults(func=cmd_discover)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ModelNotFoundError, UnsupportedModelError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_MODEL
    except GenerationCancelled:
        print("\n[Остановлено]", file=sys.stderr)
        return EXIT_OK
    except KeyboardInterrupt:
        print("\n[Прервано]", file=sys.stderr)
        return EXIT_RUNTIME
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_RUNTIME


if __name__ == "__main__":
    raise SystemExit(main())
