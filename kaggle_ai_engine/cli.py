"""
Novel Visualiser — Kaggle AI Engine CLI Utility

Command-line interface to authenticate, check spend quota ($10/day),
test frontier model inference, generate character locks, and inspect images.

Usage:
  python -m kaggle_ai_engine.cli status
  python -m kaggle_ai_engine.cli auth
  python -m kaggle_ai_engine.cli quota
  python -m kaggle_ai_engine.cli chat -p "Hello from Novel Visualiser"
  python -m kaggle_ai_engine.cli lock-character --name "Sunny"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Force UTF-8 on Windows console for emoji/unicode compatibility
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from kaggle_ai_engine.config import get_kaggle_settings
from kaggle_ai_engine.consistency_architect import VisualConsistencyArchitect
from kaggle_ai_engine.proxy_client import KaggleModelProxyClient
from kaggle_ai_engine.quota_manager import KaggleQuotaManager
from kaggle_ai_engine.semantic_qc import KaggleSemanticQCInspector


def cmd_status(args: argparse.Namespace) -> None:
    """Print configuration and proxy connection status."""
    settings = get_kaggle_settings()
    summary = settings.get_summary()

    print("\n=======================================================")
    print("      KAGGLE MODEL PROXY AI ENGINE STATUS              ")
    print("=======================================================")
    print(f" Proxy Configured  : {summary['proxy_configured']}")
    print(f" Proxy URL         : {summary['proxy_url']}")
    print(f" API Key Present   : {summary['has_api_key']}")
    print(f" Expiry Time       : {summary['expiry_time']}")
    print(f" Is Token Expired  : {summary['is_expired']}")
    print(f" Default Model     : {summary['default_model']}")
    print(f" Daily Quota Limit : {summary['daily_quota_limit']}")
    print(f" Available Models  : {', '.join(summary['available_models'])}")
    print("=======================================================\n")


def cmd_auth(args: argparse.Namespace) -> None:
    """Authenticate or refresh Model Proxy token."""
    settings = get_kaggle_settings()
    print("[*] Refreshing Kaggle Model Proxy credentials via `kaggle benchmarks auth -y`...")
    success = settings.refresh_proxy_token()
    if success:
        print("[+] Successfully refreshed Model Proxy credentials!")
        print(f"    Proxy URL   : {settings.model_proxy_url}")
        print(f"    Expiry Time : {settings.model_proxy_expiry_time}")
    else:
        print("[-] Failed to refresh credentials via CLI.")
        print("    Ensure you are logged in to Kaggle: run 'kaggle auth login' first.")


def cmd_quota(args: argparse.Namespace) -> None:
    """Display Kaggle AI inference quota."""
    manager = KaggleQuotaManager()
    print(manager.format_status_table())


def cmd_chat(args: argparse.Namespace) -> None:
    """Test inference through the Model Proxy."""
    client = KaggleModelProxyClient()
    model = args.model or client.settings.llm_default
    prompt = args.prompt

    print(f"[*] Querying Model Proxy [{model}]...")
    messages = [
        {"role": "system", "content": "You are an anime visual director assisting Novel Visualiser."},
        {"role": "user", "content": prompt},
    ]
    try:
        reply = client.chat_completion(messages, model=model)
        print("\n--- MODEL RESPONSE ---")
        print(reply)
        print("----------------------\n")
    except Exception as exc:
        print(f"[!] Request error: {exc}")


def cmd_lock_character(args: argparse.Namespace) -> None:
    """Synthesize an immutable Danbooru character identity card."""
    architect = VisualConsistencyArchitect()
    name = args.name
    print(f"[*] Synthesizing Danbooru Character Identity Lock for '{name}'...")

    sample_prose = [
        "A delicate, pale young man with jet-black messy hair and pitch-dark eyes.",
        "He looked younger than his years, slender and gaunt from living in the outskirts.",
        "His shadow seemed to move with a strange sentience of its own.",
    ]
    card = architect.build_character_identity(
        character_name=name,
        novel_context="Shadow Slave by Guiltythree",
        prose_descriptions=sample_prose,
    )
    print("\n[+] Character Identity Card Synthesized:")
    print(card.model_dump_json(indent=2))


def cmd_qc(args: argparse.Namespace) -> None:
    """Run multimodal QC inspection on an image."""
    inspector = KaggleSemanticQCInspector()
    img_path = Path(args.image)
    ref_path = Path(args.ref) if args.ref else None

    print(f"[*] Inspecting shot image: {img_path}")
    verdict = inspector.evaluate_shot(
        generated_image_path=img_path,
        reference_image_path=ref_path,
        expected_character=args.character or "Sunny",
        expected_outfit=args.outfit or "worn dark tunic",
    )
    print("\n[+] Multimodal QC Verdict:")
    print(verdict.model_dump_json(indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="kaggle_ai_engine",
        description="Novel Visualiser — Kaggle Model Proxy AI Engine CLI",
    )
    subparsers = parser.add_subparsers(dest="command")

    # status
    p_status = subparsers.add_parser("status", help="Show Kaggle Model Proxy connection status")
    p_status.set_defaults(func=cmd_status)

    # auth
    p_auth = subparsers.add_parser("auth", help="Fetch/refresh Kaggle Model Proxy token")
    p_auth.set_defaults(func=cmd_auth)

    # quota
    p_quota = subparsers.add_parser("quota", help="Display $10/day spend quota")
    p_quota.set_defaults(func=cmd_quota)

    # chat
    p_chat = subparsers.add_parser("chat", help="Run a test chat completion")
    p_chat.add_argument("-p", "--prompt", required=True, help="Prompt text")
    p_chat.add_argument("-m", "--model", default=None, help="Model slug (gemini-2.5-pro, claude-sonnet-4)")
    p_chat.set_defaults(func=cmd_chat)

    # lock-character
    p_lock = subparsers.add_parser("lock-character", help="Generate immutable Danbooru character card")
    p_lock.add_argument("--name", default="Sunny", help="Character name")
    p_lock.set_defaults(func=cmd_lock_character)

    # qc
    p_qc = subparsers.add_parser("qc", help="Run multimodal Gate 3 QC inspection")
    p_qc.add_argument("--image", required=True, help="Path to generated shot image")
    p_qc.add_argument("--ref", default=None, help="Path to canonical reference image")
    p_qc.add_argument("--character", default="Sunny", help="Expected character name")
    p_qc.add_argument("--outfit", default="worn dark tunic", help="Expected outfit")
    p_qc.set_defaults(func=cmd_qc)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

