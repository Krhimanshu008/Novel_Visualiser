"""
Novel Visualiser — Kaggle AI Engine Configuration

Manages Kaggle Benchmarks Model Proxy credentials ($10/day quota),
authentication tokens, model slugs, and automatic token refresh.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

# Search for .env files in kaggle_ai_engine, novel_visualiser, and workspace root
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
ENV_CANDIDATES = [
    Path(__file__).resolve().parent / ".env",
    WORKSPACE_ROOT / ".env",
    WORKSPACE_ROOT / "novel_visualiser" / ".env",
]

for env_path in ENV_CANDIDATES:
    if env_path.exists():
        load_dotenv(env_path, override=False)


class KaggleAISettings:
    """Settings and credentials for Kaggle Model Proxy and Benchmarks tasks."""

    def __init__(self, env_file: Optional[Path] = None):
        self.env_file = env_file or (WORKSPACE_ROOT / ".env")
        self.reload()

    def reload(self) -> None:
        """Reload environment variables from the env file and os.environ."""
        if self.env_file.exists():
            load_dotenv(self.env_file, override=True)

        # Model Proxy credentials (written by `kaggle benchmarks auth` or `init`)
        self.model_proxy_url: str = os.getenv("MODEL_PROXY_URL", "").rstrip("/")
        self.model_proxy_api_key: str = os.getenv("MODEL_PROXY_API_KEY", "")
        self.model_proxy_expiry_time: str = os.getenv("MODEL_PROXY_EXPIRY_TIME", "")

        # Default model selection
        self.llm_default: str = os.getenv("LLM_DEFAULT", "gemini-2.5-pro")
        self.llm_default_eval: str = os.getenv("LLM_DEFAULT_EVAL", "claude-sonnet-4")
        self.llms_available_raw: str = os.getenv(
            "LLMS_AVAILABLE",
            "gemini-2.5-pro,gemini-2.5-flash,claude-sonnet-4,claude-haiku-4-5,deepseek-r1,qwen2.5-72b",
        )

        # Kaggle user credentials
        self.kaggle_username: str = os.getenv("KAGGLE_USERNAME", "")
        self.kaggle_key: str = os.getenv("KAGGLE_KEY", "")
        self.kaggle_api_token: str = os.getenv("KAGGLE_API_TOKEN", "")

        # Optional fallback to direct Gemini if proxy is unconfigured
        self.google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
        self.gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

        # Quota safeguards
        self.daily_quota_usd: float = float(os.getenv("KAGGLE_DAILY_QUOTA_USD", "10.00"))
        self.reserve_margin_usd: float = float(os.getenv("KAGGLE_RESERVE_MARGIN_USD", "1.00"))

    @property
    def available_models(self) -> list[str]:
        """List of available models configured for the proxy."""
        return [m.strip() for m in self.llms_available_raw.split(",") if m.strip()]

    def is_proxy_configured(self) -> bool:
        """Return True if Model Proxy URL and API key are populated."""
        return bool(self.model_proxy_url and self.model_proxy_api_key)

    def is_token_expired(self) -> bool:
        """
        Check if the Model Proxy API token has passed its expiration time.
        Kaggle tokens typically expire within 24 hours.
        """
        if not self.model_proxy_expiry_time:
            # If no expiry time is recorded, treat as expired if missing key
            return not bool(self.model_proxy_api_key)

        try:
            # Format usually ISO 8601: e.g. 2026-08-08T00:00:00+00:00
            expiry_dt = datetime.fromisoformat(self.model_proxy_expiry_time.replace("Z", "+00:00"))
            now_utc = datetime.now(timezone.utc)
            return now_utc >= expiry_dt
        except Exception:
            return False

    def refresh_proxy_token(self, env_file: Optional[Path] = None) -> bool:
        """
        Run `kaggle benchmarks auth -y` via subprocess to mint a fresh
        Model Proxy token and append it to the target .env file.
        """
        target_env = env_file or self.env_file
        cmd = [
            sys.executable,
            "-m",
            "kaggle",
            "benchmarks",
            "auth",
            "-y",
            "--env-file",
            str(target_env),
        ]

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if res.returncode == 0:
                self.reload()
                return True
            else:
                # Log stderr for diagnostics
                return False
        except Exception:
            return False

    def get_summary(self) -> dict[str, Any]:
        """Diagnostic state summary."""
        return {
            "proxy_configured": self.is_proxy_configured(),
            "proxy_url": self.model_proxy_url or "Not Set",
            "has_api_key": bool(self.model_proxy_api_key),
            "expiry_time": self.model_proxy_expiry_time or "Unknown",
            "is_expired": self.is_token_expired(),
            "default_model": self.llm_default,
            "available_models": self.available_models,
            "daily_quota_limit": f"${self.daily_quota_usd:.2f}",
        }


# Global singleton instance
_settings: Optional[KaggleAISettings] = None


def get_kaggle_settings() -> KaggleAISettings:
    """Return the cached KaggleAISettings singleton."""
    global _settings
    if _settings is None:
        _settings = KaggleAISettings()
    return _settings

