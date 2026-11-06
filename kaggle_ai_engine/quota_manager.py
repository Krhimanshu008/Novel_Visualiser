"""
Novel Visualiser — Kaggle Model Proxy Quota Manager

Monitors the user's Kaggle Model Proxy AI inference spend quota ($10/day),
tracks token expenditures, prevents overages, and reports refill windows.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Optional

from kaggle_ai_engine.config import get_kaggle_settings

logger = logging.getLogger("kaggle_ai_engine.quota")


@dataclass
class QuotaTier:
    """Represents a quota tier (Daily or Monthly)."""
    period: str
    used: float
    remaining: float
    total: float
    refill_at: str

    @property
    def percentage_used(self) -> float:
        if self.total <= 0:
            return 0.0
        return min(100.0, (self.used / self.total) * 100.0)


@dataclass
class QuotaReport:
    """Consolidated Model Proxy quota report."""
    daily: Optional[QuotaTier] = None
    monthly: Optional[QuotaTier] = None
    raw_output: str = ""
    error: Optional[str] = None
    is_valid: bool = False


class KaggleQuotaManager:
    """
    Manages and monitors Kaggle Model Proxy spend quota.
    Invokes `kaggle benchmarks quota` under the hood.
    """

    def __init__(self):
        self.settings = get_kaggle_settings()

    def fetch_quota(self) -> QuotaReport:
        """
        Fetch the current spend quota directly from Kaggle CLI.
        Attempts JSON first, falls back to CSV or table parsing.
        """
        cmd_json = [
            sys.executable,
            "-m",
            "kaggle",
            "benchmarks",
            "quota",
            "--format",
            "json",
        ]

        try:
            res = subprocess.run(cmd_json, capture_output=True, text=True, timeout=15)
            if res.returncode == 0 and res.stdout.strip():
                return self._parse_json(res.stdout.strip())
        except Exception as e:
            logger.debug("JSON quota query failed or timed out: %s", e)

        # Fallback to CSV
        cmd_csv = [
            sys.executable,
            "-m",
            "kaggle",
            "benchmarks",
            "quota",
            "--format",
            "csv",
        ]
        try:
            res = subprocess.run(cmd_csv, capture_output=True, text=True, timeout=15)
            if res.returncode == 0 and res.stdout.strip():
                return self._parse_csv(res.stdout.strip())
            else:
                return QuotaReport(
                    raw_output=res.stderr or res.stdout,
                    error=res.stderr.strip() or "Failed to retrieve quota",
                    is_valid=False,
                )
        except Exception as e:
            return QuotaReport(
                error=f"Exception executing kaggle quota command: {e}",
                is_valid=False,
            )

    def _parse_json(self, raw_json: str) -> QuotaReport:
        """Parse JSON response from `kaggle b quota --format json`."""
        try:
            data = json.loads(raw_json)
            report = QuotaReport(raw_output=raw_json, is_valid=True)
            # JSON is typically a list of objects or a dict of tiers
            items = data if isinstance(data, list) else data.get("quotas", [data])

            for row in items:
                period = str(row.get("period", "")).capitalize()
                used = float(str(row.get("used", "0")).replace("$", ""))
                remaining = float(str(row.get("remaining", "0")).replace("$", ""))
                total = float(str(row.get("total", "0")).replace("$", ""))
                refill = str(row.get("refillAt", row.get("refill_at", "")))

                tier = QuotaTier(
                    period=period,
                    used=used,
                    remaining=remaining,
                    total=total,
                    refill_at=refill,
                )
                if period == "Daily":
                    report.daily = tier
                elif period == "Monthly":
                    report.monthly = tier

            return report
        except Exception as e:
            return QuotaReport(raw_output=raw_json, error=f"JSON parse error: {e}", is_valid=False)

    def _parse_csv(self, raw_csv: str) -> QuotaReport:
        """Parse CSV response from `kaggle b quota --format csv`."""
        try:
            reader = csv.DictReader(io.StringIO(raw_csv))
            report = QuotaReport(raw_output=raw_csv, is_valid=True)

            for row in reader:
                period = row.get("period", "").strip().capitalize()
                used = float(row.get("used", "0").replace("$", "").strip() or 0.0)
                remaining = float(row.get("remaining", "0").replace("$", "").strip() or 0.0)
                total = float(row.get("total", "0").replace("$", "").strip() or 0.0)
                refill = row.get("refillAt", "").strip()

                tier = QuotaTier(
                    period=period,
                    used=used,
                    remaining=remaining,
                    total=total,
                    refill_at=refill,
                )
                if period == "Daily":
                    report.daily = tier
                elif period == "Monthly":
                    report.monthly = tier

            return report
        except Exception as e:
            return QuotaReport(raw_output=raw_csv, error=f"CSV parse error: {e}", is_valid=False)

    def can_spend(self, estimated_usd: float = 0.05) -> tuple[bool, str]:
        """
        Safety check before dispatching high-token LLM calls.
        Ensures sufficient daily remaining quota and reserves a minimum buffer.
        """
        report = self.fetch_quota()
        if not report.is_valid:
            # If CLI authentication is missing or unreachable, allow local proxy attempt
            # if user provided static credentials in .env
            if self.settings.is_proxy_configured():
                return True, "Proxy credentials present; live quota fetch bypassed."
            return False, f"Cannot verify Kaggle quota: {report.error}"

        if report.daily:
            avail = report.daily.remaining - self.settings.reserve_margin_usd
            if avail < estimated_usd:
                return (
                    False,
                    f"Insufficient daily quota: ${report.daily.remaining:.2f} remaining "
                    f"(Reserve margin: ${self.settings.reserve_margin_usd:.2f}, Required: ${estimated_usd:.2f}). "
                    f"Refills at: {report.daily.refill_at}",
                )
            return True, f"Daily quota OK: ${report.daily.remaining:.2f} available."

        return True, "No daily quota tier detected; proceeding."

    def format_status_table(self) -> str:
        """Return a formatted string representing quota status for CLI display."""
        report = self.fetch_quota()
        lines = [
            "===========================================================",
            "       KAGGLE MODEL PROXY AI SPEND QUOTA ($10/DAY)         ",
            "===========================================================",
        ]

        if not report.is_valid:
            lines.append(f" Status: UNAVAILABLE ({report.error or 'Check Kaggle auth'})")
            lines.append(" Tip: Run 'kaggle auth login' or 'kaggle benchmarks auth -y'")
            lines.append("===========================================================")
            return "\n".join(lines)

        lines.append(f"{'Period':<10} | {'Used':<10} | {'Remaining':<10} | {'Total':<10} | {'Refill Window'}")
        lines.append("-" * 65)

        for tier in [report.daily, report.monthly]:
            if tier:
                used_str = f"${tier.used:.2f}"
                rem_str = f"${tier.remaining:.2f}"
                tot_str = f"${tier.total:.2f}"
                lines.append(
                    f"{tier.period:<10} | {used_str:<10} | {rem_str:<10} | {tot_str:<10} | {tier.refill_at}"
                )

        lines.append("===========================================================")
        return "\n".join(lines)

