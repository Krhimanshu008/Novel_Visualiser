"""
Novel Visualiser — Kaggle AI Integration Bridge

Provides seamless, non-breaking adapters connecting the Kaggle Model Proxy ($10/day quota)
with the existing Novel Visualiser architecture:
- Replaces naive LLM calls in StoryIntelligenceAgent with Gemini 2.5 Pro / Claude Sonnet 4
- Augments QC Gate 3 with multimodal VLM inspection
- Supplies Danbooru-locked prompt blocks to PromptCompiler
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from kaggle_ai_engine.config import get_kaggle_settings
from kaggle_ai_engine.consistency_architect import VisualConsistencyArchitect
from kaggle_ai_engine.proxy_client import KaggleModelProxyClient
from kaggle_ai_engine.quota_manager import KaggleQuotaManager
from kaggle_ai_engine.semantic_qc import KaggleSemanticQCInspector
from kaggle_ai_engine.story_extractor import KaggleStoryExtractor

logger = logging.getLogger("kaggle_ai_engine.integration")


class KagglePipelineBridge:
    """
    Unified entry point for Novel Visualiser agents to consume
    the Kaggle $10/day advance AI capabilities.
    """

    def __init__(self):
        self.settings = get_kaggle_settings()
        self.quota = KaggleQuotaManager()
        self.client = KaggleModelProxyClient()
        self.story_extractor = KaggleStoryExtractor(self.client)
        self.consistency = VisualConsistencyArchitect(self.client)
        self.qc_inspector = KaggleSemanticQCInspector(self.client)

    def is_operational(self) -> bool:
        """Check if Kaggle Model Proxy is authenticated and has spend quota remaining."""
        if not self.settings.is_proxy_configured() and not self.settings.google_api_key:
            return False
        can_run, _ = self.quota.can_spend(0.01)
        return can_run

    def enhance_chapter_analysis(
        self,
        chapter_number: int,
        passages: list[dict[str, Any]],
        novel_title: str = "Shadow Slave",
    ) -> dict[str, Any]:
        """
        Drop-in replacement for Story Intelligence Agent chapter extraction.
        Returns a dict directly serializable to SQLite tables.
        """
        logger.info("Running Kaggle Model Proxy Story Intelligence on Chapter %d...", chapter_number)
        result = self.story_extractor.analyze_chapter(
            chapter_number=chapter_number,
            passages=passages,
            novel_title=novel_title,
        )
        return result.model_dump()

    def enhance_gate3_qc(
        self,
        shot_image_path: Path | str,
        reference_image_path: Optional[Path | str] = None,
        character_name: str = "Sunny",
        outfit_name: str = "worn dark tunic",
        expected_elements: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Drop-in enhancement for QC Gate 3. Evaluates actual pixels using VLM.
        """
        logger.info("Executing Kaggle Multimodal Gate 3 QC on %s...", shot_image_path)
        verdict = self.qc_inspector.evaluate_shot(
            generated_image_path=shot_image_path,
            reference_image_path=reference_image_path,
            expected_character=character_name,
            expected_outfit=outfit_name,
            expected_elements=expected_elements,
        )
        return verdict.model_dump()

    def get_system_health(self) -> dict[str, Any]:
        """Returns diagnostic health of the Kaggle AI integration."""
        quota_report = self.quota.fetch_quota()
        return {
            "proxy_configured": self.settings.is_proxy_configured(),
            "proxy_url": self.settings.model_proxy_url or "Unset",
            "token_expired": self.settings.is_token_expired(),
            "default_model": self.settings.llm_default,
            "daily_quota_valid": quota_report.is_valid,
            "daily_remaining": f"${quota_report.daily.remaining:.2f}" if quota_report.daily else "Unknown",
            "daily_used": f"${quota_report.daily.used:.2f}" if quota_report.daily else "Unknown",
        }

