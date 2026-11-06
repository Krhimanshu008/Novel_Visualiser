"""
Novel Visualiser — Kaggle Frontier Story Extractor

Leverages Kaggle's Model Proxy ($10/day quota) with Gemini 2.5 Pro or Claude Sonnet 4
to perform deep narrative intelligence extraction on raw novel chapters.

Features:
- Psychological and dramatic tension curve analysis
- Exact scene segmentation with start and end passage references
- Visual state delta extraction (outfit tearing, wounds, weather accumulation)
- Direct compatibility with Novel Visualiser's SQLite schema
"""

from __future__ import annotations

import logging
from typing import Any, Optional
from pydantic import BaseModel, Field

from kaggle_ai_engine.proxy_client import KaggleModelProxyClient

logger = logging.getLogger("kaggle_ai_engine.story")


class ExtractedScene(BaseModel):
    """A distinct cinematic scene extracted from the chapter."""
    scene_number: int
    start_passage_id: str
    end_passage_id: str
    location_name: str
    time_of_day: str
    weather: str
    mood: str
    dramatic_intensity: float = Field(
        ge=0.0, le=1.0, description="Pacing and intensity score (0.0=calm, 1.0=climax)"
    )
    summary: str
    characters_present: list[str]


class ExtractedVisualDelta(BaseModel):
    """Visual condition change occurring within a scene."""
    character_name: str
    scene_number: int
    delta_type: str = Field(description="'injury', 'clothing_damage', 'weather_effect', 'item_acquired'")
    description: str
    danbooru_modifier: str = Field(
        description="Suggested prompt tag, e.g. 'blood on cheek', 'torn sleeve', 'soaked wet clothes'"
    )


class ChapterAnalysisResult(BaseModel):
    """Structured chapter analysis produced by frontier LLM."""
    chapter_number: int
    chapter_title: Optional[str] = None
    scenes: list[ExtractedScene]
    visual_deltas: list[ExtractedVisualDelta]
    narrative_summary: str
    key_themes: list[str] = Field(default_factory=list)


class KaggleStoryExtractor:
    """
    Analyzes novel chapters using Kaggle Model Proxy reasoning models.
    """

    def __init__(self, client: Optional[KaggleModelProxyClient] = None):
        self.client = client or KaggleModelProxyClient()

    def analyze_chapter(
        self,
        chapter_number: int,
        passages: list[dict[str, Any]],
        novel_title: str = "Shadow Slave",
    ) -> ChapterAnalysisResult:
        """
        Extract complete cinematic scenes, pacing intensity, and visual deltas
        from numbered chapter passages.

        Args:
            chapter_number: Chapter integer.
            passages: List of dicts [{"passage_id": "CH01_P001", "text": "..."}]
            novel_title: Title of source novel for lore anchoring.

        Returns:
            Strongly-typed ChapterAnalysisResult.
        """
        passages_text = "\n".join(
            f"[{p.get('passage_id', f'P{i:03d}')}] {p.get('text', '')}"
            for i, p in enumerate(passages, start=1)
        )

        prompt = (
            f"You are the Lead Visual Story Director for an anime adaptation of '{novel_title}'.\n"
            f"Analyze Chapter {chapter_number} text below and decompose it into a sequence of cinematic scenes.\n\n"
            f"CHAPTER PASSAGES:\n"
            f"{passages_text[:25000]}\n\n"
            "INSTRUCTIONS:\n"
            "1. Identify logical scene cuts based on physical movement, shifts in time/location, or dramatic mood changes.\n"
            "2. Note exact start_passage_id and end_passage_id for each scene.\n"
            "3. Extract EVERY visual change (injuries, blood, dirt, clothing tears) that must appear on character models.\n"
            "4. Provide suggested Danbooru tags for those visual changes."
        )

        return self.client.structured_extract(
            prompt=prompt,
            schema=ChapterAnalysisResult,
            system_instruction="You are an expert anime director and narrative extractor.",
            model="gemini-2.5-pro",  # Gemini Pro excels at large passage ingestion
        )

