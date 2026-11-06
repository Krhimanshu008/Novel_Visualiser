"""
Novel Visualiser — Kaggle Multimodal Semantic QC Inspector (Gate 3)

Uses Kaggle Model Proxy's frontier Vision-Language Models (Gemini 2.5 Pro Vision / Claude Sonnet 4)
to inspect generated shot frames against locked reference art and scene requirements.

Addresses the core defect where OpenCV histograms and SSIM fail to verify
whether the correct person, outfit, or story action was actually drawn.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional
from pydantic import BaseModel, Field

from kaggle_ai_engine.proxy_client import KaggleModelProxyClient

logger = logging.getLogger("kaggle_ai_engine.qc")


class SemanticQCVerdict(BaseModel):
    """Structured inspection report for Gate 3 Quality Control."""
    overall_passed: bool = Field(description="True if the image meets all minimum visual standards")
    identity_consistency_score: float = Field(
        ge=0.0, le=1.0, description="Score evaluating facial & hair consistency against reference (0.0 to 1.0)"
    )
    attire_consistency_score: float = Field(
        ge=0.0, le=1.0, description="Score evaluating clothing fidelity (0.0 to 1.0)"
    )
    story_checklist_matches: list[str] = Field(
        default_factory=list, description="Requested scene items confirmed present in the pixels"
    )
    story_checklist_missing: list[str] = Field(
        default_factory=list, description="Requested scene items absent from the image"
    )
    detected_defects: list[str] = Field(
        default_factory=list,
        description="Defects observed: extra fingers, mutated limbs, modern artifacts, incorrect lighting"
    )
    remediation_suggestion: Optional[str] = Field(
        default=None, description="Actionable adjustment for negative prompt or IP-Adapter weights"
    )


class KaggleSemanticQCInspector:
    """
    Multimodal visual continuity inspector utilizing Kaggle's $10/day AI proxy.
    """

    def __init__(self, client: Optional[KaggleModelProxyClient] = None):
        self.client = client or KaggleModelProxyClient()

    def evaluate_shot(
        self,
        generated_image_path: Path | str,
        reference_image_path: Optional[Path | str] = None,
        expected_character: str = "Sunny",
        expected_outfit: str = "worn dark tunic",
        expected_elements: Optional[list[str]] = None,
        lighting_mood: str = "tense and dark",
    ) -> SemanticQCVerdict:
        """
        Inspect a generated shot frame using multimodal frontier AI.

        Args:
            generated_image_path: Path to the newly rendered ComfyUI output image.
            reference_image_path: Path to the approved canonical reference sheet or portrait.
            expected_character: Target character name.
            expected_outfit: Expected clothing description.
            expected_elements: List of visual features required in this shot (e.g. ['coffee cup', 'snow blizzard']).
            lighting_mood: Target lighting atmosphere.

        Returns:
            SemanticQCVerdict containing pass/fail, identity score, and defect report.
        """
        gen_path = Path(generated_image_path)
        if not gen_path.exists():
            return SemanticQCVerdict(
                overall_passed=False,
                identity_consistency_score=0.0,
                attire_consistency_score=0.0,
                detected_defects=[f"Generated image file not found: {gen_path}"],
            )

        images = [gen_path]
        ref_text = ""
        if reference_image_path and Path(reference_image_path).exists():
            images.append(Path(reference_image_path))
            ref_text = (
                "Image 1 is the GENERATED SHOT being evaluated.\n"
                "Image 2 is the APPROVED CANONICAL REFERENCE ART for identity comparison.\n\n"
            )
        else:
            ref_text = "Image 1 is the GENERATED SHOT being evaluated (no reference image provided).\n\n"

        req_items = expected_elements or []
        prompt = (
            f"You are a Senior Visual Quality Control Inspector for a cinematic anime production.\n"
            f"{ref_text}"
            f"EVALUATION CRITERIA:\n"
            f"- Target Character: {expected_character}\n"
            f"- Expected Attire: {expected_outfit}\n"
            f"- Expected Scene Elements: {', '.join(req_items) if req_items else 'Standard framing'}\n"
            f"- Atmospheric Mood: {lighting_mood}\n\n"
            "Examine the pixels carefully:\n"
            "1. Facial & Physical Identity: Does Image 1 match the face, hair, and physique of the reference?\n"
            "2. Attire: Is the clothing consistent with the expected outfit?\n"
            "3. Required Elements: Are the requested props, wounds, and weather effects visible?\n"
            "4. Anatomy & Quality: Check for distorted hands, extra fingers, malformed eyes, or blurry smudges.\n\n"
            "Output your findings matching the SemanticQCVerdict JSON schema."
        )

        return self.client.structured_extract(
            prompt=prompt,
            schema=SemanticQCVerdict,
            system_instruction="You perform rigorous multimodal visual inspection for anime generation.",
            model="gemini-2.5-pro",
        )

