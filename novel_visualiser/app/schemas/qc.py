"""Novel Visualiser — Multi-Gate Quality Assurance Data Contracts.

Defines schemas for:
1. QCVerdictEnum (APPROVED, NEEDS_INPAINT_PATCH, FATAL_FAIL)
2. Gate1TagAuditResult (Pre-render prompt token & negative constraint validation)
3. Gate2OpenCVMetricResult (Laplacian sharpness, exposure clipping, bbox boundaries)
4. Gate3VisionAuditResult (Multimodal vision scoring on identity, lighting, anatomy)
5. ComprehensiveQCAudit (Unified multi-gate quality report with targeted patch spec)
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class QCVerdictEnum(str, Enum):
    """Overall quality decision."""
    APPROVED = "approved"                     # High fidelity, passes all gates
    NEEDS_INPAINT_PATCH = "needs_inpaint_patch" # Minor flaw (face blur, deformed hand); repair via inpaint
    FATAL_FAIL = "fatal_fail"                 # Fundamental defect; re-render required


class Gate1TagAuditResult(BaseModel):
    """Gate 1: Pre-Render Prompt & Contract Adherence Audit (Deterministic)."""

    passed: bool = Field(description="True if prompt meets all identity and negative rules")
    missing_mandatory_identity_tags: list[str] = Field(
        default_factory=list, description="Required biological tags missing from prompt"
    )
    detected_illegal_traits: list[str] = Field(
        default_factory=list, description="Contradictory tokens found in prompt"
    )
    promoted_negative_compliant: bool = Field(
        default=True, description="True if all retired biological tags are in negative prompt"
    )
    environmental_harmony_compliant: bool = Field(
        default=True, description="True if no daylight/sun tokens leak into night/smog scenes"
    )
    feedback: str = Field(default="Gate 1 validation passed")


class Gate2OpenCVMetricResult(BaseModel):
    """Gate 2: OpenCV Metric & Spatial Integrity Audit (Automated Local Analysis)."""

    passed: bool = Field(description="True if image passes sharpness and exposure thresholds")
    laplacian_sharpness_score: float = Field(
        default=150.0, description="Laplacian variance (threshold >= 100.0)"
    )
    is_blurry: bool = Field(default=False)
    highlights_clipped_pct: float = Field(
        default=0.5, description="Percentage of over-exposed pixels (> 252)"
    )
    shadows_crushed_pct: float = Field(
        default=5.0, description="Percentage of under-exposed pixels (< 5)"
    )
    bbox_bounds_compliant: bool = Field(
        default=True, description="True if character mask stays inside canvas bounds"
    )
    feedback: str = Field(default="Gate 2 technical checks passed")


class Gate3VisionAuditResult(BaseModel):
    """Gate 3: Multimodal Vision Continuity Audit (Gemini-3 Flash Multimodal)."""

    passed: bool = Field(description="True if composite score >= 75.0")
    identity_fidelity_score: float = Field(
        default=90.0, description="Facial contours, eye color, hair style match (0-100)"
    )
    lighting_consistency_score: float = Field(
        default=88.0, description="Key light Kelvin temp and shadow direction match (0-100)"
    )
    wardrobe_fidelity_score: float = Field(
        default=92.0, description="Clothing texture, tears, and poverty wear match (0-100)"
    )
    anatomical_integrity_score: float = Field(
        default=95.0, description="Freedom from extra fingers or deformed anatomy (0-100)"
    )
    composite_score: float = Field(
        default=91.2, description="Weighted average quality score"
    )
    critique_notes: str = Field(default="Visual quality meets production standards")
    defect_bounding_box: Optional[tuple[int, int, int, int]] = Field(
        default=None, description="[x, y, w, h] of defect region for inpaint patching"
    )


class ComprehensiveQCAudit(BaseModel):
    """Master Multi-Gate Quality Audit Card combining Gates 1, 2, and 3."""

    shot_id: str
    verdict: QCVerdictEnum = Field(default=QCVerdictEnum.APPROVED)
    overall_score: float = Field(default=90.0, description="Composite score (0-100)")
    
    gate_1: Gate1TagAuditResult
    gate_2: Gate2OpenCVMetricResult
    gate_3: Optional[Gate3VisionAuditResult] = None
    
    remediation_inpaint_prompt: Optional[str] = Field(
        default=None, description="Targeted inpaint prompt if NEEDS_INPAINT_PATCH"
    )
    defect_bbox: Optional[tuple[int, int, int, int]] = Field(
        default=None, description="Coordinates for targeted inpainting"
    )

