"""Novel Visualiser — Compositing & Shot Synthesis Subagent Data Contracts.

Defines schemas for:
1. CompositingModeEnum (Inpaint Differential vs. ControlNet+IPAdapter Fusion)
2. BoundingBox2D (Normalized or pixel bounding box coordinates)
3. ShotCompositionSpec (Multi-subagent unified contract for compositing a shot)
4. RenderedShotArtifact (Resulting composited image and production telemetry)
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field

from app.schemas.character_subagent import ShotCharacterAnchor
from app.schemas.creature_subagent import ShotCreatureAnchor
from app.schemas.environment_subagent import ShotEnvironmentAnchor
from app.schemas.location_subagent import ArcHierarchyTag, ShotLocationAnchor


class CompositingModeEnum(str, Enum):
    """Compositing execution mode."""
    INPAINT_DIFFERENTIAL = "inpaint_differential"        # Freeze stage plate, inpaint character into bbox
    CONTROLNET_IPADAPTER_FUSION = "controlnet_ipadapter" # Unified render conditioned on depth map & face chip
    MULTI_SUBJECT_BATTLE = "multi_subject_battle"        # Regional split for Character vs. Creature


class BoundingBox2D(BaseModel):
    """2D rectangular bounding box on the 1216x832 canvas."""
    x: int = Field(default=320, description="Top-left X coordinate")
    y: int = Field(default=240, description="Top-left Y coordinate")
    width: int = Field(default=576, description="Width in pixels")
    height: int = Field(default=512, description="Height in pixels")


class ShotCompositionSpec(BaseModel):
    """Unified shot contract assembling outputs from all 4 specialized subagents."""

    shot_id: str = Field(description="Unique shot ID (e.g. 'shot_ch01_sc01_001')")
    novel_arc: ArcHierarchyTag = Field(description="Narrative arc hierarchy coordinate")
    mode: CompositingModeEnum = Field(default=CompositingModeEnum.CONTROLNET_IPADAPTER_FUSION)
    
    # 1. Location Subagent Inputs
    location_plate_path: Optional[str] = Field(
        default=None, description="Path to rendered Action Stage plate (1216x832)"
    )
    location_depth_path: Optional[str] = Field(
        default=None, description="Path to OpenCV/ControlNet Depth Map (1216x832)"
    )
    location_anchor: Optional[ShotLocationAnchor] = Field(default=None)

    # 2. Character Subagent Inputs
    character_anchor: ShotCharacterAnchor = Field(
        description="Character biology, wardrobe, facial chip path, and inpaint bbox"
    )

    # 3. Environment Subagent Inputs
    environment_anchor: ShotEnvironmentAnchor = Field(
        description="Lighting vectors, Kelvin temp, weather particles, moisture rating"
    )

    # 4. Creature / Monster Subagent Inputs (Optional, for combat scenes)
    creature_anchor: Optional[ShotCreatureAnchor] = Field(
        default=None, description="Monster battle prompt, scale ratio, and predatory head chip"
    )

    # Spatial Canvas Framing
    character_bbox: BoundingBox2D = Field(
        default_factory=lambda: BoundingBox2D(x=320, y=200, width=576, height=550),
        description="Inpainting / Regional conditioning mask coordinates"
    )
    canvas_width: int = Field(default=1216, description="Standard canvas width")
    canvas_height: int = Field(default=832, description="Standard canvas height")
    deterministic_seed: Optional[int] = Field(default=None)


class RenderedShotArtifact(BaseModel):
    """Production artifact output from the dual-GPU ComfyUI cluster."""

    shot_id: str
    image_path: str = Field(description="Local path to downloaded composited PNG image")
    mode_used: CompositingModeEnum
    worker_url_used: str = Field(description="Remote worker URL that processed the job")
    prompt_id: Optional[str] = Field(default=None, description="ComfyUI job prompt ID")
    execution_time_seconds: float = Field(default=0.0)
    final_positive_prompt: str = Field(description="Exact prompt dispatched to diffusion")
    final_negative_prompt: str = Field(description="Exact negative prompt dispatched")
    qc_passed: bool = Field(default=True, description="Whether the shot passed sanity validation")

