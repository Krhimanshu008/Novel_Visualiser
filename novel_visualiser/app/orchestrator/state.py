"""
Novel Visualiser — LangGraph Production State Definition

Defines the shared state that flows through the LangGraph orchestrator.
Holds only lightweight IDs and status flags — never raw text, base64 images,
or full LLM responses. Heavy data lives in SQLite.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ProductionWave(str, Enum):
    """VRAM wave execution stages."""
    TEXT_INTELLIGENCE = "text_intelligence"       # Wave 1: LLM loaded, no ComfyUI
    ASSET_GENERATION = "asset_generation"         # Wave 2: ComfyUI for ref sheets
    SCENE_GENERATION = "scene_generation"         # Wave 3: ComfyUI for story shots
    QUALITY_CONTROL = "quality_control"           # Wave 4: InsightFace + VLM
    AUDIO_FILM = "audio_film"                     # Wave 5: TTS + FFmpeg
    COMPLETE = "complete"


class PipelineStage(str, Enum):
    """Individual pipeline stages within a wave."""
    INIT = "init"
    RESEARCH = "research"
    STORY_INTEL = "story_intelligence"
    WORLD_ARCHITECT = "world_architect"
    RECONCILIATION = "reconciliation"
    SCENE_DIRECTION = "scene_direction"
    PROMPT_COMPILATION = "prompt_compilation"
    IMAGE_GENERATION = "image_generation"
    QUALITY_CONTROL = "quality_control"
    HUMAN_REVIEW = "human_review"
    NARRATION = "narration"
    SUBTITLE_ALIGNMENT = "subtitle_alignment"
    FILM_EDITING = "film_editing"
    COMPLETE = "complete"
    ERROR = "error"


class ApprovalRequest(BaseModel):
    """A pending human approval item."""
    request_id: str
    request_type: str  # "new_entity", "character_sheet", "location_plate", "shot_review"
    entity_id: str | None = None
    shot_id: str | None = None
    description: str
    options: list[str] = Field(default_factory=lambda: ["approve", "reject", "regenerate"])


class ProductionState(BaseModel):
    """
    The shared state object for the LangGraph Production Manager.

    This flows through every node in the graph. It contains ONLY lightweight
    references (IDs, counts, flags). All heavy data is read from / written to
    SQLite by the individual agent nodes.

    Designed for SqliteSaver checkpointing — the entire state is serializable.
    """

    # === Project Identity ===
    project_id: str
    project_name: str = ""

    # === Current Position ===
    current_wave: ProductionWave = ProductionWave.TEXT_INTELLIGENCE
    current_stage: PipelineStage = PipelineStage.INIT
    current_chapter: int = 1
    chapters_in_batch: list[int] = Field(
        default_factory=list,
        description="Chapter numbers being processed in the current wave (e.g., [1,2,3,4,5])"
    )

    # === Progress Tracking ===
    chapters_parsed: list[int] = Field(default_factory=list)
    chapters_scenes_extracted: list[int] = Field(default_factory=list)
    chapters_assets_generated: list[int] = Field(default_factory=list)
    chapters_shots_generated: list[int] = Field(default_factory=list)
    chapters_qc_passed: list[int] = Field(default_factory=list)
    chapters_narrated: list[int] = Field(default_factory=list)
    chapters_rendered: list[int] = Field(default_factory=list)

    # === Entity Tracking ===
    new_entities_detected: list[str] = Field(
        default_factory=list,
        description="Entity IDs flagged for review in the current batch"
    )
    entities_pending_approval: list[str] = Field(default_factory=list)
    entities_approved: list[str] = Field(default_factory=list)

    # === Shot Tracking ===
    total_shots_planned: int = 0
    shots_generated: int = 0
    shots_passed_qc: int = 0
    shots_failed_qc: list[str] = Field(
        default_factory=list,
        description="Shot IDs that failed QC and need retry"
    )
    shots_pending_review: list[str] = Field(default_factory=list)

    # === Human Review Queue ===
    pending_approvals: list[ApprovalRequest] = Field(default_factory=list)
    human_review_needed: bool = False

    # === Error Handling ===
    error_message: str | None = None
    retry_count: int = 0
    max_retries: int = 3
    last_successful_stage: PipelineStage = PipelineStage.INIT

    # === Metadata ===
    messages: list[str] = Field(
        default_factory=list,
        description="Log of key actions for audit trail"
    )

    def log(self, message: str) -> None:
        """Append a message to the state's audit log."""
        self.messages.append(message)

    def advance_stage(self, new_stage: PipelineStage) -> None:
        """Move to the next pipeline stage, recording the last successful one."""
        self.last_successful_stage = self.current_stage
        self.current_stage = new_stage
        self.error_message = None
        self.retry_count = 0

    def record_error(self, error: str) -> None:
        """Record an error and increment the retry counter."""
        self.error_message = error
        self.retry_count += 1
        self.current_stage = PipelineStage.ERROR
        self.log(f"ERROR (attempt {self.retry_count}): {error}")

