"""Novel Visualiser — Master Scene Director & Orchestration Data Contracts.

Defines schemas for:
1. SceneBreakdownBeat (Micro-narrative beat with character action and dialogue)
2. SceneOrchestrationPlan (Unified plan for a single scene orchestrating all 5 subagents)
3. ChapterProductionPlan (Complete multi-scene storyboard ledger for an entire chapter)
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field

from app.schemas.character_subagent import CharacterMatrixSpec, ShotCharacterAnchor
from app.schemas.compositing_subagent import ShotCompositionSpec
from app.schemas.creature_subagent import CreatureCard, ItemPropCard, ShotCreatureAnchor
from app.schemas.environment_subagent import EnvironmentalAtmosphereCard, ShotEnvironmentAnchor
from app.schemas.location_subagent import ArcHierarchyTag, LocationPlateMatrixSpec, ShotLocationAnchor


class SceneBreakdownBeat(BaseModel):
    """Micro-narrative beat within a scene."""

    beat_index: int = Field(description="Order of beat in scene")
    narration_text: str = Field(description="Narration paragraph or dialogue")
    primary_character: Optional[str] = Field(default=None, description="Acting character")
    dramatic_action: str = Field(description="Visual action occurring in beat")
    camera_intent: str = Field(default="medium", description="'wide', 'medium', 'close_up', 'action_wide'")


class SceneOrchestrationPlan(BaseModel):
    """Complete orchestration package for a single scene across all specialized subagents."""

    scene_id: str = Field(description="Unique scene ID (e.g. 'ch01_sc01')")
    scene_number: int = Field(description="Scene sequence number")
    scene_title: str = Field(description="Descriptive title")
    arc_hierarchy: ArcHierarchyTag = Field(description="Hierarchical 3-tier arc coordinates")
    scene_type: str = Field(default="dialogue", description="'combat', 'dialogue', 'exposition', 'climax'")
    
    # Narration & Budget
    full_scene_text: str = Field(description="Complete raw text of the scene")
    word_count: int = Field(description="Narration word count")
    target_shots: int = Field(description="Calculated shot budget")
    estimated_duration_sec: float = Field(description="Estimated audio/visual duration")
    pacing_curve: str = Field(description="'fast', 'moderate', 'slow'")

    # Subagent Assets
    location_matrix: LocationPlateMatrixSpec = Field(description="Location 4-plate standard & guides")
    environment_card: EnvironmentalAtmosphereCard = Field(description="Lighting rig & weather layer")
    active_characters: list[ShotCharacterAnchor] = Field(
        default_factory=list, description="Active character anchors with active epochs"
    )
    active_creatures: list[ShotCreatureAnchor] = Field(
        default_factory=list, description="Active creatures/monsters in combat"
    )
    active_props: list[ItemPropCard] = Field(
        default_factory=list, description="Hero weapons and Memories in scene"
    )

    # Sequenced Shot Storyboard
    shots: list[ShotCompositionSpec] = Field(
        default_factory=list, description="Sequenced list of shot composition contracts"
    )


class ChapterProductionPlan(BaseModel):
    """Master production plan for an entire novel chapter."""

    project_id: str
    novel_title: str
    chapter_number: int
    chapter_title: str
    total_words: int
    total_shots: int
    total_duration_sec: float
    scenes: list[SceneOrchestrationPlan] = Field(default_factory=list)
    created_at: str = Field(description="ISO timestamp of plan creation")

