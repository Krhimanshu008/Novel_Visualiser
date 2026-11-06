"""Arc Hierarchy & Timeline Contracts.

Defines schemas for:
1. ArcType & ArcNode (Hierarchical 4-tier tree: Major Arc > Sub Arc > Mini Arc > Chapter)
2. StoryTimeline & ParallelThread (In-universe time & multi-POV tracking)
3. ChapterBudgetSpec & SceneBudgetSpec (Mathematical shot budgeting & pacing)
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


class ArcType(str, Enum):
    """Tier level in the story hierarchy."""
    MAJOR = "major"       # Tier 1 (Volume / Macro saga: 50-150 chapters)
    SUB = "sub"           # Tier 2 (Regional phase: 15-45 chapters)
    MINI = "mini"         # Tier 3 (Mission / Tactical quest beat: 3-8 chapters)


class StoryTimeline(BaseModel):
    """In-universe story timeline coordinate."""

    in_universe_day: int = Field(default=1, description="Day number since story or arc inception")
    time_of_day: Literal["dawn", "morning", "noon", "afternoon", "dusk", "night", "midnight"] = Field(
        default="noon", description="Time of day"
    )
    global_event_tag: Optional[str] = Field(
        default=None, description="Dominant world event (e.g. 'The Great Eclipse', 'Acid Rain Tide')"
    )


class ParallelThread(BaseModel):
    """Parallel storyline branch / POV thread within an arc."""

    thread_id: str = Field(description="Unique thread identifier (e.g. 'thread_sunny_solo')")
    thread_name: str = Field(description="Descriptive name (e.g. 'Sunny Solo Labyrinth Exploration')")
    pov_entity_id: str = Field(description="Primary POV character entity ID")
    active_chapter_numbers: list[int] = Field(
        default_factory=list, description="Chapters that take place along this thread"
    )
    isolated_outfit_id: Optional[str] = Field(
        default=None, description="Outfit ID worn specifically on this branch"
    )
    timeline_day_range: tuple[int, int] = Field(
        default=(1, 1), description="(start_day, end_day) in-universe"
    )


class ArcNode(BaseModel):
    """Hierarchical Arc tree node."""

    arc_id: str
    project_id: str
    name: str
    arc_type: ArcType
    arc_number: int
    start_chapter: int
    end_chapter: int
    description: Optional[str] = None
    parent_arc_id: Optional[str] = None
    parallel_thread_id: Optional[str] = None
    timeline_day_start: Optional[int] = None
    timeline_day_end: Optional[int] = None
    in_universe_weather: Optional[str] = None
    children: list["ArcNode"] = Field(default_factory=list)


class SceneBudgetSpec(BaseModel):
    """Shot and pacing budget for a single scene."""

    scene_id: Optional[str] = None
    scene_number: int
    scene_type: str = Field(
        default="dialogue",
        description="Category: 'combat', 'climax', 'dialogue', 'tension', 'exposition', 'establishing'"
    )
    word_count: int
    target_shots: int
    avg_hold_duration_sec: float
    total_duration_sec: float
    pacing_class: Literal["fast", "medium_fast", "medium", "steady", "slow"]
    target_words_per_shot: float
    suggested_rhythm: list[str] = Field(
        default_factory=list,
        description="Suggested sequence of shot framings (e.g. ['wide', 'medium', 'close_up'])"
    )


class ChapterBudgetSpec(BaseModel):
    """Complete visual and audio pacing budget for a chapter."""

    chapter_number: int
    total_words: int
    estimated_audio_duration_sec: float
    total_shot_count: int
    scenes: list[SceneBudgetSpec] = Field(default_factory=list)
    motion_tier_distribution: dict[str, int] = Field(
        default_factory=lambda: {"static_parallax": 0, "i2v_climax": 0}
    )

