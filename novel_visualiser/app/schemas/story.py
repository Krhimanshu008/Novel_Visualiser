"""Story Intelligence Agent I/O contracts.

Input/output schemas for chapter analysis, scene breaking, event extraction,
visual state tracking, and new entity detection. These are the structured
contracts that the Story Intelligence Agent consumes and produces.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PassageTag(BaseModel):
    """A pre-tagged passage with entity references."""

    passage_id: str = Field(description="Unique passage identifier, e.g. 'CH01_P014'")
    text: str = Field(description="Raw passage text content")
    passage_type: Literal["narration", "dialogue", "description", "action"] = Field(
        description="Functional classification of this passage"
    )
    tagged_entities: list[str] = Field(
        default_factory=list,
        description="Entity IDs found by FlashText pre-tagger",
    )


class SceneBreak(BaseModel):
    """A logical scene boundary within a chapter."""

    scene_number: int = Field(description="Sequential scene number within the chapter")
    start_passage: str = Field(description="Passage ID where this scene begins")
    end_passage: str = Field(description="Passage ID where this scene ends")
    location_name: str | None = Field(
        default=None,
        description="Human-readable location name; resolved to location entity later",
    )
    time_of_day: str | None = Field(
        default=None, description="Time of day descriptor, e.g. 'dawn', 'midnight'"
    )
    weather: str | None = Field(
        default=None, description="Weather conditions during this scene"
    )
    mood: (
        Literal[
            "calm",
            "tense",
            "dark",
            "epic",
            "melancholic",
            "mysterious",
            "chaotic",
            "peaceful",
        ]
        | None
    ) = Field(default=None, description="Dominant emotional tone of the scene")
    description: str | None = Field(
        default=None, description="Brief narrative description of the scene"
    )


class EventExtraction(BaseModel):
    """A key plot event extracted from the chapter."""

    event_type: Literal[
        "plot_beat",
        "character_intro",
        "battle",
        "discovery",
        "dialogue_exchange",
        "environmental",
        "power_awakening",
    ] = Field(description="Category of the narrative event")
    description: str = Field(description="Human-readable description of the event")
    source_passages: list[str] = Field(
        description="Passage IDs that support this event extraction"
    )
    characters_involved: list[str] = Field(
        description="Entity names or IDs of characters participating in this event"
    )
    importance: Literal["low", "medium", "high", "climax"] = Field(
        description="Narrative weight of this event"
    )


class VisualStateDelta(BaseModel):
    """A change to a character or location's visual state in this chapter."""

    entity_name: str = Field(description="Display name of the affected entity")
    entity_id: str | None = Field(
        default=None, description="Resolved entity ID after entity matching"
    )
    delta_type: Literal[
        "permanent_change",
        "temporary_condition",
        "new_equipment",
        "injury",
        "environmental_effect",
    ] = Field(description="Category of the visual change")
    description: str = Field(description="What changed visually")
    source_passages: list[str] = Field(
        description="Passage IDs where this change is described"
    )
    reverts_after_scene: bool = Field(
        default=False,
        description="True for temporary conditions like mud/blood that revert",
    )


class NewEntityDetection(BaseModel):
    """A newly discovered entity not in the pre-tagger dictionary."""

    suggested_name: str = Field(
        description="Proposed canonical name for this new entity"
    )
    entity_type: Literal[
        "character", "location", "creature", "item", "power", "effect", "faction"
    ] = Field(description="Classification of the new entity")
    description: str = Field(description="Visual and narrative description of the entity")
    source_passages: list[str] = Field(
        description="Passage IDs where this entity appears"
    )
    first_mention_passage: str = Field(
        description="Passage ID of the entity's first mention"
    )


class ChapterAnalysis(BaseModel):
    """Complete structured output from Story Intelligence Agent for one chapter."""

    project_id: str = Field(description="Parent project identifier")
    chapter_number: int = Field(description="Sequential chapter number")
    chapter_title: str | None = Field(
        default=None, description="Chapter title if available"
    )
    word_count: int = Field(description="Total word count of the chapter")
    scenes: list[SceneBreak] = Field(
        description="Ordered list of scene breaks identified in the chapter"
    )
    events: list[EventExtraction] = Field(
        description="Key plot events extracted from the chapter"
    )
    visual_state_deltas: list[VisualStateDelta] = Field(
        description="Visual state changes detected in this chapter"
    )
    new_entities: list[NewEntityDetection] = Field(
        description="Newly discovered entities not previously catalogued"
    )
    emotion_curve: list[str] = Field(
        description="Sequence of dominant emotions across the chapter, one per scene"
    )
    summary: str = Field(
        description="2-3 sentence chapter summary for context"
    )
