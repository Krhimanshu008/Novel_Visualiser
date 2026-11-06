"""Novel Visualiser — Location Continuity Subagent Data Contracts.

Defines schemas for:
1. ArcHierarchyTag (Main Arc > Sub Arc > Mini Arc + Parallel Threads)
2. DeepObservationalAnalysis (5-dimension CoT scene reasoning)
3. PlateRole, SinglePlateSpec, and LocationPlateMatrixSpec (4-plate standard)
4. ShotLocationAnchor (Runtime camera angle and structural guide resolution)
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class PlateRole(str, Enum):
    """The 4 standardized cinematic camera roles for location plates."""
    WIDE_ESTABLISHING = "WIDE_ESTABLISHING"  # Plate 1: Master Vista (1216x832)
    ACTION_STAGE = "ACTION_STAGE"            # Plate 2: Character Interaction Zone (1216x832)
    REVERSE_RECIPROCAL = "REVERSE_RECIPROCAL"# Plate 3: 180-deg Counter Shot (1216x832)
    MACRO_TEXTURE = "MACRO_TEXTURE"          # Plate 4: Material / Prop Anchor (1024x1024)


class ArcHierarchyTag(BaseModel):
    """Hierarchical Arc Coordinate tracking location lifespan and narrative continuity."""

    main_arc_id: str = Field(description="Tier 1 Major Arc (e.g. 'ARC_01_FIRST_NIGHTMARE')")
    sub_arc_id: Optional[str] = Field(
        default=None, description="Tier 2 Sub Arc (e.g. 'SUBARC_01_01_OUTSKIRTS')"
    )
    mini_arc_id: Optional[str] = Field(
        default=None, description="Tier 3 Mini Arc (e.g. 'MINIARC_01_01_01_AWAKENING')"
    )
    parallel_thread_id: Optional[str] = Field(
        default=None, description="Parallel storyline POV thread ID if divergent"
    )
    in_universe_day: int = Field(default=1, description="Day index in story timeline")
    time_of_day: str = Field(default="night", description="Current time of day")
    weather: Optional[str] = Field(default=None, description="Atmospheric weather condition")


class DeepObservationalAnalysis(BaseModel):
    """5-dimension reasoning output from extended thinking agent."""

    socioeconomic_reality: str = Field(
        description="Poverty physics, class signifiers, material access, repair/decay quality"
    )
    architectural_geometry: str = Field(
        description="Exact dimensional scales, curb heights, road widths, fixed landmarks"
    )
    environmental_physics: str = Field(
        description="Ground saturation, precipitation, puddle oil-sheens, atmospheric haze"
    )
    volumetric_lighting: str = Field(
        description="Primary and secondary light sources, Kelvin temperatures, shadow falloff"
    )
    botanical_micro_debris: str = Field(
        description="Hardy vegetation in fissures, fungal decay, discarded refuse, street clutter"
    )


class SinglePlateSpec(BaseModel):
    """Specification and artifact paths for a single plate in the location matrix."""

    role: PlateRole = Field(description="Role within the 4-plate standard")
    title: str = Field(description="Human-readable plate description")
    prompt: str = Field(description="Compiled Danbooru / SDXL scenic prompt")
    negative_prompt: str = Field(
        default="worst quality, low quality, blurry, 1boy, 1girl, people, crowds, character, human, distorted",
        description="Negative constraint prompt ensuring no hallucinated characters",
    )
    width: int = Field(default=1216, description="Pixel width")
    height: int = Field(default=832, description="Pixel height")
    azimuth_degrees: int = Field(default=0, description="Camera azimuth angle (0=front, 180=reverse)")
    image_path: Optional[str] = Field(default=None, description="Path to rendered master RGB plate")
    depth_map_path: Optional[str] = Field(default=None, description="Path to extracted structural depth guide")
    lineart_map_path: Optional[str] = Field(default=None, description="Path to extracted Canny wireframe")
    seed: Optional[int] = Field(default=None, description="Deterministic generation seed")


class LocationPlateMatrixSpec(BaseModel):
    """Complete 4-plate location reference matrix with memory and arc hierarchy."""

    location_id: str = Field(description="Canonical entity ID (e.g. 'loc_outskirts_bench_n24')")
    canonical_name: str = Field(description="Canonical display name of the environment")
    arc_hierarchy: ArcHierarchyTag = Field(description="Hierarchical arc coordinates")
    observation: DeepObservationalAnalysis = Field(description="Extended thinking observational analysis")
    plate_1_vista: SinglePlateSpec = Field(description="Wide exterior establishing vista")
    plate_2_stage: SinglePlateSpec = Field(description="Action/interaction zone plate")
    plate_3_reverse: SinglePlateSpec = Field(description="Reciprocal counter-shot plate")
    plate_4_macro: SinglePlateSpec = Field(description="Material / prop macro texture plate")
    is_existing: bool = Field(default=False, description="Whether this location was retrieved from historical memory")
    first_appearance_chapter: int = Field(default=1, description="Chapter where location was first introduced")
    last_appearance_chapter: int = Field(default=1, description="Most recent chapter where location appeared")
    temporal_delta: Optional[str] = Field(
        default=None, description="Delta from previous appearance (e.g. day->night or battle damage)"
    )


class ShotLocationAnchor(BaseModel):
    """Resolved camera angle and geometric anchor handed off to Main Agent for a specific shot."""

    location_id: str = Field(description="Canonical location ID")
    canonical_name: str = Field(description="Location name")
    resolved_role: PlateRole = Field(description="The chosen camera plate role")
    selected_plate_image_path: str = Field(description="Path to approved RGB plate")
    depth_map_path: Optional[str] = Field(default=None, description="Structural depth map for ControlNet")
    lineart_map_path: Optional[str] = Field(default=None, description="Lineart guide for edge locking")
    lighting_baseline: str = Field(description="Dominant lighting description for character shading harmonization")
    weather_condition: str = Field(description="Weather condition for character wetness / cloak physics")
    recommended_inpaint_box: Optional[list[float]] = Field(
        default=None, description="Normalized [x1, y1, x2, y2] bounding box where subject should be placed"
    )
    temporal_delta_tags: list[str] = Field(
        default_factory=list, description="Additional environment modifiers (e.g. ['scorched wall', 'broken bench'])"
    )

