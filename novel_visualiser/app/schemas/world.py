"""World Architecture & Asset Inheritance contracts.

Schemas for entity management, visual asset inheritance layers (canonical →
arc → chapter → scene), the visual resolver output, alias mappings, and
canonical visual profiles produced by the World Architect Agent.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CanonicalIdentity(BaseModel):
    """Base physical identity of a character that rarely changes."""

    age_appearance: str | None = Field(
        default=None, description="Apparent age descriptor, e.g. '16-year-old slender youth'"
    )
    face: str | None = Field(
        default=None,
        description="Facial features description, e.g. 'pale gaunt skin, sharp dark eyes, messy jet-black hair'",
    )
    physique: str | None = Field(
        default=None,
        description="Body build descriptor, e.g. 'short, lean, malnourished frame'",
    )
    distinguishing_marks: list[str] = Field(
        default_factory=list, description="Scars, tattoos, birthmarks, etc."
    )
    species: str | None = Field(default=None, description="Species or race if non-human")
    gender: str | None = Field(default=None, description="Gender presentation")


class ArcAppearance(BaseModel):
    """Arc-level default outfit/equipment state."""

    clothing: str | None = Field(
        default=None, description="Default clothing for this story arc"
    )
    armor: str | None = Field(
        default=None, description="Armor description if applicable"
    )
    weapons: list[str] = Field(
        default_factory=list, description="Weapons carried during this arc"
    )
    accessories: list[str] = Field(
        default_factory=list, description="Accessories worn during this arc"
    )
    rank_title: str | None = Field(
        default=None, description="Rank or title if visual rank insignia exists"
    )


class ChapterDelta(BaseModel):
    """Chapter-specific temporary visual modifications."""

    conditions: list[str] = Field(
        default_factory=list,
        description="Temporary visual conditions, e.g. 'bleeding forehead', 'torn tunic'",
    )
    environment_effects: list[str] = Field(
        default_factory=list,
        description="Environment-driven effects, e.g. 'covered in mud', 'soaked from rain'",
    )
    temporary_items: list[str] = Field(
        default_factory=list, description="Temporarily held or worn items"
    )
    injuries: list[str] = Field(
        default_factory=list, description="Visible injuries sustained this chapter"
    )


class SceneAction(BaseModel):
    """Scene-specific pose, expression, and camera directives."""

    pose: str | None = Field(default=None, description="Character body pose")
    expression: str | None = Field(
        default=None, description="Facial expression directive"
    )
    action: str | None = Field(
        default=None, description="Active motion, e.g. 'swinging sword overhead'"
    )
    lighting_modifier: str | None = Field(
        default=None, description="Per-subject lighting, e.g. 'backlit by fire'"
    )


class ResolvedVisualAsset(BaseModel):
    """The FINAL merged visual description for rendering a specific entity in a specific scene.

    This is the output of resolver.py — consumed by the Jinja2 prompt compiler.
    """

    entity_id: str = Field(description="Canonical entity identifier")
    entity_name: str = Field(description="Display name of the entity")
    entity_type: Literal["character", "location", "creature", "item"] = Field(
        description="Entity classification"
    )

    # Merged layers
    canonical: CanonicalIdentity = Field(
        description="Base physical identity layer"
    )
    arc: ArcAppearance = Field(description="Arc-level appearance layer")
    chapter_delta: ChapterDelta = Field(
        description="Chapter-specific temporary modifications"
    )
    scene_action: SceneAction = Field(
        description="Scene-specific pose and expression directives"
    )

    # ComfyUI reference data
    reference_image_path: str | None = Field(
        default=None, description="Path to the reference image for this entity"
    )
    lora_name: str | None = Field(
        default=None, description="LoRA model name for entity-specific fine-tuning"
    )
    lora_weight: float = Field(
        default=0.8, description="LoRA influence weight"
    )
    ip_adapter_image_path: str | None = Field(
        default=None, description="Path to IP-Adapter reference image"
    )
    ip_adapter_weight: float = Field(
        default=0.7, description="IP-Adapter influence weight"
    )
    canonical_seed: int | None = Field(
        default=None, description="Fixed seed for deterministic generation of this entity"
    )
    face_embedding_path: str | None = Field(
        default=None, description="Path to InsightFace embedding file"
    )

    # Style
    color_palette: list[str] = Field(
        default_factory=list, description="Dominant colors associated with this entity"
    )
    art_style_tags: list[str] = Field(
        default_factory=list, description="Stylistic tags, e.g. 'anime', 'dark_fantasy'"
    )


class EntityAliasMapping(BaseModel):
    """Maps an alias to its canonical entity, respecting spoiler boundaries."""

    alias: str = Field(description="The alias string")
    entity_id: str = Field(description="Canonical entity ID this alias resolves to")
    canonical_name: str = Field(description="The entity's canonical display name")
    alias_type: Literal[
        "name", "title", "nickname", "description_reference", "codename"
    ] = Field(description="Classification of this alias")
    valid_from_chapter: int = Field(
        default=1, description="Chapter from which this alias is active"
    )
    reveal_chapter: int | None = Field(
        default=None,
        description="Chapter in which the reader learns this alias connection",
    )


class CanonicalVisualProfile(BaseModel):
    """Full canonical visual profile for an entity, produced by World Architect Agent."""

    entity_id: str = Field(description="Unique entity identifier")
    entity_type: str = Field(description="Entity classification")
    canonical_name: str = Field(description="Primary display name")
    base_identity: CanonicalIdentity = Field(
        description="Base physical identity of the entity"
    )
    initial_appearance: ArcAppearance = Field(
        description="Initial arc-level appearance"
    )
    art_style_notes: str | None = Field(
        default=None, description="Free-text art style guidance"
    )
    color_palette: list[str] = Field(
        default_factory=list, description="Signature colours for this entity"
    )
    visual_reference_urls: list[str] = Field(
        default_factory=list, description="External reference image URLs"
    )
    source_description: str = Field(
        description="The novel text that describes this entity"
    )
    confidence: Literal["confirmed", "likely", "speculative"] = Field(
        default="likely",
        description="Confidence level in the visual profile accuracy",
    )
