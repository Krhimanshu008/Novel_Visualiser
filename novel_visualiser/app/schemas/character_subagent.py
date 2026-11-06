"""Novel Visualiser — Character Continuity Subagent Data Contracts.

Defines schemas for:
1. ArchetypalDNA (Tier 1: Lifetime Invariable Genetic Core)
2. BiologicalEpoch (Tier 2: Arc-Level Physical Evolution & Maturation)
3. CharacterDeepObservation (5-dimension socioeconomic and biological CoT reasoning)
4. DanbooruIdentityLock (Active biological facial/body identity tags for the current epoch)
5. OutfitCard (Tier 3: Arc-scoped clothing with poverty wear, tear, and material specifics)
6. ItemCard (Hero props and handheld accessories)
7. CharacterSheetRole, SingleCharacterSheetSpec, and CharacterMatrixSpec (4-sheet standard)
8. ShotCharacterAnchor (Runtime character prompt, facial reference, and inpaint bounds)
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field

from app.schemas.location_subagent import ArcHierarchyTag


class CharacterSheetRole(str, Enum):
    """The 4 standardized reference sheet roles for character canon."""
    CANONICAL_TURNAROUND = "CANONICAL_TURNAROUND"  # Sheet 1: Front, 3/4, Profile, Back (1024x1024)
    EXPRESSION_GRID = "EXPRESSION_GRID"            # Sheet 2: 4-Expression Emotion Matrix (1024x1024)
    OUTFIT_PLATE = "OUTFIT_PLATE"                  # Sheet 3: Isolated Arc Wardrobe on T-pose (1024x1024)
    ITEM_SHEET = "ITEM_SHEET"                      # Sheet 4: Isolated Hero Props & Equipment (1024x1024)


class ArchetypalDNA(BaseModel):
    """Tier 1: Lifetime Invariable Archetypal Genetic Core.
    
    Traits that NEVER change across thousands of chapters regardless of power rank or age.
    """
    gender_tag: str = Field(default="1boy", description="Gender tag for diffusion (e.g. '1boy', '1man', '1girl')")
    hair_color_dna: str = Field(default="raven-black hair", description="Invariable hair pigmentation")
    eye_color_dna: str = Field(default="dark onyx eyes", description="Invariable ocular iris pigmentation")
    facial_bone_structure: str = Field(
        default="sharp delicate jawline, porcelain bone structure",
        description="Core bone structure contours"
    )
    permanent_negative_tags: list[str] = Field(
        default_factory=lambda: ["blonde hair", "blue eyes", "beard", "round face", "chubby", "fat"],
        description="Universal anti-features that never enter the character latent space"
    )


class BiologicalEpoch(BaseModel):
    """Tier 2: Arc-Level Physical Evolution & Maturation Stage.
    
    Managed by the Main Orchestrator Agent using high-end frontier AI.
    Controls age progression, physique shifts, skin purification, and supernatural glow.
    """
    epoch_id: str = Field(description="Unique epoch identifier (e.g. 'epoch_01_slum_orphan')")
    epoch_name: str = Field(description="Display title (e.g. 'Outskirts Malnourished Slum Orphan')")
    valid_from_chapter: int = Field(default=1, description="First chapter where this biology takes effect")
    valid_to_chapter: Optional[int] = Field(default=None, description="Last chapter (None if currently active)")
    age_appearance: str = Field(default="16", description="Apparent age bracket ('adolescent', 'young adult', 'adult')")
    physique_tags: list[str] = Field(
        description="Build/stature tags (e.g. ['scrawny', 'slender build'] -> ['lean sinewy'] -> ['athletic build'])"
    )
    skin_complexion_tags: list[str] = Field(
        description="Dermal traits (e.g. ['sallow skin', 'pale skin'] -> ['porcelain skin', 'alabaster skin', 'subtle glow'])"
    )
    ocular_fatigue_tags: list[str] = Field(
        default_factory=list,
        description="Eye condition (e.g. ['dark circles under eyes', 'sunken eyes'] -> ['obsidian eyes', 'shadow depth'])"
    )
    supernatural_manifestation_tags: list[str] = Field(
        default_factory=list,
        description="Auras/aspect marks (e.g. ['ethereal shadow glow', 'ambient shadow mist'])"
    )
    retired_negative_tags: list[str] = Field(
        default_factory=list,
        description="Tags from previous outgrown epochs that must now be FORBIDDEN (e.g. 'scrawny', 'sallow skin')"
    )


class CharacterDeepObservation(BaseModel):
    """5-dimension reasoning output from extended thinking agent for a character."""

    socioeconomic_nutrition: str = Field(
        description="Physical indicators of economic status, malnutrition, sallow skin, skeletal lean build"
    )
    facial_morphology: str = Field(
        description="Exact facial contours, raven-black hair bangs, sharp jaw, dark hollow onyx eyes"
    )
    wardrobe_poverty_textile: str = Field(
        description="Specific fabric textures, frayed seams, broken zippers, oil/rain stains, patches"
    )
    emotional_expressions: str = Field(
        description="Psychological emotional markers: cynical smirk, grimace, exhausted drooping eyelids"
    )
    posture_body_language: str = Field(
        description="Kinesics: hunched defensive shoulders, clutching paper cup, guarded seated posture"
    )


class DanbooruIdentityLock(BaseModel):
    """Active biological identity of a character for the current epoch."""

    entity_id: str = Field(description="Canonical entity ID (e.g. 'char_sunny')")
    entity_name: str = Field(description="Display name of the character")
    archetypal_dna: ArchetypalDNA = Field(default_factory=ArchetypalDNA)
    active_epoch: Optional[BiologicalEpoch] = None
    gender_tag: str = Field(default="1boy", description="Gender tag for diffusion (e.g. '1boy', '1man')")
    core_danbooru_tags: list[str] = Field(
        description="Active biological Danbooru tags for this epoch"
    )
    negative_identity_tags: list[str] = Field(
        default_factory=list,
        description="Anti-features to prevent drift, combining DNA negatives with retired epoch tags",
    )
    token_weights: dict[str, float] = Field(
        default_factory=dict,
        description="Explicit token weights (e.g. {'(messy black hair:1.2)': 1.2})",
    )
    canonical_seed: int = Field(default=42, description="Deterministic base seed for character facial latent stability")
    reference_face_path: Optional[str] = Field(
        default=None, description="Path to approved canonical face crop for IP-Adapter Plus"
    )


class OutfitCard(BaseModel):
    """Tier 3: Modular outfit definition scoped to a specific story arc."""

    outfit_id: str = Field(description="Unique outfit identifier (e.g. 'outfit_sunny_outskirts_rags')")
    outfit_name: str = Field(description="Display name (e.g. 'Outskirts Frayed Utility Jacket & Rags')")
    danbooru_clothing_tags: list[str] = Field(
        description="Exact Danbooru tags for this attire, e.g. ['oversized dark grey utility jacket', 'frayed collar', 'patched trousers', 'worn combat boots']"
    )
    poverty_wear_tags: list[str] = Field(
        default_factory=list,
        description="Wear-and-tear descriptors, e.g. ['frayed seams', 'dirt stains', 'tarnished zipper', 'faded fabric']",
    )
    negative_clothing_tags: list[str] = Field(
        default_factory=list,
        description="Forbidden clothing tags, e.g. ['clean clothes', 'business suit', 'armor', 'hoodie']",
    )
    reference_image_path: Optional[str] = Field(
        default=None, description="Path to isolated outfit reference sheet"
    )


class ItemCard(BaseModel):
    """Modular definition for hero handheld props and accessories."""

    item_id: str = Field(description="Unique item identifier (e.g. 'item_synth_coffee_cup')")
    item_name: str = Field(description="Item name (e.g. 'Crushed Paper Synth-Coffee Cup')")
    danbooru_item_tags: list[str] = Field(
        description="Exact Danbooru tags describing the object, e.g. ['holding paper coffee cup', 'dented paper cup', 'cheap white cup with plastic lid']"
    )
    negative_item_tags: list[str] = Field(
        default_factory=list,
        description="Forbidden item tags (e.g. ['ceramic mug', 'sword', 'steampunk cup'])",
    )
    reference_image_path: Optional[str] = Field(default=None, description="Path to isolated item reference sheet")


class SingleCharacterSheetSpec(BaseModel):
    """Specification and artifact paths for a single sheet in the character matrix."""

    role: CharacterSheetRole = Field(description="Role within the 4-sheet standard")
    title: str = Field(description="Human-readable sheet title")
    prompt: str = Field(description="Compiled Danbooru prompt for sheet generation")
    negative_prompt: str = Field(
        default="worst quality, low quality, blurry, extra limbs, bad anatomy, deformed fingers, text, watermark",
        description="Negative constraint prompt",
    )
    width: int = Field(default=1024, description="Pixel width")
    height: int = Field(default=1024, description="Pixel height")
    image_path: Optional[str] = Field(default=None, description="Path to rendered master RGB sheet")
    seed: Optional[int] = Field(default=None, description="Deterministic generation seed")


class CharacterMatrixSpec(BaseModel):
    """Complete character reference matrix with memory, identity locks, and wardrobe."""

    character_id: str = Field(description="Canonical entity ID (e.g. 'char_sunny')")
    canonical_name: str = Field(description="Canonical character name (e.g. 'Sunny / Sunless')")
    gender: str = Field(default="male", description="Gender of the character")
    arc_hierarchy: ArcHierarchyTag = Field(description="Hierarchical arc coordinate")
    archetypal_dna: ArchetypalDNA = Field(default_factory=ArchetypalDNA)
    active_epoch: Optional[BiologicalEpoch] = None
    observation: CharacterDeepObservation = Field(description="5-dimension deep observational reasoning")
    identity_lock: DanbooruIdentityLock = Field(description="Immutable biological identity lock")
    active_outfit: OutfitCard = Field(description="Current wardrobe scoped to this arc")
    held_items: list[ItemCard] = Field(default_factory=list, description="Hero props active in this scene")
    sheet_1_turnaround: SingleCharacterSheetSpec = Field(description="Neutral turnaround model sheet")
    sheet_2_expressions: SingleCharacterSheetSpec = Field(description="4-expression emotion matrix")
    sheet_3_outfit: SingleCharacterSheetSpec = Field(description="Arc-specific wardrobe plate")
    sheet_4_items: SingleCharacterSheetSpec = Field(description="Hero props reference plate")
    is_existing: bool = Field(default=False, description="Whether this character was retrieved from historical memory")
    first_appearance_chapter: int = Field(default=1, description="Chapter where character was introduced")
    last_appearance_chapter: int = Field(default=1, description="Most recent chapter where character appeared")
    temporal_deltas: list[str] = Field(
        default_factory=list,
        description="Scene/chapter micro-deltas (e.g. ['dark circles under eyes', 'rain-soaked hair', 'grimace'])",
    )


class ShotCharacterAnchor(BaseModel):
    """Resolved character prompt block and references ready for inpainting into a location stage canvas."""

    entity_id: str = Field(description="Character entity ID")
    entity_name: str = Field(description="Character name")
    full_positive_prompt: str = Field(description="Ordered, locked positive prompt block")
    full_negative_prompt: str = Field(description="Comprehensive negative prompt block")
    face_reference_path: Optional[str] = Field(
        default=None, description="Image path to approved face crop for IP-Adapter Plus"
    )
    expression_tag: str = Field(description="Resolved facial expression tag for this shot")
    pose_action_tag: str = Field(description="Resolved posture/action tag for this shot")
    recommended_inpaint_box: Optional[list[float]] = Field(
        default=None, description="Normalized [x1, y1, x2, y2] bounding box within the location stage canvas"
    )
