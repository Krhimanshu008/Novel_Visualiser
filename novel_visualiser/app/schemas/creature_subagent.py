"""Novel Visualiser — Creature, Monster & Power System Subagent Data Contracts.

Defines schemas for:
1. CreatureRankEnum & CreatureClassEnum (2-axis power and scale hierarchy)
2. CreatureTaxonomyEnum (Biological and eldritch clades)
3. CreatureDeepObservation (5-dimension CoT reasoning on creature biology and threat physics)
4. CreatureScaleSpec (Scale ratios against human silhouettes, mass, and metric dimensions)
5. CreatureCard (Canonical monster asset with 4-sheet bestiary matrix and head chip)
6. ItemPropCard (Memories, weapons, artifacts with material physics and rank tiers)
7. ShotCreatureAnchor (Runtime battle prompt tokens, scale anchor, and IP-Adapter chip)
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field

from app.schemas.location_subagent import ArcHierarchyTag


class CreatureRankEnum(str, Enum):
    """Axis 1: Qualitative Soul Rank & Supernatural Power Tier."""
    DORMANT = "dormant"          # Rank 1: Mundane feral biology, unawakened
    AWAKENED = "awakened"        # Rank 2: Supernatural hardening, elemental sparks
    FALLEN = "fallen"            # Rank 3: High predatory cunning, dark sorcery, lethal to teams
    CORRUPTED = "corrupted"      # Rank 4: Reality-distorting miasma, city-level devastation
    GREAT = "great"              # Rank 5: Calamity level, godlike physical density and dread
    CURSED = "cursed"            # Rank 6: Continental bio-corruption, conceptual madness
    UNHOLY = "unholy"            # Rank 7: Deity / Titan tier, world extinction class


class CreatureClassEnum(str, Enum):
    """Axis 2: Soul Cores, Structural Scale & Swarm Command."""
    BEAST = "beast"              # 1 Soul Core: Small to Medium (1-3m), animalistic instinct
    MONSTER = "monster"          # 2 Soul Cores: Large (3-6m), reinforced bone/chitin, cunning
    DEMON = "demon"              # 3 Soul Cores: Large Humanoid / Quadruped (4-8m), martial weapons
    DEVIL = "devil"              # 4 Soul Cores: Variable size, dark magic, deceitful illusions
    TYRANT = "tyrant"            # 5 Soul Cores: Massive (8-15m), hive brood glands, spawns minions
    TERROR = "terror"            # 6 Soul Cores: Gigantic (15-30m), passive area madness/terror aura
    TITAN = "titan"              # 7 Soul Cores: Colossal / Mountain Scale (50-500m+), terrain leveler


class CreatureTaxonomyEnum(str, Enum):
    """Biological, elemental, and eldritch clades."""
    INSECTOID = "insectoid"                # Chitinous centipedes, scorpions, arachnids
    REPTILIAN_DRACONIC = "reptilian_draconic" # Obsidian scales, wyrms, horned basilisks
    ABYSSAL_AQUATIC = "abyssal_aquatic"    # Bioluminescent tentacles, deep-sea maws
    UNDEAD_NECROTIC = "undead_necrotic"    # Calcified bone, decaying stitched hide, soul flame
    MAMMALIAN_BEAST = "mammalian_beast"    # Dire wolves, sabertooth apes, matted blood fur
    ELDRITCH_ABERRATION = "eldritch_aberration" # Non-Euclidean limbs, shifting shadow flesh, eye clusters
    ELEMENTAL = "elemental"                # Living frost, magma rock, lightning wraith


class CreatureDeepObservation(BaseModel):
    """5-Dimension Observational Reasoning on monster anatomy and power manifestation."""

    power_system_soul_anatomy: str = Field(
        description="Rank, Class, soul core count, and supernatural aura manifestations"
    )
    skeletal_scale_mass: str = Field(
        description="Physical dimensions, height/length ratio against human silhouette, center of mass"
    )
    biomaterial_integument: str = Field(
        description="Carapace textures, scales, calcified bone, rotting tissue, quills, eye count"
    )
    locomotion_kinesics_weaponry: str = Field(
        description="Natural weaponry (mandibles, scythe limbs, tail stinger), stalking posture, kinetic strike speed"
    )
    environmental_corruption: str = Field(
        description="Acid slime melting stone, localized frost crystallization, shadow miasma"
    )


class CreatureScaleSpec(BaseModel):
    """Precise scale metrics for composition and human-relative framing."""

    estimated_length_meters: float = Field(default=3.0, description="Length/span in meters")
    estimated_height_meters: float = Field(default=1.8, description="Height at shoulder/crest in meters")
    human_scale_ratio: str = Field(
        default="1:2",
        description="Visual size ratio against a standard 1.75m human silhouette (e.g. '1:1', '1:4', '1:10')"
    )
    danbooru_scale_tags: list[str] = Field(
        default_factory=lambda: ["large creature", "towering"],
        description="Tags enforcing scale against surroundings"
    )


class SingleBestiaryPlateSpec(BaseModel):
    """Specification for one plate in the 4-sheet bestiary reference matrix."""

    sheet_index: int = Field(description="Sheet number 1 to 4")
    title: str = Field(description="Descriptive sheet title")
    role: str = Field(description="TURNAROUND, ACTION_MATRIX, CARAPACE_DETAIL, SCALE_COMPARISON")
    prompt: str = Field(description="Compiled Danbooru / SDXL prompt")
    negative_prompt: str = Field(
        default="worst quality, low quality, blurry, 1boy, 1girl, human, crowds, deformed limbs",
        description="Anti-features"
    )
    image_path: Optional[str] = Field(default=None, description="Path to rendered sheet artifact")
    seed: Optional[int] = Field(default=None)


class CreatureCard(BaseModel):
    """Canonical Creature/Monster Asset Card stored in SQLite."""

    creature_id: str = Field(description="Unique ID (e.g. 'creature_mountain_centipede_larva')")
    creature_name: str = Field(description="Canonical display name (e.g. 'Centipede of the Mountain')")
    novel_arc: ArcHierarchyTag = Field(description="Arc hierarchy where creature is active")
    valid_from_chapter: int = Field(default=1)
    valid_to_chapter: Optional[int] = Field(default=None)

    # Power System Classification
    rank: CreatureRankEnum = Field(default=CreatureRankEnum.AWAKENED)
    creature_class: CreatureClassEnum = Field(default=CreatureClassEnum.BEAST)
    taxonomy: CreatureTaxonomyEnum = Field(default=CreatureTaxonomyEnum.INSECTOID)
    soul_core_count: int = Field(default=1, description="Number of soul cores (1 for Beast, 5 for Tyrant, 7 for Titan)")

    # 5-Dimension Reasoning & Scale
    observation: CreatureDeepObservation
    scale_spec: CreatureScaleSpec

    # Danbooru Identity Locking Tokens
    core_danbooru_biology_tags: list[str] = Field(
        default_factory=list,
        description="Anatomical tokens (e.g. 'segmented black chitin', 'cluster of crimson eyes', 'triple mandibles')"
    )
    supernatural_aura_tags: list[str] = Field(
        default_factory=list,
        description="Aura tokens (e.g. 'dark mist', 'shadow aura', 'oozing green acid', 'soul flame')"
    )
    negative_identity_tags: list[str] = Field(
        default_factory=lambda: ["1boy", "1girl", "human face", "cute", "fluffy", "anime girl", "bipedal human"],
        description="Strict anti-features preventing humanoid or cute drift"
    )

    # 4-Sheet Bestiary Matrix (delegated via Character Agent engine)
    sheet_1_turnaround: SingleBestiaryPlateSpec
    sheet_2_action_threat: SingleBestiaryPlateSpec
    sheet_3_carapace_detail: SingleBestiaryPlateSpec
    sheet_4_scale_comparison: SingleBestiaryPlateSpec

    # Auto-cropped Predatory Head Chip for IP-Adapter Plus
    head_reference_chip_path: Optional[str] = Field(
        default=None,
        description="Path to cropped predatory head chip for ComfyUI IP-Adapter consistency"
    )


class ItemPropCard(BaseModel):
    """Canonical Hero Prop, Memory, or Weapon Asset Card."""

    item_id: str = Field(description="Unique ID (e.g. 'prop_memory_midnight_shard')")
    item_name: str = Field(description="Canonical item name (e.g. 'Midnight Shard')")
    item_type: str = Field(default="weapon", description="'weapon', 'armor', 'charm', 'relic', 'tool'")
    rank_tier: str = Field(default="Awakened", description="Soul Rank of memory (e.g. 'Dormant', 'Awakened', 'Ascended')")
    material_physics: str = Field(
        description="Material quality (e.g. 'reflective obsidian glass', 'damascus steel', 'etched silver runes')"
    )
    dimensions: str = Field(description="Physical scale (e.g. '85cm single-edged straight blade, 3cm wide')")
    danbooru_item_tags: list[str] = Field(
        default_factory=list,
        description="Tags for diffusion (e.g. 'holding sword', 'black blade', 'obsidian sword', 'glowing runes')"
    )
    negative_item_tags: list[str] = Field(
        default_factory=lambda: ["broken blade", "rust", "wooden sword", "blunt"],
        description="Anti-features"
    )
    reference_plate_path: Optional[str] = Field(
        default=None,
        description="Path to rendered 1024x1024 isolated prop plate on neutral background"
    )


class ShotCreatureAnchor(BaseModel):
    """Runtime monster anchor for compositing into combat battle scenes."""

    creature_card_id: str
    resolved_creature_prompt: str = Field(
        description="Compiled Danbooru tokens (biology + aura + scale + action)"
    )
    resolved_negative_prompt: str = Field(
        description="Compiled negative constraint tokens"
    )
    head_chip_path: Optional[str] = Field(
        default=None,
        description="Path to IP-Adapter predatory head chip"
    )
    suggested_spatial_anchor: str = Field(
        default="right_flank",
        description="Position in shot (e.g. 'background_looming', 'center_foreground', 'left_flank')"
    )
    human_scale_ratio: str = Field(
        default="1:2",
        description="Scale multiplier relative to human character"
    )

