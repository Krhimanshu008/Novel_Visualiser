"""Visual Consistency Engine (VCE) Data Contracts.

Defines schemas for:
1. VisualStyleProfile (Global style lock)
2. DanbooruIdentityLock (Immutable character identity tags)
3. OutfitCard (Arc-specific attire specifications)
4. LocationPlateSpec (Environment anchor tags & plate references)
5. SubjectVisualState & ShotVisualContext (Intermediate representation for prompt compilation)
6. PromptBlockCollection (8-block structured prompt representation)
"""

from __future__ import annotations

import hashlib
from typing import Any, Literal
from pydantic import BaseModel, Field


class VisualStyleProfile(BaseModel):
    """Locks the global visual style across chapters.
    
    Prevents mid-chapter aesthetic drift by enforcing fixed checkpoint,
    VAE, sampler, cfg, quality tags, and negative baseline.
    """

    profile_id: str = Field(default="STYLE-001", description="Unique style profile identifier")
    name: str = Field(default="Cinematic Manhwa", description="Descriptive profile name")
    checkpoint: str = Field(
        default="waiNSFWIllustrious_v110.safetensors",
        description="Target diffusion checkpoint",
    )
    vae: str = Field(
        default="sdxl_vae_fp16_fix.safetensors",
        description="Target VAE model",
    )
    sampler: str = Field(default="euler_ancestral", description="Sampler algorithm")
    scheduler: str = Field(default="normal", description="Noise scheduler")
    steps: int = Field(default=28, ge=15, le=60, description="Diffusion steps")
    cfg_scale: float = Field(default=6.0, ge=1.0, le=15.0, description="CFG guidance scale")
    width: int = Field(default=1024, description="Default image width")
    height: int = Field(default=1024, description="Default image height")
    master_quality_tags: list[str] = Field(
        default_factory=lambda: [
            "masterpiece",
            "best quality",
            "cinematic manhwa illustration",
            "clean lineart",
            "detailed shading",
            "vibrant color grading",
        ],
        description="Global positive quality anchor tags",
    )
    master_negative_tags: list[str] = Field(
        default_factory=lambda: [
            "low quality",
            "worst quality",
            "blurry",
            "bad anatomy",
            "deformed",
            "extra limbs",
            "flat shading",
            "photorealistic",
            "3d render",
            "watermark",
            "signature",
            "text",
            "jpeg artifacts",
        ],
        description="Global negative constraint tags",
    )
    locked: bool = Field(default=True, description="Whether this profile is locked against runtime mutation")
    version: int = Field(default=1, description="Version number of this style profile")


class DanbooruIdentityLock(BaseModel):
    """Immutable identity lock for a character using Danbooru tags.
    
    Decouples permanent facial and biological structure from temporal
    outfits, injuries, and arc gear.
    """

    entity_id: str = Field(description="Canonical entity ID (e.g. 'char_sunny')")
    entity_name: str = Field(description="Display name of the character")
    identity_version: str = Field(default="CANON-V1", description="Canon identity version")
    core_danbooru_tags: list[str] = Field(
        description="Core immutable Danbooru tags (e.g. ['1boy', 'pale skin', 'messy black hair', 'sharp dark onyx eyes', 'slender build'])"
    )
    negative_identity_tags: list[str] = Field(
        default_factory=list,
        description="Anti-features to prevent drift (e.g. ['blonde hair', 'blue eyes', 'muscular', 'beard'])",
    )
    token_weights: dict[str, float] = Field(
        default_factory=dict,
        description="Explicit token weights (e.g. {'(messy black hair:1.2)': 1.2})",
    )
    canonical_seed: int = Field(
        default=42,
        description="Deterministic base seed for character facial latent stability",
    )
    reference_image_path: str | None = Field(
        default=None,
        description="Path to approved character reference sheet / faceplate",
    )


class OutfitCard(BaseModel):
    """Arc-specific attire specifications.
    
    Treats clothing as an independent, modular asset that characters wear,
    preventing clothing tags from contaminating the character's base biology.
    """

    outfit_id: str = Field(description="Unique outfit identifier (e.g. 'OUTFIT-SUNNY-ARC1-01')")
    entity_id: str = Field(description="Entity this outfit belongs to")
    outfit_name: str = Field(description="Name (e.g. 'Outskirts Scavenger Tunic')")
    valid_from_chapter: int = Field(default=1, description="Earliest story chapter this outfit appears")
    valid_to_chapter: int | None = Field(default=None, description="Latest chapter, or None if ongoing")
    danbooru_clothing_tags: list[str] = Field(
        description="Danbooru tags for the clothing (e.g. ['black sleeveless tunic', 'leather vambraces', 'dark trousers', 'combat boots'])"
    )
    negative_clothing_tags: list[str] = Field(
        default_factory=list,
        description="Tags to suppress unwanted attire (e.g. ['modern clothes', 'hoodie', 'sneakers', 'plate armor'])",
    )
    reference_image_path: str | None = Field(
        default=None,
        description="Path to approved outfit reference sheet",
    )


class LocationPlateSpec(BaseModel):
    """Environment anchor specification.
    
    Maintains architectural and atmospheric consistency for scene backdrops.
    """

    location_id: str = Field(description="Canonical location ID (e.g. 'loc_forgotten_shore')")
    location_name: str = Field(description="Location name")
    plate_tags: list[str] = Field(
        description="Danbooru tags defining architecture, materials, and atmosphere"
    )
    negative_environment_tags: list[str] = Field(
        default_factory=list,
        description="Unwanted background elements (e.g. ['trees', 'modern buildings', 'sunlight'])",
    )
    plate_reference_path: str | None = Field(
        default=None,
        description="Approved master plate image path for ControlNet depth/lineart conditioning",
    )


class SubjectVisualState(BaseModel):
    """Resolved visual state for one subject in a specific shot."""

    entity_id: str = Field(description="Canonical entity identifier")
    entity_name: str = Field(description="Subject display name")
    identity_lock: DanbooruIdentityLock | None = Field(
        default=None, description="Resolved immutable Danbooru identity tags"
    )
    outfit: OutfitCard | None = Field(
        default=None, description="Resolved outfit tags for this arc/chapter"
    )
    held_equipment: list[str] = Field(
        default_factory=list,
        description="Weapons or items held in this shot (e.g. ['holding silver sword', 'leather scabbard'])",
    )
    temporal_deltas: list[str] = Field(
        default_factory=list,
        description="Chapter/scene specific wounds, dirt, sweat (e.g. ['scratch on cheek', 'dirt smudges'])",
    )
    pose_expression: str = Field(
        default="standing, looking at viewer, serious expression",
        description="Pose, action, and facial expression tags",
    )
    region_box: list[float] = Field(
        default_factory=lambda: [0.0, 0.0, 1.0, 1.0],
        description="[x1, y1, x2, y2] normalised coordinates for regional conditioning",
    )
    z_order: int = Field(default=0, description="0 = foreground, 1 = midground, 2 = background")


class ShotVisualContext(BaseModel):
    """The central intermediate representation for a single visual shot.
    
    Decouples the high-level scene direction and lore resolution from the
    low-level prompt compilation and ComfyUI workflow generation.
    """

    shot_id: str = Field(description="Unique shot ID (e.g. 'CH01_SC002_SH001')")
    chapter_number: int = Field(description="Chapter number (spoiler boundary)")
    style_profile_id: str = Field(default="STYLE-001", description="Style profile reference")
    workflow_id: str = Field(default="WF-SHOT-001", description="Target ComfyUI workflow")
    
    # Framing and Camera
    framing: Literal[
        "close_up",
        "medium_close_up",
        "medium",
        "medium_wide",
        "wide",
        "extreme_wide",
        "two_shot",
        "group",
        "pov",
        "over_the_shoulder",
        "insert",
    ] = Field(default="medium", description="Camera framing classification")
    camera_angle: Literal[
        "eye_level",
        "low_angle_dramatic",
        "high_angle",
        "dutch_angle",
        "birds_eye",
        "worms_eye",
        "profile",
    ] = Field(default="eye_level", description="Camera angle")
    focal_subject_id: str | None = Field(default=None, description="Primary entity in focus")
    
    # Environment
    environment: LocationPlateSpec | None = Field(
        default=None, description="Resolved location plate specification"
    )
    
    # Subjects
    subjects: list[SubjectVisualState] = Field(
        default_factory=list, description="Subjects present in this shot"
    )
    
    # Atmosphere & Lighting
    lighting_atmosphere: list[str] = Field(
        default_factory=lambda: ["cinematic lighting", "atmospheric depth"],
        description="Lighting, time of day, and environmental atmospheric tags",
    )
    vfx_particles: list[str] = Field(
        default_factory=list,
        description="Particle effects (e.g. ['falling snow', 'shadow tendrils', 'fire embers'])",
    )
    
    # Negative constraints
    negative_constraints: list[str] = Field(
        default_factory=list,
        description="Scene-specific negative prompt constraints",
    )
    
    # Deterministic Seed
    seed: int | None = Field(
        default=None,
        description="Deterministic latent seed. If None, auto-computed from hash",
    )

    def compute_deterministic_seed(self, base_seed: int = 42) -> int:
        """Deterministically derive a latent seed from shot attributes."""
        if self.seed is not None:
            return self.seed
        primary_id = self.focal_subject_id or (self.subjects[0].entity_id if self.subjects else "env")
        seed_source = f"{self.style_profile_id}:{primary_id}:{self.shot_id}:{base_seed}"
        digest = hashlib.sha256(seed_source.encode("utf-8")).hexdigest()
        computed = int(digest[:8], 16) % (2**31 - 1)
        return computed


class PromptBlockCollection(BaseModel):
    """The 8-Block structured representation of a compiled prompt."""

    block1_style: str = Field(description="Block 1: Global Style Anchor")
    block2_identity: str = Field(description="Block 2: Character Identity Anchor")
    block3_outfit: str = Field(description="Block 3: Appearance & Costume Block")
    block4_equipment: str = Field(description="Block 4: Equipment & Accessories Block")
    block5_environment: str = Field(description="Block 5: Environment Plate Anchor")
    block6_composition: str = Field(description="Block 6: Framing, Camera & Pose Block")
    block7_deltas_lighting: str = Field(description="Block 7: Temporal Deltas & Lighting Block")
    block8_negative: str = Field(description="Block 8: Negative Constraints Block")

    def full_positive(self) -> str:
        """Assemble blocks 1-7 into the final positive prompt string."""
        blocks = [
            self.block1_style,
            self.block2_identity,
            self.block3_outfit,
            self.block4_equipment,
            self.block5_environment,
            self.block6_composition,
            self.block7_deltas_lighting,
        ]
        active = [b.strip().rstrip(",") for b in blocks if b.strip()]
        return ", ".join(active)

    def full_negative(self) -> str:
        """Return block 8 as the final negative prompt string."""
        return self.block8_negative.strip().rstrip(",")

