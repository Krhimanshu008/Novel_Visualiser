"""
Novel Visualiser — Visual Consistency Architect (VCA)

Powered by Kaggle's $10/day Model Proxy (Claude Sonnet 4 / Gemini 2.5 Pro).
Solves the root causes of character drift, clothing drift, and environment instability.

Key Innovations:
1. Separation of Immutable Identity (face, physique, hair) from Mutable Appearance (outfits, damage).
2. Danbooru Tag Translation for Illustrious / SDXL checkpoints (replaces ambiguous prose).
3. Modular Outfit & Accessory Anchors.
4. Master Location Plate Generation Specs for Depth ControlNet anchoring.
5. Deterministic Ordered Prompt Block Assembly.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Optional
from pydantic import BaseModel, Field

from kaggle_ai_engine.proxy_client import KaggleModelProxyClient

logger = logging.getLogger("kaggle_ai_engine.consistency")


class CharacterIdentityCard(BaseModel):
    """Immutable physical identity of a character (never drifts across chapters)."""
    character_name: str
    gender: str
    age_appearance: str
    danbooru_identity_tags: list[str] = Field(
        description="Core immutable Danbooru tags, e.g. ['1boy', 'pale skin', 'messy black hair', 'dark eyes', 'lean youth']"
    )
    distinguishing_features: list[str] = Field(
        default_factory=list,
        description="Permanent scars, eye shapes, tattoos, or marks"
    )
    forbidden_drift_tags: list[str] = Field(
        default_factory=list,
        description="Tags that must NEVER be generated (e.g. 'muscular', 'blonde hair', 'cheerful smile')"
    )


class OutfitCard(BaseModel):
    """Modular outfit definition with independent versioning."""
    outfit_id: str
    outfit_name: str
    danbooru_clothing_tags: list[str] = Field(
        description="Exact Danbooru tags for this attire, e.g. ['black leather tunic', 'dark trousers', 'fingerless gloves']"
    )
    accessories: list[str] = Field(
        default_factory=list,
        description="Specific gear worn with this outfit, e.g. ['silver chain necklace', 'worn scabbard']"
    )
    color_palette: list[str] = Field(
        default_factory=list,
        description="Primary colors of the outfit, e.g. ['charcoal black', 'dull silver']"
    )


class LocationPlateSpec(BaseModel):
    """Master plate prompt specification to anchor room/landscape geometry."""
    location_name: str
    environment_type: str
    danbooru_landscape_tags: list[str] = Field(
        description="Tags for empty establishing plate without characters"
    )
    architectural_features: list[str] = Field(
        default_factory=list,
        description="Key landmarks or structural anchors that must stay stationary"
    )
    lighting_baseline: str = Field(
        default="dim cold ambient lighting, shadows",
        description="Atmospheric lighting for this setting"
    )


class VisualConsistencyArchitect:
    """
    Synthesizes and enforces strict visual continuity specifications
    using the Kaggle Model Proxy frontier reasoning engine.
    """

    def __init__(self, client: Optional[KaggleModelProxyClient] = None):
        self.client = client or KaggleModelProxyClient()

    def build_character_identity(
        self,
        character_name: str,
        novel_context: str,
        prose_descriptions: list[str],
    ) -> CharacterIdentityCard:
        """
        Synthesize an immutable Danbooru identity card for a novel character.
        Converts literary descriptions into high-precision tags recognized by SDXL / Illustrious.
        """
        prompt = (
            f"You are a master anime visual director and prompt engineer specializing in SDXL and Illustrious models.\n"
            f"Character: {character_name}\n"
            f"Novel / Universe: {novel_context}\n\n"
            f"Source prose excerpts describing this character's physical traits:\n"
            + "\n".join(f"- {p}" for p in prose_descriptions)
            + "\n\nSynthesize an immutable CharacterIdentityCard containing EXACT Danbooru-style tags for their "
            "permanent physical face, hair, eyes, skin, and body build. Exclude all clothing, weapons, and temporary injuries!"
        )

        return self.client.structured_extract(
            prompt=prompt,
            schema=CharacterIdentityCard,
            system_instruction="You build permanent anime character identity locks with Danbooru tags.",
        )

    def build_outfit_card(
        self,
        character_name: str,
        outfit_name: str,
        story_arc: str,
        prose_excerpt: str,
    ) -> OutfitCard:
        """
        Synthesize a modular outfit specification for a character in a specific story arc.
        """
        prompt = (
            f"Character: {character_name}\n"
            f"Story Arc / Context: {story_arc}\n"
            f"Outfit Name: {outfit_name}\n"
            f"Prose description:\n\"{prose_excerpt}\"\n\n"
            f"Extract the exact clothing and accessory items into Danbooru tags for SDXL generation."
        )

        return self.client.structured_extract(
            prompt=prompt,
            schema=OutfitCard,
            system_instruction="You extract anime outfit and accessory tags.",
        )

    def build_location_plate(
        self,
        location_name: str,
        scene_description: str,
    ) -> LocationPlateSpec:
        """
        Synthesize a master environment plate specification (no characters)
        to be used for ControlNet depth / geometry anchoring.
        """
        prompt = (
            f"Location: {location_name}\n"
            f"Description: \"{scene_description}\"\n\n"
            f"Create a Master Location Plate specification. The tags must describe ONLY the setting, "
            f"architecture, textures, and lighting with NO humans, NO characters, NO crowd."
        )

        return self.client.structured_extract(
            prompt=prompt,
            schema=LocationPlateSpec,
            system_instruction="You generate master anime background and architecture plate prompts.",
        )

    @staticmethod
    def compile_locked_prompt(
        identity: CharacterIdentityCard,
        outfit: Optional[OutfitCard],
        location: Optional[LocationPlateSpec],
        shot_framing: str = "medium shot",
        camera_angle: str = "eye level",
        action_pose: Optional[str] = None,
        expression: Optional[str] = None,
        state_deltas: Optional[list[str]] = None,
        lighting_vfx: Optional[str] = None,
    ) -> str:
        """
        Deterministic Ordered Block Compilation for Illustrious / SDXL:
        [1. MASTER QUALITY] -> [2. FRAMING & ANGLE] -> [3. IMMUTABLE IDENTITY] ->
        [4. OUTFIT] -> [5. ACTION & EXPRESSION] -> [6. STATE DELTAS] ->
        [7. ENVIRONMENT] -> [8. LIGHTING & VFX]
        """
        blocks = []

        # 1. Master Quality
        blocks.append("masterpiece, best quality, highly detailed anime illustration")

        # 2. Framing & Angle
        blocks.append(f"{shot_framing}, {camera_angle}")

        # 3. Immutable Identity Lock (verbatim tags, never randomized)
        blocks.extend(identity.danbooru_identity_tags)

        # 4. Outfit Lock
        if outfit and outfit.danbooru_clothing_tags:
            blocks.extend(outfit.danbooru_clothing_tags)

        # 5. Pose & Expression
        if action_pose:
            blocks.append(action_pose)
        if expression:
            blocks.append(expression)

        # 6. State Deltas (Injuries, dirt, blood, torn clothes)
        if state_deltas:
            blocks.extend(state_deltas)

        # 7. Environment Anchor
        if location and location.danbooru_landscape_tags:
            blocks.extend(location.danbooru_landscape_tags[:4])

        # 8. Lighting & VFX
        if lighting_vfx:
            blocks.append(lighting_vfx)
        elif location and location.lighting_baseline:
            blocks.append(location.lighting_baseline)

        return ", ".join(dict.fromkeys(blocks))  # Deduplicate preserving order

    @staticmethod
    def compute_deterministic_seed(project_name: str, chapter_num: int, shot_num: int) -> int:
        """
        Compute a reproducible integer seed based on project, chapter, and shot number.
        Guarantees that re-running the exact shot produces consistent seed conditions.
        """
        key = f"{project_name}_ch{chapter_num:03d}_shot{shot_num:03d}"
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()
        # SDXL seed range [0, 2147483647]
        return int(h[:8], 16) % 2147483647

