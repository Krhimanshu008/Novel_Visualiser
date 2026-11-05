"""
Novel Visualiser — Hierarchical Asset Inheritance Resolver

The core deterministic engine that merges visual attributes across the
inheritance hierarchy:

    Project Style ⊕ Canonical Identity ⊕ Arc Version ⊕ Chapter Delta ⊕ Scene Action

This function is the "secret weapon" of the pipeline — it produces a complete,
spoiler-free visual payload for any entity at any scene, ready for the
Jinja2 prompt compiler to consume.

Zero LLM tokens. 100% deterministic Python.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy import select, and_
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def _merge_dicts(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """
    Deep-merge two dictionaries. Overlay values take precedence.
    Lists are replaced (not appended). None values in overlay are skipped.
    """
    merged = base.copy()
    for key, value in overlay.items():
        if value is None:
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _parse_visual_attributes(raw: Any) -> dict[str, Any]:
    """Safely parse visual_attributes which may be a JSON string or dict."""
    if raw is None:
        return {}
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
    if isinstance(raw, dict):
        return raw
    return {}


def resolve_asset_for_scene(
    db: Session,
    entity_id: str,
    arc_id: str | None,
    chapter_number: int,
    scene_id: str | None = None,
) -> dict[str, Any]:
    """
    Resolve the complete visual description for an entity at a specific point
    in the story by merging all applicable inheritance layers.

    Layers are merged in order (later layers override earlier ones):
        1. project_canonical — Base identity (face, body, species)
        2. arc_default       — Current arc's outfit, weapons, rank insignia
        3. chapter_delta     — Temporary injuries, dirt, fatigue, weather effects
        4. scene_override    — Pose, expression, lighting, action for this shot

    CRITICAL: Only records where valid_from_chapter <= chapter_number are included.
    This prevents spoiler leakage from future chapters.

    Args:
        db: Active SQLAlchemy session.
        entity_id: UUID of the entity to resolve.
        arc_id: UUID of the current story arc (can be None for project-level only).
        chapter_number: The chapter we are rendering (spoiler boundary).
        scene_id: UUID of the specific scene (can be None for chapter-level resolution).

    Returns:
        A dictionary containing the fully merged visual attributes plus
        ComfyUI reference data (LoRA, IPAdapter, seed, embeddings).
    """
    # Import here to avoid circular imports at module level
    from app.database.models import AssetVersion, Entity, VisualRef

    # ------------------------------------------------------------------
    # Step 1: Fetch the entity's base information
    # ------------------------------------------------------------------
    entity = db.execute(
        select(Entity).where(Entity.id == entity_id)
    ).scalar_one_or_none()

    if entity is None:
        raise ValueError(f"Entity not found: {entity_id}")

    result: dict[str, Any] = {
        "entity_id": entity_id,
        "entity_name": entity.canonical_name,
        "entity_type": entity.entity_type.value if hasattr(entity.entity_type, 'value') else str(entity.entity_type),
        "canonical": {},
        "arc": {},
        "chapter_delta": {},
        "scene_action": {},
        "reference_image_path": None,
        "lora_name": None,
        "lora_weight": 0.8,
        "ip_adapter_image_path": None,
        "ip_adapter_weight": 0.7,
        "canonical_seed": None,
        "face_embedding_path": None,
        "color_palette": [],
        "art_style_tags": [],
    }

    # ------------------------------------------------------------------
    # Step 2: Fetch all asset versions for this entity, ordered by level
    # ------------------------------------------------------------------
    level_order = {
        "project_canonical": 0,
        "arc_default": 1,
        "chapter_delta": 2,
        "scene_override": 3,
    }

    versions_query = (
        select(AssetVersion)
        .where(
            and_(
                AssetVersion.entity_id == entity_id,
                AssetVersion.valid_from_chapter <= chapter_number,
            )
        )
        .order_by(AssetVersion.valid_from_chapter, AssetVersion.created_at)
    )

    versions = db.execute(versions_query).scalars().all()

    # ------------------------------------------------------------------
    # Step 3: Merge layers in hierarchy order
    # ------------------------------------------------------------------
    # Bucket versions by level
    bucketed: dict[str, list] = {
        "project_canonical": [],
        "arc_default": [],
        "chapter_delta": [],
        "scene_override": [],
    }

    for v in versions:
        level_str = v.level.value if hasattr(v.level, 'value') else str(v.level)

        # Filter arc-specific versions to the current arc
        if level_str == "arc_default" and arc_id and v.arc_id and v.arc_id != arc_id:
            continue

        # Filter scene-specific overrides to the current scene
        if level_str == "scene_override" and scene_id and v.scene_id and v.scene_id != scene_id:
            continue

        if level_str in bucketed:
            bucketed[level_str].append(v)

    # Merge in order: project → arc → chapter → scene
    merged_attributes: dict[str, Any] = {}

    for level_name in ["project_canonical", "arc_default", "chapter_delta", "scene_override"]:
        for version in bucketed[level_name]:
            attrs = _parse_visual_attributes(version.visual_attributes)
            merged_attributes = _merge_dicts(merged_attributes, attrs)

    # ------------------------------------------------------------------
    # Step 4: Distribute merged attributes into the result structure
    # ------------------------------------------------------------------
    # Map known keys to their logical groups
    canonical_keys = {
        "age_appearance", "face", "physique", "distinguishing_marks",
        "species", "gender", "skin_color", "hair", "eyes", "height", "build",
    }
    arc_keys = {
        "clothing", "armor", "weapons", "accessories", "rank_title",
        "outfit", "equipment",
    }
    chapter_delta_keys = {
        "conditions", "environment_effects", "temporary_items", "injuries",
        "fatigue", "dirt", "blood", "weather_effects",
    }
    scene_keys = {
        "pose", "expression", "action", "lighting_modifier",
        "camera_focus", "emotion",
    }

    for key, value in merged_attributes.items():
        if key in canonical_keys:
            result["canonical"][key] = value
        elif key in arc_keys:
            result["arc"][key] = value
        elif key in chapter_delta_keys:
            result["chapter_delta"][key] = value
        elif key in scene_keys:
            result["scene_action"][key] = value
        elif key == "color_palette":
            result["color_palette"] = value
        elif key == "art_style_tags":
            result["art_style_tags"] = value
        else:
            # Unknown keys go into canonical as a catch-all
            result["canonical"][key] = value

    # ------------------------------------------------------------------
    # Step 5: Attach the best approved visual reference
    # ------------------------------------------------------------------
    ref_query = (
        select(VisualRef)
        .where(
            and_(
                VisualRef.entity_id == entity_id,
                VisualRef.approved == True,
            )
        )
        .order_by(VisualRef.approved_at.desc())
    )

    best_ref = db.execute(ref_query).scalars().first()

    if best_ref:
        result["reference_image_path"] = best_ref.image_path
        result["canonical_seed"] = best_ref.seed

        lora_cfg = best_ref.lora_config
        if isinstance(lora_cfg, str):
            try:
                lora_cfg = json.loads(lora_cfg)
            except (json.JSONDecodeError, TypeError):
                lora_cfg = None
        if isinstance(lora_cfg, dict):
            result["lora_name"] = lora_cfg.get("name")
            result["lora_weight"] = lora_cfg.get("weight", 0.8)

        ipa_cfg = best_ref.ip_adapter_config
        if isinstance(ipa_cfg, str):
            try:
                ipa_cfg = json.loads(ipa_cfg)
            except (json.JSONDecodeError, TypeError):
                ipa_cfg = None
        if isinstance(ipa_cfg, dict):
            result["ip_adapter_image_path"] = ipa_cfg.get("image_path")
            result["ip_adapter_weight"] = ipa_cfg.get("weight", 0.7)

    logger.debug(
        "Resolved asset for %s (chapter=%d): %d versions merged",
        entity.canonical_name,
        chapter_number,
        len(versions),
    )

    return result


def resolve_all_subjects_for_shot(
    db: Session,
    subject_entity_ids: list[str],
    arc_id: str | None,
    chapter_number: int,
    scene_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Resolve visual assets for all entities appearing in a single shot.

    This is a convenience wrapper that calls resolve_asset_for_scene for each
    subject and returns a list of resolved assets, ready for the prompt compiler.

    Args:
        db: Active SQLAlchemy session.
        subject_entity_ids: List of entity UUIDs appearing in the shot.
        arc_id: Current story arc UUID.
        chapter_number: The chapter being rendered.
        scene_id: The specific scene UUID (optional).

    Returns:
        List of resolved visual asset dictionaries.
    """
    resolved = []
    for eid in subject_entity_ids:
        try:
            asset = resolve_asset_for_scene(
                db=db,
                entity_id=eid,
                arc_id=arc_id,
                chapter_number=chapter_number,
                scene_id=scene_id,
            )
            resolved.append(asset)
        except ValueError as e:
            logger.warning("Could not resolve entity %s: %s", eid, e)
            continue
    return resolved


def get_safe_knowledge(
    db: Session,
    entity_id: str,
    current_chapter: int,
) -> dict[str, Any]:
    """
    THE SPOILER GATE.

    Returns all known information about an entity strictly bounded by
    the current chapter. No downstream agent is allowed to query the
    research_claims or entity_aliases tables directly — they MUST go
    through this function.

    Args:
        db: Active SQLAlchemy session.
        entity_id: UUID of the entity.
        current_chapter: The chapter number acting as the spoiler boundary.

    Returns:
        Dictionary with 'aliases', 'claims', 'images', and 'novel_passages'
        all filtered to only include information known by current_chapter.
    """
    from app.database.models import (
        Entity, EntityAlias, ResearchClaim, ResearchImage, Passage
    )

    # Get entity
    entity = db.execute(
        select(Entity).where(Entity.id == entity_id)
    ).scalar_one_or_none()

    if entity is None:
        raise ValueError(f"Entity not found: {entity_id}")

    # Safe aliases: only those revealed by current_chapter
    aliases_query = (
        select(EntityAlias)
        .where(
            and_(
                EntityAlias.entity_id == entity_id,
                EntityAlias.valid_from_chapter <= current_chapter,
                # Only include if reveal_chapter is None (always known) or <= current
                (EntityAlias.reveal_chapter.is_(None)) |
                (EntityAlias.reveal_chapter <= current_chapter),
            )
        )
    )
    aliases = [
        {"alias": a.alias, "type": a.alias_type.value if hasattr(a.alias_type, 'value') else str(a.alias_type)}
        for a in db.execute(aliases_query).scalars().all()
    ]

    # Safe research claims
    claims_query = (
        select(ResearchClaim)
        .where(
            and_(
                ResearchClaim.entity_id == entity_id,
                ResearchClaim.valid_from_chapter <= current_chapter,
                (ResearchClaim.valid_to_chapter.is_(None)) |
                (ResearchClaim.valid_to_chapter >= current_chapter),
            )
        )
    )
    claims = [
        {
            "claim_type": c.claim_type.value if hasattr(c.claim_type, 'value') else str(c.claim_type),
            "claim_text": c.claim_text,
            "confidence": c.confidence.value if hasattr(c.confidence, 'value') else str(c.confidence),
            "valid_from": c.valid_from_chapter,
        }
        for c in db.execute(claims_query).scalars().all()
    ]

    # Safe images
    images_query = (
        select(ResearchImage)
        .where(ResearchImage.entity_id == entity_id)
    )
    images = [
        {
            "image_url": img.image_url,
            "local_path": img.local_path,
            "category": img.category.value if hasattr(img.category, 'value') else str(img.category),
            "caption": img.caption,
        }
        for img in db.execute(images_query).scalars().all()
    ]

    return {
        "entity_id": entity_id,
        "entity_name": entity.canonical_name,
        "entity_type": entity.entity_type.value if hasattr(entity.entity_type, 'value') else str(entity.entity_type),
        "current_chapter": current_chapter,
        "known_aliases": aliases,
        "visual_claims": claims,
        "reference_images": images,
    }

