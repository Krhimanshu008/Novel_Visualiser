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


def _parse_json_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        return [str(x).strip() for x in parsed if str(x).strip()] if isinstance(parsed, list) else []
    except Exception:
        return [x.strip() for x in raw.split(",") if x.strip()]


def get_safe_knowledge(
    db: Session,
    entity_id: str,
    current_chapter: int,
) -> dict[str, Any]:
    """
    THE SPOILER GATE & TEMPORAL ERA RESOLVER (0 LLM Tokens).

    Returns all known information about an entity strictly bounded by
    the current chapter. No downstream agent is allowed to query the
    research_claims or entity_aliases tables directly — they MUST go
    through this function.

    - Blocks entities before their debut chapter.
    - Filters aliases where valid_from_chapter <= current_chapter and reveal_chapter <= current_chapter.
    - Filters claims active during current_chapter: valid_from <= current_chapter <= valid_until.
    - Automatically turns FUTURE claims (valid_from > current_chapter) into anti_spoiler_negative_tags!
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

    first_appearance = getattr(entity, "first_appearance_chapter", 1) or 1
    if first_appearance > current_chapter:
        return {
            "status": "locked_spoiler",
            "allowed": False,
            "entity_id": entity.id,
            "entity_name": entity.canonical_name,
            "canonical_name": entity.canonical_name,
            "entity_type": entity.entity_type.value if hasattr(entity.entity_type, 'value') else str(entity.entity_type),
            "current_chapter": current_chapter,
            "reason": f"Entity '{entity.canonical_name}' debuts at Ch. {first_appearance} (Current: Ch. {current_chapter})",
            "known_aliases": [],
            "visual_claims": [],
            "reference_images": [],
            "safe_aliases": [],
            "blocked_spoiler_aliases": [],
            "active_claims": [],
            "danbooru_prompt": "",
            "flux_natural_prompt": "",
            "anti_spoiler_negative_prompt": "",
            "canonical_reference_image": None,
        }

    # 1. Resolve Safe vs Blocked Aliases
    all_aliases = db.execute(
        select(EntityAlias).where(EntityAlias.entity_id == entity_id)
    ).scalars().all()

    safe_aliases: list[str] = []
    blocked_aliases: list[str] = []
    aliases_legacy: list[dict[str, str]] = []

    for a in all_aliases:
        reveal_ch = a.reveal_chapter if a.reveal_chapter is not None else a.valid_from_chapter
        is_safe = (a.valid_from_chapter <= current_chapter) and (reveal_ch <= current_chapter)
        if is_safe:
            safe_aliases.append(a.alias)
            aliases_legacy.append({
                "alias": a.alias,
                "type": a.alias_type.value if hasattr(a.alias_type, 'value') else str(a.alias_type),
            })
        else:
            blocked_aliases.append(a.alias)

    # 2. Resolve Active Claims vs Future Spoiler Claims (Auto-Inverted into Negatives)
    all_claims = db.execute(
        select(ResearchClaim).where(ResearchClaim.entity_id == entity_id)
    ).scalars().all()

    active_claims: list[dict[str, Any]] = []
    claims_legacy: list[dict[str, Any]] = []
    safe_danbooru_tags: list[str] = []
    flux_descriptions: list[str] = []
    anti_spoiler_negative_tags: list[str] = []

    for c in all_claims:
        until_ch = getattr(c, "valid_until_chapter", None)
        if until_ch is None:
            until_ch = getattr(c, "valid_to_chapter", None)

        is_started = c.valid_from_chapter <= current_chapter
        is_not_expired = (until_ch is None) or (until_ch >= current_chapter)

        c_tags = _parse_json_list(getattr(c, "danbooru_tags", None))
        c_neg = _parse_json_list(getattr(c, "negative_tags", None))

        if is_started and is_not_expired:
            active_claims.append({
                "id": c.id,
                "era_name": getattr(c, "era_name", None) or "Base Era",
                "claim_type": c.claim_type.value if hasattr(c.claim_type, 'value') else str(c.claim_type),
                "claim_text": c.claim_text,
                "valid_from_chapter": c.valid_from_chapter,
                "valid_until_chapter": until_ch,
            })
            claims_legacy.append({
                "claim_type": c.claim_type.value if hasattr(c.claim_type, 'value') else str(c.claim_type),
                "claim_text": c.claim_text,
                "confidence": c.confidence.value if hasattr(c.confidence, 'value') else str(c.confidence),
                "valid_from": c.valid_from_chapter,
            })
            for t in c_tags:
                if t not in safe_danbooru_tags:
                    safe_danbooru_tags.append(t)
            for nt in c_neg:
                if nt not in anti_spoiler_negative_tags:
                    anti_spoiler_negative_tags.append(nt)

            flux_p = getattr(c, "flux_prompt", None)
            if flux_p and flux_p not in flux_descriptions:
                flux_descriptions.append(flux_p)

        elif not is_started:
            # Future spoiler claim -> automatically push its visual tags into negative prompts!
            for future_tag in c_tags:
                if future_tag not in safe_danbooru_tags and future_tag not in anti_spoiler_negative_tags:
                    anti_spoiler_negative_tags.append(future_tag)

    anti_spoiler_negative_tags = [t for t in anti_spoiler_negative_tags if t not in safe_danbooru_tags]

    # 3. Resolve Reference Images
    images_db = db.execute(
        select(ResearchImage).where(ResearchImage.entity_id == entity_id)
    ).scalars().all()

    images_legacy = [
        {
            "image_url": img.image_url,
            "local_path": img.local_path,
            "category": img.category.value if hasattr(img.category, 'value') else str(img.category),
            "caption": img.caption,
        }
        for img in images_db
    ]

    canonical_image = None
    for preferred_cat in ("official_cover", "official_illustration", "reference_photo", "community_art"):
        for img in images_db:
            c_val = img.category.value if hasattr(img.category, 'value') else str(img.category)
            if c_val == preferred_cat:
                canonical_image = img.local_path or img.image_url
                break
        if canonical_image:
            break
    if not canonical_image and images_db:
        canonical_image = images_db[0].local_path or images_db[0].image_url

    flux_text = " ".join(flux_descriptions) if flux_descriptions else "; ".join(c["claim_text"] for c in active_claims)

    return {
        "status": "ok",
        "allowed": True,
        "entity_id": entity_id,
        "entity_name": entity.canonical_name,
        "canonical_name": entity.canonical_name,
        "entity_type": entity.entity_type.value if hasattr(entity.entity_type, 'value') else str(entity.entity_type),
        "current_chapter": current_chapter,
        "safe_aliases": safe_aliases,
        "blocked_spoiler_aliases": blocked_aliases,
        "known_aliases": aliases_legacy,
        "active_claims": active_claims,
        "visual_claims": claims_legacy,
        "reference_images": images_legacy,
        "danbooru_prompt": ", ".join(safe_danbooru_tags),
        "flux_natural_prompt": flux_text,
        "anti_spoiler_negative_prompt": ", ".join(anti_spoiler_negative_tags),
        "canonical_reference_image": canonical_image,
    }


def resolve_shot_visual_context(
    db: Session,
    shot: Any,
    chapter_number: int,
    arc_id: str | None = None,
    style_profile_id: str = "STYLE-001",
) -> Any:
    """Resolve a ShotSpec or Shot model into a fully resolved ShotVisualContext.
    
    Acts as the bridge between scene direction and the 8-block prompt compiler.
    """
    from app.schemas.visual_context import (
        ShotVisualContext,
        SubjectVisualState,
        DanbooruIdentityLock,
        OutfitCard,
        LocationPlateSpec,
    )

    shot_id = getattr(shot, "id", None) or getattr(shot, "shot_id", "shot_001")
    workflow_id = getattr(shot, "workflow_id", "WF-SHOT-001")
    shot_type = getattr(shot, "shot_type", "medium")
    camera_angle = getattr(shot, "camera_angle", "eye_level")
    lighting = getattr(shot, "lighting_override", None)
    vfx_tag = getattr(shot, "vfx_overlay_tag", None)
    raw_subjects = getattr(shot, "subjects", [])

    resolved_subjects: list[SubjectVisualState] = []
    for sub in raw_subjects:
        eid = getattr(sub, "entity_id", None) or (sub.get("entity_id") if isinstance(sub, dict) else None)
        ename = getattr(sub, "entity_name", None) or (sub.get("entity_name") if isinstance(sub, dict) else eid)
        pose = getattr(sub, "pose", None) or (sub.get("pose") if isinstance(sub, dict) else None)
        expr = getattr(sub, "expression", None) or (sub.get("expression") if isinstance(sub, dict) else None)
        box = getattr(sub, "region_box", None) or (sub.get("region_box") if isinstance(sub, dict) else [0.0, 0.0, 1.0, 1.0])

        pose_expr = f"{pose or 'standing'}, {expr or 'looking at viewer'}"
        
        # Resolve assets
        res_data = {}
        if eid:
            try:
                res_data = resolve_asset_for_scene(db, eid, arc_id, chapter_number)
            except Exception as exc:
                logger.warning("Could not resolve asset for entity %s: %s", eid, exc)

        canon = res_data.get("canonical", {})
        arc = res_data.get("arc", {})
        delta = res_data.get("chapter_delta", {})

        # Identity lock
        id_lock = None
        if "danbooru_identity_lock" in canon:
            id_lock = DanbooruIdentityLock(**canon["danbooru_identity_lock"])
        else:
            tags = []
            if canon.get("species"): tags.append(canon["species"])
            if canon.get("physique"): tags.append(canon["physique"])
            if canon.get("hair"): tags.append(canon["hair"])
            if canon.get("eyes"): tags.append(canon["eyes"])
            if canon.get("skin_color"): tags.append(canon["skin_color"])
            id_lock = DanbooruIdentityLock(
                entity_id=eid or "unknown",
                entity_name=ename or "Subject",
                core_danbooru_tags=tags or [ename or "character"],
                canonical_seed=res_data.get("canonical_seed", 42),
                reference_image_path=res_data.get("reference_image_path"),
            )

        # Outfit card
        outfit = None
        if "outfit_card" in arc:
            outfit = OutfitCard(**arc["outfit_card"])
        else:
            c_tags = []
            if arc.get("clothing"): c_tags.append(arc["clothing"])
            if arc.get("armor"): c_tags.append(arc["armor"])
            outfit = OutfitCard(
                outfit_id=f"OUTFIT-{eid or 'UNK'}",
                entity_id=eid or "unknown",
                outfit_name="Arc Attire",
                valid_from_chapter=chapter_number,
                danbooru_clothing_tags=c_tags or ["standard clothes"],
            )

        # Held equipment
        held = []
        if arc.get("weapons"):
            held.append(f"holding {arc['weapons']}")
        if arc.get("accessories"):
            acc = arc["accessories"]
            if isinstance(acc, list):
                held.extend([str(a) for a in acc])
            else:
                held.append(str(acc))

        # Temporal deltas
        deltas = []
        if delta.get("injuries"):
            inj = delta["injuries"]
            deltas.extend(inj if isinstance(inj, list) else [str(inj)])
        if delta.get("conditions"):
            cond = delta["conditions"]
            deltas.extend(cond if isinstance(cond, list) else [str(cond)])

        resolved_subjects.append(
            SubjectVisualState(
                entity_id=eid or "unknown",
                entity_name=ename or "Subject",
                identity_lock=id_lock,
                outfit=outfit,
                held_equipment=held,
                temporal_deltas=deltas,
                pose_expression=pose_expr,
                region_box=box,
            )
        )

    # Resolve environment
    env_spec = None
    loc_id = getattr(shot, "location_id", None)
    if loc_id:
        try:
            loc_res = resolve_asset_for_scene(db, loc_id, arc_id, chapter_number)
            p_tags = [str(v) for v in loc_res.get("canonical", {}).values() if v]
            env_spec = LocationPlateSpec(
                location_id=loc_id,
                location_name="Scene Location",
                plate_tags=p_tags or ["dungeon corridor"],
                plate_reference_path=loc_res.get("reference_image_path"),
            )
        except Exception:
            pass

    return ShotVisualContext(
        shot_id=str(shot_id),
        chapter_number=chapter_number,
        style_profile_id=style_profile_id,
        workflow_id=str(workflow_id),
        framing=shot_type if shot_type in [
            "close_up", "medium_close_up", "medium", "medium_wide", "wide",
            "extreme_wide", "two_shot", "group", "pov", "over_the_shoulder", "insert"
        ] else "medium",
        camera_angle=camera_angle if camera_angle in [
            "eye_level", "low_angle_dramatic", "high_angle", "dutch_angle",
            "birds_eye", "worms_eye", "profile"
        ] else "eye_level",
        environment=env_spec,
        subjects=resolved_subjects,
        lighting_atmosphere=[lighting] if lighting else ["cinematic lighting"],
        vfx_particles=[vfx_tag] if vfx_tag else [],
    )

