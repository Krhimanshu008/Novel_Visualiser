import json
from typing import Any
from sqlalchemy.orm import Session

from app.database.models import Entity, EntityAlias, ResearchClaim, ResearchImage


def _parse_json_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        return [str(x).strip() for x in parsed if str(x).strip()] if isinstance(parsed, list) else []
    except Exception:
        return [x.strip() for x in raw.split(",") if x.strip()]


def get_safe_knowledge(db: Session, entity_id: str, current_chapter: int) -> dict[str, Any]:
    """
    Universal 4-Tier Spoiler Gate & Temporal Era Resolver (0 LLM Tokens).
    - Blocks entities before their debut chapter.
    - Filters aliases where `valid_from_chapter <= current_chapter` and `reveal_chapter <= current_chapter`.
    - Filters claims active during `current_chapter`: `valid_from_chapter <= current_chapter <= valid_until_chapter`.
    - Automatically turns FUTURE claims (`valid_from_chapter > current_chapter`) into `anti_spoiler_negative_tags`!
    """
    entity = db.query(Entity).filter_by(id=entity_id).first()
    if not entity:
        return {"status": "not_found", "allowed": False}

    if entity.first_appearance_chapter > current_chapter:
        return {
            "status": "locked_spoiler",
            "allowed": False,
            "entity_id": entity.id,
            "reason": f"Entity '{entity.canonical_name}' debuts at Ch. {entity.first_appearance_chapter} (Current: Ch. {current_chapter})",
        }

    # 1. Resolve Spoiler-Safe Aliases
    all_aliases = db.query(EntityAlias).filter_by(entity_id=entity.id).all()
    safe_aliases = [
        a.alias
        for a in all_aliases
        if a.valid_from_chapter <= current_chapter and a.reveal_chapter <= current_chapter
    ]
    blocked_aliases = [
        a.alias
        for a in all_aliases
        if a.alias not in safe_aliases
    ]

    # 2. Resolve Active Claims vs. Future Spoiler Claims vs. Expired Past Claims
    all_claims = db.query(ResearchClaim).filter_by(entity_id=entity.id).all()
    active_claims: list[dict[str, Any]] = []
    safe_danbooru_tags: list[str] = []
    flux_descriptions: list[str] = []
    anti_spoiler_negative_tags: list[str] = []

    for c in all_claims:
        is_started = c.valid_from_chapter <= current_chapter
        is_not_expired = (c.valid_until_chapter is None) or (c.valid_until_chapter >= current_chapter)

        c_tags = _parse_json_list(c.danbooru_tags)
        c_neg = _parse_json_list(c.negative_tags)

        if is_started and is_not_expired:
            # Active in the current chapter!
            active_claims.append({
                "id": c.id,
                "era_name": c.era_name,
                "claim_type": c.claim_type,
                "claim_text": c.claim_text,
                "valid_from_chapter": c.valid_from_chapter,
                "valid_until_chapter": c.valid_until_chapter,
            })
            for t in c_tags:
                if t not in safe_danbooru_tags:
                    safe_danbooru_tags.append(t)
            for nt in c_neg:
                if nt not in anti_spoiler_negative_tags:
                    anti_spoiler_negative_tags.append(nt)
            if c.flux_prompt and c.flux_prompt not in flux_descriptions:
                flux_descriptions.append(c.flux_prompt)

        elif not is_started:
            # FUTURE SPOILER CLAIM (e.g. Ch 350 armor or Ch 800 scar when querying at Ch 1)
            # Automatically push its visual tags into the negative prompt so ComfyUI never draws it early!
            for future_tag in c_tags:
                if future_tag not in safe_danbooru_tags and future_tag not in anti_spoiler_negative_tags:
                    anti_spoiler_negative_tags.append(future_tag)

    # Remove any overlap if a base tag is still active
    anti_spoiler_negative_tags = [t for t in anti_spoiler_negative_tags if t not in safe_danbooru_tags]

    # 3. Resolve Canonical Reference Image
    images = db.query(ResearchImage).filter_by(entity_id=entity.id).all()
    canonical_image = None
    for preferred_cat in ("official_cover", "official_illustration", "reference_photo", "community_art"):
        match = next((img.local_path or img.image_url for img in images if img.category == preferred_cat), None)
        if match:
            canonical_image = match
            break

    return {
        "status": "ok",
        "allowed": True,
        "entity_id": entity.id,
        "canonical_name": entity.canonical_name,
        "entity_type": entity.entity_type,
        "current_chapter": current_chapter,
        "safe_aliases": safe_aliases,
        "blocked_spoiler_aliases": blocked_aliases,
        "active_claims": active_claims,
        "danbooru_prompt": ", ".join(safe_danbooru_tags),
        "flux_natural_prompt": " ".join(flux_descriptions) if flux_descriptions else "; ".join(c["claim_text"] for c in active_claims),
        "anti_spoiler_negative_prompt": ", ".join(anti_spoiler_negative_tags),
        "canonical_reference_image": canonical_image,
    }