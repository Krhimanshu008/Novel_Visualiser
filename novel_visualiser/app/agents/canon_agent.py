"""Novel Visualiser — Visual Canon Architect Agent (Agent 6 / VCE).

Responsible for:
1. Translating narrative entity lore into immutable Danbooru tag identity locks.
2. Decoupling permanent facial/body geometry from temporal outfits and equipment.
3. Synthesizing OutfitCard specifications for specific story arcs.
4. Synthesizing LocationPlateSpec specifications for environment anchors.
5. Powered by Kaggle Model Proxy (Gemini Flash preview) via kaggle_ai_engine.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional
from sqlalchemy.orm import Session

from app.database.models import AssetLevel, AssetVersion, Entity, VisualRef
from app.schemas.visual_context import (
    DanbooruIdentityLock,
    OutfitCard,
    LocationPlateSpec,
)

logger = logging.getLogger("novel_visualiser.canon_agent")


class VisualCanonAgent:
    """Consistency Architect Agent that manages permanent visual identity locks."""

    def __init__(self, use_kaggle_proxy: bool = True):
        self.use_kaggle_proxy = use_kaggle_proxy
        self._kaggle_client = None
        if self.use_kaggle_proxy:
            try:
                import sys
                from pathlib import Path
                # Add workspace root so kaggle_ai_engine can be imported
                workspace_root = Path(__file__).resolve().parent.parent.parent.parent
                if str(workspace_root) not in sys.path:
                    sys.path.insert(0, str(workspace_root))
                from kaggle_ai_engine.proxy_client import KaggleModelProxyClient
                self._kaggle_client = KaggleModelProxyClient()
                logger.info("VisualCanonAgent initialized with Kaggle Model Proxy Client")
            except Exception as exc:
                logger.warning("Could not initialize Kaggle Model Proxy Client: %s. Using deterministic fallback.", exc)
                self._kaggle_client = None

    def synthesize_identity_lock(
        self,
        entity_name: str,
        entity_id: str,
        description: str,
        gender: Optional[str] = None,
        canonical_seed: int = 42,
    ) -> DanbooruIdentityLock:
        """Synthesize an immutable Danbooru tag identity lock for a character."""
        # Check if Kaggle Proxy is available
        if self._kaggle_client:
            try:
                prompt = f"""
You are an expert anime character designer and prompt engineer specializing in Danbooru tags for Illustrious / SDXL anime checkpoints.
Analyze the following novel character description and generate an IMMUTABLE Danbooru identity lock.

Character Name: {entity_name}
Gender: {gender or 'unspecified'}
Description: {description}

Rules:
1. ONLY include permanent biological and facial attributes (gender tag, skin tone, hair color/style, eye color/shape, body build).
2. DO NOT include clothing, armor, weapons, temporary wounds, or scene actions.
3. Format tags in lowercase Danbooru standard syntax (e.g. '1boy', 'pale skin', 'messy black hair', 'sharp dark onyx eyes', 'slender build').
4. List negative anti-features (e.g. if he has black hair, negative is 'blonde hair, brown hair').

Respond ONLY in valid JSON matching this schema:
{{
  "core_danbooru_tags": ["tag1", "tag2", "tag3"],
  "negative_identity_tags": ["neg1", "neg2"],
  "token_weights": {{"(messy black hair:1.2)": 1.2}}
}}
"""
                resp_str = self._kaggle_client.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
                raw_text = resp_str.strip()
                if "```json" in raw_text:
                    raw_text = raw_text.split("```json")[1].split("```")[0].strip()
                elif "```" in raw_text:
                    raw_text = raw_text.split("```")[1].split("```")[0].strip()
                data = json.loads(raw_text)

                return DanbooruIdentityLock(
                    entity_id=entity_id,
                    entity_name=entity_name,
                    identity_version="CANON-V1",
                    core_danbooru_tags=data.get("core_danbooru_tags", []),
                    negative_identity_tags=data.get("negative_identity_tags", []),
                    token_weights=data.get("token_weights", {}),
                    canonical_seed=canonical_seed,
                )
            except Exception as exc:
                logger.warning("Kaggle Model Proxy identity synthesis failed (%s), using deterministic fallback", exc)

        # Deterministic fallback
        core_tags = []
        neg_tags = []
        desc_lower = description.lower()

        # Subject count & gender
        if gender and gender.lower() == "male" or "boy" in desc_lower or "man" in desc_lower or "he" in desc_lower:
            core_tags.append("1boy")
            neg_tags.extend(["1girl", "female", "breasts"])
        elif gender and gender.lower() == "female" or "girl" in desc_lower or "woman" in desc_lower or "she" in desc_lower:
            core_tags.append("1girl")
            neg_tags.extend(["1boy", "male", "facial hair"])

        # Skin tone
        if "pale" in desc_lower or "fair" in desc_lower:
            core_tags.append("pale skin")
            neg_tags.append("dark skin")
        elif "tan" in desc_lower or "dark" in desc_lower and "skin" in desc_lower:
            core_tags.append("dark skin")
            neg_tags.append("pale skin")

        # Hair
        if "black hair" in desc_lower or "dark hair" in desc_lower or "jet-black" in desc_lower:
            core_tags.append("messy black hair" if "messy" in desc_lower else "black hair")
            neg_tags.extend(["blonde hair", "silver hair", "brown hair", "blue hair"])
        elif "silver hair" in desc_lower or "white hair" in desc_lower:
            core_tags.append("silver hair")
            neg_tags.extend(["black hair", "brown hair"])

        # Eyes
        if "black eyes" in desc_lower or "dark eyes" in desc_lower or "onyx" in desc_lower:
            core_tags.append("sharp dark onyx eyes")
            neg_tags.extend(["blue eyes", "green eyes", "golden eyes"])
        elif "blue eyes" in desc_lower:
            core_tags.append("blue eyes")
            neg_tags.append("black eyes")

        # Physique
        if "slender" in desc_lower or "lean" in desc_lower or "thin" in desc_lower:
            core_tags.append("slender build")
            neg_tags.append("muscular")
        elif "muscular" in desc_lower or "broad" in desc_lower:
            core_tags.append("muscular build")

        return DanbooruIdentityLock(
            entity_id=entity_id,
            entity_name=entity_name,
            identity_version="CANON-V1",
            core_danbooru_tags=core_tags or [entity_name.lower(), "character portrait"],
            negative_identity_tags=neg_tags,
            canonical_seed=canonical_seed,
        )

    def synthesize_outfit_card(
        self,
        entity_id: str,
        outfit_id: str,
        outfit_name: str,
        raw_clothing_desc: str,
        valid_from_chapter: int = 1,
        valid_to_chapter: Optional[int] = None,
    ) -> OutfitCard:
        """Synthesize an OutfitCard isolating clothing tags from base character biology."""
        if self._kaggle_client:
            try:
                prompt = f"""
You are an expert anime character designer. Convert this clothing description into Danbooru tags for an anime diffusion model.

Outfit Name: {outfit_name}
Description: {raw_clothing_desc}

Rules:
1. ONLY include clothing, footwear, belts, bracers, cloaks, and jewelry.
2. DO NOT include hair, eyes, skin, face, or physical body descriptions.
3. List negative tags (e.g. 'modern clothes, sneakers, hoodie' if it's fantasy).

Respond ONLY in valid JSON matching this schema:
{{
  "danbooru_clothing_tags": ["tag1", "tag2"],
  "negative_clothing_tags": ["neg1", "neg2"]
}}
"""
                resp_str = self._kaggle_client.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
                raw_text = resp_str.strip()
                if "```json" in raw_text:
                    raw_text = raw_text.split("```json")[1].split("```")[0].strip()
                elif "```" in raw_text:
                    raw_text = raw_text.split("```")[1].split("```")[0].strip()
                data = json.loads(raw_text)

                return OutfitCard(
                    outfit_id=outfit_id,
                    entity_id=entity_id,
                    outfit_name=outfit_name,
                    valid_from_chapter=valid_from_chapter,
                    valid_to_chapter=valid_to_chapter,
                    danbooru_clothing_tags=data.get("danbooru_clothing_tags", []),
                    negative_clothing_tags=data.get("negative_clothing_tags", []),
                )
            except Exception as exc:
                logger.warning("Kaggle Model Proxy outfit synthesis failed: %s", exc)

        # Deterministic fallback
        clothing_tags = [t.strip() for t in raw_clothing_desc.split(",") if t.strip()]
        return OutfitCard(
            outfit_id=outfit_id,
            entity_id=entity_id,
            outfit_name=outfit_name,
            valid_from_chapter=valid_from_chapter,
            valid_to_chapter=valid_to_chapter,
            danbooru_clothing_tags=clothing_tags,
            negative_clothing_tags=["modern clothing", "hoodie", "sneakers"],
        )

    def synthesize_location_plate(
        self,
        location_id: str,
        location_name: str,
        raw_description: str,
    ) -> LocationPlateSpec:
        """Synthesize environment tags for a location backdrop plate."""
        if self._kaggle_client:
            try:
                prompt = f"""
Convert this location description into clean Danbooru scenery tags for an anime diffusion model:
Location: {location_name}
Description: {raw_description}

Respond ONLY in valid JSON:
{{
  "plate_tags": ["tag1", "tag2"],
  "negative_environment_tags": ["neg1"]
}}
"""
                resp_str = self._kaggle_client.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
                raw_text = resp_str.strip()
                if "```json" in raw_text:
                    raw_text = raw_text.split("```json")[1].split("```")[0].strip()
                elif "```" in raw_text:
                    raw_text = raw_text.split("```")[1].split("```")[0].strip()
                data = json.loads(raw_text)
                return LocationPlateSpec(
                    location_id=location_id,
                    location_name=location_name,
                    plate_tags=data.get("plate_tags", []),
                    negative_environment_tags=data.get("negative_environment_tags", []),
                )
            except Exception:
                pass

        return LocationPlateSpec(
            location_id=location_id,
            location_name=location_name,
            plate_tags=[t.strip() for t in raw_description.split(",") if t.strip()],
            negative_environment_tags=["blurry background", "modern buildings"],
        )

    def lock_entity_canon(
        self,
        db: Session,
        entity_id: str,
        identity_lock: DanbooruIdentityLock,
        outfit_card: Optional[OutfitCard] = None,
        chapter_number: int = 1,
    ) -> None:
        """Persist the Danbooru identity lock into AssetVersion in the SQLite world canon."""
        # Find or create CANONICAL asset version
        v_attrs: dict[str, Any] = {
            "danbooru_identity_lock": identity_lock.model_dump(),
            "hair": next((t for t in identity_lock.core_danbooru_tags if "hair" in t), None),
            "eyes": next((t for t in identity_lock.core_danbooru_tags if "eye" in t), None),
            "skin_color": next((t for t in identity_lock.core_danbooru_tags if "skin" in t), None),
            "physique": next((t for t in identity_lock.core_danbooru_tags if "build" in t or "frame" in t), None),
            "canonical_seed": identity_lock.canonical_seed,
        }

        # Check existing canonical version
        from sqlalchemy import select, and_
        q = select(AssetVersion).where(
            and_(
                AssetVersion.entity_id == entity_id,
                AssetVersion.level == AssetLevel.CANONICAL,
            )
        )
        existing = db.execute(q).scalars().first()
        if existing:
            existing_attrs = json.loads(existing.visual_attributes) if isinstance(existing.visual_attributes, str) else (existing.visual_attributes or {})
            existing_attrs.update(v_attrs)
            existing.visual_attributes = json.dumps(existing_attrs)
        else:
            new_v = AssetVersion(
                entity_id=entity_id,
                level=AssetLevel.CANONICAL,
                valid_from_chapter=chapter_number,
                visual_attributes=json.dumps(v_attrs),
                change_reason="Locked canonical Danbooru identity via VisualCanonAgent",
            )
            db.add(new_v)

        # If outfit provided, add ARC level version
        if outfit_card:
            arc_attrs = {
                "outfit_card": outfit_card.model_dump(),
                "clothing": ", ".join(outfit_card.danbooru_clothing_tags),
            }
            arc_v = AssetVersion(
                entity_id=entity_id,
                level=AssetLevel.ARC,
                valid_from_chapter=outfit_card.valid_from_chapter,
                valid_to_chapter=outfit_card.valid_to_chapter,
                visual_attributes=json.dumps(arc_attrs),
                change_reason=f"Locked outfit {outfit_card.outfit_id} via VisualCanonAgent",
            )
            db.add(arc_v)

        db.commit()
        logger.info("Successfully locked visual canon for entity %s into SQLite DB", entity_id)
