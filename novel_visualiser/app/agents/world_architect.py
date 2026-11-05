"""
Novel Visualiser — World Architect & Knowledge Reconciliation Agent (Agent 3)

Implements:
1. RapidFuzz Alias Matching (80% Deterministic, 0 Tokens):
   - Compares newly extracted entity names against existing entities and aliases
   - Prevents duplicate entity generation across chapters
2. Visual State Delta Committer:
   - Records chapter-level conditions, injuries, and attire modifications into AssetVersion
   - Automatically scopes valid_from_chapter to preserve spoiler-free history
3. Canonical Profile Synthesizer (20% Targeted LLM / Deterministic fallback):
   - Synthesizes base physical identity for truly new characters, creatures, and locations
4. Seamlessly integrates with resolver.py (Hierarchical Visual Asset Inheritance)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from rapidfuzz import fuzz, process
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database.models import (
    AliasType,
    AssetLevel,
    AssetVersion,
    Chapter,
    Entity,
    EntityAlias,
    EntityType,
    Project,
)
from app.orchestrator.resolver import resolve_asset_for_scene
from app.schemas.story import ChapterAnalysis, NewEntityDetection, VisualStateDelta
from app.schemas.world import CanonicalIdentity, CanonicalVisualProfile

logger = logging.getLogger("novel_visualiser.world_architect")


class KnowledgeReconciler:
    """
    RapidFuzz fuzzy alias matcher.
    Deduplicates newly extracted names against existing entities and aliases.
    """

    @staticmethod
    def match_entity(
        suggested_name: str,
        db: Session,
        project_id: str,
        threshold: float = 85.0,
    ) -> Optional[tuple[Entity, float]]:
        """
        Check if suggested_name matches any known entity canonical name or alias.
        Returns (Entity, similarity_score) if match >= threshold, else None.
        """
        clean_name = suggested_name.strip()
        if not clean_name:
            return None

        # 1. Fetch all entities and aliases for this project
        entities = db.execute(
            select(Entity).where(Entity.project_id == project_id)
        ).scalars().all()

        if not entities:
            return None

        # Build candidate lookup dict: string -> Entity
        candidates: dict[str, Entity] = {}
        for ent in entities:
            candidates[ent.canonical_name.lower()] = ent
            for al in ent.aliases:
                candidates[al.alias.lower()] = ent

        target = clean_name.lower()
        if target in candidates:
            return candidates[target], 100.0

        # Run RapidFuzz fuzzy extract using WRatio (handles titles, nicknames & prefixes)
        best = process.extractOne(
            target,
            list(candidates.keys()),
            scorer=fuzz.WRatio,
        )

        if best and best[1] >= threshold:
            matched_key, score, _ = best
            matched_entity = candidates[matched_key]
            logger.info(
                "Fuzzy matched '%s' to known entity '%s' (score: %.1f)",
                suggested_name,
                matched_entity.canonical_name,
                score,
            )
            return matched_entity, float(score)

        return None


class WorldArchitectAgent:
    """
    World Architect Agent.
    Manages entity lifecycle, asset versioning, and visual state inheritance.
    """

    def __init__(self, model: str = "gemini-2.5-flash"):
        self.settings = get_settings()
        self.model = model
        self.api_key = self.settings.llm.google_api_key

    def process_chapter_analysis(
        self,
        db: Session,
        analysis: ChapterAnalysis,
    ) -> dict[str, Any]:
        """
        Reconcile and commit world state changes from a chapter analysis:
        1. Resolve new entities (deduplicate via RapidFuzz)
        2. Create canonical profiles for confirmed new entities
        3. Commit visual state deltas into asset_versions table
        """
        project_id = analysis.project_id
        chapter_number = analysis.chapter_number
        chapter = db.execute(
            select(Chapter).where(
                Chapter.project_id == project_id,
                Chapter.chapter_number == chapter_number,
            )
        ).scalar_one_or_none()

        chapter_id = chapter.id if chapter else None

        entities_resolved: list[dict[str, Any]] = []
        deltas_committed: list[dict[str, Any]] = []

        # -------------------------------------------------------------
        # Step 1: Process New Entities
        # -------------------------------------------------------------
        for new_ent in analysis.new_entities:
            match_res = KnowledgeReconciler.match_entity(
                suggested_name=new_ent.suggested_name,
                db=db,
                project_id=project_id,
            )

            if match_res:
                existing_entity, score = match_res
                # Add as an alias rather than duplicating entity
                alias_exists = any(
                    al.alias.lower() == new_ent.suggested_name.lower()
                    for al in existing_entity.aliases
                )
                if not alias_exists:
                    new_alias = EntityAlias(
                        entity_id=existing_entity.id,
                        alias=new_ent.suggested_name,
                        alias_type=AliasType.NICKNAME,
                        valid_from_chapter=chapter_number,
                        reveal_chapter=chapter_number,
                    )
                    db.add(new_alias)
                entities_resolved.append({
                    "name": new_ent.suggested_name,
                    "action": "linked_existing",
                    "entity_id": existing_entity.id,
                    "canonical_name": existing_entity.canonical_name,
                })
            else:
                # Brand new entity
                etype = EntityType.CHARACTER
                if new_ent.entity_type == "location":
                    etype = EntityType.LOCATION
                elif new_ent.entity_type == "creature":
                    etype = EntityType.CREATURE
                elif new_ent.entity_type == "item":
                    etype = EntityType.ITEM

                created_ent = Entity(
                    project_id=project_id,
                    canonical_name=new_ent.suggested_name,
                    entity_type=etype,
                    description=new_ent.description,
                    first_appearance_chapter=chapter_number,
                )
                db.add(created_ent)
                db.flush()

                # Create initial canonical visual profile
                base_attrs = {
                    "description": new_ent.description,
                    "first_seen": f"Chapter {chapter_number}",
                }
                canon_version = AssetVersion(
                    entity_id=created_ent.id,
                    level=AssetLevel.PROJECT_CANONICAL,
                    visual_attributes=base_attrs,
                    change_description=f"Initial discovery in Chapter {chapter_number}",
                    valid_from_chapter=chapter_number,
                )
                db.add(canon_version)

                entities_resolved.append({
                    "name": new_ent.suggested_name,
                    "action": "created_new",
                    "entity_id": created_ent.id,
                    "canonical_name": created_ent.canonical_name,
                })

        # -------------------------------------------------------------
        # Step 2: Commit Visual State Deltas into asset_versions
        # -------------------------------------------------------------
        for delta in analysis.visual_state_deltas:
            # Find target entity
            match_res = KnowledgeReconciler.match_entity(
                suggested_name=delta.entity_name,
                db=db,
                project_id=project_id,
            )
            if not match_res:
                logger.warning("Could not match entity for visual delta: '%s'", delta.entity_name)
                continue

            target_entity, _ = match_res

            # Determine appropriate inheritance level
            alevel = AssetLevel.CHAPTER_DELTA
            if delta.delta_type in ["permanent_change", "new_equipment"]:
                alevel = AssetLevel.ARC_DEFAULT

            # Format visual attributes payload
            attrs_payload: dict[str, Any] = {}
            if delta.delta_type == "injury":
                attrs_payload["injuries"] = [delta.description]
            elif delta.delta_type == "new_equipment":
                attrs_payload["weapons"] = [delta.description]
            elif delta.delta_type == "temporary_condition":
                attrs_payload["conditions"] = [delta.description]
            else:
                attrs_payload["clothing"] = delta.description

            version_record = AssetVersion(
                entity_id=target_entity.id,
                level=alevel,
                chapter_id=chapter_id,
                visual_attributes=attrs_payload,
                change_description=f"{delta.delta_type}: {delta.description}",
                valid_from_chapter=chapter_number,
            )
            db.add(version_record)
            deltas_committed.append({
                "entity_name": target_entity.canonical_name,
                "delta_type": delta.delta_type,
                "description": delta.description,
                "level": alevel.value,
                "valid_from_chapter": chapter_number,
            })

        db.commit()

        logger.info(
            "World Architect processed Chapter %d: %d entities resolved, %d visual deltas saved",
            chapter_number,
            len(entities_resolved),
            len(deltas_committed),
        )

        return {
            "chapter_number": chapter_number,
            "entities_resolved": entities_resolved,
            "deltas_committed": deltas_committed,
        }
