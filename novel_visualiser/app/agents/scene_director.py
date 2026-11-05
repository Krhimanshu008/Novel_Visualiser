"""
Novel Visualiser — Scene Director Agent (Phase 4)

Responsible for:
1. Deterministic Shot-Budget Calculation (0 LLM tokens)
   - Narration word-count to shot budget mapping based on scene intensity/pacing
2. Cinematic Storyboard Generation
   - Generates list[ShotSpec] with camera angles, motion tiers, VFX overlay tags,
     and regional subject placements ([x1, y1, x2, y2] bounding boxes)
3. Database Persistence
   - Saves Shot and ShotSubject records to SQLite linked to Scene and Chapter
"""

from __future__ import annotations

import logging
import math
import uuid
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.models import (
    Chapter,
    Entity,
    EntityType,
    MotionTier,
    Passage,
    Scene,
    Shot,
    ShotStatus,
    ShotSubject,
)
from app.schemas.scene import ShotSpec, SubjectSpec

logger = logging.getLogger("novel_visualiser.scene_director")


def calculate_shot_budget(
    text_or_passages: str | list[str] | list[dict[str, Any]],
    scene_type: str = "dialogue",
    min_shots: int = 1,
    max_shots: int = 20,
) -> dict[str, Any]:
    """
    Deterministic Shot-Budget Calculator (Zero LLM Tokens).

    Calculates the exact number of visual shots, target duration, and pacing curve
    based on the narration word count and scene dramatic category.

    Pacing standards:
    - combat / action / climax: 1 shot per ~20-25 words (~3.0 - 4.5 sec/shot)
    - dialogue / conversation / tension: 1 shot per ~35-45 words (~5.5 - 7.5 sec/shot)
    - exposition / atmospheric / landscape: 1 shot per ~45-55 words (~7.0 - 9.0 sec/shot)
    - default: 1 shot per ~35 words (~6.0 sec/shot)

    Args:
        text_or_passages: Raw string or list of passage texts or passage dicts.
        scene_type: Narrative category ('combat', 'dialogue', 'exposition', 'climax', etc.)
        min_shots: Minimum shots allowed for the scene.
        max_shots: Maximum shots ceiling to prevent budget bloat.

    Returns:
        dict containing shot_count, avg_duration_sec, total_duration_sec, pacing, words_per_shot.
    """
    if isinstance(text_or_passages, str):
        full_text = text_or_passages
    elif isinstance(text_or_passages, list):
        items = []
        for p in text_or_passages:
            if isinstance(p, dict):
                items.append(p.get("text", ""))
            elif hasattr(p, "text"):
                items.append(getattr(p, "text"))
            else:
                items.append(str(p))
        full_text = " ".join(items)
    else:
        full_text = str(text_or_passages)

    words = full_text.split()
    total_words = len(words)

    # Normalise scene type
    st = scene_type.lower().strip()
    if any(k in st for k in ["combat", "fight", "action", "battle", "climax"]):
        pacing = "fast"
        target_words_per_shot = 22
        avg_duration = 3.8
    elif any(k in st for k in ["exposition", "landscape", "atmospheric", "worldbuilding", "travel"]):
        pacing = "slow"
        target_words_per_shot = 48
        avg_duration = 8.0
    else:  # dialogue, monologue, investigation, generic
        pacing = "moderate"
        target_words_per_shot = 36
        avg_duration = 6.2

    if total_words == 0:
        shot_count = min_shots
    else:
        raw_shots = round(total_words / target_words_per_shot)
        shot_count = max(min_shots, min(raw_shots, max_shots))

    total_duration = round(shot_count * avg_duration, 1)

    # Prescribed cinematic shot rhythm sequence
    # E.g. Establishing wide -> Mid -> Close-up -> Action/Reaction -> Climax wide
    shot_rhythm: list[str] = []
    framings = ["wide", "medium", "medium_close_up", "close_up", "two_shot", "over_the_shoulder"]
    for i in range(shot_count):
        if i == 0:
            shot_rhythm.append("wide")  # Always establish
        elif i == shot_count - 1 and pacing == "fast":
            shot_rhythm.append("medium_close_up")
        else:
            shot_rhythm.append(framings[i % len(framings)])

    return {
        "shot_count": shot_count,
        "total_words": total_words,
        "target_words_per_shot": target_words_per_shot,
        "avg_duration_sec": avg_duration,
        "total_duration_sec": total_duration,
        "pacing": pacing,
        "shot_rhythm": shot_rhythm,
    }


class SceneDirectorAgent:
    """
    Scene Director Agent.

    Plans cinematic visual shots for scenes, assigning camera angles,
    framing classifications, regional entity positions, VFX tags, and
    motion tiers.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    def plan_shots_for_scene(
        self,
        scene: Scene | dict[str, Any],
        passages: list[Any],
        entities_present: list[dict[str, Any]] | None = None,
    ) -> list[ShotSpec]:
        """
        Produce a list of ShotSpecs for a scene.
        Uses deterministic cinematic composition rules.
        """
        # Extract metadata
        scene_type = getattr(scene, "scene_type", None) or (
            scene.get("scene_type") if isinstance(scene, dict) else "dialogue"
        )
        if hasattr(scene_type, "value"):
            scene_type = scene_type.value

        location_id = getattr(scene, "location_id", None) or (
            scene.get("location_id") if isinstance(scene, dict) else None
        )
        mood = getattr(scene, "mood", None) or (
            scene.get("mood") if isinstance(scene, dict) else "tense"
        )
        conflict_type = getattr(scene, "conflict_type", None) or (
            scene.get("conflict_type") if isinstance(scene, dict) else "internal"
        )

        # Calculate shot budget
        budget_info = calculate_shot_budget(passages, scene_type=str(scene_type))
        shot_count = budget_info["shot_count"]
        pacing = budget_info["pacing"]

        # Parse passage IDs & texts
        passage_ids: list[str] = []
        passage_texts: list[str] = []
        for p in passages:
            pid = getattr(p, "id", None) or (p.get("passage_id") if isinstance(p, dict) else str(uuid.uuid4()))
            ptext = getattr(p, "text", None) or (p.get("text") if isinstance(p, dict) else str(p))
            passage_ids.append(pid)
            passage_texts.append(ptext)

        # Distribute passages across shots
        passages_per_shot = max(1, math.ceil(len(passage_texts) / shot_count)) if passage_texts else 1

        # Determine entities present
        known_entities = entities_present or []
        if not known_entities:
            # Fallback to Sunny as primary entity if none passed
            known_entities = [{"entity_id": "Sunny", "entity_name": "Sunny"}]

        # Cinematic camera angles rotation
        camera_angles_pool = [
            "eye_level",
            "low_angle_dramatic",
            "high_angle",
            "dutch_angle",
            "profile",
            "worms_eye",
        ]

        # VFX overlays map based on mood/conflict
        vfx_tag: Optional[str] = None
        if any(w in str(mood).lower() for w in ["cold", "snow", "frost", "frozen", "blizzard"]):
            vfx_tag = "falling_snow_heavy"
        elif any(w in str(mood).lower() for w in ["shadow", "dark", "grim", "night", "desolate"]):
            vfx_tag = "shadow_tendrils"
        elif any(w in str(conflict_type).lower() for w in ["combat", "physical", "violence"]):
            vfx_tag = "blood_spatter"

        shots: list[ShotSpec] = []

        for shot_idx in range(shot_count):
            shot_num = shot_idx + 1

            # Slice passages for this shot
            start_p = shot_idx * passages_per_shot
            end_p = min(start_p + passages_per_shot, len(passage_texts))
            sub_passages = passage_ids[start_p:end_p] if passage_ids else [f"P_{shot_num}"]
            sub_text = " ".join(passage_texts[start_p:end_p]) if passage_texts else ""

            # Framing & Camera Angle
            if shot_idx == 0:
                shot_type = "wide"
                cam_angle = "eye_level"
                workflow_id = "WF-003" if not known_entities else "WF-006"
                cam_motion = "pan_left_to_right"
                motion_tier = "static_parallax"
            elif shot_idx == shot_count - 1 and pacing == "fast":
                shot_type = "medium_close_up"
                cam_angle = "low_angle_dramatic"
                workflow_id = "WF-004"
                cam_motion = "fast_push_in"
                motion_tier = "i2v_climax"
            else:
                framings = ["medium", "close_up", "two_shot", "over_the_shoulder"]
                shot_type = framings[shot_idx % len(framings)]
                cam_angle = camera_angles_pool[shot_idx % len(camera_angles_pool)]
                workflow_id = "WF-005" if len(known_entities) >= 2 else "WF-002"
                cam_motion = "slow_zoom_in" if shot_type in ["close_up", "medium_close_up"] else "dolly_up"
                motion_tier = "i2v_climax" if (pacing == "fast" and shot_idx % 2 == 1) else "static_parallax"

            # Assign Subject placements with bounding boxes [x1, y1, x2, y2]
            subjects: list[SubjectSpec] = []
            if len(known_entities) == 1:
                # Single character centered/left-third
                ent = known_entities[0]
                box = [0.25, 0.15, 0.75, 0.95] if shot_type != "wide" else [0.35, 0.30, 0.65, 0.85]
                subjects.append(
                    SubjectSpec(
                        entity_id=ent.get("entity_id", ent.get("id", "Sunny")),
                        entity_name=ent.get("entity_name", ent.get("canonical_name", "Sunny")),
                        region_box=box,
                        pose="alert standing defensive posture" if pacing == "fast" else "cautious observant stance",
                        expression="focused cold gaze" if pacing == "fast" else "stoic watchful eyes",
                        z_order=0,
                    )
                )
            elif len(known_entities) >= 2:
                # Dual character composition (e.g. Left vs Right regional boxes)
                ent1 = known_entities[0]
                ent2 = known_entities[1]
                subjects.append(
                    SubjectSpec(
                        entity_id=ent1.get("entity_id", ent1.get("id", "Sunny")),
                        entity_name=ent1.get("entity_name", ent1.get("canonical_name", "Sunny")),
                        region_box=[0.05, 0.15, 0.48, 0.95],  # Left frame
                        pose="cautious stance, body slightly angled",
                        expression="guarded intense gaze",
                        z_order=0,
                    )
                )
                subjects.append(
                    SubjectSpec(
                        entity_id=ent2.get("entity_id", ent2.get("id", "Nephis")),
                        entity_name=ent2.get("entity_name", ent2.get("canonical_name", "Nephis")),
                        region_box=[0.52, 0.12, 0.95, 0.95],  # Right frame
                        pose="poised upright posture, authoritative",
                        expression="calm composed expression",
                        z_order=0,
                    )
                )

            # Lighting directive
            lighting = "cold moonlight with harsh rim shadows"
            if "twilight" in sub_text.lower():
                lighting = "dim dusky twilight with pale blue ambient fill"
            elif "flame" in sub_text.lower() or "fire" in sub_text.lower():
                lighting = "dramatic warm orange firelight cutting through dark shadows"

            shot = ShotSpec(
                shot_number=shot_num,
                source_passages=sub_passages,
                workflow_id=workflow_id,
                shot_type=shot_type,
                camera_angle=cam_angle,
                location_id=location_id,
                subjects=subjects,
                lighting_override=lighting,
                vfx_overlay_tag=vfx_tag,
                motion_tier=motion_tier,
                camera_motion_preset=cam_motion,
                narration_text=sub_text[:300] if sub_text else None,
                mood=str(mood),
            )
            shots.append(shot)

        logger.info(
            "Planned %d shots for scene (type=%s, pacing=%s, words=%d)",
            len(shots),
            scene_type,
            pacing,
            budget_info["total_words"],
        )
        return shots

    def direct_chapter(
        self,
        db: Session,
        chapter_id: str,
    ) -> list[Shot]:
        """
        Direct all scenes within a chapter and commit Shot + ShotSubject records to SQLite.

        Args:
            db: SQLAlchemy session.
            chapter_id: UUID of Chapter record.

        Returns:
            list[Shot]: Database Shot records created.
        """
        chapter = db.execute(
            select(Chapter).where(Chapter.id == chapter_id)
        ).scalar_one_or_none()
        if not chapter:
            raise ValueError(f"Chapter not found: {chapter_id}")

        # Fetch scenes for this chapter
        scenes = db.execute(
            select(Scene).where(Scene.chapter_id == chapter_id).order_by(Scene.scene_number)
        ).scalars().all()

        if not scenes:
            logger.warning("No scenes found for chapter %d; cannot direct shots", chapter.chapter_number)
            return []

        all_created_shots: list[Shot] = []
        global_shot_counter = 1

        for scene in scenes:
            # Fetch passages between start_passage and end_passage
            start_p = db.get(Passage, scene.start_passage_id) if scene.start_passage_id else None
            end_p = db.get(Passage, scene.end_passage_id) if scene.end_passage_id else None

            if start_p and end_p:
                passages = db.execute(
                    select(Passage).where(
                        Passage.chapter_id == chapter_id,
                        Passage.paragraph_index >= min(start_p.paragraph_index, end_p.paragraph_index),
                        Passage.paragraph_index <= max(start_p.paragraph_index, end_p.paragraph_index),
                    ).order_by(Passage.paragraph_index)
                ).scalars().all()
            else:
                passages = chapter.passages or []

            # Find entities present in this scene from tagged passages
            known_entities: list[dict[str, Any]] = []
            seen_eids = set()
            for p in passages:
                if p.tagged_entity_ids:
                    for te in p.tagged_entity_ids:
                        if te not in seen_eids:
                            seen_eids.add(te)
                            known_entities.append({"entity_id": te, "entity_name": te})

            if not known_entities:
                known_entities = [{"entity_id": "Sunny", "entity_name": "Sunny"}]

            shot_specs = self.plan_shots_for_scene(
                scene=scene,
                passages=passages,
                entities_present=known_entities,
            )

            for spec in shot_specs:
                shot_row = Shot(
                    scene_id=scene.id,
                    chapter_id=chapter_id,
                    shot_number=global_shot_counter,
                    workflow_id=spec.workflow_id,
                    shot_type=spec.shot_type,
                    camera_angle=spec.camera_angle,
                    lighting_override=spec.lighting_override,
                    vfx_overlay_tag=spec.vfx_overlay_tag,
                    motion_tier=MotionTier.I2V_CLIMAX if spec.motion_tier == "i2v_climax" else MotionTier.STATIC_PARALLAX,
                    camera_motion_preset=spec.camera_motion_preset,
                    source_passage_ids=spec.source_passages,
                    status=ShotStatus.PLANNED,
                )
                db.add(shot_row)
                db.flush()  # populate shot_row.id

                # Save shot subjects
                for sub in spec.subjects:
                    ent = db.execute(
                        select(Entity).where(
                            (Entity.id == sub.entity_id) | (Entity.canonical_name == sub.entity_id)
                        )
                    ).scalars().first()

                    if not ent:
                        ent = Entity(
                            project_id=chapter.project_id,
                            canonical_name=sub.entity_name or sub.entity_id,
                            entity_type=EntityType.CHARACTER,
                        )
                        db.add(ent)
                        db.flush()

                    sub_row = ShotSubject(
                        shot_id=shot_row.id,
                        entity_id=ent.id,
                        region_box=sub.region_box,
                        pose=sub.pose,
                        expression=sub.expression,
                        z_order=sub.z_order,
                    )
                    db.add(sub_row)

                all_created_shots.append(shot_row)
                global_shot_counter += 1

        db.commit()
        logger.info(
            "Directed chapter %d: created %d shots across %d scenes",
            chapter.chapter_number,
            len(all_created_shots),
            len(scenes),
        )
        return all_created_shots
