"""Novel Visualiser — Master Scene Director & Production Orchestrator Agent.

Autonomous Main Orchestrator responsible for:
1. Decomposing chapters into dramatic scene beats with word-count & shot budgeting.
2. Synchronizing and coordinating all 5 specialized subagents:
   - LocationContinuitySubagent (Immutable 3D spatial geometry & ControlNet depth guides)
   - CharacterContinuitySubagent (Biological DNA, active epochs, temporal wardrobes, IP-Adapter head chips)
   - BiologicalEvolutionAuditor (High-end frontier reasoning for character physical maturation)
   - EnvironmentContinuitySubagent (Dynamic lighting rigs, Kelvin temps, weather particles over frozen geometry)
   - CreatureMonsterSubagent (2-axis power hierarchy, bestiary turnarounds, predatory head chips)
   - CompositingSubagent (De-duplicated multi-layer prompt assembly, inpaint canvases, ComfyUI dispatch)
3. Directing the Multi-Gate QC Agent (QCContinuityAgent) to audit and verify pixel-level visual fidelity.
4. Persisting complete production manifests and shot cards into the SQLite canon database.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from sqlalchemy.orm import Session

from app.database.models import (
    Chapter,
    ChapterStatus,
    Passage,
    PassageType,
    Project,
    Scene,
    Shot,
    ShotStatus,
)
from app.agents.biological_evolution_auditor import BiologicalEvolutionAuditor
from app.agents.character_subagent import CharacterContinuitySubagent
from app.agents.compositing_subagent import CompositingSubagent
from app.agents.creature_subagent import CreatureMonsterSubagent
from app.agents.environment_subagent import EnvironmentContinuitySubagent
from app.agents.location_subagent import LocationContinuitySubagent
from app.agents.qc_continuity_agent import QCContinuityAgent

from app.schemas.character_subagent import CharacterMatrixSpec, ShotCharacterAnchor
from app.schemas.compositing_subagent import (
    BoundingBox2D,
    CompositingModeEnum,
    RenderedShotArtifact,
    ShotCompositionSpec,
)
from app.schemas.creature_subagent import (
    CreatureCard,
    ItemPropCard,
    ShotCreatureAnchor,
)
from app.schemas.environment_subagent import (
    EnvironmentalAtmosphereCard,
    ShotEnvironmentAnchor,
)
from app.schemas.location_subagent import (
    ArcHierarchyTag,
    LocationPlateMatrixSpec,
    ShotLocationAnchor,
)
from app.schemas.orchestrator import (
    ChapterProductionPlan,
    SceneBreakdownBeat,
    SceneOrchestrationPlan,
)
from app.schemas.qc import ComprehensiveQCAudit, QCVerdictEnum

logger = logging.getLogger("novel_visualiser.scene_director")


class SceneDirectorAgent:
    """Master Orchestrator Agent for Chapter Breakdown, Subagent Delegation, and Shot Synthesis."""

    def __init__(
        self,
        db: Optional[Session] = None,
        use_ai_thinking: bool = True,
        output_dir: Optional[Path] = None,
    ):
        """Initialize SceneDirectorAgent and all 5 autonomous subagents."""
        self.db = db
        self.use_ai_thinking = use_ai_thinking
        self.output_dir = output_dir or Path("output/production_plans")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Specialized Subagents
        self.location_subagent = LocationContinuitySubagent(
            db=self.db, use_ai_thinking=self.use_ai_thinking
        )
        self.character_subagent = CharacterContinuitySubagent(
            db=self.db, use_ai_thinking=self.use_ai_thinking
        )
        self.evolution_auditor = BiologicalEvolutionAuditor(
            db=self.db, use_ai_thinking=self.use_ai_thinking
        )
        self.environment_subagent = EnvironmentContinuitySubagent(
            db=self.db, use_ai_thinking=self.use_ai_thinking
        )
        self.creature_subagent = CreatureMonsterSubagent(
            db=self.db, use_ai_thinking=self.use_ai_thinking
        )
        self.compositing_subagent = CompositingSubagent(
            db=self.db, allow_offline_fallback=True
        )
        self.qc_agent = QCContinuityAgent(
            db=self.db, use_ai_thinking=self.use_ai_thinking
        )

        # High-End Frontier AI for Chapter Decomposition
        self._kaggle_client = None
        if self.use_ai_thinking:
            try:
                workspace_root = Path(__file__).resolve().parent.parent.parent.parent
                if str(workspace_root) not in sys.path:
                    sys.path.insert(0, str(workspace_root))
                from kaggle_ai_engine.proxy_client import KaggleModelProxyClient
                self._kaggle_client = KaggleModelProxyClient()
                logger.info("SceneDirectorAgent initialized with Kaggle Model Proxy client.")
            except Exception as exc:
                logger.warning("Could not initialize Kaggle Model Proxy: %s. Using heuristic scene parsing.", exc)
                self._kaggle_client = None

    # =========================================================================
    # 1. CHAPTER DECOMPOSITION & SCENE BREAKDOWN
    # =========================================================================

    def break_down_chapter_into_scenes(
        self,
        chapter_number: int,
        chapter_title: str,
        novel_text: str,
        arc_tag: ArcHierarchyTag,
    ) -> list[dict[str, Any]]:
        """
        Decompose a novel chapter into dramatic scene chunks with location and character tags.
        """
        # Frontier AI Decomposition if available
        if self.use_ai_thinking and self._kaggle_client is not None:
            try:
                prompt = (
                    "You are the Director and Storyboard Supervisor for a top-tier anime/manhwa production.\n"
                    f"Chapter {chapter_number}: '{chapter_title}'\n"
                    f"Arc Hierarchy: {arc_tag.main_arc_id} -> {arc_tag.sub_arc_id} -> {arc_tag.mini_arc_id}\n\n"
                    f"Chapter Text:\n{novel_text}\n\n"
                    "Break down this chapter into 2 to 4 dramatic, visually distinct scenes.\n"
                    "For each scene, output a JSON object with:\n"
                    "- scene_number (int)\n"
                    "- scene_title (str)\n"
                    "- scene_type ('combat', 'dialogue', 'exposition', or 'climax')\n"
                    "- location_name (str, e.g. 'Outskirts Transit Bench N24')\n"
                    "- primary_character (str, e.g. 'Sunny')\n"
                    "- secondary_characters (list of str)\n"
                    "- creatures (list of str)\n"
                    "- weapons_or_items (list of str)\n"
                    "- full_scene_text (the exact corresponding passage from the chapter text)\n"
                    "Return ONLY a JSON list of scene objects."
                )
                raw_response = self._kaggle_client.chat_completion(
                    messages=[
                        {"role": "system", "content": "You are a master storyboard director. Output JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                )
                clean = raw_response.strip("` \n\r")
                if clean.startswith("json"):
                    clean = clean[4:].strip()
                parsed = json.loads(clean)
                if isinstance(parsed, list) and len(parsed) > 0:
                    logger.info("SceneDirector: AI successfully broke down Chapter %d into %d scenes.", chapter_number, len(parsed))
                    return parsed
            except Exception as exc:
                logger.warning("SceneDirector AI chapter breakdown failed: %s; falling back to heuristic chunking.", exc)

        # Deterministic Heuristic Fallback
        paragraphs = [p.strip() for p in novel_text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [novel_text]

        half = len(paragraphs) // 2
        scene_1_text = "\n\n".join(paragraphs[:half]) if half > 0 else novel_text
        scene_2_text = "\n\n".join(paragraphs[half:]) if half > 0 else novel_text

        # Detect primary entities heuristically
        has_centipede = "centipede" in novel_text.lower() or "larva" in novel_text.lower()
        has_shard = "midnight shard" in novel_text.lower() or "dagger" in novel_text.lower()

        scenes = [
            {
                "scene_number": 1,
                "scene_title": f"{chapter_title} - Establishing Exposition & Slum Solitude",
                "scene_type": "exposition",
                "location_name": "Outskirts Transit Bench N24",
                "primary_character": "Sunny",
                "secondary_characters": [],
                "creatures": [],
                "weapons_or_items": ["Tattered Coat"],
                "full_scene_text": scene_1_text,
            }
        ]

        if half > 0 and scene_2_text.strip():
            scenes.append(
                {
                    "scene_number": 2,
                    "scene_title": f"{chapter_title} - Midnight Confrontation & Awakening",
                    "scene_type": "combat" if has_centipede else "dialogue",
                    "location_name": "Outskirts Transit Bench N24",
                    "primary_character": "Sunny",
                    "secondary_characters": [],
                    "creatures": ["Mountain Centipede Larva"] if has_centipede else [],
                    "weapons_or_items": ["Midnight Shard"] if has_shard else [],
                    "full_scene_text": scene_2_text,
                }
            )

        return scenes

    # =========================================================================
    # 2. SCENE ORCHESTRATION & SHOT STORYBOARDING
    # =========================================================================

    def orchestrate_scene(
        self,
        scene_id: str,
        scene_number: int,
        scene_info: dict[str, Any],
        arc_tag: ArcHierarchyTag,
        chapter_number: int = 1,
        project_id: str = "shadow_slave",
    ) -> SceneOrchestrationPlan:
        """
        Orchestrate all 5 subagents for a single scene and compile sequenced shots.
        """
        scene_title = scene_info.get("scene_title", f"Scene {scene_number}")
        scene_type = scene_info.get("scene_type", "dialogue").lower()
        full_scene_text = scene_info.get("full_scene_text", "")
        location_name = scene_info.get("location_name", "Outskirts Transit Bench N24")
        char_name = scene_info.get("primary_character", "Sunny")
        creature_names = scene_info.get("creatures", [])
        item_names = scene_info.get("weapons_or_items", [])

        # Word count & duration budgeting
        words = len(full_scene_text.split()) if full_scene_text else 50
        duration_sec = round(words / 2.5, 1)  # ~150 words/min audio pace

        # Calculate shot budget based on dramatic pacing
        if scene_type == "combat":
            target_shots = max(2, min(6, words // 35))
            pacing_curve = "fast"
        elif scene_type == "exposition":
            target_shots = max(2, min(4, words // 80))
            pacing_curve = "slow"
        else:
            target_shots = max(2, min(5, words // 55))
            pacing_curve = "moderate"

        logger.info(
            "Orchestrating Scene %s: type=%s, words=%d, shots=%d, pace=%s",
            scene_id, scene_type, words, target_shots, pacing_curve
        )

        # ---------------------------------------------------------------------
        # SUBAGENT 1: LOCATION CONTINUITY (Immutable Geometry)
        # ---------------------------------------------------------------------
        clean_loc = re.sub(r"[^a-zA-Z0-9_]+", "_", location_name.lower()).strip("_")
        loc_dir = Path("output/location_plates") / clean_loc
        loc_matrix = self.location_subagent.observe_and_synthesize_location(
            location_name=location_name,
            passage_text=full_scene_text,
            arc_hierarchy=arc_tag,
            deterministic_seed=1000 + chapter_number * 10 + scene_number,
        )
        loc_matrix = self.location_subagent.generate_location_matrix(
            loc_matrix, output_dir=loc_dir
        )
        self.location_subagent.save_to_canon(
            project_id=project_id, matrix_spec=loc_matrix, current_chapter=chapter_number
        )

        # ---------------------------------------------------------------------
        # SUBAGENT 2: ENVIRONMENT CONTINUITY (Lighting Rig & Weather)
        # ---------------------------------------------------------------------
        env_card = self.environment_subagent.observe_and_synthesize_environment(
            location_name=location_name,
            passage_text=full_scene_text,
            arc_hierarchy=arc_tag,
            chapter_number=chapter_number,
        )
        env_anchor = self.environment_subagent.resolve_shot_environment_anchor(env_card)

        # ---------------------------------------------------------------------
        # SUBAGENT 3: CHARACTER CONTINUITY & BIOLOGICAL EVOLUTION AUDIT
        # ---------------------------------------------------------------------
        char_dir = Path("output/character_sheets") / char_name.lower()
        char_matrix = self.character_subagent.observe_and_synthesize_character(
            character_name=char_name,
            passage_text=full_scene_text,
            arc_hierarchy=arc_tag,
            gender="male",
            deterministic_seed=42,
        )
        char_matrix = self.character_subagent.generate_character_matrix(
            char_matrix, output_dir=char_dir
        )

        # Run Biological Evolution Auditor to detect physical transformations across chapters
        new_epoch = self.evolution_auditor.audit_evolution(
            character_name=char_name,
            current_chapter=chapter_number,
            passage_text=full_scene_text,
            current_spec=char_matrix,
        )
        if new_epoch:
            char_matrix.active_epoch = new_epoch
            epoch_name_str = getattr(new_epoch, "epoch_name", getattr(new_epoch, "epoch_title", "Epoch"))
            logger.info("SceneDirector: Character %s transitioned into epoch '%s'", char_name, epoch_name_str)

        self.character_subagent.save_to_canon(
            project_id=project_id, matrix_spec=char_matrix, current_chapter=chapter_number
        )
        active_char_anchor = self.character_subagent.resolve_shot_character_anchor(
            matrix_spec=char_matrix,
            shot_action="slumped exhausted on concrete bench, shivering from acid mist",
            emotion="fatigued and wary",
            camera_framing="MEDIUM",
        )

        # ---------------------------------------------------------------------
        # SUBAGENT 4: CREATURE & MONSTER CONTINUITY (Power System)
        # ---------------------------------------------------------------------
        active_creatures: list[ShotCreatureAnchor] = []
        for cr_name in creature_names:
            cr_dir = Path("output/bestiary_canon") / cr_name.lower().replace(" ", "_")
            cr_card = self.creature_subagent.observe_and_synthesize_creature(
                creature_name=cr_name,
                passage_text=full_scene_text,
                arc_hierarchy=arc_tag,
                chapter_number=chapter_number,
            )
            cr_card = self.creature_subagent.render_bestiary_matrix_and_head_chip(
                cr_card, output_dir=cr_dir
            )
            cr_anchor = self.creature_subagent.resolve_shot_creature_anchor(
                cr_card,
                combat_action="coiling aggressively around decaying concrete, mandibles dripping venom",
                spatial_anchor="midground_right",
            )
            active_creatures.append(cr_anchor)

        # Hero Weapons / Props
        active_props: list[ItemPropCard] = []
        for it_name in item_names:
            item_card = self.creature_subagent.observe_and_synthesize_item(
                item_name=it_name,
                passage_text=full_scene_text,
                arc_hierarchy=arc_tag,
                chapter_number=chapter_number,
            )
            active_props.append(item_card)

        # ---------------------------------------------------------------------
        # SUBAGENT 5: COMPOSITING & SEQUENCED SHOT STORYBOARD
        # ---------------------------------------------------------------------
        shots: list[ShotCompositionSpec] = []
        camera_sequence = [
            ("wide", "eye_level", "establishing environment and bleak dystopian vista"),
            ("medium", "eye_level", "subject grounded in environment with clear lighting key"),
            ("close_up", "slightly_low", "intense dramatic facial focus with rim highlight"),
            ("action_wide", "low_angle", "dynamic combat motion and silhouette contrast"),
            ("over_the_shoulder", "high_angle", "reaction beat and environmental depth"),
        ]

        for s_idx in range(1, target_shots + 1):
            cam_type, cam_angle, action_beat = camera_sequence[(s_idx - 1) % len(camera_sequence)]
            shot_id = f"{scene_id}_shot_{s_idx:02d}"

            # Select appropriate plate from location matrix
            loc_anchor = self.location_subagent.resolve_shot_angle_anchor(
                matrix_spec=loc_matrix,
                shot_cinematic_intent="wide_establishing" if cam_type == "wide" else "stage_action",
            )

            # Character framing specific anchor
            char_anchor_copy = self.character_subagent.resolve_shot_character_anchor(
                matrix_spec=char_matrix,
                shot_action=f"action beat {s_idx}: {action_beat}",
                emotion="intense" if scene_type == "combat" else "wary",
                camera_framing=cam_type.upper(),
            )

            # Assemble composition spec matching CompositingSubagent data contract
            temp_spec = ShotCompositionSpec(
                shot_id=shot_id,
                novel_arc=arc_tag,
                mode=(
                    CompositingModeEnum.MULTI_SUBJECT_BATTLE
                    if active_creatures
                    else (
                        CompositingModeEnum.INPAINT_DIFFERENTIAL
                        if s_idx > 1
                        else CompositingModeEnum.CONTROLNET_IPADAPTER_FUSION
                    )
                ),
                location_anchor=loc_anchor,
                environment_anchor=env_anchor,
                character_anchor=char_anchor_copy,
                creature_anchor=active_creatures[0] if active_creatures else None,
                character_bbox=BoundingBox2D(x=320, y=200, width=576, height=550),
            )

            shots.append(temp_spec)

        orchestration_plan = SceneOrchestrationPlan(
            scene_id=scene_id,
            scene_number=scene_number,
            scene_title=scene_title,
            arc_hierarchy=arc_tag,
            scene_type=scene_type,
            full_scene_text=full_scene_text,
            word_count=words,
            target_shots=target_shots,
            estimated_duration_sec=duration_sec,
            pacing_curve=pacing_curve,
            location_matrix=loc_matrix,
            environment_card=env_card,
            active_characters=[active_char_anchor],
            active_creatures=active_creatures,
            active_props=active_props,
            shots=shots,
        )

        return orchestration_plan

    # =========================================================================
    # 3. CHAPTER PRODUCTION ORCHESTRATION & DATABASE PERSISTENCE
    # =========================================================================

    def orchestrate_chapter(
        self,
        chapter_number: int,
        chapter_title: str,
        novel_text: str,
        arc_tag: ArcHierarchyTag,
        project_id: str = "shadow_slave",
    ) -> ChapterProductionPlan:
        """
        Orchestrate an entire novel chapter end-to-end and persist to SQLite canon.
        """
        logger.info("=== Starting Master Orchestration for Chapter %d: '%s' ===", chapter_number, chapter_title)

        # 1. Break down chapter into dramatic scenes
        scene_dicts = self.break_down_chapter_into_scenes(
            chapter_number=chapter_number,
            chapter_title=chapter_title,
            novel_text=novel_text,
            arc_tag=arc_tag,
        )

        total_words = len(novel_text.split())
        scenes_plans: list[SceneOrchestrationPlan] = []
        total_shots = 0
        total_duration = 0.0

        # 2. Orchestrate each scene across all subagents
        for s_idx, sc_info in enumerate(scene_dicts, start=1):
            sc_id = f"ch{chapter_number:02d}_sc{s_idx:02d}"
            sc_plan = self.orchestrate_scene(
                scene_id=sc_id,
                scene_number=s_idx,
                scene_info=sc_info,
                arc_tag=arc_tag,
                chapter_number=chapter_number,
                project_id=project_id,
            )
            scenes_plans.append(sc_plan)
            total_shots += sc_plan.target_shots
            total_duration += sc_plan.estimated_duration_sec

        production_plan = ChapterProductionPlan(
            project_id=project_id,
            novel_title=getattr(arc_tag, "novel_title", project_id),
            chapter_number=chapter_number,
            chapter_title=chapter_title,
            total_words=total_words,
            total_shots=total_shots,
            total_duration_sec=round(total_duration, 1),
            scenes=scenes_plans,
            created_at=datetime.utcnow().isoformat(),
        )

        # 3. Persist Production Plan to Database if Session is attached
        if self.db is not None:
            try:
                # Ensure Project exists
                proj = self.db.query(Project).filter(Project.id == project_id).first()
                novel_title_str = getattr(arc_tag, "novel_title", project_id)
                if not proj:
                    proj = Project(
                        id=project_id,
                        name=novel_title_str,
                        novel_title=novel_title_str,
                        author="Guiltythree",
                        description=f"Automated Visual Production for {novel_title_str}",
                    )
                    self.db.add(proj)
                    self.db.flush()

                # Ensure Chapter exists
                ch_rec = (
                    self.db.query(Chapter)
                    .filter(Chapter.project_id == project_id, Chapter.chapter_number == chapter_number)
                    .first()
                )
                if not ch_rec:
                    ch_rec = Chapter(
                        project_id=project_id,
                        chapter_number=chapter_number,
                        title=chapter_title,
                        raw_text=novel_text,
                        word_count=total_words,
                        status=ChapterStatus.PRODUCTION_READY,
                    )
                    self.db.add(ch_rec)
                    self.db.flush()
                else:
                    ch_rec.title = chapter_title
                    ch_rec.word_count = total_words
                    ch_rec.status = ChapterStatus.PRODUCTION_READY

                # Persist Scenes and Shots
                for sc in scenes_plans:
                    # Check or create Scene
                    db_scene = (
                        self.db.query(Scene)
                        .filter(Scene.chapter_id == ch_rec.id, Scene.scene_number == sc.scene_number)
                        .first()
                    )
                    if not db_scene:
                        # Create dummy passages for bounding if none
                        p1 = Passage(chapter_id=ch_rec.id, paragraph_index=1, text=sc.full_scene_text[:200])
                        p2 = Passage(chapter_id=ch_rec.id, paragraph_index=2, text=sc.full_scene_text[-200:])
                        self.db.add_all([p1, p2])
                        self.db.flush()

                        db_scene = Scene(
                            id=sc.scene_id,
                            chapter_id=ch_rec.id,
                            scene_number=sc.scene_number,
                            start_passage_id=p1.id,
                            end_passage_id=p2.id,
                            time_of_day=str(getattr(sc.environment_card, "time_of_day", "night")),
                            weather=str(getattr(sc.environment_card, "weather_condition", "overcast")),
                            mood=sc.scene_type,
                            description=sc.scene_title,
                            shot_count_target=sc.target_shots,
                        )
                        self.db.add(db_scene)
                        self.db.flush()

                    # Persist Shots
                    for s_spec in sc.shots:
                        existing_shot = self.db.query(Shot).filter(Shot.id == s_spec.shot_id).first()
                        if not existing_shot:
                            new_shot = Shot(
                                id=s_spec.shot_id,
                                scene_id=db_scene.id,
                                chapter_id=ch_rec.id,
                                shot_number=int(s_spec.shot_id.split("_")[-1]),
                                workflow_id="WF-MULTI-LAYER",
                                shot_type=s_spec.mode.value,
                                camera_angle="eye_level",
                                status=ShotStatus.PLANNED,
                            )
                            self.db.add(new_shot)
                self.db.commit()
                logger.info("Persisted Chapter %d production plan to SQLite database.", chapter_number)
            except Exception as db_err:
                logger.warning("Could not persist Chapter %d production plan to SQLite: %s", chapter_number, db_err)
                self.db.rollback()

        # Save Plan JSON to disk
        plan_path = self.output_dir / f"chapter_{chapter_number:02d}_production_plan.json"
        try:
            with open(plan_path, "w", encoding="utf-8") as f:
                f.write(production_plan.model_dump_json(indent=2))
            logger.info("Saved Chapter %d production plan to %s", chapter_number, plan_path)
        except Exception as err:
            logger.warning("Failed to save production plan to %s: %s", plan_path, err)

        return production_plan

    # =========================================================================
    # 4. EXECUTION & VISUAL QA PIPELINE
    # =========================================================================

    def execute_and_audit_chapter(
        self,
        production_plan: ChapterProductionPlan,
        force_mock: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Execute composited rendering for every budgeted shot and run 3-gate visual QA.
        """
        results: list[dict[str, Any]] = []

        for scene in production_plan.scenes:
            for shot_spec in scene.shots:
                logger.info("Rendering shot %s (%s)...", shot_spec.shot_id, shot_spec.mode.value)
                
                # 1. Render Shot via Compositing Subagent
                artifact = self.compositing_subagent.execute_composite_shot(
                    spec=shot_spec,
                    force_mock=force_mock,
                )

                # 2. Multi-Gate Visual QA Audit
                char_matrix = getattr(scene, "location_matrix", None)  # fallback
                qc_audit = self.qc_agent.audit_shot(
                    shot_artifact=artifact,
                    spec=shot_spec,
                    env_card=scene.environment_card,
                )

                logger.info(
                    "Shot %s QA Verdict: %s (Score: %.1f)",
                    shot_spec.shot_id, qc_audit.verdict.value, qc_audit.overall_score
                )

                results.append({
                    "shot_id": shot_spec.shot_id,
                    "artifact": artifact,
                    "qc_audit": qc_audit,
                })

        return results
