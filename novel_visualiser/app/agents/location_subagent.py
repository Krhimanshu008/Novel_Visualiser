"""Novel Visualiser — Location Continuity Subagent.

Autonomous Subagent responsible for:
1. Detecting all physical locations in a novel chapter.
2. Querying historical SQLite canon & arc memory across past chapters and arcs.
3. Conducting 5-dimension Deep Observational Reasoning on new locations using frontier AI.
4. Synthesizing and rendering the standardized 4-Plate Reference Matrix (Vista, Stage, Reverse, Macro).
5. Extracting structural 3D Depth maps and Canny Lineart wireframes.
6. Resolving exact camera angle anchors for downstream inpainting and Qwen Edit.
7. Enforcing the 3-tier Arc Hierarchy: Main Arc > Sub Arc > Mini Arc (+ Parallel Threads).
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Optional
import cv2
import numpy as np
from sqlalchemy.orm import Session

from app.database.models import (
    AssetLevel,
    AssetVersion,
    Entity,
    EntityAlias,
    EntityType,
    RefType,
    VisualRef,
)
from app.schemas.location_subagent import (
    ArcHierarchyTag,
    DeepObservationalAnalysis,
    LocationPlateMatrixSpec,
    PlateRole,
    ShotLocationAnchor,
    SinglePlateSpec,
)
from app.services.comfyui_client import ComfyUIWorkflowCompiler, DualGPUComfyUIDispatcher

logger = logging.getLogger("novel_visualiser.location_subagent")


class LocationContinuitySubagent:
    """Autonomous Subagent dedicated to location canon, memory, and spatial continuity."""

    def __init__(
        self,
        db: Optional[Session] = None,
        comfy_dispatcher: Optional[DualGPUComfyUIDispatcher] = None,
        use_ai_thinking: bool = True,
    ):
        self.db = db
        self.comfy_dispatcher = comfy_dispatcher
        self.use_ai_thinking = use_ai_thinking
        self._kaggle_client = None

        if self.use_ai_thinking:
            try:
                # Add workspace root to sys.path if not present
                workspace_root = Path(__file__).resolve().parent.parent.parent.parent
                if str(workspace_root) not in sys.path:
                    sys.path.insert(0, str(workspace_root))
                from kaggle_ai_engine.proxy_client import KaggleModelProxyClient
                self._kaggle_client = KaggleModelProxyClient()
                logger.info("LocationContinuitySubagent initialized with Kaggle Model Proxy reasoning engine.")
            except Exception as exc:
                logger.warning("Could not initialize Kaggle Model Proxy: %s. Using heuristic analysis.", exc)
                self._kaggle_client = None

    # =========================================================================
    # 1. HISTORICAL MEMORY & CANON LOOKUP
    # =========================================================================

    def query_historical_memory(
        self,
        project_id: str,
        candidate_name: str,
    ) -> Optional[LocationPlateMatrixSpec]:
        """Check if an environment has already been established in prior chapters or arcs."""
        if not self.db:
            return None

        clean_name = candidate_name.strip().lower()

        # Query Entity by canonical_name or alias
        entity = (
            self.db.query(Entity)
            .filter(
                Entity.project_id == project_id,
                Entity.entity_type == EntityType.LOCATION,
            )
            .filter(
                (Entity.canonical_name.ilike(f"%{clean_name}%"))
                | (Entity.id.in_(
                    self.db.query(EntityAlias.entity_id).filter(
                        EntityAlias.alias.ilike(f"%{clean_name}%")
                    )
                ))
            )
            .first()
        )

        if not entity:
            return None

        # Fetch latest AssetVersion with level=ARC_DEFAULT or PROJECT_CANONICAL
        latest_version = (
            self.db.query(AssetVersion)
            .filter(
                AssetVersion.entity_id == entity.id,
                AssetVersion.level.in_([AssetLevel.ARC_DEFAULT, AssetLevel.PROJECT_CANONICAL]),
            )
            .order_by(AssetVersion.valid_from_chapter.desc())
            .first()
        )

        if not latest_version or not latest_version.visual_attributes:
            return None

        try:
            matrix_data = latest_version.visual_attributes
            if "plate_1_vista" in matrix_data:
                spec = LocationPlateMatrixSpec.model_validate(matrix_data)
                spec.is_existing = True
                return spec
        except Exception as exc:
            logger.warning("Failed to parse cached location matrix spec for %s: %s", entity.canonical_name, exc)

        return None

    # =========================================================================
    # 2. DEEP OBSERVATIONAL REASONING ENGINE (AI THINKING)
    # =========================================================================

    def observe_and_synthesize_location(
        self,
        location_name: str,
        passage_text: str,
        arc_hierarchy: ArcHierarchyTag,
        deterministic_seed: int = 1000,
    ) -> LocationPlateMatrixSpec:
        """
        Conduct deep 5-dimension Chain-of-Thought observation and synthesize the 4-plate standard.
        """
        import re
        clean_name = re.sub(r"[^a-zA-Z0-9_]+", "_", location_name.lower()).strip("_")
        location_id = f"loc_{clean_name[:32]}"

        # If frontier reasoning client is available, execute 5-dimension CoT
        if self._kaggle_client:
            try:
                system_prompt = (
                    "You are an Elite Anime Visual Director and Environmental Worldbuilder for a high-end anime production.\n"
                    "Analyze the novel scene text and perform a deep 5-dimension observational analysis:\n"
                    "1. Socioeconomic Reality & Poverty Physics: Material quality, rust, broken concrete, cheap synthetic fixtures, patched repairs.\n"
                    "2. Architectural Geometry: Dimensions, curb heights, road widths, fixed structural anchors.\n"
                    "3. Environmental Physics: Rain saturation, ground wetness, puddle oil sheens, atmospheric mist.\n"
                    "4. Volumetric Lighting: Color temperatures (sodium amber vs cold cyan fluorescent), shadow depth.\n"
                    "5. Botanical & Micro-Debris: Weeds in fissures, peeling posters, discarded litter.\n\n"
                    "Then compile the standardized 4-Plate Reference Matrix (Danbooru tags, anime aesthetic):\n"
                    "- Plate 1: Wide Exterior Establishing Vista (1216x832)\n"
                    "- Plate 2: Action Stage (interaction zone empty of people, 1216x832)\n"
                    "- Plate 3: Reciprocal Reverse Angle (180-deg counter view, 1216x832)\n"
                    "- Plate 4: Macro Material & Prop Texture (1024x1024)\n"
                    "CRITICAL: All plates must specify 'no humans, empty street/room' in prompt and negative prompt."
                )

                user_prompt = f"""
Location Name: {location_name}
Arc Hierarchy: Main Arc={arc_hierarchy.main_arc_id}, Sub Arc={arc_hierarchy.sub_arc_id}, Mini Arc={arc_hierarchy.mini_arc_id}
Time/Weather: {arc_hierarchy.time_of_day}, {arc_hierarchy.weather}
Novel Passage Text:
\"\"\"{passage_text[:3000]}\"\"\"

Respond ONLY in valid JSON matching this exact structure:
{{
  "observation": {{
    "socioeconomic_reality": "...",
    "architectural_geometry": "...",
    "environmental_physics": "...",
    "volumetric_lighting": "...",
    "botanical_micro_debris": "..."
  }},
  "plate_1_vista_prompt": "masterpiece, anime aesthetic, wide angle, exterior vista, ...",
  "plate_2_stage_prompt": "masterpiece, anime aesthetic, medium shot, action stage, ...",
  "plate_3_reverse_prompt": "masterpiece, anime aesthetic, 180-degree reverse shot, ...",
  "plate_4_macro_prompt": "masterpiece, anime aesthetic, extreme close-up macro texture, ..."
}}
"""
                resp_text = self._kaggle_client.chat_completion(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                )
                import re
                clean_text = resp_text.strip()
                if "```json" in clean_text:
                    clean_text = clean_text.split("```json")[1].split("```")[0].strip()
                elif "```" in clean_text:
                    clean_text = clean_text.split("```")[1].split("```")[0].strip()
                else:
                    start_idx = clean_text.find("{")
                    end_idx = clean_text.rfind("}")
                    if start_idx != -1 and end_idx != -1:
                        clean_text = clean_text[start_idx:end_idx + 1].strip()

                # Clean trailing commas which LLMs frequently emit before closing braces
                clean_text = re.sub(r",\s*([\]\}])", r"\1", clean_text)

                try:
                    data = json.loads(clean_text)
                except Exception as parse_err:
                    match = re.search(r"(\{.*\})", resp_text, re.DOTALL)
                    if match:
                        repaired = re.sub(r",\s*([\]\}])", r"\1", match.group(1))
                        data = json.loads(repaired)
                    else:
                        raise parse_err
                obs_data = data.get("observation", {})
                obs = DeepObservationalAnalysis(
                    socioeconomic_reality=obs_data.get("socioeconomic_reality", "Dystopian industrial decay"),
                    architectural_geometry=obs_data.get("architectural_geometry", "Brutalist concrete and asphalt"),
                    environmental_physics=obs_data.get("environmental_physics", "Wet rain-slicked surfaces"),
                    volumetric_lighting=obs_data.get("volumetric_lighting", "Cold streetlights and dark shadows"),
                    botanical_micro_debris=obs_data.get("botanical_micro_debris", "Sparse weeds and discarded refuse"),
                )

                neg = "worst quality, low quality, blurry, 1boy, 1girl, character, person, human, crowd, text, watermark"

                return LocationPlateMatrixSpec(
                    location_id=location_id,
                    canonical_name=location_name,
                    arc_hierarchy=arc_hierarchy,
                    observation=obs,
                    plate_1_vista=SinglePlateSpec(
                        role=PlateRole.WIDE_ESTABLISHING,
                        title=f"{location_name} — Wide Establishing Vista",
                        prompt=data.get("plate_1_vista_prompt", f"wide establishing shot of {location_name}, no humans"),
                        negative_prompt=neg,
                        width=1216,
                        height=832,
                        azimuth_degrees=0,
                        seed=deterministic_seed + 1,
                    ),
                    plate_2_stage=SinglePlateSpec(
                        role=PlateRole.ACTION_STAGE,
                        title=f"{location_name} — Action Stage Interaction Zone",
                        prompt=data.get("plate_2_stage_prompt", f"medium shot stage of {location_name}, no humans"),
                        negative_prompt=neg,
                        width=1216,
                        height=832,
                        azimuth_degrees=0,
                        seed=deterministic_seed + 2,
                    ),
                    plate_3_reverse=SinglePlateSpec(
                        role=PlateRole.REVERSE_RECIPROCAL,
                        title=f"{location_name} — 180-Degree Reciprocal View",
                        prompt=data.get("plate_3_reverse_prompt", f"reverse counter shot of {location_name}, no humans"),
                        negative_prompt=neg,
                        width=1216,
                        height=832,
                        azimuth_degrees=180,
                        seed=deterministic_seed + 3,
                    ),
                    plate_4_macro=SinglePlateSpec(
                        role=PlateRole.MACRO_TEXTURE,
                        title=f"{location_name} — Macro Material & Texture Anchor",
                        prompt=data.get("plate_4_macro_prompt", f"macro close-up texture of {location_name}"),
                        negative_prompt=neg,
                        width=1024,
                        height=1024,
                        azimuth_degrees=0,
                        seed=deterministic_seed + 4,
                    ),
                    is_existing=False,
                )
            except Exception as exc:
                logger.warning("Kaggle AI thinking analysis failed: %s; falling back to heuristic engine", exc)

        # Heuristic Rule-Based Deep Fallback
        obs = DeepObservationalAnalysis(
            socioeconomic_reality="Dystopian brutalist outskirts; decaying ferroconcrete, corrugated iron, exposed cables",
            architectural_geometry="Standard 15cm road curb, 1.8m bolted industrial angle-iron bench, wide traffic boulevard",
            environmental_physics="Continuous cold precipitation, saturated asphalt creating mirror-like puddles with oily sheens",
            volumetric_lighting="Low-pressure sodium vapor amber overhead contrasted with stark cyan security floodlights",
            botanical_micro_debris="Dark lichen and dandelions cracking through asphalt joints, bent iron tree-guard",
        )
        neg = "worst quality, low quality, blurry, 1boy, 1girl, character, person, human, crowd, text, watermark"

        return LocationPlateMatrixSpec(
            location_id=location_id,
            canonical_name=location_name,
            arc_hierarchy=arc_hierarchy,
            observation=obs,
            plate_1_vista=SinglePlateSpec(
                role=PlateRole.WIDE_ESTABLISHING,
                title=f"{location_name} — Wide Establishing Vista",
                prompt=f"masterpiece, anime aesthetic, wide establishing shot, exterior landscape of {location_name}, brutalist architecture, rain-slicked wet asphalt with puddle reflections, neon streetlights, atmospheric volumetric mist, no humans, empty street",
                negative_prompt=neg,
                width=1216,
                height=832,
                seed=deterministic_seed + 1,
            ),
            plate_2_stage=SinglePlateSpec(
                role=PlateRole.ACTION_STAGE,
                title=f"{location_name} — Action Stage Interaction Zone",
                prompt=f"masterpiece, anime aesthetic, eye-level medium shot, action stage of {location_name}, rusted metal bench on cracked sidewalk curb, wet ground, streetlamp glow, empty bench, no humans",
                negative_prompt=neg,
                width=1216,
                height=832,
                seed=deterministic_seed + 2,
            ),
            plate_3_reverse=SinglePlateSpec(
                role=PlateRole.REVERSE_RECIPROCAL,
                title=f"{location_name} — 180-Degree Reciprocal View",
                prompt=f"masterpiece, anime aesthetic, 180-degree counter shot looking opposite from {location_name}, wide rainy street, fortified building facade, security barriers, cold cyan lights, no humans",
                negative_prompt=neg,
                width=1216,
                height=832,
                azimuth_degrees=180,
                seed=deterministic_seed + 3,
            ),
            plate_4_macro=SinglePlateSpec(
                role=PlateRole.MACRO_TEXTURE,
                title=f"{location_name} — Macro Material & Texture Anchor",
                prompt=f"masterpiece, anime aesthetic, extreme close-up macro texture, corroded steel rivets, flaking paint with orange rust, wet surface with water droplets, dramatic lighting",
                negative_prompt=neg,
                width=1024,
                height=1024,
                seed=deterministic_seed + 4,
            ),
            is_existing=False,
        )

    # =========================================================================
    # 3. STRUCTURAL DEPTH & CANNY WIREFRAME GENERATION
    # =========================================================================

    def compute_structural_guides(self, image_path: Path) -> tuple[Path, Path]:
        """Compute structural 3D depth map and Canny lineart wireframe from an RGB plate."""
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read plate image for structural guide extraction: {image_path}")

        # Compute Depth Map (Gradient + Luminance proxy with bilateral smoothing)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        y_gradient = np.linspace(0, 255, h, dtype=np.uint8)
        y_gradient = np.tile(y_gradient[:, np.newaxis], (1, w))
        depth_raw = cv2.addWeighted(gray, 0.45, y_gradient, 0.55, 0)
        depth_smooth = cv2.bilateralFilter(depth_raw, 9, 75, 75)

        depth_path = image_path.parent / f"{image_path.stem}_depth.png"
        cv2.imwrite(str(depth_path), depth_smooth)

        # Compute Canny Lineart Wireframe
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 60, 160)
        # Invert to black lines on white background standard for lineart
        lineart_inverted = cv2.bitwise_not(edges)

        lineart_path = image_path.parent / f"{image_path.stem}_lineart.png"
        cv2.imwrite(str(lineart_path), lineart_inverted)

        return depth_path, lineart_path

    # =========================================================================
    # 4. MULTI-PLATE AUTOMATED FOUNDRY EXECUTION
    # =========================================================================

    def generate_location_matrix(
        self,
        matrix_spec: LocationPlateMatrixSpec,
        output_dir: Path,
    ) -> LocationPlateMatrixSpec:
        """Render all 4 plates across the ComfyUI cluster and extract structural guides."""
        output_dir.mkdir(parents=True, exist_ok=True)
        plates = [
            ("plate_1_vista", matrix_spec.plate_1_vista),
            ("plate_2_stage", matrix_spec.plate_2_stage),
            ("plate_3_reverse", matrix_spec.plate_3_reverse),
            ("plate_4_macro", matrix_spec.plate_4_macro),
        ]

        # Check if ComfyUI cluster is available
        if self.comfy_dispatcher and self.comfy_dispatcher.clients:
            logger.info("Executing 4-plate generation on ComfyUI Dual-GPU cluster...")
            jobs = []
            for key, plate in plates:
                job_id = f"{matrix_spec.location_id}_{plate.role.value.lower()}"
                workflow = ComfyUIWorkflowCompiler.compile_shot_env(
                    shot_id=job_id,
                    env_prompt=plate.prompt,
                    negative_prompt=plate.negative_prompt,
                    width=plate.width,
                    height=plate.height,
                    seed=plate.seed or random.randint(1000, 9999),
                )
                jobs.append((job_id, workflow))

            # Dispatch across dual-GPU workers concurrently
            results = self.comfy_dispatcher.dispatch_batch_parallel(jobs, output_dir=output_dir)

            for (key, plate), (job_id, img_path) in zip(plates, results):
                if img_path and img_path.exists():
                    plate.image_path = str(img_path)
                    depth_p, lineart_p = self.compute_structural_guides(img_path)
                    plate.depth_map_path = str(depth_p)
                    plate.lineart_map_path = str(lineart_p)
                else:
                    logger.warning("Plate %s failed to render; generating mock fallback", key)
                    self._create_mock_plate(plate, output_dir, matrix_spec.location_id)
        else:
            # Fallback mock plate generation for offline/testing mode
            logger.info("ComfyUI cluster not connected; generating local test plates")
            for key, plate in plates:
                self._create_mock_plate(plate, output_dir, matrix_spec.location_id)

        return matrix_spec

    def _create_mock_plate(self, plate: SinglePlateSpec, out_dir: Path, location_id: str) -> None:
        """Create a placeholder plate with high-contrast structural guide for tests."""
        img_path = out_dir / f"{location_id}_{plate.role.value.lower()}.png"
        img = np.zeros((plate.height, plate.width, 3), dtype=np.uint8)
        img[:] = (35, 30, 25)  # dark moody background
        # Draw mock horizon and structural guidelines
        cv2.line(img, (0, int(plate.height * 0.6)), (plate.width, int(plate.height * 0.6)), (80, 75, 70), 3)
        cv2.putText(
            img,
            f"{plate.title} ({plate.role.value})",
            (50, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (200, 200, 200),
            2,
        )
        cv2.imwrite(str(img_path), img)

        plate.image_path = str(img_path)
        depth_p, lineart_p = self.compute_structural_guides(img_path)
        plate.depth_map_path = str(depth_p)
        plate.lineart_map_path = str(lineart_p)

    # =========================================================================
    # 5. CAMERA ANGLE RESOLUTION FOR SHOTS
    # =========================================================================

    def resolve_shot_angle_anchor(
        self,
        matrix_spec: LocationPlateMatrixSpec,
        shot_cinematic_intent: str,  # 'WIDE', 'STAGE_ACTION', 'REVERSE_POV', 'MACRO'
        camera_description: str = "",
    ) -> ShotLocationAnchor:
        """Resolve which plate and structural guides anchor a specific shot."""
        intent = shot_cinematic_intent.upper()

        if "WIDE" in intent or "ESTABLISHING" in intent:
            plate = matrix_spec.plate_1_vista
            inpaint_box = [0.1, 0.4, 0.9, 0.95]
        elif "REVERSE" in intent or "POV" in intent or "COUNTER" in intent:
            plate = matrix_spec.plate_3_reverse
            inpaint_box = [0.2, 0.3, 0.8, 0.9]
        elif "MACRO" in intent or "CLOSEUP" in intent or "DETAIL" in intent:
            plate = matrix_spec.plate_4_macro
            inpaint_box = [0.25, 0.25, 0.75, 0.75]
        else:  # Default to Action Stage (Plate 2)
            plate = matrix_spec.plate_2_stage
            inpaint_box = [0.25, 0.35, 0.75, 0.95]

        return ShotLocationAnchor(
            location_id=matrix_spec.location_id,
            canonical_name=matrix_spec.canonical_name,
            resolved_role=plate.role,
            selected_plate_image_path=plate.image_path or "",
            depth_map_path=plate.depth_map_path,
            lineart_map_path=plate.lineart_map_path,
            lighting_baseline=matrix_spec.observation.volumetric_lighting,
            weather_condition=matrix_spec.observation.environmental_physics,
            recommended_inpaint_box=inpaint_box,
            temporal_delta_tags=[matrix_spec.temporal_delta] if matrix_spec.temporal_delta else [],
        )

    # =========================================================================
    # 6. DATABASE CANON PERSISTENCE
    # =========================================================================

    def save_to_canon(
        self,
        project_id: str,
        matrix_spec: LocationPlateMatrixSpec,
        current_chapter: int = 1,
    ) -> Entity:
        """Persist location into SQLite entities, asset_versions, and visual_refs."""
        if not self.db:
            logger.warning("No database session provided; skipping canon persistence.")
            return Entity(id=matrix_spec.location_id, canonical_name=matrix_spec.canonical_name)

        # 1. Upsert Entity
        entity = (
            self.db.query(Entity)
            .filter(Entity.id == matrix_spec.location_id)
            .first()
        )
        if not entity:
            entity = Entity(
                id=matrix_spec.location_id,
                project_id=project_id,
                entity_type=EntityType.LOCATION,
                canonical_name=matrix_spec.canonical_name,
                description=matrix_spec.observation.socioeconomic_reality,
                first_appearance_chapter=current_chapter,
                is_active=True,
            )
            self.db.add(entity)
            self.db.flush()

        # 2. Add AssetVersion (Level = ARC_DEFAULT or PROJECT_CANONICAL)
        asset_level = AssetLevel.ARC_DEFAULT if matrix_spec.arc_hierarchy.main_arc_id else AssetLevel.PROJECT_CANONICAL
        asset_version = AssetVersion(
            entity_id=entity.id,
            level=asset_level,
            arc_id=matrix_spec.arc_hierarchy.main_arc_id if self.db.query(Entity).filter(Entity.id == matrix_spec.arc_hierarchy.main_arc_id).first() else None,
            valid_from_chapter=current_chapter,
            change_description=matrix_spec.temporal_delta or "Initial 4-plate location reference matrix",
            visual_attributes=matrix_spec.model_dump(),
        )
        self.db.add(asset_version)
        self.db.flush()

        # 3. Add VisualRef records for each plate & structural guide
        plates = [
            matrix_spec.plate_1_vista,
            matrix_spec.plate_2_stage,
            matrix_spec.plate_3_reverse,
            matrix_spec.plate_4_macro,
        ]

        for p in plates:
            if p.image_path:
                vref_img = VisualRef(
                    entity_id=entity.id,
                    asset_version_id=asset_version.id,
                    ref_type=RefType.LOCATION_PLATE,
                    image_path=p.image_path,
                    seed=p.seed,
                )
                self.db.add(vref_img)
            if p.depth_map_path:
                vref_depth = VisualRef(
                    entity_id=entity.id,
                    asset_version_id=asset_version.id,
                    ref_type=RefType.LOCATION_PLATE,
                    image_path=p.depth_map_path,
                )
                self.db.add(vref_depth)
            if p.lineart_map_path:
                vref_line = VisualRef(
                    entity_id=entity.id,
                    asset_version_id=asset_version.id,
                    ref_type=RefType.LOCATION_PLATE,
                    image_path=p.lineart_map_path,
                )
                self.db.add(vref_line)

        self.db.commit()
        logger.info("Successfully persisted location %s to SQLite canon.", matrix_spec.canonical_name)
        return entity
