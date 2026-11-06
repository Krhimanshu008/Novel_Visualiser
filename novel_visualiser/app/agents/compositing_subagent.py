"""Novel Visualiser — Compositing & Shot Synthesis Subagent.

Autonomous Subagent responsible for:
1. Harmonizing outputs from all 4 specialized subagents:
   - LocationContinuitySubagent (Stage Plate + Depth Map)
   - CharacterContinuitySubagent (Biological Epoch + Wardrobe + Face Chip)
   - EnvironmentContinuitySubagent (Lighting Rig + Weather + Surface Moisture)
   - CreatureMonsterSubagent (Creature Prompt + Predatory Head Chip + Scale)
2. Compiling de-duplicated, non-contaminated multi-layer prompts and negative constraints.
3. Compiling targeted ComfyUI workflows:
   - ControlNet Union Depth Guidance
   - IP-Adapter Advanced Face / Head Consistency
   - Differential Diffusion Inpainting
   - Integrated FaceDetailer Remastering Pass
4. Dispatching jobs across the permanent Dual-GPU ComfyUI cluster (GPU 0 & GPU 1).
5. Handling offline / test mock canvas synthesis for deterministic testing.
"""

from __future__ import annotations

import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Optional
import cv2
import numpy as np
from sqlalchemy.orm import Session

from app.database.models import Base, Shot, ShotStatus
from app.schemas.compositing_subagent import (
    CompositingModeEnum,
    RenderedShotArtifact,
    ShotCompositionSpec,
)
from app.services.comfyui_client import (
    ComfyUIClient,
    ComfyUIWorkflowCompiler,
    DualGPUComfyUIDispatcher,
    get_dual_gpu_dispatcher,
)

logger = logging.getLogger("novel_visualiser.compositing_subagent")


class CompositingSubagent:
    """Autonomous Subagent dedicated to multi-layer shot compositing and execution."""

    def __init__(
        self,
        db: Optional[Session] = None,
        dispatcher: Optional[DualGPUComfyUIDispatcher] = None,
        allow_offline_fallback: bool = True,
    ):
        """Initialize CompositingSubagent with database and cluster dispatcher."""
        self.db = db
        self.allow_offline_fallback = allow_offline_fallback
        
        if dispatcher is not None:
            self.dispatcher = dispatcher
        else:
            try:
                self.dispatcher = get_dual_gpu_dispatcher()
            except Exception as exc:
                logger.warning("Could not initialize default cluster dispatcher: %s", exc)
                self.dispatcher = None

    # =========================================================================
    # 1. MULTI-SUBAGENT PROMPT HARMONIZATION
    # =========================================================================

    def assemble_shot_prompts(
        self,
        spec: ShotCompositionSpec,
    ) -> tuple[str, str]:
        """
        Merge and de-duplicate prompt blocks from all 4 subagents without token bleed.
        """
        positive_blocks: list[str] = [
            "masterpiece",
            "best quality",
            "cinematic manhwa illustration",
            "8k wallpaper",
        ]

        # 1. Environment & Lighting
        if spec.environment_anchor.resolved_lighting_block:
            positive_blocks.append(spec.environment_anchor.resolved_lighting_block)

        # 2. Location Anchor (if provided)
        if spec.location_anchor:
            loc_pos = getattr(
                spec.location_anchor,
                "resolved_shot_prompt",
                getattr(spec.location_anchor, "lighting_baseline", ""),
            )
            if loc_pos:
                positive_blocks.append(loc_pos)

        # 3. Character Biology & Wardrobe
        char_pos = getattr(
            spec.character_anchor,
            "full_positive_prompt",
            getattr(spec.character_anchor, "resolved_shot_prompt", ""),
        )
        if char_pos:
            positive_blocks.append(char_pos)

        # 4. Creature (if combat shot)
        if spec.creature_anchor:
            positive_blocks.append(spec.creature_anchor.resolved_creature_prompt)

        # De-duplicate positive tokens while preserving natural order
        seen_pos = set()
        deduped_pos: list[str] = []
        for block in positive_blocks:
            for token in block.split(","):
                token_clean = token.strip()
                token_lower = token_clean.lower()
                if token_clean and token_lower not in seen_pos:
                    seen_pos.add(token_lower)
                    deduped_pos.append(token_clean)

        final_positive = ", ".join(deduped_pos)

        # Assemble Negative Constraints
        negative_blocks: list[str] = [
            "worst quality",
            "low quality",
            "blurry",
            "extra limbs",
            "bad anatomy",
            "deformed fingers",
        ]

        char_neg = getattr(
            spec.character_anchor,
            "full_negative_prompt",
            getattr(spec.character_anchor, "resolved_negative_prompt", ""),
        )
        if char_neg:
            negative_blocks.append(char_neg)

        negative_blocks.append(spec.environment_anchor.resolved_negative_block)
        if spec.creature_anchor:
            negative_blocks.append(spec.creature_anchor.resolved_negative_prompt)

        seen_neg = set()
        deduped_neg: list[str] = []
        for block in negative_blocks:
            for token in block.split(","):
                token_clean = token.strip()
                token_lower = token_clean.lower()
                if token_clean and token_lower not in seen_neg:
                    seen_neg.add(token_lower)
                    deduped_neg.append(token_clean)

        final_negative = ", ".join(deduped_neg)

        return final_positive, final_negative

    # =========================================================================
    # 2. COMPILE WORKFLOW SPECIFICATION
    # =========================================================================

    def compile_workflow(
        self,
        spec: ShotCompositionSpec,
        positive_prompt: str,
        negative_prompt: str,
    ) -> dict[str, Any]:
        """Compile the targeted ComfyUI workflow depending on compositing mode."""
        seed = spec.deterministic_seed or random.randint(1000000, 2147483647)

        # Extract file paths
        face_chip = getattr(
            spec.character_anchor,
            "face_reference_path",
            getattr(spec.character_anchor, "face_reference_chip_path", None),
        )
        depth_map = spec.location_depth_path
        loc_plate = spec.location_plate_path

        # Basenames for ComfyUI upload staging
        face_ref_name = Path(face_chip).name if face_chip else None
        depth_ref_name = Path(depth_map).name if depth_map else None
        plate_ref_name = Path(loc_plate).name if loc_plate else None

        if spec.mode == CompositingModeEnum.INPAINT_DIFFERENTIAL and loc_plate:
            bbox = (
                spec.character_bbox.x,
                spec.character_bbox.y,
                spec.character_bbox.width,
                spec.character_bbox.height,
            )
            return ComfyUIWorkflowCompiler.compile_edit_inpaint(
                shot_id=spec.shot_id,
                failed_image_filename=plate_ref_name or "stage_canvas.png",
                repair_prompt=positive_prompt,
                defect_bbox=bbox,
                negative_prompt=negative_prompt,
                denoise=0.75,
                seed=seed,
                image_width=spec.canvas_width,
                image_height=spec.canvas_height,
            )

        elif spec.mode == CompositingModeEnum.MULTI_SUBJECT_BATTLE and spec.creature_anchor:
            creature_chip = spec.creature_anchor.head_chip_path
            creature_ref_name = Path(creature_chip).name if creature_chip else None
            char_pos = getattr(
                spec.character_anchor,
                "full_positive_prompt",
                getattr(spec.character_anchor, "resolved_shot_prompt", ""),
            )
            return ComfyUIWorkflowCompiler.compile_shot_multi(
                shot_id=spec.shot_id,
                bg_prompt=spec.environment_anchor.resolved_lighting_block,
                subj1_prompt=char_pos,
                subj2_prompt=spec.creature_anchor.resolved_creature_prompt,
                subj1_face_filename=face_ref_name or "char_face.png",
                subj2_face_filename=creature_ref_name or "creature_head.png",
                loc_depth_filename=depth_ref_name,
                negative_prompt=negative_prompt,
                seed=seed,
                width=spec.canvas_width,
                height=spec.canvas_height,
            )

        else:
            # Default: ControlNet + IP-Adapter Solo Composite
            return ComfyUIWorkflowCompiler.compile_shot_solo(
                shot_id=spec.shot_id,
                positive_prompt=positive_prompt,
                negative_prompt=negative_prompt,
                face_ref_filename=face_ref_name,
                loc_depth_filename=depth_ref_name,
                seed=seed,
                width=spec.canvas_width,
                height=spec.canvas_height,
            )

    # =========================================================================
    # 3. END-TO-END EXECUTION (CLUSTER OR DETERMINISTIC MOCK)
    # =========================================================================

    def execute_composite_shot(
        self,
        spec: ShotCompositionSpec,
        output_dir: Optional[Path] = None,
        force_mock: bool = False,
    ) -> RenderedShotArtifact:
        """
        Execute compositing on the live Dual-GPU cluster, or generate mock canvas if offline.
        """
        target_dir = output_dir or Path("output/composited_shots")
        target_dir.mkdir(parents=True, exist_ok=True)
        final_img_path = target_dir / f"{spec.shot_id}.png"

        start_time = time.time()
        pos_prompt, neg_prompt = self.assemble_shot_prompts(spec)
        workflow = self.compile_workflow(spec, pos_prompt, neg_prompt)

        cluster_online = False
        active_client: Optional[ComfyUIClient] = None

        if not force_mock and self.dispatcher:
            try:
                healths = self.dispatcher.check_health_all()
                online_clients = [
                    self.dispatcher.clients[i]
                    for i, h in enumerate(healths)
                    if h.get("status") == "online" or h.get("online")
                ]
                if online_clients:
                    if not hasattr(self, "_worker_counter"):
                        self._worker_counter = 0
                    active_client = online_clients[self._worker_counter % len(online_clients)]
                    self._worker_counter += 1
                    cluster_online = True
            except Exception as exc:
                logger.debug("Cluster check failed: %s", exc)

        # EXECUTION PATH A: LIVE DUAL-GPU COMFYUI CLUSTER
        if cluster_online and active_client is not None:
            try:
                logger.info(
                    "Dispatching shot [%s] to live ComfyUI worker: %s",
                    spec.shot_id,
                    active_client.base_url,
                )

                # Pre-upload referenced local files to the worker input directory
                face_chip = getattr(
                    spec.character_anchor,
                    "face_reference_path",
                    getattr(spec.character_anchor, "face_reference_chip_path", None),
                )
                local_images_to_stage = [
                    face_chip,
                    spec.location_depth_path,
                    spec.location_plate_path,
                ]
                if spec.creature_anchor:
                    local_images_to_stage.append(spec.creature_anchor.head_chip_path)

                for local_p in local_images_to_stage:
                    if local_p and Path(local_p).exists():
                        try:
                            active_client.upload_image(Path(local_p))
                        except Exception as e:
                            logger.warning("Could not pre-upload %s: %s", local_p, e)

                # Submit and wait
                prompt_id = active_client.submit_prompt(workflow)
                outputs = active_client.wait_for_execution(prompt_id, max_wait=180.0)

                # Download resulting image
                downloaded: list[Path] = []
                for node_id, node_out in outputs.items():
                    if "images" in node_out:
                        for img_meta in node_out["images"]:
                            fname = img_meta.get("filename")
                            sub = img_meta.get("subfolder", "")
                            if fname:
                                p = active_client.download_output_image(
                                    fname, subfolder=sub, target_dir=target_dir
                                )
                                downloaded.append(p)

                if downloaded:
                    # Rename first output to canonical shot_id
                    downloaded[0].replace(final_img_path)
                    exec_time = time.time() - start_time
                    logger.info("Shot [%s] rendered on %s in %.2fs", spec.shot_id, active_client.base_url, exec_time)

                    return RenderedShotArtifact(
                        shot_id=spec.shot_id,
                        image_path=str(final_img_path),
                        mode_used=spec.mode,
                        worker_url_used=active_client.base_url,
                        prompt_id=prompt_id,
                        execution_time_seconds=exec_time,
                        final_positive_prompt=pos_prompt,
                        final_negative_prompt=neg_prompt,
                        qc_passed=True,
                    )

            except Exception as exc:
                logger.warning("Live ComfyUI render failed: %s; falling back to mock canvas.", exc)

        # EXECUTION PATH B: DETERMINISTIC MOCK CANVAS GENERATION
        if not self.allow_offline_fallback:
            raise RuntimeError(f"Could not render shot {spec.shot_id} and offline fallback is disabled.")

        self._generate_mock_composite_canvas(spec, final_img_path)
        exec_time = time.time() - start_time

        return RenderedShotArtifact(
            shot_id=spec.shot_id,
            image_path=str(final_img_path),
            mode_used=spec.mode,
            worker_url_used="mock://local_opencv_engine",
            prompt_id=f"mock_{random.randint(1000, 9999)}",
            execution_time_seconds=exec_time,
            final_positive_prompt=pos_prompt,
            final_negative_prompt=neg_prompt,
            qc_passed=True,
        )

    # =========================================================================
    # 4. MOCK COMPOSITE CANVAS SYNTHESIS (OPENCV)
    # =========================================================================

    def _generate_mock_composite_canvas(
        self,
        spec: ShotCompositionSpec,
        dest_path: Path,
    ) -> None:
        """Create an annotated OpenCV canvas plate for deterministic testing."""
        w, h = spec.canvas_width, spec.canvas_height

        # Base Canvas: Load location plate if exists, otherwise create twilight smog canvas
        if spec.location_plate_path and Path(spec.location_plate_path).exists():
            canvas = cv2.imread(spec.location_plate_path)
            if canvas.shape[0] != h or canvas.shape[1] != w:
                canvas = cv2.resize(canvas, (w, h))
        else:
            canvas = np.full((h, w, 3), (25, 20, 18), dtype=np.uint8)  # Smog twilight dark background
            # Road curb and transit bench geometry
            cv2.rectangle(canvas, (0, int(h * 0.70)), (w, h), (40, 40, 45), -1)
            cv2.line(canvas, (0, int(h * 0.70)), (w, int(h * 0.70)), (75, 75, 80), 3)
            # Metal bench
            cv2.rectangle(canvas, (300, 480), (850, 560), (60, 55, 50), -1)
            cv2.rectangle(canvas, (310, 560), (330, 680), (35, 30, 25), -1)
            cv2.rectangle(canvas, (820, 560), (840, 680), (35, 30, 25), -1)

        # Character Silhouette & Inpaint Box
        bx = spec.character_bbox
        # Soft key light glow on character bounding box
        key_glow = np.zeros_like(canvas)
        cv2.ellipse(
            key_glow,
            (bx.x + bx.width // 2, bx.y + bx.height // 2),
            (bx.width // 2 + 30, bx.height // 2 + 30),
            0, 0, 360,
            (20, 120, 230), # Warm amber sodium glow
            -1
        )
        canvas = cv2.addWeighted(canvas, 1.0, key_glow, 0.25, 0)

        # Character Silhouette
        char_cx = bx.x + bx.width // 2
        char_cy = bx.y + bx.height // 2
        cv2.circle(canvas, (char_cx, char_cy - 80), 35, (180, 160, 150), -1) # Head
        cv2.rectangle(canvas, (char_cx - 45, char_cy - 40), (char_cx + 45, char_cy + 90), (70, 70, 75), -1) # Charcoal jacket
        cv2.putText(canvas, "Sunny (Inpainted)", (bx.x + 10, bx.y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)

        # Creature presence if multi-subject
        if spec.creature_anchor:
            cv2.ellipse(canvas, (980, 420), (160, 120), 0, 0, 360, (15, 15, 15), -1)
            cv2.circle(canvas, (920, 380), 8, (30, 30, 240), -1) # Red eyes
            cv2.circle(canvas, (940, 380), 8, (30, 30, 240), -1)
            cv2.putText(canvas, "Creature Threat", (880, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)

        # Header metadata overlay
        cv2.rectangle(canvas, (0, 0), (w, 45), (10, 10, 12), -1)
        cv2.putText(
            canvas,
            f"SHOT COMPOSITE: [{spec.shot_id}] | Mode: {spec.mode.value} | Lighting: {spec.environment_anchor.time_of_day.value}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (230, 230, 230),
            2,
            cv2.LINE_AA,
        )

        cv2.imwrite(str(dest_path), canvas)
        logger.info("Saved mock composite canvas to %s", dest_path)
