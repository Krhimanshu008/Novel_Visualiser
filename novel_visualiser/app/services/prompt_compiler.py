"""Novel Visualiser — Structured 8-Block Prompt Compiler Service (Phase 4 / VCE).

Responsible for:
1. 100% Deterministic prompt compilation (0 LLM tokens during generation).
2. Converts ShotVisualContext or ShotSpec + Hierarchical Asset Resolver outputs
   into the fixed 8-block Danbooru prompt architecture:
     Block 1: Global Style Anchor (VisualStyleProfile)
     Block 2: Subject Identity Anchor (Immutable Danbooru face/body tags)
     Block 3: Appearance & Outfit Block (Arc-specific attire)
     Block 4: Equipment & Accessories Block (Weapons, handheld items)
     Block 5: Environment Plate Anchor (Architecture, backdrop)
     Block 6: Framing, Camera & Pose Block (Composition, angle, posture)
     Block 7: Atmosphere & Temporal Deltas Block (Injuries, dirt, lighting, VFX)
     Block 8: Negative Constraints Block (Anti-features, quality barriers)
3. Produces regional conditioning blocks for multi-character shots to avoid tag bleeding.
4. Generates direct ComfyUI API workflow payloads (comfyui_workflow_params).
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Optional

from app.database.models import Shot
from app.orchestrator.resolver import resolve_asset_for_scene
from app.schemas.scene import CompiledPrompt, ShotSpec
from app.schemas.visual_context import (
    VisualStyleProfile,
    DanbooruIdentityLock,
    OutfitCard,
    LocationPlateSpec,
    SubjectVisualState,
    ShotVisualContext,
    PromptBlockCollection,
)
from app.services.comfyui_client import ComfyUIWorkflowCompiler

logger = logging.getLogger("novel_visualiser.prompt_compiler")

DEFAULT_NEGATIVE_PROMPT = (
    "low quality, worst quality, blurry, bad anatomy, deformed limbs, extra arms, "
    "extra legs, missing fingers, extra fingers, floating limbs, disconnected limbs, "
    "mutation, poorly drawn face, poorly drawn hands, cropped, out of frame, "
    "watermark, signature, text, modern clothing, oversaturated, flat shading"
)

VFX_PROMPT_MAP = {
    "falling_snow_heavy": "heavy blizzard falling snow, swirling frost particles, cold white snowflakes",
    "shadow_tendrils": "writhing ethereal shadow tendrils rising from ground, dark shadow essence, shadowy aura",
    "blood_spatter": "fresh blood spatter droplets, battle wounds, dark crimson staining",
    "fire_embers": "drifting glowing orange fire embers, heat shimmer, sparks rising in the air",
    "divine_flame": "brilliant white-gold holy flames, radiant divine luminescence",
}


def _clean_block(tags: list[str]) -> str:
    """Join tag list into a clean, comma-separated string without empty entries or double commas."""
    cleaned = []
    for t in tags:
        if not t:
            continue
        stripped = str(t).strip().strip(",")
        if stripped and stripped not in cleaned:
            cleaned.append(stripped)
    return ", ".join(cleaned)


class PromptCompiler:
    """Structured 8-Block Danbooru Prompt Compiler.
    
    Compiles shot specifications and resolved visual entity records into
    ComfyUI-ready CompiledPrompt objects adhering to the Visual Consistency Engine.
    """

    def __init__(self, default_style_profile: Optional[VisualStyleProfile] = None):
        self.default_style_profile = default_style_profile or VisualStyleProfile()

    def build_8_blocks(
        self,
        style_profile: VisualStyleProfile,
        subjects: list[SubjectVisualState | dict[str, Any]],
        environment_tags: list[str],
        camera_framing: str,
        camera_angle: str,
        lighting: Optional[str] = None,
        vfx_tag: Optional[str] = None,
        mood: Optional[str] = None,
        action_cue: Optional[str] = None,
        custom_negatives: Optional[list[str]] = None,
    ) -> PromptBlockCollection:
        """Assemble the 8 ordered prompt blocks for diffusion conditioning."""
        # -------------------------------------------------------------
        # Block 1: Global Style Anchor
        # -------------------------------------------------------------
        block1_style = _clean_block(style_profile.master_quality_tags)

        # -------------------------------------------------------------
        # Block 2: Character Identity Anchor (Immutable Danbooru tags)
        # -------------------------------------------------------------
        block2_tags: list[str] = []
        for sub in subjects:
            if isinstance(sub, SubjectVisualState):
                if sub.identity_lock and sub.identity_lock.core_danbooru_tags:
                    block2_tags.extend(sub.identity_lock.core_danbooru_tags)
                else:
                    block2_tags.append(sub.entity_name)
            elif isinstance(sub, dict):
                # Legacy dictionary structure
                ename = sub.get("entity_name") or "character"
                block2_tags.append(ename)
                canon = sub.get("resolved", {}).get("canonical", {})
                for k in ["species", "physique", "hair", "eyes", "skin_color"]:
                    v = canon.get(k)
                    if v:
                        block2_tags.append(str(v))
        block2_identity = _clean_block(block2_tags)

        # -------------------------------------------------------------
        # Block 3: Appearance & Costume Block (Arc-specific attire)
        # -------------------------------------------------------------
        block3_tags: list[str] = []
        for sub in subjects:
            if isinstance(sub, SubjectVisualState):
                if sub.outfit and sub.outfit.danbooru_clothing_tags:
                    block3_tags.extend(sub.outfit.danbooru_clothing_tags)
            elif isinstance(sub, dict):
                arc = sub.get("resolved", {}).get("arc", {})
                if arc.get("clothing"):
                    block3_tags.append(f"wearing {arc['clothing']}")
                if arc.get("armor"):
                    block3_tags.append(f"armored with {arc['armor']}")
        block3_outfit = _clean_block(block3_tags)

        # -------------------------------------------------------------
        # Block 4: Equipment & Accessories Block (Weapons, handhelds)
        # -------------------------------------------------------------
        block4_tags: list[str] = []
        for sub in subjects:
            if isinstance(sub, SubjectVisualState):
                block4_tags.extend(sub.held_equipment)
            elif isinstance(sub, dict):
                arc = sub.get("resolved", {}).get("arc", {})
                if arc.get("weapons"):
                    block4_tags.append(f"carrying {arc['weapons']}")
                if arc.get("accessories"):
                    acc = arc["accessories"]
                    if isinstance(acc, list):
                        block4_tags.extend([str(a) for a in acc])
                    else:
                        block4_tags.append(str(acc))
        block4_equipment = _clean_block(block4_tags)

        # -------------------------------------------------------------
        # Block 5: Environment Plate Anchor
        # -------------------------------------------------------------
        block5_environment = _clean_block(environment_tags)

        # -------------------------------------------------------------
        # Block 6: Framing, Camera & Pose Block
        # -------------------------------------------------------------
        block6_tags: list[str] = []
        framing_str = camera_framing.replace("_", " ") + " framing" if camera_framing else ""
        angle_str = camera_angle.replace("_", " ") + " shot" if camera_angle else ""
        if angle_str:
            block6_tags.append(angle_str)
        if framing_str:
            block6_tags.append(framing_str)
        if action_cue:
            block6_tags.append(action_cue)

        for sub in subjects:
            if isinstance(sub, SubjectVisualState):
                if sub.pose_expression:
                    block6_tags.append(sub.pose_expression)
            elif isinstance(sub, dict):
                if sub.get("pose"):
                    block6_tags.append(sub["pose"])
                if sub.get("expression"):
                    block6_tags.append(sub["expression"])
        block6_composition = _clean_block(block6_tags)

        # -------------------------------------------------------------
        # Block 7: Atmosphere, Temporal Deltas & Lighting Block
        # -------------------------------------------------------------
        block7_tags: list[str] = []
        # Temporal injuries / dirt
        for sub in subjects:
            if isinstance(sub, SubjectVisualState):
                block7_tags.extend(sub.temporal_deltas)
            elif isinstance(sub, dict):
                delta = sub.get("resolved", {}).get("chapter_delta", {})
                if delta.get("injuries"):
                    inj = delta["injuries"]
                    if isinstance(inj, list):
                        block7_tags.extend([str(i) for i in inj])
                    else:
                        block7_tags.append(f"injuries: {inj}")
                if delta.get("conditions"):
                    cond = delta["conditions"]
                    if isinstance(cond, list):
                        block7_tags.extend([str(c) for c in cond])
                    else:
                        block7_tags.append(f"conditions: {cond}")

        # Lighting & mood
        if lighting:
            block7_tags.append(lighting)
        if mood:
            block7_tags.append(f"{mood} atmosphere")
        block7_tags.append("cinematic depth of field, volumetric lighting")

        # VFX Particles
        if vfx_tag:
            vfx_str = VFX_PROMPT_MAP.get(vfx_tag, vfx_tag.replace("_", " "))
            block7_tags.append(vfx_str)

        block7_deltas_lighting = _clean_block(block7_tags)

        # -------------------------------------------------------------
        # Block 8: Negative Constraints Block
        # -------------------------------------------------------------
        block8_tags: list[str] = list(style_profile.master_negative_tags)
        for sub in subjects:
            if isinstance(sub, SubjectVisualState):
                if sub.identity_lock:
                    block8_tags.extend(sub.identity_lock.negative_identity_tags)
                if sub.outfit:
                    block8_tags.extend(sub.outfit.negative_clothing_tags)
        if custom_negatives:
            block8_tags.extend(custom_negatives)
        block8_negative = _clean_block(block8_tags)

        return PromptBlockCollection(
            block1_style=block1_style,
            block2_identity=block2_identity,
            block3_outfit=block3_outfit,
            block4_equipment=block4_equipment,
            block5_environment=block5_environment,
            block6_composition=block6_composition,
            block7_deltas_lighting=block7_deltas_lighting,
            block8_negative=block8_negative,
        )

    def compile_shot(
        self,
        shot: ShotSpec | Shot,
        db: Any,
        chapter_number: int,
        arc_id: Optional[str] = None,
        default_checkpoint: Optional[str] = None,
    ) -> CompiledPrompt:
        """Compile a single Shot into a ComfyUI-ready CompiledPrompt using 8-block architecture."""
        ckpt = default_checkpoint or ComfyUIWorkflowCompiler.DEFAULT_CKPT
        shot_id = getattr(shot, "id", None) or getattr(shot, "shot_id", "shot_001")
        workflow_id = getattr(shot, "workflow_id", "WF-004")
        shot_type = getattr(shot, "shot_type", "medium")
        camera_angle = getattr(shot, "camera_angle", "eye_level")
        lighting = getattr(shot, "lighting_override", None)
        vfx_tag = getattr(shot, "vfx_overlay_tag", None)
        mood = getattr(shot, "mood", "tense")

        # Extract subjects
        raw_subjects = getattr(shot, "subjects", [])
        subjects_data: list[dict[str, Any]] = []
        ip_adapter_images: list[str] = []
        canonical_seed: Optional[int] = None

        for sub in raw_subjects:
            eid = getattr(sub, "entity_id", None) or (sub.get("entity_id") if isinstance(sub, dict) else None)
            ename = getattr(sub, "entity_name", None) or (sub.get("entity_name") if isinstance(sub, dict) else eid)
            pose = getattr(sub, "pose", None) or (sub.get("pose") if isinstance(sub, dict) else None)
            expression = getattr(sub, "expression", None) or (sub.get("expression") if isinstance(sub, dict) else None)
            box = getattr(sub, "region_box", None) or (sub.get("region_box") if isinstance(sub, dict) else [0.2, 0.2, 0.8, 0.8])

            resolved: dict[str, Any] = {}
            if eid and db:
                try:
                    resolved = resolve_asset_for_scene(
                        db=db,
                        entity_id=eid,
                        arc_id=arc_id,
                        chapter_number=chapter_number,
                    )
                except Exception as exc:
                    logger.warning("Could not resolve asset for entity %s: %s", eid, exc)
                    resolved = {"canonical": {}, "arc": {}, "chapter_delta": {}, "scene_action": {}}

            # Check reference image
            ref_path = resolved.get("reference_image_path") or resolved.get("ip_adapter_image_path")
            if ref_path and ref_path not in ip_adapter_images:
                ip_adapter_images.append(ref_path)

            if resolved.get("canonical_seed") and canonical_seed is None:
                canonical_seed = resolved["canonical_seed"]

            subjects_data.append({
                "entity_id": eid,
                "entity_name": ename or "Unknown Entity",
                "pose": pose,
                "expression": expression,
                "region_box": box,
                "resolved": resolved,
            })

        # Dynamic visual action cue from narration text
        narration = getattr(shot, "narration_text", "") or ""
        action_cue = ""
        n_lower = narration.lower()
        if "coffee" in n_lower or "cup" in n_lower:
            action_cue = "holding real coffee paper cup, taking cautious sip, rusty metal bench in slums"
        elif "bench" in n_lower or "rusty" in n_lower:
            action_cue = "sitting on weathered rusty metal bench in dystopian slums"
        elif "police" in n_lower or "station" in n_lower or "officer" in n_lower or "desk" in n_lower:
            action_cue = "inside fortified police station reception with armor plates and ceiling turrets"
        elif "cell" in n_lower or "detention" in n_lower or "bars" in n_lower or "bunk" in n_lower:
            action_cue = "inside dimly lit reinforced concrete holding cell, sitting on cold bunk"
        elif "drowsiness" in n_lower or "sleep" in n_lower or "tired" in n_lower:
            action_cue = "drowsy heavy eyelids, head slumping forward, slipping into trance"
        elif "black" in n_lower or "nightmare" in n_lower or "spell" in n_lower or "trial" in n_lower or "aspirant" in n_lower:
            action_cue = "engulfed in pure dark void, mystical glowing runes and writhing shadowy wisps"

        # Determine environment tags
        env_tags: list[str] = []
        loc_id = getattr(shot, "location_id", None)
        if loc_id and db:
            try:
                loc_res = resolve_asset_for_scene(db=db, entity_id=loc_id, arc_id=arc_id, chapter_number=chapter_number)
                loc_canon = loc_res.get("canonical", {})
                for k, v in loc_canon.items():
                    if v:
                        env_tags.append(str(v))
            except Exception:
                pass
        if not env_tags and not subjects_data:
            env_tags = ["ancient snow-swept mountain landscape"]

        # Deterministic shot seed calculation
        shot_num = getattr(shot, "shot_number", 1) or 1
        if canonical_seed is not None:
            if int(shot_num) == 1:
                run_seed = int(canonical_seed)
            else:
                run_seed = (int(canonical_seed) + (int(shot_num) - 1) * 31337) % 2147483647
        else:
            primary_id = subjects_data[0]["entity_id"] if subjects_data else "scene"
            hash_src = f"{primary_id}:{arc_id}:{shot_id}:{shot_num}"
            digest = hashlib.sha256(hash_src.encode("utf-8")).hexdigest()
            run_seed = int(digest[:8], 16) % 2147483647

        # Assemble the 8 prompt blocks
        blocks = self.build_8_blocks(
            style_profile=self.default_style_profile,
            subjects=subjects_data,
            environment_tags=env_tags,
            camera_framing=shot_type,
            camera_angle=camera_angle,
            lighting=lighting,
            vfx_tag=vfx_tag,
            mood=mood,
            action_cue=action_cue,
        )

        clean_positive = blocks.full_positive()
        clean_negative = blocks.full_negative()

        # Regional prompts for multi-character shots
        regional_prompts: list[dict[str, Any]] = []
        if len(subjects_data) > 1:
            for s in subjects_data:
                sub_res = s.get("resolved", {})
                canon = sub_res.get("canonical", {})
                arc = sub_res.get("arc", {})
                delta = sub_res.get("chapter_delta", {})
                sub_tokens = [
                    s["entity_name"],
                    canon.get("physique", ""),
                    canon.get("hair", ""),
                    canon.get("eyes", ""),
                    arc.get("clothing", ""),
                    delta.get("injuries", ""),
                    s.get("pose", ""),
                    s.get("expression", ""),
                ]
                regional_prompts.append({
                    "region_box": s.get("region_box", [0.0, 0.0, 1.0, 1.0]),
                    "prompt": ", ".join(t for t in sub_tokens if t),
                    "ip_adapter_path": sub_res.get("reference_image_path"),
                })

        # Resolution & workflow sizing
        width, height = 1024, 768
        if shot_type in ["wide", "extreme_wide"]:
            width, height = 1216, 832
        elif shot_type in ["close_up", "medium_close_up"]:
            width, height = 1024, 1024

        # Compile ComfyUI workflow payload using the 6+2 Specialized Architecture
        comfyui_params = self.dispatch_comfyui_workflow(
            workflow_id=workflow_id,
            shot_id=str(shot_id),
            positive_prompt=clean_positive,
            negative_prompt=clean_negative,
            seed=run_seed,
            width=width,
            height=height,
            checkpoint=ckpt,
            ip_adapter_images=ip_adapter_images,
            subjects_data=subjects_data,
        )

        compiled = CompiledPrompt(
            shot_id=str(shot_id),
            workflow_id=str(workflow_id),
            positive_prompt=clean_positive,
            negative_prompt=clean_negative,
            width=width,
            height=height,
            seed=run_seed,
            steps=24,
            cfg_scale=6.0,
            sampler="euler_ancestral",
            scheduler="normal",
            checkpoint=ckpt,
            regional_prompts=regional_prompts if regional_prompts else None,
            ip_adapter_images=ip_adapter_images,
            comfyui_workflow_params=comfyui_params,
        )

        logger.debug(
            "Compiled 8-block prompt for shot %s: workflow=%s, subjects=%d, len=%d chars",
            shot_id,
            workflow_id,
            len(subjects_data),
            len(clean_positive),
        )
        return compiled

    def compile_visual_context(
        self,
        context: ShotVisualContext,
        style_profile: Optional[VisualStyleProfile] = None,
    ) -> tuple[CompiledPrompt, PromptBlockCollection]:
        """Compile a resolved ShotVisualContext directly into CompiledPrompt and PromptBlockCollection."""
        profile = style_profile or self.default_style_profile
        env_tags = context.environment.plate_tags if context.environment else []
        
        blocks = self.build_8_blocks(
            style_profile=profile,
            subjects=context.subjects,
            environment_tags=env_tags,
            camera_framing=context.framing,
            camera_angle=context.camera_angle,
            lighting=", ".join(context.lighting_atmosphere) if context.lighting_atmosphere else None,
            vfx_tag=context.vfx_particles[0] if context.vfx_particles else None,
            custom_negatives=context.negative_constraints,
        )

        positive = blocks.full_positive()
        negative = blocks.full_negative()
        seed = context.compute_deterministic_seed()

        width, height = 1024, 768
        if context.framing in ["wide", "extreme_wide"]:
            width, height = 1216, 832
        elif context.framing in ["close_up", "medium_close_up"]:
            width, height = 1024, 1024

        ip_images = []
        for s in context.subjects:
            if s.identity_lock and s.identity_lock.reference_image_path:
                ip_images.append(s.identity_lock.reference_image_path)
            if s.outfit and s.outfit.reference_image_path:
                ip_images.append(s.outfit.reference_image_path)

        comfyui_params = self.dispatch_comfyui_workflow(
            workflow_id=context.workflow_id,
            shot_id=context.shot_id,
            positive_prompt=positive,
            negative_prompt=negative,
            seed=seed,
            width=width,
            height=height,
            checkpoint=profile.checkpoint,
            ip_adapter_images=ip_images,
            subjects_data=context.subjects,
        )

        compiled = CompiledPrompt(
            shot_id=context.shot_id,
            workflow_id=context.workflow_id,
            positive_prompt=positive,
            negative_prompt=negative,
            width=width,
            height=height,
            seed=seed,
            steps=profile.steps,
            cfg_scale=profile.cfg_scale,
            sampler=profile.sampler,
            scheduler=profile.scheduler,
            checkpoint=profile.checkpoint,
            ip_adapter_images=ip_images,
            comfyui_workflow_params=comfyui_params,
        )

        return compiled, blocks

    @classmethod
    def dispatch_comfyui_workflow(
        cls,
        workflow_id: str,
        shot_id: str,
        positive_prompt: str,
        negative_prompt: str,
        seed: int,
        width: int,
        height: int,
        checkpoint: Optional[str] = None,
        ip_adapter_images: Optional[list[str]] = None,
        loc_depth_filename: Optional[str] = None,
        subjects_data: Optional[list[Any]] = None,
    ) -> dict[str, Any]:
        """Dispatch workflow ID to one of the 6+2 specialized ComfyUI workflows."""
        wf_upper = str(workflow_id).upper()
        images = ip_adapter_images or []
        subjects = subjects_data or []
        ckpt = checkpoint or ComfyUIWorkflowCompiler.DEFAULT_CKPT

        def _get_fname(idx: int, default: Optional[str] = None) -> Optional[str]:
            if idx < len(images) and images[idx]:
                return images[idx].replace("\\", "/").split("/")[-1]
            return default

        # 1. WF-FOUNDRY-CHAR (Character Turnaround & Expressions)
        if "FOUNDRY-CHAR" in wf_upper or "WF-01" in wf_upper or "WF-001" in wf_upper:
            return ComfyUIWorkflowCompiler.compile_foundry_char(
                char_id=str(shot_id),
                identity_tags=positive_prompt,
                mode="turnaround",
                seed=seed,
                width=width,
                height=height,
                checkpoint=ckpt,
            )

        # 2. WF-FOUNDRY-LOC (Location Master Plate & Guides)
        if "FOUNDRY-LOC" in wf_upper or "LOCATION" in wf_upper or "WF-02" in wf_upper or "WF-002" in wf_upper:
            return ComfyUIWorkflowCompiler.compile_foundry_loc(
                loc_id=str(shot_id),
                landscape_tags=positive_prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                width=width,
                height=height,
                checkpoint=ckpt,
            )

        # 3. WF-FOUNDRY-PROP (Isolated Item / Weapon)
        if "FOUNDRY-PROP" in wf_upper or "PROP" in wf_upper or "ITEM" in wf_upper:
            return ComfyUIWorkflowCompiler.compile_foundry_prop(
                prop_id=str(shot_id),
                item_tags=positive_prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                width=width,
                height=height,
                checkpoint=ckpt,
            )

        # 4. WF-EDIT-INPAINT (Defect Remediation)
        if "INPAINT" in wf_upper or "EDIT" in wf_upper or "WF-07" in wf_upper or "WF-EDIT" in wf_upper:
            img_file = _get_fname(0, "failed_image.png") or "failed_image.png"
            return ComfyUIWorkflowCompiler.compile_edit_inpaint(
                shot_id=str(shot_id),
                failed_image_filename=img_file,
                repair_prompt=positive_prompt,
                defect_bbox=(680, 480, 320, 280),
                negative_prompt=negative_prompt,
                seed=seed,
                image_width=width,
                image_height=height,
            )

        # 5. WF-MASTER-UPSCALE (4K Master Tiled Polish)
        if "UPSCALE" in wf_upper or "MASTER" in wf_upper or "WF-08" in wf_upper or "WF-MASTER" in wf_upper:
            img_file = _get_fname(0, "approved_image.png") or "approved_image.png"
            return ComfyUIWorkflowCompiler.compile_master_upscale(
                shot_id=str(shot_id),
                input_image_filename=img_file,
                prompt_tags=positive_prompt,
                negative_prompt=negative_prompt,
                scale_factor=2.0,
                seed=seed,
            )

        # 6. WF-SHOT-MULTI (Multi-Subject Regional Isolation)
        if ("MULTI" in wf_upper or "GRP" in wf_upper or len(subjects) >= 2) and "LOC" not in wf_upper and "ENV" not in wf_upper:
            s1_name = "Subject 1"
            s2_name = "Subject 2"
            if len(subjects) >= 1:
                s1_name = getattr(subjects[0], "entity_name", None) or (subjects[0].get("entity_name") if isinstance(subjects[0], dict) else "Subject 1")
            if len(subjects) >= 2:
                s2_name = getattr(subjects[1], "entity_name", None) or (subjects[1].get("entity_name") if isinstance(subjects[1], dict) else "Subject 2")
            f1 = _get_fname(0, f"{s1_name}_face.png") or f"{s1_name}_face.png"
            f2 = _get_fname(1, f"{s2_name}_face.png") or f"{s2_name}_face.png"
            return ComfyUIWorkflowCompiler.compile_shot_multi(
                shot_id=str(shot_id),
                bg_prompt=positive_prompt,
                subj1_prompt=f"{s1_name}, detailed face, signature outfit",
                subj2_prompt=f"{s2_name}, detailed face, signature outfit",
                subj1_face_filename=f1,
                subj2_face_filename=f2,
                negative_prompt=negative_prompt,
                loc_depth_filename=loc_depth_filename,
                seed=seed,
                width=width,
                height=height,
            )

        # 7. WF-SHOT-ENV (Pure Scenery / Establishing Vista)
        if "ENV" in wf_upper or "VISTA" in wf_upper or (len(subjects) == 0 and "LOC" not in wf_upper and "PROP" not in wf_upper and "CHAR" not in wf_upper):
            return ComfyUIWorkflowCompiler.compile_shot_env(
                shot_id=str(shot_id),
                env_prompt=positive_prompt,
                negative_prompt=negative_prompt,
                loc_depth_filename=loc_depth_filename,
                width=width,
                height=height,
                seed=seed,
                checkpoint=ckpt,
            )

        # 8. WF-SHOT-SOLO (Single-Character Workhorse — Default)
        face_file = _get_fname(0, None)
        return ComfyUIWorkflowCompiler.compile_shot_solo(
            shot_id=str(shot_id),
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            face_ref_filename=face_file,
            loc_depth_filename=loc_depth_filename,
            seed=seed,
            width=width,
            height=height,
            checkpoint=ckpt,
        )
