"""
Novel Visualiser — Jinja2 Prompt Compiler Service (Phase 4)

Responsible for:
1. 100% Deterministic prompt compilation (0 LLM tokens).
2. Merges ShotSpec + Hierarchical Asset Resolver outputs (Project -> Canonical -> Arc -> Chapter Delta -> Scene Action).
3. Produces positive and negative prompt strings, regional conditioning blocks, and IP-Adapter reference links.
4. Generates direct ComfyUI API workflow payloads (comfyui_workflow_params) matching the user-tested workflows.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
from jinja2 import Environment, BaseLoader

from app.database.models import Shot, ShotSubject
from app.orchestrator.resolver import resolve_asset_for_scene
from app.schemas.scene import CompiledPrompt, ShotSpec, SubjectSpec
from app.services.comfyui_client import ComfyUIWorkflowCompiler

logger = logging.getLogger("novel_visualiser.prompt_compiler")

# Jinja2 templates for positive prompt generation
POSITIVE_PROMPT_TEMPLATE = """
{{ master_quality_tags }},
{{ shot.camera_angle | replace('_', ' ') }} shot, {{ shot.shot_type | replace('_', ' ') }} framing,
{% if location_prompt %}{{ location_prompt }},{% endif %}
{% if shot.action_cue %}{{ shot.action_cue }},{% endif %}
{% for sub in subjects %}
  {{ sub.entity_name }},
  {% if sub.resolved.canonical.get('species') %}{{ sub.resolved.canonical.species }},{% endif %}
  {% if sub.resolved.canonical.get('physique') %}{{ sub.resolved.canonical.physique }},{% endif %}
  {% if sub.resolved.canonical.get('hair') %}{{ sub.resolved.canonical.hair }},{% endif %}
  {% if sub.resolved.canonical.get('eyes') %}{{ sub.resolved.canonical.eyes }},{% endif %}
  {% if sub.resolved.canonical.get('skin_color') %}{{ sub.resolved.canonical.skin_color }},{% endif %}
  {% if sub.resolved.arc.get('clothing') %}wearing {{ sub.resolved.arc.clothing }},{% endif %}
  {% if sub.resolved.arc.get('armor') %}armored with {{ sub.resolved.arc.armor }},{% endif %}
  {% if sub.resolved.arc.get('weapons') %}carrying {{ sub.resolved.arc.weapons }},{% endif %}
  {% if sub.resolved.chapter_delta.get('injuries') %}injuries: {{ sub.resolved.chapter_delta.injuries }},{% endif %}
  {% if sub.resolved.chapter_delta.get('conditions') %}conditions: {{ sub.resolved.chapter_delta.conditions }},{% endif %}
  {% if sub.pose %}{{ sub.pose }},{% endif %}
  {% if sub.expression %}{{ sub.expression }},{% endif %}
{% endfor %}
{% if shot.lighting_override %}{{ shot.lighting_override }},{% endif %}
{% if shot.mood %}{{ shot.mood }} atmosphere, cinematic depth of field,{% endif %}
{% if vfx_prompt %}{{ vfx_prompt }},{% endif %}
volumetric lighting, highly detailed textures
""".strip()

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


class PromptCompiler:
    """
    Deterministic Jinja2 Prompt Compiler.
    Compiles shot specifications and resolved visual entity records into
    ComfyUI-ready CompiledPrompt objects.
    """

    def __init__(self):
        self.jinja_env = Environment(loader=BaseLoader(), autoescape=False)
        self.template = self.jinja_env.from_string(POSITIVE_PROMPT_TEMPLATE)

    def compile_shot(
        self,
        shot: ShotSpec | Shot,
        db: Any,
        chapter_number: int,
        arc_id: Optional[str] = None,
        default_checkpoint: Optional[str] = None,
    ) -> CompiledPrompt:
        """
        Compile a single Shot into a ComfyUI-ready CompiledPrompt.

        Args:
            shot: ShotSpec Pydantic object or Shot SQLAlchemy model.
            db: Active SQLAlchemy session for resolver lookups.
            chapter_number: Chapter number acting as spoiler gate boundary.
            arc_id: Current arc ID (optional).
            default_checkpoint: ComfyUI checkpoint name (defaults to Illustrious SDXL).

        Returns:
            CompiledPrompt ready for execution.
        """
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

        # VFX Prompt
        vfx_prompt = VFX_PROMPT_MAP.get(vfx_tag, "") if vfx_tag else ""

        # Deterministic shot-specific seed calculation (Shot 1 uses canonical seed, subsequent shots vary)
        shot_num = getattr(shot, "shot_number", 1) or 1
        if canonical_seed is not None:
            run_seed = (int(canonical_seed) + (int(shot_num) - 1) * 31337) % 2147483647
        else:
            run_seed = (123456 + (int(shot_num) - 1) * 31337) % 2147483647

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

        # Master quality tokens
        master_quality = "masterpiece, best quality, highly detailed, dark fantasy anime digital painting, Unreal Engine 5 render style"

        # Render positive prompt via Jinja2
        shot_obj = type("ShotObj", (), {
            "shot_type": shot_type,
            "camera_angle": camera_angle,
            "lighting_override": lighting,
            "mood": mood,
            "action_cue": action_cue,
        })()

        rendered_positive = self.template.render(
            master_quality_tags=master_quality,
            shot=shot_obj,
            subjects=subjects_data,
            location_prompt="ancient snow-swept mountain landscape" if not subjects_data else None,
            vfx_prompt=vfx_prompt,
        )

        # Clean up whitespace / multiple commas
        clean_positive = ", ".join(
            part.strip() for part in rendered_positive.replace("\n", " ").split(",") if part.strip()
        )

        # Regional prompts for dual subjects or multi-character scenes
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

        # Compile ComfyUI workflow payload based on workflow_id (WF 00 to WF 07)
        comfyui_params: dict[str, Any]
        wf_upper = str(workflow_id).upper()

        if ("WF-01" in wf_upper or "WF-001" in wf_upper or "IDENTITY" in wf_upper or "CONSISTENCY" in wf_upper) and ip_adapter_images:
            ref_filename = ip_adapter_images[0].replace("\\", "/").split("/")[-1]
            comfyui_params = ComfyUIWorkflowCompiler.compile_character_identity(
                reference_image_filename=ref_filename,
                positive_prompt=clean_positive,
                negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                seed=run_seed,
                width=width,
                height=height,
                checkpoint=ckpt,
            )
        elif "WF-02" in wf_upper or "WF-002" in wf_upper or "LOCATION" in wf_upper:
            comfyui_params = ComfyUIWorkflowCompiler.compile_location_plate(
                positive_prompt=clean_positive,
                negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                seed=run_seed,
                width=width,
                height=height,
                checkpoint=ckpt,
            )
        elif "WF-03" in wf_upper or "WF-003" in wf_upper or "SCENE" in wf_upper or "COMPOSITION" in wf_upper:
            ref_filename = ip_adapter_images[0].replace("\\", "/").split("/")[-1] if ip_adapter_images else "Sunny_portrait.png"
            comfyui_params = ComfyUIWorkflowCompiler.compile_scene_composition(
                reference_image_filename=ref_filename,
                positive_prompt=clean_positive,
                negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                seed=run_seed,
                width=width,
                height=height,
            )
        elif "WF-04" in wf_upper or "WF-004" in wf_upper or "ITEM" in wf_upper:
            comfyui_params = ComfyUIWorkflowCompiler.compile_item_plate(
                positive_prompt=clean_positive,
                negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                seed=run_seed,
                width=width,
                height=height,
            )
        elif "WF-05" in wf_upper or "WF-005" in wf_upper or "BATTLE" in wf_upper:
            ref_filename = ip_adapter_images[0].replace("\\", "/").split("/")[-1] if ip_adapter_images else "Sunny_portrait.png"
            comfyui_params = ComfyUIWorkflowCompiler.compile_cinematic_battle(
                reference_image_filename=ref_filename,
                positive_prompt=clean_positive,
                negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                seed=run_seed,
                width=width,
                height=height,
            )
        else:
            comfyui_params = ComfyUIWorkflowCompiler.compile_diagnostic_base(
                positive_prompt=clean_positive,
                negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                seed=run_seed,
                width=width,
                height=height,
                checkpoint=ckpt,
            )

        compiled = CompiledPrompt(
            shot_id=str(shot_id),
            workflow_id=str(workflow_id),
            positive_prompt=clean_positive,
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
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
            "Compiled prompt for shot %s: workflow=%s, subjects=%d, len=%d chars",
            shot_id,
            workflow_id,
            len(subjects_data),
            len(clean_positive),
        )
        return compiled

