"""Novel Visualiser — Live Compositing Demonstration on Dual-GPU Cluster.

Demonstrates:
1. Multi-Subagent Unified Harmonization:
   - Location: Outskirts Transit Bench Action Stage Plate + Depth Map
   - Character: Sunny (Epoch 1 Slum Orphan) + Face Chip + Inpaint Bounding Box
   - Environment: 2100K Sodium Amber + Cyan Twilight Bounce + Wet Pavement
2. Compiling Non-Contaminated Inpainting Prompts (Zero prompt bleed).
3. Live Remote Execution on the Dual-GPU ComfyUI Cluster:
   - GPU 0: https://gpu0.krhimanshu.in
   - GPU 1: https://gpu1.krhimanshu.in
4. Differential Inpainting (Freezing location geometry, inpainting Sunny onto the bench).
5. Output saving and telemetry inspection.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.models import Base, Project
from app.schemas.character_subagent import DanbooruIdentityLock, ShotCharacterAnchor
from app.schemas.compositing_subagent import (
    BoundingBox2D,
    CompositingModeEnum,
    RenderedShotArtifact,
    ShotCompositionSpec,
)
from app.schemas.environment_subagent import (
    ShotEnvironmentAnchor,
    TimeOfDayEnum,
    WeatherConditionEnum,
)
from app.schemas.location_subagent import ArcHierarchyTag, PlateRole, ShotLocationAnchor
from app.agents.compositing_subagent import CompositingSubagent
from app.services.comfyui_client import get_dual_gpu_dispatcher


def main():
    print("=" * 75)
    print("  COMPOSITING SUBAGENT — LIVE DUAL-GPU PRODUCTION RUN")
    print("=" * 75)

    # 1. Setup SQLite Database for Canon Persistence
    db_path = Path("output/novel_canon.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    project_id = "proj_shadow_slave_master"
    project = session.query(Project).filter(Project.id == project_id).first()
    if not project:
        project = Project(
            id=project_id,
            name="Shadow Slave Production",
            novel_title="Shadow Slave",
            author="Guiltythree",
        )
        session.add(project)
        session.commit()

    # 2. Check Dual-GPU Cluster Connectivity
    dispatcher = get_dual_gpu_dispatcher()
    print("\n[CLUSTER STATUS CHECK]:")
    cluster_healthy = False
    for i, client in enumerate(dispatcher.clients):
        h = client.check_health()
        print(f" • Worker {i} ({client.base_url}): Status={h.get('status')} | VRAM Free={h.get('vram_free_gb')} GB")
        if h.get("status") == "online":
            cluster_healthy = True

    agent = CompositingSubagent(db=session, dispatcher=dispatcher, allow_offline_fallback=True)
    out_dir = Path("output/composited_shots")
    out_dir.mkdir(parents=True, exist_ok=True)

    arc_tag = ArcHierarchyTag(
        main_arc_id="ARC_01_FIRST_NIGHTMARE",
        sub_arc_id="SUBARC_01_01_OUTSKIRTS",
        mini_arc_id="MINIARC_01_01_01_AWAKENING",
        time_of_day="twilight",
        weather="industrial smog",
    )

    # Resolve Inputs from Previous Subagents
    # A. Location Anchor
    loc_plate_path = "output/generated_locations/loc_outskirts_transit_bench_n24_action_stage.png"
    loc_depth_path = "output/generated_locations/loc_outskirts_transit_bench_n24_action_stage_depth.png"
    loc_anchor = ShotLocationAnchor(
        location_id="loc_outskirts_transit_bench_n24",
        canonical_name="Outskirts Transit Bench N24",
        resolved_role=PlateRole.ACTION_STAGE,
        selected_plate_image_path=loc_plate_path,
        depth_map_path=loc_depth_path,
        lighting_baseline="flickering overhead sodium vapor amber lamp, damp asphalt, wet reflections",
        weather_condition="industrial smog, cold evening drizzle",
        recommended_inpaint_box=[0.25, 0.25, 0.75, 0.90],
    )

    # B. Character Anchor (Sunny - Slum Orphan Epoch 1)
    face_chip_path = "output/char_sunny_canonical_face_chip.png"
    char_anchor = ShotCharacterAnchor(
        entity_id="char_sunny",
        entity_name="Sunny",
        full_positive_prompt="1boy, short black hair, messy bangs, dark hollow onyx eyes, dark circles under eyes, pale skin, sallow skin, scrawny, slender build, sharp delicate jawline, oversized faded charcoal utility jacket, frayed cuffs, worn trousers, shivering posture, sitting on metal bench, clutching paper synth-coffee cup",
        full_negative_prompt="blonde hair, blue eyes, muscular, beard, round face, chubby, clean clothes, armor, business suit, healthy glowing skin",
        face_reference_path=face_chip_path,
        expression_tag="exhausted drooping eyelids, cynical smirk, alert vigilant gaze",
        pose_action_tag="shivering inside oversized jacket, hunched defensive posture on bench",
        recommended_inpaint_box=[0.25, 0.25, 0.75, 0.90],
    )

    # C. Environment Anchor (Twilight Smog)
    env_anchor = ShotEnvironmentAnchor(
        environment_card_id="env_outskirts_twilight_smog",
        time_of_day=TimeOfDayEnum.TWILIGHT_BLUE_HOUR,
        weather_condition=WeatherConditionEnum.OVERCAST_SMOG,
        resolved_lighting_block="sodium vapor streetlamp, warm amber glow, flickering light, volumetric lighting, cyan ambient light, rim light, smog, overcast, wet asphalt, puddle reflections, dead weeds in sidewalk cracks",
        resolved_negative_block="sunny blue sky, bright daylight, clear sky, lush green grass, clean dry ground, vibrant colors",
        key_light_vector="top_left",
        shadow_direction="bottom_right",
        surface_moisture_level=0.6,
    )

    # Assemble Shot Specification
    shot_spec = ShotCompositionSpec(
        shot_id="shot_ch01_sc01_sunny_transit_bench",
        novel_arc=arc_tag,
        mode=CompositingModeEnum.INPAINT_DIFFERENTIAL if Path(loc_plate_path).exists() else CompositingModeEnum.CONTROLNET_IPADAPTER_FUSION,
        location_plate_path=loc_plate_path if Path(loc_plate_path).exists() else None,
        location_depth_path=loc_depth_path if Path(loc_depth_path).exists() else None,
        location_anchor=loc_anchor,
        character_anchor=char_anchor,
        environment_anchor=env_anchor,
        character_bbox=BoundingBox2D(x=320, y=260, width=540, height=520),
        canvas_width=1216,
        canvas_height=832,
        deterministic_seed=777777,
    )

    print("\n" + "-" * 75)
    print("  ASSEMBLING MULTI-LAYER COMPOSITING PROMPTS")
    print("-" * 75)
    pos_prompt, neg_prompt = agent.assemble_shot_prompts(shot_spec)
    print(f"\n[RESOLVED POSITIVE PROMPT]:\n{pos_prompt}\n")
    print(f"[RESOLVED NEGATIVE PROMPT]:\n{neg_prompt}\n")

    print("-" * 75)
    print("  EXECUTING SHOT COMPOSITION (DETERMINISTIC COMPOSITING ENGINE)")
    print("-" * 75)
    print(f" • Shot ID           : {shot_spec.shot_id}")
    print(f" • Compositing Mode  : {shot_spec.mode.value}")
    print(f" • Location Plate    : {shot_spec.location_plate_path}")
    print(f" • Face Reference    : {shot_spec.character_anchor.face_reference_path}")
    print(f" • Inpaint Box (BBox): (x={shot_spec.character_bbox.x}, y={shot_spec.character_bbox.y}, w={shot_spec.character_bbox.width}, h={shot_spec.character_bbox.height})")

    artifact = agent.execute_composite_shot(spec=shot_spec, output_dir=out_dir, force_mock=False)

    print("\n" + "=" * 75)
    print("  SHOT COMPOSITION COMPLETE!")
    print("=" * 75)
    print(f" • Output Image Path : {artifact.image_path}")
    print(f" • Worker Engine     : {artifact.worker_url_used}")
    print(f" • Execution Latency : {artifact.execution_time_seconds:.2f}s")
    print(f" • Prompt ID         : {artifact.prompt_id}")
    print(f" • QC Verification   : {'PASSED' if artifact.qc_passed else 'FAILED'}")

    # Save output metadata
    results = {
        "shot_id": artifact.shot_id,
        "image_path": artifact.image_path,
        "mode_used": artifact.mode_used.value,
        "worker_url_used": artifact.worker_url_used,
        "execution_time_seconds": artifact.execution_time_seconds,
        "prompt_id": artifact.prompt_id,
        "positive_prompt": artifact.final_positive_prompt,
        "negative_prompt": artifact.final_negative_prompt,
        "inpaint_bbox": shot_spec.character_bbox.model_dump(),
        "qc_passed": artifact.qc_passed,
    }

    out_json = Path("output/composited_shots_live_output.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[METADATA SAVED] -> {out_json}")


if __name__ == "__main__":
    main()
