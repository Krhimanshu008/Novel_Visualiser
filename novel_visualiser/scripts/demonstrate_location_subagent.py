"""Novel Visualiser — Demonstration of LocationContinuitySubagent.

Processes all 3 locations of Chapter 1 ("Without a Name"):
1. LOC-001: Outskirts Slum Transit Bench (Scene 1)
2. LOC-002: Fortified Police Reception Lobby (Scene 2)
3. LOC-003: Detention Holding Cell 07 (Scene 3)

Demonstrates:
- 5-Dimension Deep Observational Reasoning
- 4-Plate Reference Matrix (Vista, Stage, Reverse, Macro)
- Structural Depth Map & Canny Lineart Extraction
- Camera Angle Anchor Resolution for scene shots
- Hierarchical Arc Tagging (Main Arc > Sub Arc > Mini Arc)
- Cross-Chapter Memory Recall (Chapter 1 -> Chapter 12 re-occurrence)
"""

import os
import shutil
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.models import Base, Project
from app.schemas.location_subagent import ArcHierarchyTag
from app.agents.location_subagent import LocationContinuitySubagent


def main():
    print("=" * 75)
    print("  LOCATION CONTINUITY SUBAGENT — PRODUCTION DEMONSTRATION")
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

    agent = LocationContinuitySubagent(db=session, use_ai_thinking=True)
    out_dir = Path("output/generated_locations")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. Define Chapter 1 Arc Hierarchy Context
    ch01_arc = ArcHierarchyTag(
        main_arc_id="ARC_01_FIRST_NIGHTMARE",
        sub_arc_id="SUBARC_01_01_OUTSKIRTS",
        mini_arc_id="MINIARC_01_01_01_AWAKENING",
        parallel_thread_id=None,
        in_universe_day=1,
        time_of_day="midnight",
        weather="cold rain and mist",
    )

    # 3. Process the 3 Core Locations of Chapter 1
    chapter_locations = [
        (
            "Outskirts Transit Bench N24",
            """
            Sunny sat on the battered, rusted metal bench across from the police station, nursing a cup of cheap lukewarm synth-coffee.
            The dreary outskirts of the N24 district were drenched in cold rain. Leaning brutalist tenements loomed over the rain-slicked boulevard.
            Flickering holographic neon signs sputtered above cracked curbs with oil-streaked puddles.
            """,
        ),
        (
            "Fortified Police Station Reception",
            """
            Inside the fortress-like police precinct, thick reinforced blast doors closed behind him.
            Harsh cold cyan fluorescent tubes cast stark shadows across the industrial concrete lobby.
            Automated ceiling security turrets hummed behind wire mesh, aimed at the heavy steel turnstiles.
            """,
        ),
        (
            "Detention Holding Cell 07",
            """
            A narrow concrete cube with thick steel bars. A solitary cold iron bunk bolted to the wall.
            The floor was cold pitted stone. The only light filtered dimly through the iron bars from the corridor.
            Sunny sat huddled against the cold corner, exhaustion and supernatural drowsiness dragging him into the dark void.
            """,
        ),
    ]

    location_manifests = []
    print("\n[PHASE 1] Observing and Generating 4-Plate Matrices...")
    for loc_name, passage in chapter_locations:
        print(f"\n--- Location: {loc_name} ---")
        # Step A: Deep Observation & 4-Plate Synthesis
        spec = agent.observe_and_synthesize_location(
            location_name=loc_name,
            passage_text=passage,
            arc_hierarchy=ch01_arc,
        )
        print(f" • Socioeconomic Reality: {spec.observation.socioeconomic_reality}")
        print(f" • Volumetric Lighting  : {spec.observation.volumetric_lighting}")
        print(f" • Environmental Physics: {spec.observation.environmental_physics}")

        # Step B: Render 4 Plates and Extract Structural Guides
        spec = agent.generate_location_matrix(spec, output_dir=out_dir)
        print(f" [OK] Plate 1 (Vista)   : {Path(spec.plate_1_vista.image_path).name} + Depth + Lineart")
        print(f" [OK] Plate 2 (Stage)   : {Path(spec.plate_2_stage.image_path).name} + Depth + Lineart")
        print(f" [OK] Plate 3 (Reverse) : {Path(spec.plate_3_reverse.image_path).name} + Depth + Lineart")
        print(f" [OK] Plate 4 (Macro)   : {Path(spec.plate_4_macro.image_path).name} + Depth + Lineart")

        # Step C: Save to SQLite Canon
        agent.save_to_canon(project_id, spec, current_chapter=1)
        location_manifests.append(spec)

    # 4. Demonstrate Dynamic Camera Angle Resolution for Chapter 1 Shots
    print("\n" + "=" * 75)
    print("[PHASE 2] Resolving Camera Angle Anchors for Chapter 1 Shots...")
    print("=" * 75)

    test_shots = [
        (location_manifests[0], "WIDE_ESTABLISHING", "Shot 1: Wide establishing vista of the rainy slum street"),
        (location_manifests[0], "STAGE_ACTION", "Shot 2: Medium shot of Sunny sitting on the bench drinking coffee"),
        (location_manifests[0], "REVERSE_POV", "Shot 3: Sunny looks across the rainy street at the police fortress"),
        (location_manifests[0], "MACRO_DETAIL", "Shot 4: Close-up of the crushed paper coffee cup on the rusted metal"),
    ]

    for loc_spec, intent, desc in test_shots:
        anchor = agent.resolve_shot_angle_anchor(loc_spec, intent)
        print(f"\n• {desc}")
        print(f"  -> Resolved Plate Role: {anchor.resolved_role.value}")
        print(f"  -> Selected Canvas    : {Path(anchor.selected_plate_image_path).name}")
        print(f"  -> ControlNet Depth   : {Path(anchor.depth_map_path).name if anchor.depth_map_path else 'None'}")
        print(f"  -> Inpaint BoundingBox: {anchor.recommended_inpaint_box}")
        print(f"  -> Lighting Shading   : {anchor.lighting_baseline}")

    # 5. Demonstrate Cross-Chapter Historical Memory Recall
    print("\n" + "=" * 75)
    print("[PHASE 3] Simulating Chapter 12 Re-Occurrence (Historical Memory Recall)...")
    print("=" * 75)

    recalled_spec = agent.query_historical_memory(project_id, "Outskirts Transit Bench N24")
    assert recalled_spec is not None, "Failed to recall location from historical memory!"
    print(" [MEMORY HIT] Location found in SQLite Canon Database!")
    print(f" • Canonical ID    : {recalled_spec.location_id}")
    print(f" • Main Arc Tag    : {recalled_spec.arc_hierarchy.main_arc_id}")
    print(f" • Sub Arc Tag     : {recalled_spec.arc_hierarchy.sub_arc_id}")
    print(f" • Pre-Established Stage Plate: {Path(recalled_spec.plate_2_stage.image_path).name}")
    print(f" • Pre-Computed Depth Guide   : {Path(recalled_spec.plate_2_stage.depth_map_path).name}")
    print(" -> ZERO location drift: Downstream shots reuse established 3D geometry!")

    # 6. Copy samples to Artifact Directory
    artifact_img_dir = Path("C:/Users/Himan/.gemini/antigravity/brain/8fe42fa3-9cb5-4680-b7b3-f411053b4161/location_matrix_samples")
    artifact_img_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.glob("*.png"):
        shutil.copy(p, artifact_img_dir / p.name)
    print(f"\n[OK] Copied all generated location plates & depth/lineart guides to:\n     {artifact_img_dir}")

    session.close()
    print("\n===========================================================================")
    print("  LOCATION CONTINUITY SUBAGENT DEMONSTRATION COMPLETE (100% SUCCESS)")
    print("===========================================================================")


if __name__ == "__main__":
    main()

