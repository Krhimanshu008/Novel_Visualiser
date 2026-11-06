"""
Novel Visualiser — Phase 4 Verification Gate Test Suite

Verifies:
1. Deterministic Shot-Budget Calculator (0 tokens, combat vs dialogue vs exposition pacing)
2. Scene Director Storyboarding (framing, camera angles, motion tiers, regional boxes)
3. SQLite Persistence of Shot and ShotSubject records
4. Deterministic Jinja2 Prompt Compiler (merges Asset Resolver, spoiler boundary, ComfyUI payload)
5. 3-Gate Automated QC Funnel (OpenCV blur/exposure, identity consistency, semantic checklist)
6. LangGraph Phase 4 Nodes & FastAPI Endpoints
"""

import os
import sys
os.environ['ALLOW_MOCK_FALLBACK'] = '1'
from pathlib import Path
import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agents.scene_director import SceneDirectorAgent, calculate_shot_budget
from app.database.models import (
    AssetLevel,
    AssetVersion,
    Chapter,
    ChapterStatus,
    Entity,
    EntityType,
    GenerationJob,
    JobStatus,
    MotionTier,
    Passage,
    PassageType,
    Project,
    RefType,
    Scene,
    Shot,
    ShotStatus,
    ShotSubject,
    VisualRef,
)
from app.database.session import get_session, init_db
from app.orchestrator.graph import compile_prompts, generate_images, run_quality_control, run_scene_director
from app.orchestrator.state import PipelineStage, ProductionState
from app.schemas.scene import ShotSpec
from app.services.prompt_compiler import PromptCompiler
from app.services.qc_service import QCService


def setup_module():
    init_db()


def test_01_deterministic_shot_budget():
    print("\n--- Test 1: Deterministic Shot-Budget Calculator (0 Tokens) ---")

    # 1. Combat Scene: fast pacing (~22 words/shot)
    combat_text = (
        "Sunny ducked under the jagged claws of the nightmare creature. "
        "The black beast roared, spraying icy mist into the freezing mountain air. "
        "He rolled across the snowy ground, drawing his crude bronze blade with a swift motion. "
        "Shadows writhed beneath his boots as he prepared his counter-thrust."
    )
    combat_budget = calculate_shot_budget(combat_text, scene_type="combat")
    assert combat_budget["pacing"] == "fast"
    assert combat_budget["shot_count"] >= 2
    assert combat_budget["avg_duration_sec"] < 4.5
    print(f" [PASS] Combat pacing: {combat_budget['shot_count']} shots, avg duration {combat_budget['avg_duration_sec']}s")

    # 2. Dialogue Scene: moderate pacing (~36 words/shot)
    dialogue_text = (
        "Sunny looked at the girl who called herself Changing Star. Her gaze was unflinching, "
        "like pure white marble that knew no fear or compromise. 'We need to survive the first trial,' "
        "she said quietly. Sunny offered a faint, crooked smile, hiding his doubts beneath a layer of cynicism."
    )
    dialogue_budget = calculate_shot_budget(dialogue_text, scene_type="dialogue")
    assert dialogue_budget["pacing"] == "moderate"
    assert dialogue_budget["shot_count"] >= 1
    print(f" [PASS] Dialogue pacing: {dialogue_budget['shot_count']} shots, avg duration {dialogue_budget['avg_duration_sec']}s")

    # 3. Exposition Scene: slow pacing (~48 words/shot)
    expo_text = (
        "The ancient mountains loomed high above the forgotten realm, their obsidian peaks "
        "shrouded in eternal blizzards. Below lay the ruins of a once-mighty civilisation, "
        "crumbled stone pillars now swallowed by creeping frost and ancient shadows that had "
        "waited millennia for an intruder to break their slumber."
    )
    expo_budget = calculate_shot_budget(expo_text, scene_type="exposition")
    assert expo_budget["pacing"] == "slow"
    assert expo_budget["avg_duration_sec"] >= 7.5
    print(f" [PASS] Exposition pacing: {expo_budget['shot_count']} shots, avg duration {expo_budget['avg_duration_sec']}s")


def test_02_scene_director_storyboarding():
    print("\n--- Test 2: Scene Director Storyboard & Regional Framing ---")

    agent = SceneDirectorAgent()
    mock_scene = {
        "scene_type": "combat",
        "location_id": "LOC_SNOWY_PASS",
        "mood": "tense and desolate",
        "conflict_type": "physical combat",
    }
    passages = [
        {"passage_id": "P_001", "text": "Sunny sprinted across the snow, pursued by the mountain beast."},
        {"passage_id": "P_002", "text": "Nephis stepped forward, her blade igniting with pure white flame."},
        {"passage_id": "P_003", "text": "Together they struck the beast from flanking angles."},
    ]
    entities = [
        {"entity_id": "Sunny", "entity_name": "Sunny"},
        {"entity_id": "Nephis", "entity_name": "Nephis"},
    ]

    shots = agent.plan_shots_for_scene(mock_scene, passages, entities_present=entities)
    assert len(shots) >= 1
    shot1 = shots[0]

    # Verify ShotSpec properties
    assert shot1.shot_number == 1
    assert shot1.shot_type in ["wide", "medium", "medium_close_up", "close_up"]
    assert shot1.camera_angle in ["eye_level", "low_angle_dramatic", "high_angle", "dutch_angle"]
    assert len(shot1.subjects) == 2  # Both Sunny & Nephis present

    # Verify Regional Bounding Boxes
    sub_sunny = next(s for s in shot1.subjects if s.entity_name == "Sunny")
    sub_nephis = next(s for s in shot1.subjects if s.entity_name == "Nephis")
    assert len(sub_sunny.region_box) == 4
    assert len(sub_nephis.region_box) == 4
    assert sub_sunny.region_box[0] < sub_nephis.region_box[0]  # Left vs Right composition
    print(f" [PASS] Storyboard planned {len(shots)} shots with dual regional bounding boxes")


def test_03_sqlite_shot_persistence():
    print("\n--- Test 3: SQLite Shot & ShotSubject Persistence ---")

    with get_session() as session:
        # Create test project & chapter
        proj = Project(name="Phase 4 Production Project", novel_title="Shadow Slave", author="Guiltythree")
        session.add(proj)
        session.flush()

        chap = Chapter(
            project_id=proj.id,
            chapter_number=1,
            title="Without a Name",
            status=ChapterStatus.PARSED,
        )
        session.add(chap)
        session.flush()

        p1 = Passage(
            chapter_id=chap.id,
            paragraph_index=1,
            text="Sunny staggered through the snow, clutching his wounded arm.",
            passage_type=PassageType.NARRATION,
            tagged_entity_ids=["Sunny"],
        )
        session.add(p1)
        session.flush()

        sc = Scene(
            chapter_id=chap.id,
            scene_number=1,
            start_passage_id=p1.id,
            end_passage_id=p1.id,
            mood="desolate",
            description="Sunny staggers through snow",
        )
        session.add(sc)
        session.commit()

        # Run scene director
        agent = SceneDirectorAgent()
        created_shots = agent.direct_chapter(session, chap.id)

        assert len(created_shots) > 0
        persisted_shot = created_shots[0]
        assert persisted_shot.status == ShotStatus.PLANNED
        assert persisted_shot.chapter_id == chap.id
        assert len(persisted_shot.subjects) >= 1
        print(f" [PASS] Persisted {len(created_shots)} shots and {len(persisted_shot.subjects)} subjects to SQLite")


def test_04_jinja2_prompt_compiler():
    print("\n--- Test 4: Jinja2 Prompt Compiler & Asset Resolver Integration ---")

    with get_session() as session:
        # Create test project
        proj = Project(name="Test Prompt Project", novel_title="Shadow Slave", author="Guiltythree")
        session.add(proj)
        session.flush()

        # Create entity and approved visual reference
        ent = Entity(
            project_id=proj.id,
            canonical_name="Sunny",
            entity_type=EntityType.CHARACTER,
        )
        session.add(ent)
        session.flush()

        ref = VisualRef(
            entity_id=ent.id,
            ref_type=RefType.TURNAROUND_SHEET,
            image_path="data/references/Sunny/Sunny_canonical.png",
            seed=445566,
            approved=True,
        )
        session.add(ref)

        # Add AssetVersion with injuries and clothing
        av = AssetVersion(
            entity_id=ent.id,
            level=AssetLevel.PROJECT_CANONICAL,
            valid_from_chapter=1,
            visual_attributes={
                "physique": "slender youth",
                "hair": "messy jet-black hair",
                "eyes": "dark eyes",
                "skin_color": "pale skin",
                "clothing": "tattered dark gray tunic",
                "injuries": "bleeding cut across cheek",
            },
        )
        session.add(av)
        session.commit()

        # Build shot
        shot_spec = ShotSpec(
            shot_number=1,
            source_passages=["P_001"],
            workflow_id="WF-002",
            shot_type="medium",
            camera_angle="low_angle_dramatic",
            lighting_override="cold moonlight with harsh rim shadows",
            vfx_overlay_tag="shadow_tendrils",
            mood="tense and dark",
            subjects=[{
                "entity_id": ent.id,
                "entity_name": "Sunny",
                "region_box": [0.2, 0.2, 0.8, 0.8],
                "pose": "alert defensive stance",
                "expression": "cautious grimace",
                "z_order": 0,
            }],
        )

        compiler = PromptCompiler()
        compiled = compiler.compile_shot(
            shot=shot_spec,
            db=session,
            chapter_number=1,
        )

        assert compiled.seed == 445566
        assert "Sunny" in compiled.positive_prompt
        assert "messy jet-black hair" in compiled.positive_prompt
        assert "tattered dark gray tunic" in compiled.positive_prompt
        assert "bleeding cut across cheek" in compiled.positive_prompt
        assert "shadow tendrils" in compiled.positive_prompt
        assert "low angle dramatic" in compiled.positive_prompt
        assert compiled.comfyui_workflow_params is not None
        assert "data/references/Sunny/Sunny_canonical.png" in compiled.ip_adapter_images[0]
        print(" [PASS] Compiled prompt contains resolved canonical attributes, delta injuries, and ComfyUI payload")


def test_05_three_gate_qc_funnel():
    print("\n--- Test 5: 3-Gate Automated QC Funnel ---")

    test_dir = Path("output/test_qc")
    test_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create a sharp, well-structured image (Should PASS Gate 1 & 2)
    sharp_img = np.zeros((512, 512, 3), dtype=np.uint8)
    for y in range(512):
        sharp_img[y, :] = (int(30 + y * 0.2), int(40 + y * 0.15), int(60 + y * 0.25))
    cv2.circle(sharp_img, (256, 256), 80, (200, 200, 220), -1)
    cv2.rectangle(sharp_img, (100, 100), (412, 412), (180, 150, 120), 4)
    cv2.putText(sharp_img, "QC Test Sharp", (120, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 40), 2)
    sharp_path = test_dir / "sharp_test.png"
    cv2.imwrite(str(sharp_path), sharp_img)

    qc = QCService(blur_threshold=50.0, gate2_threshold=0.45)
    g1_sharp = qc.evaluate_gate1(sharp_path)
    assert g1_sharp["passed"] is True
    assert g1_sharp["blur_score"] > 50.0
    print(f" [PASS] Gate 1 Sharp Image: blur score {g1_sharp['blur_score']:.1f} (PASS)")

    # 2. Create an intentionally blurry image (Should FAIL Gate 1)
    blurry_img = cv2.GaussianBlur(sharp_img, (55, 55), 0)
    blurry_path = test_dir / "blurry_test.png"
    cv2.imwrite(str(blurry_path), blurry_img)

    g1_blurry = qc.evaluate_gate1(blurry_path)
    assert g1_blurry["passed"] is False
    print(f" [PASS] Gate 1 Blurry Image: blur score {g1_blurry['blur_score']:.1f} (Correctly Rejected)")

    # 3. Create a blown-out image (Should FAIL Gate 1 Exposure)
    blown_img = np.full((512, 512, 3), 254, dtype=np.uint8)
    blown_path = test_dir / "blown_test.png"
    cv2.imwrite(str(blown_path), blown_img)

    g1_blown = qc.evaluate_gate1(blown_path)
    assert g1_blown["passed"] is False
    print(" [PASS] Gate 1 Overexposed Image correctly rejected")

    # 4. Gate 2 Identity Consistency check
    g2_res = qc.evaluate_gate2(sharp_path, reference_image_path=sharp_path)
    assert g2_res["passed"] is True
    assert g2_res["similarity"] > 0.8
    print(f" [PASS] Gate 2 Consistency against canonical reference: similarity {g2_res['similarity']:.3f} (PASS)")

    # 5. Gate 3 Semantic Checklist
    g3_res = qc.evaluate_gate3(sharp_path, checklist=["character_visible", "shadow_tendrils"])
    assert g3_res["passed"] is True
    print(f" [PASS] Gate 3 Semantic Story Checklist: {g3_res['checklist_results']} (PASS)")

    # 6. Full evaluate_job with database persistence
    with get_session() as session:
        proj = Project(name="QC Test Proj", novel_title="Shadow Slave", author="Guiltythree")
        session.add(proj)
        session.flush()

        chap = Chapter(project_id=proj.id, chapter_number=1, title="QC Test Chapter", status=ChapterStatus.PARSED)
        session.add(chap)
        session.flush()

        p1 = Passage(chapter_id=chap.id, paragraph_index=1, text="Test text for QC", passage_type=PassageType.NARRATION)
        session.add(p1)
        session.flush()

        sc = Scene(chapter_id=chap.id, scene_number=1, start_passage_id=p1.id, end_passage_id=p1.id, mood="desolate")
        session.add(sc)
        session.flush()

        shot = Shot(
            scene_id=sc.id,
            chapter_id=chap.id,
            shot_number=1,
            workflow_id="WF-004",
            shot_type="medium",
            camera_angle="eye_level",
            status=ShotStatus.PLANNED,
        )
        session.add(shot)
        session.flush()

        job = GenerationJob(
            shot_id=shot.id,
            workflow_id="WF-004",
            prompt_text="Sunny standing in snow",
            output_rgb_path=str(sharp_path),
            status=JobStatus.QUEUED,
        )
        session.add(job)
        session.commit()

        qc_eval = qc.evaluate_job(job, reference_image_path=str(sharp_path), db=session)
        assert qc_eval.overall_passed is True
        assert qc_eval.recommended_action == "approve"
        assert job.status == JobStatus.COMPLETED
        print(f" [PASS] GenerationJob QC: overall_score={qc_eval.overall_score}, status={job.status.value}")


def test_06_langgraph_pipeline_nodes():
    print("\n--- Test 6: LangGraph Phase 4 Nodes Orchestration ---")

    with get_session() as session:
        # Create minimal end-to-end project, chapter, scene, passage
        proj = Project(name="Phase 4 Pipeline Run", novel_title="Shadow Slave", author="Guiltythree")
        session.add(proj)
        session.flush()

        chap = Chapter(
            project_id=proj.id,
            chapter_number=1,
            title="The Mountain Pass",
            status=ChapterStatus.PARSED,
        )
        session.add(chap)
        session.flush()

        passg = Passage(
            chapter_id=chap.id,
            paragraph_index=1,
            text="Sunny raised his hands as shadows leaped to his command.",
            passage_type=PassageType.ACTION,
            tagged_entity_ids=["Sunny"],
        )
        session.add(passg)
        session.flush()

        sc = Scene(
            chapter_id=chap.id,
            scene_number=1,
            start_passage_id=passg.id,
            end_passage_id=passg.id,
            mood="desolate",
            description="Combat with shadows",
        )
        session.add(sc)
        session.commit()

        # Step 1: Run scene director node
        state = ProductionState(project_id=proj.id, project_name=proj.name, chapters_in_batch=[1])
        state = run_scene_director(state)
        assert state.current_stage == PipelineStage.PROMPT_COMPILATION
        assert state.total_shots_planned > 0
        print(f" [PASS] Node run_scene_director: planned {state.total_shots_planned} shots")

        # Step 2: Run compile prompts node
        state = compile_prompts(state)
        assert state.current_stage == PipelineStage.IMAGE_GENERATION
        print(" [PASS] Node compile_prompts: generated ComfyUI payloads into GenerationJobs")

        # Step 3: Run generate images node
        state = generate_images(state)
        assert state.current_stage == PipelineStage.QUALITY_CONTROL
        assert state.shots_generated > 0
        print(f" [PASS] Node generate_images: generated {state.shots_generated} frames")

        # Step 4: Run quality control node
        state = run_quality_control(state)
        assert state.current_stage == PipelineStage.HUMAN_REVIEW
        assert state.shots_passed_qc > 0
        print(f" [PASS] Node run_quality_control: {state.shots_passed_qc} shots passed 3-Gate QC funnel")


if __name__ == "__main__":
    init_db()
    test_01_deterministic_shot_budget()
    test_02_scene_director_storyboarding()
    test_03_sqlite_shot_persistence()
    test_04_jinja2_prompt_compiler()
    test_05_three_gate_qc_funnel()
    test_06_langgraph_pipeline_nodes()
    print("\n=======================================================")
    print(" ALL 6/6 PHASE 4 VERIFICATION TESTS PASSED SUCCESSFULLY! ")
    print("=======================================================\n")
