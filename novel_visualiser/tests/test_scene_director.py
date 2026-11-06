"""Unit and integration tests for SceneDirectorAgent (Option 1 Master Orchestrator).

Verifies:
1. Chapter decomposition into scene beats with word-count and pacing budgets.
2. Synchronized subagent delegation:
   - LocationContinuitySubagent (Immutable geometry & plates)
   - EnvironmentContinuitySubagent (Lighting rig & weather)
   - CharacterContinuitySubagent & BiologicalEvolutionAuditor (DNA, epochs & anchors)
   - CreatureMonsterSubagent (Bestiary turnarounds & props)
   - CompositingSubagent (Sequenced shot specifications)
3. End-to-end chapter production orchestration with SQLite canon persistence.
4. Render execution and multi-gate QC visual auditing across all budgeted shots.
"""

from pathlib import Path
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.models import Base, Chapter, Project, Scene, Shot
from app.schemas.location_subagent import ArcHierarchyTag
from app.schemas.qc import QCVerdictEnum
from app.agents.scene_director import SceneDirectorAgent


@pytest.fixture
def test_db_session():
    """In-memory SQLite database session for unit testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def sample_arc_tag():
    return ArcHierarchyTag(
        main_arc_id="ARC_01_FIRST_NIGHTMARE",
        sub_arc_id="SUBARC_01_01_OUTSKIRTS",
        mini_arc_id="MINIARC_01_01_01_BENCH_WAITING",
    )


@pytest.fixture
def chapter_1_text():
    return (
        "Sunny sat shivering on the rusted transit bench in the outskirts. A toxic twilight smog "
        "hung heavy over the dilapidated concrete towers, casting long cyan shadows across the damp street. "
        "He pulled his oversized, frayed utility jacket tighter around his scrawny frame. His pale, sallow skin "
        "was slick with cold rain, and dark circles shadowed his jet-black eyes.\n\n"
        "Suddenly, the sickening screech of chitin scraped against asphalt. From the shadowy alleyway, "
        "a monstrous centipede larva with jagged mandibles emerged, its many legs twitching aggressively. "
        "Sunny gasped and drew the Midnight Shard, the dark obsidian blade gleaming faintly under the broken streetlamp."
    )


def test_break_down_chapter_into_scenes(sample_arc_tag, chapter_1_text):
    director = SceneDirectorAgent(use_ai_thinking=False)
    scenes = director.break_down_chapter_into_scenes(
        chapter_number=1,
        chapter_title="Chapter 1: Outskirts Slum",
        novel_text=chapter_1_text,
        arc_tag=sample_arc_tag,
    )
    assert len(scenes) >= 2
    assert scenes[0]["scene_number"] == 1
    assert scenes[0]["primary_character"] == "Sunny"
    assert "Outskirts" in scenes[0]["location_name"]
    assert len(scenes[1]["creatures"]) > 0 or "Centipede" in scenes[1]["scene_title"] or len(scenes[1]["weapons_or_items"]) > 0


def test_orchestrate_scene_delegation(sample_arc_tag, chapter_1_text, tmp_path):
    director = SceneDirectorAgent(use_ai_thinking=False, output_dir=tmp_path)
    scene_info = {
        "scene_number": 1,
        "scene_title": "Scene 1: Waiting on the Transit Bench",
        "scene_type": "exposition",
        "location_name": "Outskirts Transit Bench N24",
        "primary_character": "Sunny",
        "secondary_characters": [],
        "creatures": [],
        "weapons_or_items": ["Tattered Coat"],
        "full_scene_text": chapter_1_text.split("\n\n")[0],
    }

    plan = director.orchestrate_scene(
        scene_id="ch01_sc01",
        scene_number=1,
        scene_info=scene_info,
        arc_tag=sample_arc_tag,
        chapter_number=1,
    )

    # 1. Location Matrix
    assert plan.location_matrix is not None
    assert plan.location_matrix.canonical_name == "Outskirts Transit Bench N24"

    # 2. Environment Card
    assert plan.environment_card is not None
    assert plan.environment_card.lighting_rig is not None

    # 3. Active Characters
    assert len(plan.active_characters) > 0
    assert plan.active_characters[0].entity_name == "Sunny"

    # 4. Sequenced Shots
    assert plan.target_shots >= 2
    assert len(plan.shots) == plan.target_shots
    assert plan.shots[0].shot_id.startswith("ch01_sc01_shot_")


def test_orchestrate_chapter_end_to_end_with_database(sample_arc_tag, chapter_1_text, test_db_session, tmp_path):
    director = SceneDirectorAgent(
        db=test_db_session,
        use_ai_thinking=False,
        output_dir=tmp_path,
    )

    chapter_plan = director.orchestrate_chapter(
        chapter_number=1,
        chapter_title="Chapter 1: Outskirts Slum",
        novel_text=chapter_1_text,
        arc_tag=sample_arc_tag,
        project_id="shadow_slave_test",
    )

    # Production Plan Assertions
    assert chapter_plan.chapter_number == 1
    assert chapter_plan.total_words > 0
    assert chapter_plan.total_shots >= 4
    assert len(chapter_plan.scenes) >= 2

    # Database Persistence Assertions
    db_proj = test_db_session.query(Project).filter(Project.id == "shadow_slave_test").first()
    assert db_proj is not None

    db_ch = test_db_session.query(Chapter).filter(Chapter.project_id == "shadow_slave_test").first()
    assert db_ch is not None
    assert db_ch.chapter_number == 1

    db_scenes = test_db_session.query(Scene).filter(Scene.chapter_id == db_ch.id).all()
    assert len(db_scenes) == len(chapter_plan.scenes)

    db_shots = test_db_session.query(Shot).filter(Shot.chapter_id == db_ch.id).all()
    assert len(db_shots) == chapter_plan.total_shots


def test_execute_and_audit_chapter_pipeline(sample_arc_tag, chapter_1_text, test_db_session, tmp_path):
    director = SceneDirectorAgent(
        db=test_db_session,
        use_ai_thinking=False,
        output_dir=tmp_path,
    )

    chapter_plan = director.orchestrate_chapter(
        chapter_number=1,
        chapter_title="Chapter 1: Outskirts Slum",
        novel_text=chapter_1_text,
        arc_tag=sample_arc_tag,
        project_id="shadow_slave_test",
    )

    # Execute all budgeted shots in mock canvas mode and run 3-gate QC
    results = director.execute_and_audit_chapter(
        production_plan=chapter_plan,
        force_mock=True,
    )

    assert len(results) == chapter_plan.total_shots
    for res in results:
        assert "artifact" in res
        assert "qc_audit" in res
        artifact = res["artifact"]
        qc_audit = res["qc_audit"]
        assert artifact.image_path is not None
        assert Path(artifact.image_path).exists()
        assert qc_audit.verdict in [QCVerdictEnum.APPROVED, QCVerdictEnum.NEEDS_INPAINT_PATCH]
