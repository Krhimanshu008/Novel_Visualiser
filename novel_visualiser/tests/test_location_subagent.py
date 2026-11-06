"""Unit and integration tests for LocationContinuitySubagent.

Verifies:
1. Arc Hierarchy tagging (Main Arc > Sub Arc > Mini Arc).
2. Deep 5-dimension observational reasoning & 4-plate standard synthesis.
3. Structural Depth and Lineart guide extraction via OpenCV.
4. Camera angle resolution matching (Wide, Stage, Reverse, Macro).
5. Historical memory lookup and SQLite canon persistence.
"""

import os
from pathlib import Path
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.models import Base, EntityType, Project, RefType
from app.schemas.location_subagent import (
    ArcHierarchyTag,
    PlateRole,
    SinglePlateSpec,
)
from app.agents.location_subagent import LocationContinuitySubagent


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
def arc_tag():
    """Standard arc hierarchy tag for testing."""
    return ArcHierarchyTag(
        main_arc_id="ARC_01_FIRST_NIGHTMARE",
        sub_arc_id="SUBARC_01_01_OUTSKIRTS",
        mini_arc_id="MINIARC_01_01_01_BENCH_WAITING",
        parallel_thread_id=None,
        in_universe_day=1,
        time_of_day="midnight",
        weather="cold drizzle",
    )


def test_deep_observational_reasoning(arc_tag):
    """Test 5-dimension observational synthesis on novel scene excerpt."""
    agent = LocationContinuitySubagent(use_ai_thinking=False)  # test deterministic fallback

    passage = """
    Sunny sat on the battered, rusted metal bench across from the fortified police station.
    The dreary outskirts of the N24 district were drenched in cold rain. Neon signs flickered
    with missing characters above leaking gutters, reflecting in greasy puddles on the cracked asphalt.
    """

    matrix_spec = agent.observe_and_synthesize_location(
        location_name="Outskirts Transit Bench N24",
        passage_text=passage,
        arc_hierarchy=arc_tag,
        deterministic_seed=1000,
    )

    # 1. Verify 5-dimension observation structure
    assert matrix_spec.observation is not None
    assert "brutalist" in matrix_spec.observation.socioeconomic_reality.lower() or "dystopian" in matrix_spec.observation.socioeconomic_reality.lower()
    assert "curb" in matrix_spec.observation.architectural_geometry.lower()
    assert "puddle" in matrix_spec.observation.environmental_physics.lower() or "precipitation" in matrix_spec.observation.environmental_physics.lower()

    # 2. Verify all 4 plates generated with correct roles
    assert matrix_spec.plate_1_vista.role == PlateRole.WIDE_ESTABLISHING
    assert matrix_spec.plate_2_stage.role == PlateRole.ACTION_STAGE
    assert matrix_spec.plate_3_reverse.role == PlateRole.REVERSE_RECIPROCAL
    assert matrix_spec.plate_4_macro.role == PlateRole.MACRO_TEXTURE

    # 3. Verify resolutions & angles
    assert matrix_spec.plate_1_vista.width == 1216
    assert matrix_spec.plate_1_vista.height == 832
    assert matrix_spec.plate_3_reverse.azimuth_degrees == 180
    assert matrix_spec.plate_4_macro.width == 1024
    assert matrix_spec.plate_4_macro.height == 1024


def test_structural_guide_extraction(tmp_path):
    """Verify OpenCV structural depth map and Canny lineart extraction."""
    agent = LocationContinuitySubagent(use_ai_thinking=False)

    spec = agent.observe_and_synthesize_location(
        location_name="Holding Cell",
        passage_text="A cold bare cell with steel bars.",
        arc_hierarchy=ArcHierarchyTag(main_arc_id="ARC_01"),
    )

    # Generate test plates
    updated_spec = agent.generate_location_matrix(spec, output_dir=tmp_path)

    # Verify each plate got RGB image, Depth map, and Lineart
    plates = [
        updated_spec.plate_1_vista,
        updated_spec.plate_2_stage,
        updated_spec.plate_3_reverse,
        updated_spec.plate_4_macro,
    ]

    for p in plates:
        assert p.image_path is not None
        assert os.path.exists(p.image_path)
        assert p.depth_map_path is not None
        assert os.path.exists(p.depth_map_path)
        assert p.lineart_map_path is not None
        assert os.path.exists(p.lineart_map_path)


def test_camera_angle_resolution(arc_tag, tmp_path):
    """Verify dynamic camera angle matching for different shot intents."""
    agent = LocationContinuitySubagent(use_ai_thinking=False)
    spec = agent.observe_and_synthesize_location("Police Lobby", "Reinforced doors", arc_tag)
    agent.generate_location_matrix(spec, output_dir=tmp_path)

    # Wide Establishing Shot
    anchor_wide = agent.resolve_shot_angle_anchor(spec, "WIDE_ESTABLISHING")
    assert anchor_wide.resolved_role == PlateRole.WIDE_ESTABLISHING
    assert anchor_wide.selected_plate_image_path == spec.plate_1_vista.image_path

    # Interaction Stage Shot (Medium)
    anchor_stage = agent.resolve_shot_angle_anchor(spec, "STAGE_ACTION")
    assert anchor_stage.resolved_role == PlateRole.ACTION_STAGE
    assert anchor_stage.selected_plate_image_path == spec.plate_2_stage.image_path
    assert anchor_stage.recommended_inpaint_box is not None

    # Reverse POV Counter Shot
    anchor_rev = agent.resolve_shot_angle_anchor(spec, "REVERSE_POV")
    assert anchor_rev.resolved_role == PlateRole.REVERSE_RECIPROCAL
    assert anchor_rev.selected_plate_image_path == spec.plate_3_reverse.image_path

    # Macro Detail Shot
    anchor_macro = agent.resolve_shot_angle_anchor(spec, "MACRO_DETAIL")
    assert anchor_macro.resolved_role == PlateRole.MACRO_TEXTURE
    assert anchor_macro.selected_plate_image_path == spec.plate_4_macro.image_path


def test_historical_memory_and_canon_persistence(test_db_session, arc_tag, tmp_path):
    """Verify storing location to DB and retrieving in later chapters with continuity."""
    agent = LocationContinuitySubagent(db=test_db_session, use_ai_thinking=False)

    spec = agent.observe_and_synthesize_location(
        location_name="Outskirts Bench N24",
        passage_text="Battered metal bench on rainy street.",
        arc_hierarchy=arc_tag,
    )
    agent.generate_location_matrix(spec, output_dir=tmp_path)

    # Save to SQLite Canon in Chapter 1
    project = Project(
        id="proj_test_01",
        name="Shadow Slave Test",
        novel_title="Shadow Slave",
        author="Guiltythree",
    )
    test_db_session.add(project)
    test_db_session.commit()

    project_id = "proj_test_01"
    entity = agent.save_to_canon(project_id, spec, current_chapter=1)
    assert entity.id == spec.location_id
    assert entity.entity_type == EntityType.LOCATION

    # Simulate Chapter 15: Look up the location in historical memory
    cached_spec = agent.query_historical_memory(project_id, "Outskirts Bench N24")
    assert cached_spec is not None
    assert cached_spec.is_existing is True
    assert cached_spec.location_id == spec.location_id
    assert cached_spec.plate_2_stage.title == spec.plate_2_stage.title
    assert cached_spec.arc_hierarchy.main_arc_id == "ARC_01_FIRST_NIGHTMARE"
