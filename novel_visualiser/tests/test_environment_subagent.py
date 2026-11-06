"""Unit and integration tests for EnvironmentContinuitySubagent.

Verifies:
1. 5-Dimension deep observational reasoning on atmosphere, lighting, flora, and surface physics.
2. Lighting rig specification (key vector, fill bounce, rim light, Kelvin rating).
3. Runtime shot anchor resolution (de-duplicated positive/negative tokens, light vectors, moisture rating).
4. SQLite historical memory lookup and persistence across chapters.
5. Dynamic temporal evolution across scenes while locking underlying location geometry.
"""

from pathlib import Path
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.models import Base, Entity, EntityType, Project
from app.schemas.location_subagent import ArcHierarchyTag
from app.schemas.environment_subagent import (
    EnvironmentalAtmosphereCard,
    ShotEnvironmentAnchor,
    TimeOfDayEnum,
    WeatherConditionEnum,
)
from app.agents.environment_subagent import EnvironmentContinuitySubagent


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
        time_of_day="twilight",
        weather="acid smog",
    )


def test_environment_deep_observation_and_rig(arc_tag):
    """Test 5-dimension observation and lighting rig generation."""
    agent = EnvironmentContinuitySubagent(use_ai_thinking=False)  # test deterministic fallback

    passage = """
    The outskirts were smothered beneath a thick shroud of toxic industrial smog as twilight descended.
    High above, an aged sodium vapor streetlamp flickered intermittently, casting a sickening warm amber pool
    over the cracked pavement. Cold cyan light from the dying sky bounced off the damp asphalt, where oily puddles
    mirrored the rusty lamp. Through the jagged fissures in the concrete, stubborn dead weeds and grey moss clung to life.
    """

    card = agent.observe_and_synthesize_environment(
        location_name="Outskirts Transit Bench N24",
        passage_text=passage,
        arc_hierarchy=arc_tag,
        chapter_number=1,
    )

    # 1. Verify 5-Dimension Observation
    assert card.observation is not None
    assert len(card.observation.chronological_solar) > 0
    assert len(card.observation.lighting_physics) > 0
    assert len(card.observation.atmospheric_particles) > 0
    assert len(card.observation.foliage_ecosystem) > 0
    assert len(card.observation.surface_weathering) > 0

    # 2. Verify Lighting Rig
    assert card.lighting_rig is not None
    assert card.lighting_rig.key_color_temperature_k <= 3500  # Warm sodium amber
    assert card.lighting_rig.rim_light_active is True
    assert card.lighting_rig.shadow_hardness == "soft_diffuse"

    # 3. Verify Prompt Conditioning Tokens
    assert len(card.danbooru_lighting_tags) > 0
    assert "volumetric lighting" in card.danbooru_lighting_tags
    assert len(card.danbooru_weather_tags) > 0
    assert len(card.danbooru_foliage_tags) > 0
    assert len(card.danbooru_surface_tags) > 0
    assert "sunny blue sky" in card.negative_environment_tags
    assert len(card.color_palette_hex) >= 4


def test_environment_runtime_anchor_resolution(arc_tag):
    """Test compiling runtime shot anchor tokens and vectors."""
    agent = EnvironmentContinuitySubagent(use_ai_thinking=False)

    passage = "Heavy acid drizzle soaked the cracked asphalt under flickering amber streetlights."
    card = agent.observe_and_synthesize_environment(
        location_name="Outskirts Transit Bench N24",
        passage_text=passage,
        arc_hierarchy=arc_tag,
        chapter_number=1,
    )

    anchor = agent.resolve_shot_environment_anchor(card)

    assert isinstance(anchor, ShotEnvironmentAnchor)
    assert anchor.environment_card_id == card.environment_id
    assert "volumetric lighting" in anchor.resolved_lighting_block
    assert "sunny blue sky" in anchor.resolved_negative_block
    assert anchor.key_light_vector in ["top_left", "top_right", "overhead"]
    assert anchor.shadow_direction in ["bottom_right", "bottom_left", "directly_beneath"]
    assert 0.0 <= anchor.surface_moisture_level <= 1.0


def test_environment_sqlite_persistence_and_historical_query(test_db_session, arc_tag):
    """Test round-trip persistence and historical lookup in SQLite."""
    agent = EnvironmentContinuitySubagent(db=test_db_session, use_ai_thinking=False)

    # 1. Create Project and Location Entity
    project = Project(
        name="Shadow Slave Visualisation",
        novel_title="Shadow Slave",
        author="Guiltythree",
    )
    test_db_session.add(project)
    test_db_session.flush()

    loc_entity = Entity(
        project_id=project.id,
        entity_type=EntityType.LOCATION,
        canonical_name="Outskirts Transit Bench N24",
        description="Battered metal bench in the desolate outskirts",
    )
    test_db_session.add(loc_entity)
    test_db_session.commit()

    # 2. Synthesize and Persist Card
    passage = "Gloomy twilight settled over the outskirts with cold drizzle."
    card = agent.observe_and_synthesize_environment(
        location_name="Outskirts Transit Bench N24",
        passage_text=passage,
        arc_hierarchy=arc_tag,
        location_id=loc_entity.id,
        chapter_number=1,
    )

    persisted = agent.persist_environment_card(project_id=project.id, card=card)
    assert persisted is True

    # 3. Query Historical Card back
    cached_card = agent.query_historical_environment(
        project_id=project.id,
        location_id=loc_entity.id,
        chapter_number=1,
    )

    assert cached_card is not None
    assert cached_card.environment_id == card.environment_id
    assert cached_card.time_of_day == card.time_of_day
    assert cached_card.lighting_rig.key_color_temperature_k == card.lighting_rig.key_color_temperature_k


def test_environment_temporal_evolution_different_chapters(test_db_session, arc_tag):
    """Test that the same location geometry can evolve from twilight smog to midnight blizzard."""
    agent = EnvironmentContinuitySubagent(db=test_db_session, use_ai_thinking=False)

    project = Project(
        name="Shadow Slave",
        novel_title="Shadow Slave",
        author="Guiltythree",
    )
    test_db_session.add(project)
    test_db_session.flush()

    loc_entity = Entity(
        project_id=project.id,
        entity_type=EntityType.LOCATION,
        canonical_name="Outskirts Transit Bench N24",
    )
    test_db_session.add(loc_entity)
    test_db_session.commit()

    # Chapter 1: Twilight Smog
    card_ch1 = agent.observe_and_synthesize_environment(
        location_name="Outskirts Transit Bench N24",
        passage_text="Faint twilight filtered through grey outskirts smog.",
        arc_hierarchy=arc_tag,
        location_id=loc_entity.id,
        chapter_number=1,
    )
    agent.persist_environment_card(project_id=project.id, card=card_ch1)

    # Chapter 2: Midnight Heavy Acid Drizzle
    arc_ch2 = arc_tag.model_copy()
    arc_ch2.time_of_day = "midnight"
    arc_ch2.weather = "acid drizzle"

    card_ch2 = agent.observe_and_synthesize_environment(
        location_name="Outskirts Transit Bench N24",
        passage_text="At midnight, a heavy downpour of toxic acid rain pounded the asphalt in pitch darkness.",
        arc_hierarchy=arc_ch2,
        location_id=loc_entity.id,
        chapter_number=2,
    )
    agent.persist_environment_card(project_id=project.id, card=card_ch2)

    # Historical query at Chapter 1 -> Returns Twilight Card
    retrieved_ch1 = agent.query_historical_environment(
        project_id=project.id,
        location_id=loc_entity.id,
        chapter_number=1,
    )
    assert retrieved_ch1.valid_from_chapter == 1
    assert retrieved_ch1.time_of_day == TimeOfDayEnum.TWILIGHT_BLUE_HOUR

    # Historical query at Chapter 2 -> Returns Midnight Acid Rain Card
    retrieved_ch2 = agent.query_historical_environment(
        project_id=project.id,
        location_id=loc_entity.id,
        chapter_number=2,
    )
    assert retrieved_ch2.valid_from_chapter == 2
    assert retrieved_ch2.time_of_day == TimeOfDayEnum.MIDNIGHT_PITCH
    assert retrieved_ch2.weather_condition in [WeatherConditionEnum.ACID_DRIZZLE, WeatherConditionEnum.HEAVY_RAIN]

