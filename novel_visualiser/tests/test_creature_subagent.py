"""Unit and integration tests for CreatureMonsterSubagent.

Verifies:
1. 5-Dimension deep observational reasoning on monster biology and power system classification.
2. 2-Axis power system: Soul Rank (Awakened, Fallen, etc.) & Class (Beast, Tyrant, etc.).
3. Bestiary 4-sheet reference matrix synthesis & predatory head reference chip cropping.
4. Hero prop / Memory item card synthesis (Midnight Shard).
5. SQLite canon persistence and historical lookup for creatures and items.
6. Runtime combat anchor resolution and spatial positioning for battle shots.
"""

from pathlib import Path
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.models import Base, Entity, EntityType, Project
from app.schemas.location_subagent import ArcHierarchyTag
from app.schemas.creature_subagent import (
    CreatureCard,
    CreatureClassEnum,
    CreatureRankEnum,
    CreatureTaxonomyEnum,
    ItemPropCard,
    ShotCreatureAnchor,
)
from app.agents.creature_subagent import CreatureMonsterSubagent


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
        sub_arc_id="SUBARC_01_02_MOUNTAIN_TEMPLE",
        mini_arc_id="MINIARC_01_02_01_SHAMAN_ALTAR",
        parallel_thread_id=None,
        in_universe_day=2,
        time_of_day="dawn",
        weather="blizzard",
    )


def test_creature_deep_observation_and_power_system(arc_tag):
    """Test 5-dimension observation and 2-axis power classification."""
    agent = CreatureMonsterSubagent(use_ai_thinking=False)  # Deterministic fallback

    passage = """
    From the frozen mountain fissures crawled a gigantic centipede monstrosity, an Awakened Beast of the high peaks.
    Its five-meter-long body was encased in heavy, interlocking plates of segmented black obsidian chitin, glistening like polished volcanic glass.
    Dozens of barbed scythe-like legs skittered with lethal speed over the snow, leaving sizzling trails of green corrosive acid.
    A terrifying cluster of six crimson eyes burned above its triple-hinged serrated mandibles, leaking dark shadow miasma into the blizzard.
    """

    card = agent.observe_and_synthesize_creature(
        creature_name="Mountain Centipede Beast",
        passage_text=passage,
        arc_hierarchy=arc_tag,
        chapter_number=3,
    )

    # 1. Verify Power Classification
    assert card.rank == CreatureRankEnum.AWAKENED
    assert card.creature_class == CreatureClassEnum.BEAST
    assert card.taxonomy == CreatureTaxonomyEnum.INSECTOID
    assert card.soul_core_count == 1

    # 2. Verify 5-Dimension Observation
    assert card.observation is not None
    assert len(card.observation.power_system_soul_anatomy) > 0
    assert len(card.observation.skeletal_scale_mass) > 0
    assert len(card.observation.biomaterial_integument) > 0
    assert len(card.observation.locomotion_kinesics_weaponry) > 0
    assert len(card.observation.environmental_corruption) > 0

    # 3. Verify Scale Spec
    assert card.scale_spec.estimated_length_meters >= 4.0
    assert "1:" in card.scale_spec.human_scale_ratio

    # 4. Verify Identity Locking Tokens
    assert len(card.core_danbooru_biology_tags) > 0
    assert "segmented black chitin" in card.core_danbooru_biology_tags
    assert "1boy" in card.negative_identity_tags


def test_creature_bestiary_matrix_and_head_chip_generation(arc_tag, tmp_path):
    """Test generating bestiary turnaround plates and predatory head chip auto-cropping."""
    agent = CreatureMonsterSubagent(use_ai_thinking=False)

    passage = "A ferocious Awakened centipede beast with segmented obsidian chitin and razor mandibles."
    card = agent.observe_and_synthesize_creature(
        creature_name="Mountain Centipede",
        passage_text=passage,
        arc_hierarchy=arc_tag,
        chapter_number=3,
    )

    rendered_card = agent.render_bestiary_matrix_and_head_chip(
        card=card,
        output_dir=tmp_path / "bestiary_output",
        deterministic_seed=1234,
    )

    # 1. Verify Sheet 1 (Turnaround) Path Exists
    assert rendered_card.sheet_1_turnaround.image_path is not None
    assert Path(rendered_card.sheet_1_turnaround.image_path).exists()

    # 2. Verify Sheet 4 (Scale Comparison) Path Exists
    assert rendered_card.sheet_4_scale_comparison.image_path is not None
    assert Path(rendered_card.sheet_4_scale_comparison.image_path).exists()

    # 3. Verify Predatory Head Chip Exists for ComfyUI IP-Adapter Plus
    assert rendered_card.head_reference_chip_path is not None
    assert Path(rendered_card.head_reference_chip_path).exists()


def test_item_memory_prop_synthesis(arc_tag):
    """Test synthesis of Sunny's Awakened Memory Weapon (Midnight Shard)."""
    agent = CreatureMonsterSubagent(use_ai_thinking=False)

    passage = """
    A translucent notification materialized in Sunny's vision: [You have slain an Awakened Beast! You have received a Memory: Midnight Shard].
    Summoning the weapon, a single-edged straight sword made of pitch-black obsidian appeared in his grip.
    The black blade was so dark it seemed to absorb ambient light, its surface as reflective as dark water with faint shadowy mist rising from the edge.
    """

    item = agent.observe_and_synthesize_item(
        item_name="Midnight Shard",
        passage_text=passage,
        arc_hierarchy=arc_tag,
        item_type="weapon",
        chapter_number=3,
    )

    assert isinstance(item, ItemPropCard)
    assert item.item_name == "Midnight Shard"
    assert item.item_type == "weapon"
    assert "obsidian" in item.material_physics.lower()
    assert "holding sword" in item.danbooru_item_tags
    assert "black blade" in item.danbooru_item_tags
    assert "broken blade" in item.negative_item_tags


def test_creature_and_item_sqlite_persistence(test_db_session, arc_tag, tmp_path):
    """Test round-trip SQLite persistence for CreatureCard and ItemPropCard."""
    agent = CreatureMonsterSubagent(db=test_db_session, use_ai_thinking=False)

    project = Project(
        name="Shadow Slave Production",
        novel_title="Shadow Slave",
        author="Guiltythree",
    )
    test_db_session.add(project)
    test_db_session.flush()

    # 1. Creature Card
    card = agent.observe_and_synthesize_creature(
        creature_name="High Mountain Centipede",
        passage_text="An Awakened Beast centipede with glossy black chitin and mandibles.",
        arc_hierarchy=arc_tag,
        chapter_number=3,
    )
    rendered_card = agent.render_bestiary_matrix_and_head_chip(card=card, output_dir=tmp_path)
    persisted_c = agent.persist_creature_card(project_id=project.id, card=rendered_card)
    assert persisted_c is True

    # 2. Item Card
    item = agent.observe_and_synthesize_item(
        item_name="Midnight Shard",
        passage_text="Obsidian single-edged sword.",
        arc_hierarchy=arc_tag,
        chapter_number=3,
    )
    persisted_i = agent.persist_item_card(project_id=project.id, item_card=item, chapter_number=3)
    assert persisted_i is True

    # 3. Query Back
    cached_creature = agent.query_historical_creature(
        project_id=project.id,
        creature_name="High Mountain Centipede",
        chapter_number=3,
    )
    assert cached_creature is not None
    assert cached_creature.rank == CreatureRankEnum.AWAKENED
    assert cached_creature.creature_class == CreatureClassEnum.BEAST

    cached_item = agent.query_historical_item(
        project_id=project.id,
        item_name="Midnight Shard",
        chapter_number=3,
    )
    assert cached_item is not None
    assert cached_item.item_name == "Midnight Shard"
    assert "black blade" in cached_item.danbooru_item_tags


def test_shot_creature_battle_anchor_resolution(arc_tag):
    """Test compilation of runtime combat shot tokens and spatial anchors."""
    agent = CreatureMonsterSubagent(use_ai_thinking=False)

    card = agent.observe_and_synthesize_creature(
        creature_name="Mountain Centipede",
        passage_text="An Awakened centipede with obsidian chitin.",
        arc_hierarchy=arc_tag,
        chapter_number=3,
    )

    anchor = agent.resolve_shot_creature_anchor(
        card=card,
        combat_action="rearing back with open razor mandibles striking down",
        spatial_anchor="center_foreground",
    )

    assert isinstance(anchor, ShotCreatureAnchor)
    assert anchor.suggested_spatial_anchor == "center_foreground"
    assert "segmented black chitin" in anchor.resolved_creature_prompt
    assert "rearing back with open razor mandibles" in anchor.resolved_creature_prompt
    assert "1boy" in anchor.resolved_negative_prompt

