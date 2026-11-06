"""Unit and integration tests for CharacterContinuitySubagent.

Verifies:
1. Arc Hierarchy tagging & 5-dimension deep observational reasoning.
2. Decoupling immutable biological locks from temporal outfits and items.
3. 4-Sheet Reference Matrix synthesis & automatic canonical face chip cropping.
4. Ordered Danbooru prompt block assembly & inpaint bounding box resolution.
5. Historical memory lookup across chapters and SQLite canon persistence.
"""

import os
from pathlib import Path
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.models import Base, EntityType, Project
from app.schemas.location_subagent import ArcHierarchyTag
from app.schemas.character_subagent import CharacterSheetRole
from app.agents.character_subagent import CharacterContinuitySubagent


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


def test_character_deep_observation_and_decoupling(arc_tag):
    """Test 5-dimension observation and decoupling of biology from clothing."""
    agent = CharacterContinuitySubagent(use_ai_thinking=False)  # test deterministic fallback

    passage = """
    Sunny sat on the metal bench, shivering inside his oversized, frayed utility jacket.
    He was an scrawny orphan of the outskirts, his face pale and sickly with sharp, dark circles
    under raven-black eyes. His unruly black hair fell across his brow as he took a sip of lukewarm synth-coffee.
    """

    matrix_spec = agent.observe_and_synthesize_character(
        character_name="Sunny",
        passage_text=passage,
        arc_hierarchy=arc_tag,
        gender="male",
        deterministic_seed=42,
    )

    # 1. Verify 5-Dimension Observation
    assert matrix_spec.observation is not None
    assert "malnutrition" in matrix_spec.observation.socioeconomic_nutrition.lower() or "synth-paste" in matrix_spec.observation.socioeconomic_nutrition.lower()
    assert "black" in matrix_spec.observation.facial_morphology.lower()
    assert "frayed" in matrix_spec.observation.wardrobe_poverty_textile.lower() or "jacket" in matrix_spec.observation.wardrobe_poverty_textile.lower()

    # 2. Verify Decoupling: Identity Lock must NOT contain clothing
    ident = matrix_spec.identity_lock
    assert ident.gender_tag == "1boy"
    assert "pale skin" in ident.core_danbooru_tags
    assert any("black hair" in tag for tag in ident.core_danbooru_tags)
    for tag in ident.core_danbooru_tags:
        assert "jacket" not in tag.lower(), "Clothing leaked into immutable biology lock!"
        assert "boots" not in tag.lower(), "Clothing leaked into immutable biology lock!"

    # 3. Verify Outfit Card contains clothing and poverty wear
    outfit = matrix_spec.active_outfit
    assert any("jacket" in tag for tag in outfit.danbooru_clothing_tags)
    assert len(outfit.poverty_wear_tags) > 0

    # 4. Verify Held Items
    assert len(matrix_spec.held_items) > 0
    assert "coffee" in matrix_spec.held_items[0].item_name.lower()

    # 5. Verify 4-Sheet Standard
    assert matrix_spec.sheet_1_turnaround.role == CharacterSheetRole.CANONICAL_TURNAROUND
    assert matrix_spec.sheet_2_expressions.role == CharacterSheetRole.EXPRESSION_GRID
    assert matrix_spec.sheet_3_outfit.role == CharacterSheetRole.OUTFIT_PLATE
    assert matrix_spec.sheet_4_items.role == CharacterSheetRole.ITEM_SHEET


def test_character_matrix_generation_and_face_chip(arc_tag, tmp_path):
    """Verify 4-sheet generation and automatic canonical face chip extraction."""
    agent = CharacterContinuitySubagent(use_ai_thinking=False)

    spec = agent.observe_and_synthesize_character(
        character_name="Sunny",
        passage_text="Scrawny boy with dark hair and torn jacket.",
        arc_hierarchy=arc_tag,
    )

    updated_spec = agent.generate_character_matrix(spec, output_dir=tmp_path)

    sheets = [
        updated_spec.sheet_1_turnaround,
        updated_spec.sheet_2_expressions,
        updated_spec.sheet_3_outfit,
        updated_spec.sheet_4_items,
    ]

    for s in sheets:
        assert s.image_path is not None
        assert os.path.exists(s.image_path)

    # Verify Face Chip was created for IP-Adapter Plus
    assert updated_spec.identity_lock.reference_face_path is not None
    assert os.path.exists(updated_spec.identity_lock.reference_face_path)


def test_shot_character_anchor_resolution(arc_tag, tmp_path):
    """Verify ordered prompt block assembly and inpaint bounding box resolution."""
    agent = CharacterContinuitySubagent(use_ai_thinking=False)
    spec = agent.observe_and_synthesize_character("Sunny", "Malnourished boy", arc_tag)
    agent.generate_character_matrix(spec, output_dir=tmp_path)

    # Resolve for Scene 1 Shot 2: Sitting on bench drinking coffee
    anchor = agent.resolve_shot_character_anchor(
        spec,
        shot_action="sitting on curb, holding paper coffee cup",
        emotion="bitter cynical grimace",
        camera_framing="MEDIUM",
    )

    assert anchor.entity_id == spec.character_id
    assert "1boy" in anchor.full_positive_prompt
    assert "pale skin" in anchor.full_positive_prompt
    assert "utility jacket" in anchor.full_positive_prompt
    assert "bitter cynical grimace" in anchor.full_positive_prompt
    assert "paper coffee cup" in anchor.full_positive_prompt
    assert anchor.recommended_inpaint_box == [0.25, 0.25, 0.75, 0.90]
    assert anchor.face_reference_path is not None


def test_historical_memory_and_canon_persistence(test_db_session, arc_tag, tmp_path):
    """Verify storing character to DB and retrieving in later chapters with 100% identity continuity."""
    agent = CharacterContinuitySubagent(db=test_db_session, use_ai_thinking=False)

    spec = agent.observe_and_synthesize_character(
        character_name="Sunny",
        passage_text="Scrawny boy with dark hair and torn clothes.",
        arc_hierarchy=arc_tag,
    )
    agent.generate_character_matrix(spec, output_dir=tmp_path)

    # 1. Setup Project & Save to Canon in Chapter 1
    project = Project(
        id="proj_test_char_01",
        name="Shadow Slave Test",
        novel_title="Shadow Slave",
        author="Guiltythree",
    )
    test_db_session.add(project)
    test_db_session.commit()

    entity = agent.save_to_canon("proj_test_char_01", spec, current_chapter=1)
    assert entity.id == spec.character_id
    assert entity.entity_type == EntityType.CHARACTER

    # 2. Simulate Chapter 15: Memory Lookup
    cached_spec = agent.query_historical_memory("proj_test_char_01", "Sunny")
    assert cached_spec is not None
    assert cached_spec.is_existing is True
    assert cached_spec.character_id == spec.character_id
    assert cached_spec.identity_lock.core_danbooru_tags == spec.identity_lock.core_danbooru_tags
    assert cached_spec.active_outfit.outfit_name == spec.active_outfit.outfit_name


def test_biological_evolution_epochs_and_progression(test_db_session, arc_tag, tmp_path):
    """Verify multi-epoch progression across chapters (Ch 1 -> Ch 3 -> Ch 25).
    
    Verifies that:
    1. Chapter 1 Sunny is malnourished/scrawny with sallow skin.
    2. Chapter 3 Sunny evolves into slave survivor.
    3. Chapter 25 Sunny evolves into Shadow Awakened with porcelain/alabaster skin and ethereal glow,
       while 'scrawny' and 'sallow skin' are promoted to NEGATIVE constraint tags!
    4. Querying by chapter retrieves the exact active epoch.
    """
    from app.agents.biological_evolution_auditor import BiologicalEvolutionAuditor
    from app.schemas.character_subagent import BiologicalEpoch, OutfitCard

    agent = CharacterContinuitySubagent(db=test_db_session, use_ai_thinking=False)
    auditor = BiologicalEvolutionAuditor(db=test_db_session, use_ai_thinking=False)

    project = Project(
        id="proj_evolution_01",
        name="Shadow Slave Evolution Test",
        novel_title="Shadow Slave",
        author="Guiltythree",
    )
    test_db_session.add(project)
    test_db_session.commit()

    # --- 1. Chapter 1: Slum Orphan ---
    ch1_passage = "Sunny sat on the bench, a scrawny boy with sallow skin and dark circles."
    spec_ch1 = agent.observe_and_synthesize_character("Sunny", ch1_passage, arc_tag)
    agent.save_to_canon("proj_evolution_01", spec_ch1, current_chapter=1)

    # --- 2. Chapter 3: First Nightmare Slave Survivor ---
    ch3_passage = "Sunny was forced into the slave caravan. Whip scars marked his lean sinewy back as he fought for survival in the snow."
    epoch_ch3 = auditor.audit_evolution("Sunny", 3, ch3_passage, spec_ch1)
    assert epoch_ch3 is not None
    assert "slave" in epoch_ch3.epoch_name.lower()

    spec_ch3 = auditor.apply_evolution_to_spec(
        spec_ch1,
        epoch_ch3,
        new_outfit=OutfitCard(
            outfit_id="outfit_slave_rags",
            outfit_name="Tattered Temple Slave Rags",
            danbooru_clothing_tags=["tattered slave rags", "hemp tunic"],
            poverty_wear_tags=["torn fabric", "bloodstains"],
            negative_clothing_tags=["clean clothes", "armor"],
        )
    )
    agent.save_to_canon("proj_evolution_01", spec_ch3, current_chapter=3)

    # --- 3. Chapter 25: Forgotten Shore Shadow Awakened ---
    ch25_passage = "Having awakened his Divine Shadow Aspect and claimed his True Name, Sunny's skin had transformed into smooth, flawless alabaster porcelain with a subtle ethereal glow. He moved with athletic grace."
    epoch_ch25 = auditor.audit_evolution("Sunny", 25, ch25_passage, spec_ch3)
    assert epoch_ch25 is not None
    assert "shadow" in epoch_ch25.epoch_name.lower() or "forgotten" in epoch_ch25.epoch_name.lower()
    assert "porcelain skin" in epoch_ch25.skin_complexion_tags
    assert "scrawny" in epoch_ch25.retired_negative_tags, "Scrawny was not retired into negative constraints!"
    assert "sallow skin" in epoch_ch25.retired_negative_tags, "Sallow skin was not retired into negative constraints!"

    spec_ch25 = auditor.apply_evolution_to_spec(spec_ch3, epoch_ch25)
    agent.save_to_canon("proj_evolution_01", spec_ch25, current_chapter=25)

    # --- 4. Verify Historical Temporal Lookups ---
    # Lookup at Chapter 1: Must retrieve Chapter 1 Slum Orphan
    recalled_ch1 = agent.query_historical_memory("proj_evolution_01", "Sunny", chapter_number=1)
    assert recalled_ch1 is not None
    assert "pale skin" in recalled_ch1.identity_lock.core_danbooru_tags
    assert "porcelain skin" not in recalled_ch1.identity_lock.core_danbooru_tags

    # Lookup at Chapter 10: Must retrieve Chapter 3 Slave Survivor
    recalled_ch10 = agent.query_historical_memory("proj_evolution_01", "Sunny", chapter_number=10)
    assert recalled_ch10 is not None
    assert "slave" in recalled_ch10.active_outfit.outfit_name.lower()

    # Lookup at Chapter 30: Must retrieve Chapter 25 Shadow Awakened
    recalled_ch30 = agent.query_historical_memory("proj_evolution_01", "Sunny", chapter_number=30)
    assert recalled_ch30 is not None
    assert "porcelain skin" in recalled_ch30.identity_lock.core_danbooru_tags
    assert "subtle ethereal glow" in recalled_ch30.identity_lock.core_danbooru_tags
    assert "scrawny" in recalled_ch30.identity_lock.negative_identity_tags
    assert "sallow skin" in recalled_ch30.identity_lock.negative_identity_tags


