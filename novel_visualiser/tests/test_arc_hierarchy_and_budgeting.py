"""Comprehensive verification tests for Chapter Budgeting, Asset Pre-Creation Factory, and the 4-Tier Arc & Timeline Hierarchy.

Tests:
1. Chapter shot budgeting calculation & cinematic hold times (140 WPM standard).
2. 4-Tier Arc Hierarchy (Major > Sub > Mini + Parallel Threads) and SQLite persistence.
3. Asset Pre-Creation Factory manifests (Turnarounds, Outfits, Location Master Plates).
"""

import sys
from pathlib import Path

# Add novel_visualiser to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from sqlalchemy import select
from app.database.session import get_session, init_db
from app.database.models import Project, Chapter, Arc, ArcType
from app.schemas.arc_hierarchy import ArcType as SchemaArcType, ChapterBudgetSpec
from app.agents.scene_director import calculate_chapter_budget, calculate_shot_budget
from app.agents.story_intel import ArcTimelineManager
from app.services.asset_factory import AssetPreCreationFactory
from app.schemas.visual_context import DanbooruIdentityLock, OutfitCard, LocationPlateSpec


def setup_module():
    init_db()


def test_01_chapter_budget_calculation():
    """Verify mathematical shot budgeting based on 140 WPM narration and pacing curves."""
    print("\n--- Test 1: Chapter Image Budgeting & Cinematic Pacing ---")

    # Sample scenes representing different dramatic types
    scenes = [
        {
            "scene_number": 1,
            "scene_type": "establishing",
            "passages": ["The ancient snow-swept mountain loomed under a heavy leaden sky. Cold wind howled across the barren jagged peaks of the Northern wastes. Desolation stretched as far as the eye could see."] * 3,  # ~90 words
        },
        {
            "scene_number": 2,
            "scene_type": "combat",
            "passages": ["Sunny lunged forward, blade singing through the freezing air. The shadow beast snapped its iron jaws, claws tearing into stone. Blood sprayed across the snow as he spun backward, panting and desperate!"] * 5,  # ~170 words
        },
        {
            "scene_number": 3,
            "scene_type": "dialogue",
            "passages": ["'Are you insane?' she whispered, eyes narrowing in disbelief. 'If we step past those gates, there is no returning alive.' Sunny wiped the dark blood from his cheek and smiled faintly."] * 4,  # ~140 words
        },
    ]

    budget = calculate_chapter_budget(chapter_number=1, scenes_data=scenes)

    assert isinstance(budget, ChapterBudgetSpec)
    assert budget.chapter_number == 1
    assert budget.total_words > 350
    assert budget.estimated_audio_duration_sec > 100.0  # ~400 words / 2.33 = ~170s
    assert budget.total_shot_count >= 8

    # Verify combat pacing gets more shots per word than establishing
    combat_scene = budget.scenes[1]
    estab_scene = budget.scenes[0]

    assert combat_scene.pacing_class == "fast"
    assert estab_scene.pacing_class == "slow"
    assert combat_scene.avg_hold_duration_sec < estab_scene.avg_hold_duration_sec
    assert combat_scene.target_words_per_shot < estab_scene.target_words_per_shot

    # Verify motion tier distribution
    assert budget.motion_tier_distribution["i2v_climax"] > 0
    assert budget.motion_tier_distribution["static_parallax"] > 0

    print(f" [PASS] Total Words: {budget.total_words}, Audio Duration: {budget.estimated_audio_duration_sec:.1f}s")
    print(f" [PASS] Shot Budget: {budget.total_shot_count} shots (Climax Motion: {budget.motion_tier_distribution['i2v_climax']})")


def test_02_arc_hierarchy_and_parallel_threads():
    """Verify 4-tier Arc decomposition (Major > Sub > Mini + Parallel Storylines) and SQLite persistence."""
    print("\n--- Test 2: 4-Tier Arc Hierarchy & Parallel Storylines ---")

    with get_session() as session:
        # Create test project and 30 chapters
        proj = Project(
            name="Arc Hierarchy Test Project",
            novel_title="Shadow Slave",
            author="Guiltythree",
        )
        session.add(proj)
        session.flush()

        for ch_num in range(1, 31):
            chap = Chapter(
                project_id=proj.id,
                chapter_number=ch_num,
                title=f"Chapter {ch_num}",
            )
            session.add(chap)
        session.flush()

        # Build 4-tier hierarchy
        arc_nodes = ArcTimelineManager.build_default_arc_hierarchy(
            project_id=proj.id,
            total_chapters=30,
            novel_title="Shadow Slave",
        )

        assert len(arc_nodes) > 0
        major_arc = arc_nodes[0]
        assert major_arc.arc_type == SchemaArcType.MAJOR
        assert len(major_arc.children) >= 2  # Sub-Arcs

        sub_arc1 = major_arc.children[0]
        assert sub_arc1.arc_type == SchemaArcType.SUB
        assert len(sub_arc1.children) >= 2  # Mini-Arcs

        mini_arc1 = sub_arc1.children[0]
        assert mini_arc1.arc_type == SchemaArcType.MINI

        # Persist to database
        persisted_map = ArcTimelineManager.persist_arc_hierarchy(
            db=session,
            project_id=proj.id,
            arc_nodes=arc_nodes,
        )

        assert len(persisted_map) >= 5

        # Verify DB records
        db_major = session.execute(
            select(Arc).where(Arc.id == major_arc.arc_id)
        ).scalar_one()
        assert db_major.arc_type == ArcType.MAJOR
        assert db_major.timeline_day_start == 1

        # Check leaf chapters link to the mini arc
        ch1 = session.execute(
            select(Chapter).where(Chapter.project_id == proj.id, Chapter.chapter_number == 1)
        ).scalar_one()
        assert ch1.arc_id == mini_arc1.arc_id

        # Check parallel thread in sub arc 2
        sub_arc2 = major_arc.children[1]
        assert len(sub_arc2.children) > 0
        parallel_mini = sub_arc2.children[0]
        assert parallel_mini.parallel_thread_id == "thread_sunny_solo"

        print(f" [PASS] 4-Tier Arc Tree persisted: {len(persisted_map)} nodes in SQLite hierarchy")
        print(f" [PASS] Parallel Thread verified: {parallel_mini.name} (Thread: {parallel_mini.parallel_thread_id})")


def test_03_asset_pre_creation_factory():
    """Verify AssetPreCreationFactory mints structured reference manifests for characters, outfits, and location plates."""
    print("\n--- Test 3: Asset Pre-Creation Factory ---")

    test_output_dir = Path("output/test_factory_canon")
    factory = AssetPreCreationFactory(base_data_dir=test_output_dir)

    project_id = "proj_test_vce"
    entity_id = "char_sunny"
    arc_id = "arc_sub_01"

    # 1. Mint Character Turnaround Package
    sunny_canon = DanbooruIdentityLock(
        entity_id=entity_id,
        entity_name="Sunny",
        core_danbooru_tags=["1boy", "pale skin", "messy black hair", "sharp dark onyx eyes", "slender build"],
        negative_identity_tags=["blonde hair", "muscular"],
        canonical_seed=10042,
    )

    char_manifest = factory.build_character_package_manifest(
        project_id=project_id,
        entity_id=entity_id,
        identity_lock=sunny_canon,
    )

    assert "face_front" in char_manifest["views"]
    assert "face_three_quarter" in char_manifest["views"]
    assert "face_profile" in char_manifest["views"]
    assert "expression_matrix" in char_manifest["views"]
    assert "full_body_turnaround" in char_manifest["views"]
    assert Path(char_manifest["views"]["face_front"]["path"]).parent.exists()
    print(" [PASS] Character Turnaround & Expression matrix manifest created")

    # 2. Mint Arc Outfit Package
    outfit = OutfitCard(
        outfit_id="OUTFIT-SUNNY-ARC1",
        entity_id=entity_id,
        outfit_name="Scavenger Tunic",
        valid_from_chapter=1,
        danbooru_clothing_tags=["tattered dark gray tunic", "cloth wraps", "rough boots"],
        negative_clothing_tags=["modern clothing", "hoodie"],
    )

    outfit_manifest = factory.build_outfit_package_manifest(
        project_id=project_id,
        arc_id=arc_id,
        outfit_card=outfit,
    )

    assert "outfit_front" in outfit_manifest["views"]
    assert "outfit_back" in outfit_manifest["views"]
    assert "pristine" in outfit_manifest["condition_variants"]
    assert "critical_damage" in outfit_manifest["condition_variants"]
    print(" [PASS] Arc Outfit Package with condition variants created")

    # 3. Mint Location Master Plate Package
    plate = LocationPlateSpec(
        location_id="loc_forgotten_shore_citadel",
        location_name="Sunken Citadel",
        plate_tags=["ancient submerged stone pillars", "dark obsidian water", "heavy fog"],
        negative_environment_tags=["modern buildings", "trees"],
    )

    loc_manifest = factory.build_location_plates_manifest(
        project_id=project_id,
        arc_id=arc_id,
        plate_spec=plate,
    )

    assert "master_plate_angle_A" in loc_manifest["views"]
    assert "controlnet_depth_A" in loc_manifest["views"]
    print(" [PASS] Location Master Plates Package with ControlNet depth guide created")

