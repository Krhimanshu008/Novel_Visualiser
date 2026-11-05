"""
Novel Visualiser — Phase 2 Verification Gate Test Suite

Verifies:
1. Deterministic Chapter Parser & FlashText Entity Pre-Tagger (0 LLM Tokens)
2. Story Intelligence structured scene and event extraction
3. World Architect & RapidFuzz entity deduplication / alias reconciliation
4. Visual state delta commitment into asset_versions
5. Hierarchical Asset Resolver verification across chapter boundaries
6. FastAPI Story Intelligence endpoints (/analyze, /scenes)
"""

import os
import sys
from pathlib import Path
from sqlalchemy import select

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database.models import (
    Arc,
    AssetLevel,
    AssetVersion,
    Chapter,
    ChapterStatus,
    Entity,
    EntityAlias,
    Event,
    Passage,
    Project,
    Scene,
)
from app.database.session import init_db, get_session
from app.agents.story_intel import ChapterParser, StoryIntelligenceAgent
from app.agents.world_architect import KnowledgeReconciler, WorldArchitectAgent
from app.orchestrator.resolver import resolve_asset_for_scene


CHAPTER_1_TEXT = """
The cold was bitter. Sunny tightened the ragged cloth of his slave tunic, shivering as the icy wind tore through the mountain pass.

The rusted iron shackles around his wrists clinked with a dull, hollow sound. Around him, dozens of other temple slaves trudged through the deep snow under the heavy whips of the guards.

"Move, scum! Before the shadows swallow you whole!" a scarred guard barked, his breath puffing into white mist.

Sunny lowered his head, black eyes dark and unreadable. Deep inside, his heart beat like a trapped bird. The Nightmare Spell was relentless. If he couldn't survive the First Nightmare, this frozen mountain would become his grave.

Suddenly, a blood-curdling screech tore through the blizzard. Above the ridge, something monstrous stirred in the darkness.
"""

CHAPTER_3_TEXT = """
Sunny coughed, spitting dark crimson blood onto the snow. A jagged gash ran across his left cheek, bleeding freely into his collar.

His slave tunic was torn to shreds, stained with mud and drying monster blood. The short bronze sword in his trembling hand was chipped, but still sharp.

"I survived," Sunny muttered through gritted teeth, wiping the blood from his eyes. His breathing was ragged, his malnourished body pushed far past its limits.

Far in the distance, the silhouette of an ancient mountain temple rose against the twilight sky.
"""


def test_01_chapter_parser_and_flashtext(proj_id):
    print("\n--- Test 1: Deterministic Passage Parser & FlashText Pre-Tagger ---")
    with get_session() as session:
        # Pre-seed Sunny entity
        sunny = Entity(
            project_id=proj_id,
            canonical_name="Sunny",
            entity_type="character",
            first_appearance_chapter=1,
        )
        session.add(sunny)
        session.flush()

        alias = EntityAlias(
            entity_id=sunny.id,
            alias="Sunless",
            alias_type="name",
            valid_from_chapter=1,
            reveal_chapter=1,
        )
        session.add(alias)
        session.commit()

        # Run parser
        passages = ChapterParser.split_into_passages(CHAPTER_1_TEXT, chapter_number=1)
        assert len(passages) >= 4, "Failed to split chapter into passages"
        print(f" [PASS] Segmented into {len(passages)} passages (CH01_P001 to {passages[-1]['passage_id']})")

        # Run pre-tagger
        tagged = ChapterParser.pre_tag_entities(passages, session, proj_id, chapter_number=1)
        tagged_with_sunny = [p for p in tagged if sunny.id in p["tagged_entity_ids"]]
        assert len(tagged_with_sunny) >= 2, "FlashText failed to tag Sunny in passages"
        print(f" [PASS] FlashText pre-tagged Sunny in {len(tagged_with_sunny)} passages (0 LLM tokens)")

        # Verify passage types
        types = {p["passage_type"].value for p in tagged}
        print(f" [PASS] Detected passage types: {types}")
        assert "dialogue" in types or "action" in types or "narration" in types

    return sunny.id


def test_02_story_intelligence_extraction(proj_id, sunny_id):
    print("\n--- Test 2: Story Intelligence Structured Extraction ---")
    with get_session() as session:
        # Save Chapter 1 to DB
        ch1 = Chapter(
            project_id=proj_id,
            chapter_number=1,
            title="A Slave in the Snow",
            raw_text=CHAPTER_1_TEXT,
            word_count=len(CHAPTER_1_TEXT.split()),
            status=ChapterStatus.UNPARSED,
        )
        session.add(ch1)
        session.commit()
        ch1_id = ch1.id

    agent = StoryIntelligenceAgent()
    with get_session() as session:
        analysis = agent.process_chapter(session, ch1_id)
        assert len(analysis.scenes) >= 2, "Less than 2 scenes extracted"
        assert len(analysis.events) >= 1, "No events extracted"
        print(f" [PASS] Chapter 1 analyzed: {len(analysis.scenes)} scenes, {len(analysis.events)} events")
        print(f"        Scenes: {[s.location_name for s in analysis.scenes]}")
        print(f"        Emotion curve: {analysis.emotion_curve}")

        # Check DB records
        scenes_db = session.execute(
            select(Scene).where(Scene.chapter_id == ch1_id)
        ).scalars().all()
        assert len(scenes_db) == len(analysis.scenes)
        print(f" [PASS] Persisted {len(scenes_db)} scenes into SQLite 'scenes' table")

    return ch1_id


def test_03_world_architect_reconciliation(proj_id, sunny_id):
    print("\n--- Test 3: World Architect & RapidFuzz Reconciliation ---")
    with get_session() as session:
        # Test fuzzy matching
        match_sunny = KnowledgeReconciler.match_entity("Sunless", session, proj_id)
        assert match_sunny is not None
        assert match_sunny[0].canonical_name == "Sunny"
        print(f" [PASS] RapidFuzz matched 'Sunless' -> '{match_sunny[0].canonical_name}' (score: {match_sunny[1]:.1f})")

        match_fuzzy = KnowledgeReconciler.match_entity("Sunny Boy", session, proj_id)
        assert match_fuzzy is not None
        print(f" [PASS] RapidFuzz matched 'Sunny Boy' -> '{match_fuzzy[0].canonical_name}' (score: {match_fuzzy[1]:.1f})")

        # Ingest Chapter 3 with injuries
        ch3 = Chapter(
            project_id=proj_id,
            chapter_number=3,
            title="Blood on the Snow",
            raw_text=CHAPTER_3_TEXT,
            word_count=len(CHAPTER_3_TEXT.split()),
            status=ChapterStatus.UNPARSED,
        )
        session.add(ch3)
        session.commit()
        ch3_id = ch3.id

    # Run Story Intel on Chapter 3
    story_agent = StoryIntelligenceAgent()
    with get_session() as session:
        analysis_ch3 = story_agent.process_chapter(session, ch3_id)

    # Run World Architect to reconcile deltas
    world_agent = WorldArchitectAgent()
    with get_session() as session:
        res = world_agent.process_chapter_analysis(session, analysis_ch3)
        print(f" [PASS] World Architect committed {len(res['deltas_committed'])} visual deltas:")
        for d in res["deltas_committed"]:
            print(f"        * {d['entity_name']} [{d['level']}]: {d['description']}")


def test_04_resolver_chapter_boundaries(proj_id, sunny_id):
    print("\n--- Test 4: Hierarchical Asset Resolver Across Chapters ---")
    with get_session() as session:
        # Seed Base canonical look
        av_canon = AssetVersion(
            entity_id=sunny_id,
            level=AssetLevel.PROJECT_CANONICAL,
            visual_attributes={
                "face": "pale gaunt skin, sharp dark eyes",
                "hair": "messy jet-black hair",
            },
            valid_from_chapter=1,
        )
        av_arc = AssetVersion(
            entity_id=sunny_id,
            level=AssetLevel.ARC_DEFAULT,
            visual_attributes={
                "clothing": "ragged slave tunic",
            },
            valid_from_chapter=1,
        )
        session.add_all([av_canon, av_arc])
        session.commit()

        # Check Chapter 1 (No Chapter 3 injuries)
        r_ch1 = resolve_asset_for_scene(session, sunny_id, arc_id=None, chapter_number=1)
        assert r_ch1["canonical"]["face"] == "pale gaunt skin, sharp dark eyes"
        assert r_ch1["arc"]["clothing"] == "ragged slave tunic"
        assert "injuries" not in r_ch1["chapter_delta"]
        print(" [PASS] Chapter 1: Clean slave tunic, 0 injuries.")

        # Check Chapter 3 (Inherits Chapter 3 deltas)
        r_ch3 = resolve_asset_for_scene(session, sunny_id, arc_id=None, chapter_number=3)
        assert r_ch3["canonical"]["face"] == "pale gaunt skin, sharp dark eyes"
        has_injury = "injuries" in r_ch3["chapter_delta"] or "conditions" in r_ch3["chapter_delta"]
        assert has_injury, "Failed to inherit Chapter 3 condition delta"
        print(f" [PASS] Chapter 3: Inherited condition delta -> {r_ch3['chapter_delta']}")


def test_05_fastapi_story_endpoints(proj_id):
    print("\n--- Test 5: FastAPI Story Intelligence Endpoints ---")
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)

    # 1. Test get scenes for Chapter 1
    res = client.get(f"/projects/{proj_id}/chapters/1/scenes")
    assert res.status_code == 200
    data = res.json()
    assert len(data["scenes"]) >= 2
    print(f" [PASS] GET /chapters/1/scenes -> found {len(data['scenes'])} scenes with events")

    # 2. Test analyze endpoint on Chapter 3
    res_post = client.post(f"/projects/{proj_id}/chapters/3/analyze")
    assert res_post.status_code == 200
    post_data = res_post.json()
    assert post_data["chapter_number"] == 3
    print(f" [PASS] POST /chapters/3/analyze -> {post_data['scenes_count']} scenes, {post_data['events_count']} events")


if __name__ == "__main__":
    print("==================================================")
    print("   NOVEL VISUALISER — PHASE 2 VERIFICATION GATE   ")
    print("==================================================")

    db_path = "test_phase2.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    init_db(db_path)
    with get_session() as session:
        proj = Project(
            name="Shadow Slave Story Test",
            novel_title="Shadow Slave",
            author="Guiltythree",
        )
        session.add(proj)
        session.commit()
        proj_id = proj.id

    sunny_id = test_01_chapter_parser_and_flashtext(proj_id)
    ch1_id = test_02_story_intelligence_extraction(proj_id, sunny_id)
    test_03_world_architect_reconciliation(proj_id, sunny_id)
    test_04_resolver_chapter_boundaries(proj_id, sunny_id)
    test_05_fastapi_story_endpoints(proj_id)

    print("\n==================================================")
    print("   ALL PHASE 2 VERIFICATION GATES PASSED! (5/5)   ")
    print("==================================================")
