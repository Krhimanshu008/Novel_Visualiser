"""
Novel Visualiser — Phase 0 Verification Gate Test Suite

Verifies:
1. Four-Layer SQLite database creation & constraints
2. Hierarchical Asset Inheritance (Project → Arc → Chapter → Scene)
3. Spoiler-Gate filtering at chapter boundaries
4. LangGraph production pipeline checkpointing & resume capabilities
5. FastAPI review endpoints
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database.models import (
    Base,
    Project,
    Arc,
    Chapter,
    Passage,
    Scene,
    Entity,
    EntityAlias,
    AssetVersion,
    VisualRef,
    ResearchSource,
    ResearchClaim,
    Shot,
    ChapterStatus,
    ShotStatus,
)
from app.database.session import init_db, get_session
from app.orchestrator.resolver import resolve_asset_for_scene, get_safe_knowledge
from app.orchestrator.graph import (
    build_production_graph,
    create_production_pipeline,
    run_production,
    ProductionState,
    PipelineStage,
)
from langgraph.checkpoint.sqlite import SqliteSaver


def test_01_database_tables():
    print("\n--- Test 1: Database Initialization & Tables ---")
    db_file = "test_phase0.db"
    if os.path.exists(db_file):
        os.remove(db_file)

    engine = init_db(db_file)
    with get_session() as session:
        # Create project
        proj = Project(
            name="Shadow Slave Production",
            novel_title="Shadow Slave",
            author="Guiltythree",
            description="High-accuracy anime/cinematic production of Shadow Slave",
        )
        session.add(proj)
        session.flush()

        # Add Arc
        arc = Arc(
            project_id=proj.id,
            name="First Nightmare",
            arc_number=1,
            start_chapter=1,
            end_chapter=15,
            description="The mountain of skulls and awakening",
        )
        session.add(arc)
        session.flush()

        # Add Chapter
        chap = Chapter(
            project_id=proj.id,
            arc_id=arc.id,
            chapter_number=1,
            title="A Slave in the Snow",
            raw_text="The cold was bitter. Sunny tightened his shackles...",
            word_count=1200,
            status=ChapterStatus.PARSED,
        )
        session.add(chap)
        session.commit()

        assert proj.id is not None
        assert arc.id is not None
        assert chap.id is not None
        print(f" [PASS] Project created: '{proj.name}' (ID: {proj.id[:8]}...)")
        print(f" [PASS] Arc created: '{arc.name}' (Arc #{arc.arc_number})")
        print(f" [PASS] Chapter created: '{chap.title}' (Ch #{chap.chapter_number})")

    return proj.id, arc.id, chap.id


def test_02_resolver_inheritance(proj_id, arc_id, chap_id):
    print("\n--- Test 2: Hierarchical Asset Inheritance Resolver ---")
    with get_session() as session:
        # 1. Base Canonical Entity
        sunny = Entity(
            project_id=proj_id,
            canonical_name="Sunny",
            entity_type="character",
            first_appearance_chapter=1,
        )
        session.add(sunny)
        session.flush()

        # Project canonical look (Face & hair stay constant)
        av_canon = AssetVersion(
            entity_id=sunny.id,
            level="project_canonical",
            visual_attributes={
                "age_appearance": "16-year-old slender youth",
                "face": "pale gaunt skin, sharp dark eyes",
                "hair": "messy jet-black hair",
                "physique": "lean, malnourished frame",
            },
            valid_from_chapter=1,
        )

        # Arc 1 default clothing (Ragged slave tunic)
        av_arc = AssetVersion(
            entity_id=sunny.id,
            arc_id=arc_id,
            level="arc_default",
            visual_attributes={
                "clothing": "ragged coarse grey slave tunic",
                "accessories": ["rusted iron wrist shackles"],
            },
            valid_from_chapter=1,
        )

        # Chapter 3 delta (Injured & bloodied)
        av_ch3 = AssetVersion(
            entity_id=sunny.id,
            level="chapter_delta",
            visual_attributes={
                "injuries": ["bleeding gash across left cheek"],
                "dirt": "mud-stained tunic",
                "clothing": "torn, blood-spattered slave tunic",
            },
            valid_from_chapter=3,
        )

        session.add_all([av_canon, av_arc, av_ch3])
        session.commit()

        # Resolve at Chapter 1
        res_ch1 = resolve_asset_for_scene(
            db=session,
            entity_id=sunny.id,
            arc_id=arc_id,
            chapter_number=1,
        )
        assert res_ch1["canonical"]["face"] == "pale gaunt skin, sharp dark eyes"
        assert res_ch1["arc"]["clothing"] == "ragged coarse grey slave tunic"
        assert "injuries" not in res_ch1["chapter_delta"]
        print(f" [PASS] Chapter 1: Clean slave tunic, no injuries.")

        # Resolve at Chapter 3 (Delta applied)
        res_ch3 = resolve_asset_for_scene(
            db=session,
            entity_id=sunny.id,
            arc_id=arc_id,
            chapter_number=3,
        )
        assert res_ch3["canonical"]["face"] == "pale gaunt skin, sharp dark eyes"
        assert res_ch3["arc"]["clothing"] == "torn, blood-spattered slave tunic"
        assert res_ch3["chapter_delta"]["injuries"] == ["bleeding gash across left cheek"]
        print(f" [PASS] Chapter 3: Bloody tunic & facial injury successfully inherited!")

    return sunny.id


def test_03_spoiler_gate(sunny_id):
    print("\n--- Test 3: Deterministic Spoiler-Gate Verification ---")
    with get_session() as session:
        # Add Aliases
        # Alias 1: Sunless (Known from Chapter 1)
        alias_sunless = EntityAlias(
            entity_id=sunny_id,
            alias="Sunless",
            alias_type="name",
            valid_from_chapter=1,
            reveal_chapter=1,
        )
        # Alias 2: Mongrel (Revealed in Chapter 25!)
        alias_mongrel = EntityAlias(
            entity_id=sunny_id,
            alias="Lord Mongrel",
            alias_type="title",
            valid_from_chapter=20,
            reveal_chapter=25,
        )
        session.add_all([alias_sunless, alias_mongrel])

        # Add Research Claims
        claim_initial = ResearchClaim(
            entity_id=sunny_id,
            claim_type="appearance",
            claim_text="Short youth with dark unkempt hair",
            valid_from_chapter=1,
        )
        claim_spoiler = ResearchClaim(
            entity_id=sunny_id,
            claim_type="equipment",
            claim_text="Acquires Puppeteers Shroud memory armor",
            valid_from_chapter=12,
        )
        session.add_all([claim_initial, claim_spoiler])
        session.commit()

        # Query at Chapter 1
        safe_ch1 = get_safe_knowledge(session, sunny_id, current_chapter=1)
        aliases_ch1 = [a["alias"] for a in safe_ch1["known_aliases"]]
        claims_ch1 = [c["claim_text"] for c in safe_ch1["visual_claims"]]

        assert "Sunless" in aliases_ch1
        assert "Lord Mongrel" not in aliases_ch1, "SPOILER LEAK: Mongrel leaked into Ch 1!"
        assert len(claims_ch1) == 1
        assert "Puppeteers Shroud" not in claims_ch1[0]
        print(" [PASS] Chapter 1: 'Sunless' known, 'Mongrel' and 'Puppeteers Shroud' strictly blocked.")

        # Query at Chapter 15
        safe_ch15 = get_safe_knowledge(session, sunny_id, current_chapter=15)
        aliases_ch15 = [a["alias"] for a in safe_ch15["known_aliases"]]
        claims_ch15 = [c["claim_text"] for c in safe_ch15["visual_claims"]]

        assert "Puppeteers Shroud memory armor" in claims_ch15[1]
        assert "Lord Mongrel" not in aliases_ch15, "SPOILER LEAK: Mongrel leaked before Chapter 25 reveal!"
        print(" [PASS] Chapter 15: Armor unlocked, 'Mongrel' still blocked until Chapter 25.")

        # Query at Chapter 25
        safe_ch25 = get_safe_knowledge(session, sunny_id, current_chapter=25)
        aliases_ch25 = [a["alias"] for a in safe_ch25["known_aliases"]]
        assert "Lord Mongrel" in aliases_ch25
        print(" [PASS] Chapter 25: 'Lord Mongrel' alias legally revealed.")


def test_04_checkpoint_resumption(proj_id):
    print("\n--- Test 4: LangGraph Checkpointed State Machine ---")
    checkpoint_file = "test_pipeline_checkpoints.db"
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    thread_id = "test-session-001"
    config = {"configurable": {"thread_id": thread_id}}

    # 1. Run pipeline through first wave
    with SqliteSaver.from_conn_string(checkpoint_file) as checkpointer:
        app = create_production_pipeline(checkpointer=checkpointer)

        state = ProductionState(
            project_id=proj_id,
            project_name="Shadow Slave",
            chapters_in_batch=[1, 2],
        )

        # Run to completion
        result = app.invoke(state, config)
        assert result["current_stage"] == PipelineStage.COMPLETE
        print(f" [PASS] Pipeline completed chapters {result['chapters_in_batch']}")
        print(f" [PASS] Total checkpoints saved to SQLite: {checkpoint_file}")

    # 2. Verify checkpoint database can be reopened and state inspected
    with SqliteSaver.from_conn_string(checkpoint_file) as checkpointer:
        app = create_production_pipeline(checkpointer=checkpointer)
        checkpoint_state = app.get_state(config)
        assert checkpoint_state is not None
        saved_stage = checkpoint_state.values.get("current_stage")
        print(f" [PASS] Checkpoint inspected from disk — saved stage: {saved_stage}")
        assert saved_stage == PipelineStage.COMPLETE


def test_05_fastapi_endpoints():
    print("\n--- Test 5: FastAPI Application Endpoints ---")
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)

    # Health check
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    print(f" [PASS] GET /health -> {data}")

    # List projects
    res = client.get("/projects")
    assert res.status_code == 200
    projects = res.json()
    assert len(projects) >= 1
    print(f" [PASS] GET /projects -> found {len(projects)} project(s)")


if __name__ == "__main__":
    print("==================================================")
    print("   NOVEL VISUALISER — PHASE 0 VERIFICATION GATE   ")
    print("==================================================")

    proj_id, arc_id, chap_id = test_01_database_tables()
    sunny_id = test_02_resolver_inheritance(proj_id, arc_id, chap_id)
    test_03_spoiler_gate(sunny_id)
    test_04_checkpoint_resumption(proj_id)
    test_05_fastapi_endpoints()

    print("\n==================================================")
    print("   ALL PHASE 0 VERIFICATION GATES PASSED! (5/5)   ")
    print("==================================================")

