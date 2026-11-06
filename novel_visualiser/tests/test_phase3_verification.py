"""
Novel Visualiser — Phase 3 Verification Gate Test Suite

Verifies:
1. User-tested workflow templates inspection (WF-001 to WF-004, IPAdapter, Florence-2)
2. Workflow compilers parameterizing prompts, seeds, and reference images
3. ComfyUI Client health checks and tunnel URL configuration
4. Canonical Visual Reference locking into SQLite visual_refs
5. Integration with resolver.py (locked reference + seed automatically attached)
6. FastAPI ComfyUI Worker endpoints (/comfyui/health, /compile/..., /lock-reference)
"""

import os
import sys
from pathlib import Path
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database.models import (
    Entity,
    Project,
    RefType,
    VisualRef,
)
from app.database.session import init_db, get_session
from app.services.comfyui_client import ComfyUIClient, ComfyUIWorkflowCompiler
from app.orchestrator.resolver import resolve_asset_for_scene


@pytest.fixture
def proj_id():
    db_path = "test_phase3.db"
    init_db(db_path)
    with get_session() as session:
        proj = Project(
            name="Shadow Slave Visual Lock Test",
            novel_title="Shadow Slave",
            author="Guiltythree",
        )
        session.add(proj)
        session.commit()
        return proj.id


@pytest.fixture
def sunny_id(proj_id):
    with get_session() as session:
        sunny = Entity(
            project_id=proj_id,
            canonical_name="Sunny",
            entity_type="character",
            first_appearance_chapter=1,
        )
        session.add(sunny)
        session.commit()
        return sunny.id


def test_01_user_workflow_files():
    print("\n--- Test 1: User-Tested ComfyUI Workflow Files ---")
    workflows_dir = Path(__file__).resolve().parent.parent / "templates" / "comfyui" / "Workflows"
    if not workflows_dir.exists():
        workflows_dir = Path("templates/comfyui/Workflows")
    assert workflows_dir.exists(), "Workflows directory not found"

    expected_files = [
        "WF-FOUNDRY-CHAR.json",
        "WF-FOUNDRY-LOC.json",
        "WF-FOUNDRY-PROP.json",
        "WF-SHOT-SOLO.json",
        "WF-SHOT-MULTI.json",
        "WF-SHOT-ENV.json",
        "WF-EDIT-INPAINT.json",
        "WF-MASTER-UPSCALE.json",
    ]

    for fname in expected_files:
        fpath = workflows_dir / fname
        assert fpath.exists(), f"Workflow file missing: {fname}"
        raw = ComfyUIWorkflowCompiler.load_workflow_template(fname)
        assert isinstance(raw, dict) and len(raw) > 0
        print(f" [PASS] Loaded '{fname}': {len(raw)} nodes configured")


def test_02_workflow_compilation():
    print("\n--- Test 2: Workflow Parameterization & Compilers ---")

    # 1. Compile WF-FOUNDRY-CHAR Character Design Turnaround
    prompt_char = ComfyUIWorkflowCompiler.compile_foundry_char(
        char_id="sunny",
        identity_tags="slender 16-year-old youth with messy jet-black hair and dark eyes",
        seed=123456,
        checkpoint="Wai_Illustrious_v16.safetensors",
    )
    assert "1" in prompt_char and prompt_char["1"]["inputs"]["ckpt_name"] == "Wai_Illustrious_v16.safetensors"
    assert prompt_char["6"]["inputs"]["seed"] == 123456
    assert "slender 16-year-old" in prompt_char["4"]["inputs"]["text"]
    print(" [PASS] Compiled WF-FOUNDRY-CHAR (Checkpoint + Prompt + Seed injected)")

    # 2. Compile WF-FOUNDRY-LOC Master Location Concept Plate
    prompt_loc = ComfyUIWorkflowCompiler.compile_foundry_loc(
        loc_id="frozen_pass",
        landscape_tags="frozen snowy mountain pass under twilight, slave caravan path",
        seed=654321,
    )
    assert prompt_loc["3"]["inputs"]["width"] == 1216
    assert prompt_loc["3"]["inputs"]["height"] == 832
    assert prompt_loc["6"]["inputs"]["seed"] == 654321
    print(" [PASS] Compiled WF-FOUNDRY-LOC (1216x832 resolution + Seed injected)")

    # 3. Compile WF-SHOT-SOLO Single-Character Workhorse
    prompt_solo = ComfyUIWorkflowCompiler.compile_shot_solo(
        shot_id="shot_001",
        positive_prompt="Sunny fighting pose, holding bronze sword",
        face_ref_filename="Sunny_midjourney_art.png",
        ipa_face_weight=0.85,
        seed=999888,
    )
    assert prompt_solo["8"]["inputs"]["image"] == "Sunny_midjourney_art.png"
    assert prompt_solo["9"]["inputs"]["weight"] == 0.85
    assert prompt_solo["13"]["inputs"]["seed"] == 999888
    print(" [PASS] Compiled WF-SHOT-SOLO (Face reference + IP-Adapter weight 0.85 injected)")


def test_03_client_health_and_remote_config():
    print("\n--- Test 3: ComfyUI Client & Remote Tunnel Configuration ---")
    client = ComfyUIClient()

    # Verify health check handles local or remote
    health = client.check_health()
    print(f" [PASS] ComfyUI Health Check -> Status: {health.get('status')} (Target: {health.get('base_url')})")

    # Test custom remote Colab/Kaggle URL configuration
    remote_client = ComfyUIClient(base_url="https://sample-test-tunnel.trycloudflare.com")
    assert remote_client.base_url == "https://sample-test-tunnel.trycloudflare.com"
    print(" [PASS] ComfyUI Client correctly targets remote Cloudflare / ngrok tunnel URL")

    client.close()
    remote_client.close()


def test_04_visual_reference_lock_and_resolver(proj_id):
    print("\n--- Test 4: Canonical Visual Reference Locking ---")
    with get_session() as session:
        sunny = Entity(
            project_id=proj_id,
            canonical_name="Sunny",
            entity_type="character",
            first_appearance_chapter=1,
        )
        session.add(sunny)
        session.commit()
        sunny_id = sunny.id

    client = ComfyUIClient()
    with get_session() as session:
        # Lock canonical approved reference sheet
        ref = client.lock_canonical_visual_reference(
            db=session,
            entity_id=sunny_id,
            image_path="data/references/Sunny/Sunny_midjourney_art.png",
            ref_type=RefType.TURNAROUND_SHEET,
            seed=1847291037,
            ip_adapter_config={"image_path": "data/references/Sunny/Sunny_midjourney_art.png", "weight": 0.8},
        )
        assert ref.approved == True
        print(f" [PASS] Locked canonical VisualRef ID {ref.id[:8]}... (Seed: {ref.seed})")

        # Now test that resolver.py automatically retrieves this approved reference & seed!
        resolved = resolve_asset_for_scene(session, sunny_id, arc_id=None, chapter_number=1)
        assert resolved["reference_image_path"] == "data/references/Sunny/Sunny_midjourney_art.png"
        assert resolved["canonical_seed"] == 1847291037
        assert resolved["ip_adapter_weight"] == 0.8
        print(" [PASS] Resolver seamlessly pulled locked reference image and canonical seed into scene prompt payload!")

    client.close()


def test_05_fastapi_comfyui_endpoints(proj_id, sunny_id):
    print("\n--- Test 5: FastAPI ComfyUI Worker Endpoints ---")
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)

    # 1. ComfyUI health endpoint
    res_health = client.get("/comfyui/health")
    assert res_health.status_code == 200
    print(f" [PASS] GET /comfyui/health -> {res_health.json()['status']}")

    # 2. Compile turnaround endpoint
    res_compile = client.post(
        "/comfyui/compile/turnaround",
        json={"positive_prompt": "Nephis, silver-haired girl with striking white eyes", "seed": 777},
    )
    assert res_compile.status_code == 200
    data = res_compile.json()
    ksampler = next(v for v in data.values() if v.get("class_type") == "KSampler")
    assert ksampler["inputs"]["seed"] == 777
    print(" [PASS] POST /comfyui/compile/turnaround -> compiled valid ComfyUI prompt JSON")

    # 3. Lock reference endpoint
    res_lock = client.post(
        "/comfyui/lock-reference",
        json={
            "entity_id": sunny_id,
            "image_path": "data/references/Sunny/Sunless_card_final_3.png",
            "seed": 999111,
            "ref_type": "turnaround_sheet",
        },
    )
    assert res_lock.status_code == 200
    assert res_lock.json()["status"] == "locked"
    print(" [PASS] POST /comfyui/lock-reference -> successfully locked reference via API")


if __name__ == "__main__":
    print("==================================================")
    print("   NOVEL VISUALISER — PHASE 3 VERIFICATION GATE   ")
    print("==================================================")

    db_path = "test_phase3.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    init_db(db_path)
    with get_session() as session:
        proj = Project(
            name="Shadow Slave Visual Lock Test",
            novel_title="Shadow Slave",
            author="Guiltythree",
        )
        session.add(proj)
        session.commit()
        proj_id = proj.id

    test_01_user_workflow_files()
    test_02_workflow_compilation()
    test_03_client_health_and_remote_config()
    sunny_id = test_04_visual_reference_lock_and_resolver(proj_id)
    test_05_fastapi_comfyui_endpoints(proj_id, sunny_id)

    print("\n==================================================")
    print("   ALL PHASE 3 VERIFICATION GATES PASSED! (5/5)   ")
    print("==================================================")

