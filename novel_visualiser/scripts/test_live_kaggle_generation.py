"""
Novel Visualiser — Live Remote ComfyUI Image Generation Test (Kaggle T4 GPU)

Executes end-to-end image generation on the remote Kaggle GPU via Cloudflare tunnel:
1. Submits WF-003 Master Location Plate prompt payload
2. Waits for remote execution and downloads generated PNG to output/generated_images/
3. Submits WF-001 Character Concept for Sunny
4. Evaluates generated images through the 3-Gate QC Funnel
5. Locks canonical visual reference into SQLite visual_refs
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.database.session import init_db, get_session
from app.services.comfyui_client import ComfyUIClient, ComfyUIWorkflowCompiler
from app.services.qc_service import QCService
from app.database.models import Entity, EntityType, RefType


def main():
    print("=" * 65)
    print(" [LIVE TEST] NOVEL VISUALISER - REMOTE GPU GENERATION")
    print("=" * 65)

    settings = get_settings()
    client = ComfyUIClient()

    # 1. Health check
    print(f"\n[1/5] Checking remote connection to {client.base_url}...")
    health = client.check_health()
    if health.get("status") != "online":
        print(f"[FAIL] Connection failed: {health}")
        return

    gpu_name = health.get("gpu_name")
    vram = health.get("vram_free_gb")
    print(f"[OK] Connected to remote GPU: {gpu_name} ({vram} GB VRAM free)")

    # 2. Compile & Submit WF-003 Location Plate
    print("\n[2/5] Compiling and generating Location Plate (WF-003)...")
    location_prompt = (
        "masterpiece, best quality, highly detailed, ancient snow-covered mountain pass, "
        "desolate dark fantasy landscape, frozen slave caravan path in blizzard, "
        "ominous black stone peaks, cold twilight luminescence, dramatic cinematic lighting, "
        "Unreal Engine 5 digital art style"
    )
    payload_loc = ComfyUIWorkflowCompiler.compile_location_plate(
        positive_prompt=location_prompt,
        seed=421099,
        width=1216,
        height=832,
        steps=25,
        cfg=5.5,
        checkpoint="Wai_Illustrious_v16.safetensors",
    )

    t0 = time.time()
    print(" -> Submitting job to remote Kaggle GPU queue...")
    downloaded_locs = client.generate_image_sync(payload_loc)
    duration_loc = time.time() - t0

    if not downloaded_locs:
        print("[FAIL] No images returned for Location Plate")
        return

    loc_path = downloaded_locs[0]
    print(f"[OK] Location Plate generated in {duration_loc:.1f}s!")
    print(f"   Saved to: {loc_path} ({loc_path.stat().st_size // 1024} KB)")

    # 3. Compile & Submit WF-001 Sunny Character Sheet
    print("\n[3/5] Compiling and generating Sunny Character Concept (WF-001)...")
    char_prompt = (
        "masterpiece, best quality, highly detailed, 1boy, slender pale youth, "
        "messy jet-black hair, sharp dark eyes, wearing ragged dark gray tunic, "
        "alert guarded expression, cold shadowy aura, anime dark fantasy digital art style, "
        "clean background, character portrait turnaround"
    )
    payload_char = ComfyUIWorkflowCompiler.compile_character_design(
        positive_prompt=char_prompt,
        seed=108422,
        width=1024,
        height=1024,
        steps=25,
        cfg=5.5,
        checkpoint="Wai_Illustrious_v16.safetensors",
    )

    t0 = time.time()
    print(" -> Submitting job to remote Kaggle GPU queue...")
    downloaded_chars = client.generate_image_sync(payload_char)
    duration_char = time.time() - t0

    if not downloaded_chars:
        print("[FAIL] No images returned for Character Sheet")
        return

    char_path = downloaded_chars[0]
    print(f"[OK] Character Concept generated in {duration_char:.1f}s!")
    print(f"   Saved to: {char_path} ({char_path.stat().st_size // 1024} KB)")

    # 4. Evaluate through 3-Gate QC Funnel
    print("\n[4/5] Running 3-Gate Quality Control on generated images...")
    qc = QCService(blur_threshold=60.0, gate2_threshold=0.50)

    # QC Location Plate
    g1_loc = qc.evaluate_gate1(loc_path)
    g3_loc = qc.evaluate_gate3(loc_path, checklist=["snow", "mountain", "twilight"])
    print(f" -> Location Gate 1 (Blur: {g1_loc['blur_score']:.1f}, Exposure: {g1_loc['exposure_ok']}): {'PASS' if g1_loc['passed'] else 'FAIL'}")
    print(f" -> Location Gate 3 (Story Checklist: {g3_loc['checklist_results']}): {'PASS' if g3_loc['passed'] else 'FAIL'}")

    # QC Character Concept
    g1_char = qc.evaluate_gate1(char_path)
    g3_char = qc.evaluate_gate3(char_path, checklist=["character_visible", "shadow"])
    print(f" -> Character Gate 1 (Blur: {g1_char['blur_score']:.1f}, Exposure: {g1_char['exposure_ok']}): {'PASS' if g1_char['passed'] else 'FAIL'}")
    print(f" -> Character Gate 3 (Story Checklist: {g3_char['checklist_results']}): {'PASS' if g3_char['passed'] else 'FAIL'}")

    # 5. Lock Canonical Reference into SQLite
    print("\n[5/5] Locking canonical reference into SQLite visual_refs...")
    init_db()
    with get_session() as session:
        # Check if entity Sunny exists, or create
        ent = session.query(Entity).filter(Entity.canonical_name == "Sunny").first()
        if not ent:
            from app.database.models import Project
            proj = session.query(Project).first()
            if not proj:
                proj = Project(name="Shadow Slave Production", novel_title="Shadow Slave", author="Guiltythree")
                session.add(proj)
                session.flush()
            ent = Entity(project_id=proj.id, canonical_name="Sunny", entity_type=EntityType.CHARACTER)
            session.add(ent)
            session.flush()

        locked_ref = client.lock_canonical_visual_reference(
            db=session,
            entity_id=ent.id,
            image_path=str(char_path),
            ref_type=RefType.TURNAROUND_SHEET,
            seed=108422,
        )
        print(f"[OK] Canonical reference locked successfully! (VisualRef ID: {locked_ref.id[:8]}...)")

    print("\n" + "=" * 65)
    print(" [SUCCESS] LIVE REMOTE GPU GENERATION TEST COMPLETED SUCCESSFULLY!")
    print(f"  Location Plate: {loc_path}")
    print(f"  Character Sheet: {char_path}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
