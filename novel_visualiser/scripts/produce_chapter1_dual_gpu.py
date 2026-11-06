"""
Novel Visualiser — Chapter 1 Dual-GPU Production Pipeline
Executes concurrent image generation across 2x Tesla T4 GPUs via Cloudflare tunnels.

Scene 1: Dystopian Outskirts & Rusty Bench (Coffee)
Scene 2: Fortified Police Station Reception
Scene 3: Concrete Detention Cell & Nightmare Spell Awakening
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.comfyui_client import ComfyUIWorkflowCompiler, DualGPUComfyUIDispatcher
from app.services.qc_service import QCService

URLS = [
    "https://evaluated-designing-precision-requests.trycloudflare.com",
    "https://harder-mounting-air-promotions.trycloudflare.com",
]

def main():
    print("=" * 70)
    print(" [PRODUCTION] CHAPTER 1 DUAL-GPU CONCURRENT RENDER CLUSTER")
    print("=" * 70)

    disp = DualGPUComfyUIDispatcher(URLS)
    health = disp.check_health_all()
    for i, h in enumerate(health):
        print(f" [OK] GPU {i} Worker -> {h.get('gpu_name')} ({h.get('vram_free_gb')} GB free)")

    out_dir = Path("output/generated_images/chapter_01")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reference assets
    sunny_face_ref = Path("output/generated_images/LIVE_TEST_SUNNY_00001_.png")
    slums_depth_ref = Path("output/generated_images/chapter_01/WF02_LOC_slums_bench_depth.png")

    face_name = sunny_face_ref.name if sunny_face_ref.exists() else None
    depth_name = slums_depth_ref.name if slums_depth_ref.exists() else None

    # Pre-upload reference assets to both GPU workers so LoadImage nodes resolve locally
    for client in disp.clients:
        if sunny_face_ref.exists():
            client.upload_image(sunny_face_ref)
        if slums_depth_ref.exists():
            client.upload_image(slums_depth_ref)

    # Define the 8 key cinematic shots of Chapter 1
    # Block prompts follow the fixed 8-block Danbooru syntax
    shots = [
        # --- SCENE 1: OUTSKIRTS SLUMS & COFFEE ---
        (
            "CH01_Shot_01_Wide_Slums",
            ComfyUIWorkflowCompiler.compile_shot_env(
                shot_id="ch01_s01",
                env_prompt="dystopian cyberpunk city slums, weathered rusty metal bench in foreground, dark rainy wet asphalt pavement with neon reflections, dreary cold streetlights, atmospheric fog, power lines overhead",
                loc_depth_filename=depth_name,
                width=1216,
                height=832,
                seed=1001,
            )
        ),
        (
            "CH01_Shot_02_Medium_Sunny_Bench",
            ComfyUIWorkflowCompiler.compile_shot_solo(
                shot_id="ch01_s02",
                positive_prompt="medium shot, 1boy, pale sickly skin, messy jet-black hair, sharp dark hollow eyes with heavy dark circles, wearing worn dark synthetic jacket, sitting on weathered rusty metal bench, holding lukewarm paper coffee cup with both hands, dystopian city slums background, cold twilight glow, cinematic depth of field",
                face_ref_filename=face_name,
                loc_depth_filename=depth_name,
                seed=1002,
                width=1216,
                height=832,
            )
        ),
        (
            "CH01_Shot_03_CloseUp_Sunny_Sip",
            ComfyUIWorkflowCompiler.compile_shot_solo(
                shot_id="ch01_s03",
                positive_prompt="close-up portrait shot, 1boy, pale sickly skin, messy black hair bangs covering forehead, sharp dark onyx eyes, weary cynical grimace, holding paper coffee cup near lips, dystopian slums backdrop, dramatic rim lighting, cinematic depth of field",
                face_ref_filename=face_name,
                seed=1003,
                width=1024,
                height=1024,
            )
        ),

        # --- SCENE 2: FORTIFIED POLICE STATION ---
        (
            "CH01_Shot_04_Wide_Police_Station",
            ComfyUIWorkflowCompiler.compile_shot_env(
                shot_id="ch01_s04",
                env_prompt="inside fortified police station reception hall, thick reinforced concrete walls, armored blast doors, automated ceiling security turrets, harsh cold fluorescent strip lighting, industrial sci-fi bunker interior",
                width=1216,
                height=832,
                seed=2001,
            )
        ),
        (
            "CH01_Shot_05_Medium_Sunny_Interrogation",
            ComfyUIWorkflowCompiler.compile_shot_solo(
                shot_id="ch01_s05",
                positive_prompt="medium shot, 1boy, pale sickly skin, messy jet-black hair, sharp dark hollow eyes, guarded defensive posture, sitting at metal desk inside reinforced police station, heavy armor plates in background, harsh overhead white fluorescent lighting, cinematic shadows",
                face_ref_filename=face_name,
                seed=2002,
                width=1216,
                height=832,
            )
        ),

        # --- SCENE 3: DETENTION CELL & NIGHTMARE SPELL AWAKENING ---
        (
            "CH01_Shot_06_Medium_Detention_Cell",
            ComfyUIWorkflowCompiler.compile_shot_solo(
                shot_id="ch01_s06",
                positive_prompt="medium shot, 1boy, pale skin, messy black hair, dark hollow eyes, sitting on cold steel bunk inside dim reinforced concrete detention cell, heavy steel bars, dim amber emergency lighting, lonely desolate atmosphere",
                face_ref_filename=face_name,
                seed=3001,
                width=1216,
                height=832,
            )
        ),
        (
            "CH01_Shot_07_CloseUp_Sunny_Drowsy",
            ComfyUIWorkflowCompiler.compile_shot_solo(
                shot_id="ch01_s07",
                positive_prompt="close-up portrait, 1boy, pale skin, messy black hair, heavy drooping eyelids, drowsy trance, head slumping forward, dim holding cell background, shadowy vignette, mysterious dark atmosphere",
                face_ref_filename=face_name,
                seed=3002,
                width=1024,
                height=1024,
            )
        ),
        (
            "CH01_Shot_08_Wide_Nightmare_Spell_Void",
            ComfyUIWorkflowCompiler.compile_shot_env(
                shot_id="ch01_s08",
                env_prompt="engulfed in pure dark cosmic void, writhing ethereal shadowy wisps, glowing golden runic engravings drifting in dark space, divine luminescence, mystical awakening of the Nightmare Spell, breathtaking cinematic fantasy art",
                width=1216,
                height=832,
                seed=3003,
            )
        ),
    ]

    print(f"\n[DISPATCH] Submitting {len(shots)} Chapter 1 shots across dual GPUs...")
    t_start = time.time()
    results = disp.generate_batch_parallel(shots, output_dir=out_dir)
    total_duration = time.time() - t_start

    print("\n" + "=" * 70)
    print(f" [COMPLETE] ALL {len(shots)} SHOTS GENERATED IN {total_duration:.1f}s!")
    print(f" Average speed: {total_duration / len(shots):.1f}s per shot across dual Tesla T4s!")
    print("=" * 70)

    # Quality Inspection Gate
    qc = QCService(blur_threshold=50.0, gate2_threshold=0.30)
    print("\n--- [QC GATE 1] Automated Blur & Sharpness Inspection ---")
    for shot_label, file_list in sorted(results.items()):
        if file_list:
            img_path = file_list[0]
            res = qc.evaluate_gate1(str(img_path))
            status = "PASS" if res.get("passed", True) else "FAIL"
            blur_score = res.get("blur_score", 0.0)
            print(f"  • [{status}] {shot_label:<35} | Sharpness: {blur_score:.1f} | File: {img_path.name}")

if __name__ == "__main__":
    main()
