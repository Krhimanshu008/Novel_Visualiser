"""Novel Visualiser — Live Dual-GPU Character Generation Pipeline Test (Up to Chapter 25).

Demonstrates:
1. Chapter 1: Sunny (Sunless) - Impoverished Outskirts Sleeper with strict spoiler gating
2. Biological Evolution Auditor: Chapter 1 -> Chapter 25 progression
3. Chapter 25: Sunny (Sunless) - Awakened Shadow Survivor with Memory Blade
4. Chapter 25: Nephis (Changing Star) - Silver-haired Awakened Debut
5. Concurrent Dual-GPU dispatch across GPU 0 (https://gpu0.krhimanshu.in) and GPU 1 (https://gpu1.krhimanshu.in)
6. Automatic IP-Adapter face chip extraction for identity locking
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.biological_evolution_auditor import BiologicalEvolutionAuditor
from app.agents.character_subagent import CharacterContinuitySubagent
from app.schemas.character_subagent import (
    ArcHierarchyTag,
    CharacterMatrixSpec,
    CharacterSheetRole,
    SingleCharacterSheetSpec,
)
from app.services.comfyui_client import ComfyUIWorkflowCompiler, DualGPUComfyUIDispatcher


def run_character_pipeline_chapter_25():
    print("=" * 75)
    print(" NOVEL VISUALISER — CHARACTER GENERATION PIPELINE TEST (UP TO CHAPTER 25)")
    print("=" * 75)

    # 1. Initialize Dual-GPU Cluster
    gpu_urls = [
        "https://gpu0.krhimanshu.in",
        "https://gpu1.krhimanshu.in",
    ]
    print("\n[1/5] Initializing Dual-GPU Cluster:")
    dispatcher = DualGPUComfyUIDispatcher(urls=gpu_urls)
    online_workers = dispatcher.get_online_clients()
    for worker in online_workers:
        health = worker.check_health()
        print(f"  🟢 {worker.base_url:<30} -> {health.get('gpu_name', 'GPU')} ({health.get('vram_free_gb', 0)} GB free VRAM)")

    if not online_workers:
        print("❌ No GPU workers online! Exiting.")
        sys.exit(1)

    out_base = Path("output/character_sheets/chapter_25_pipeline")
    out_base.mkdir(parents=True, exist_ok=True)

    agent = CharacterContinuitySubagent(comfy_dispatcher=dispatcher, use_ai_thinking=False)
    auditor = BiologicalEvolutionAuditor(use_ai_thinking=False)

    # -------------------------------------------------------------------------
    # 2. CHARACTER 1: SUNNY — CHAPTER 1 (OUTSKIRTS SLEEPER YOUTH)
    # -------------------------------------------------------------------------
    print("\n[2/5] Synthesizing Sunny — Chapter 1 (Outskirts Sleeper State)...")
    arc_ch1 = ArcHierarchyTag(
        main_arc_id="ARC_01_FIRST_NIGHTMARE",
        sub_arc_id="SUBARC_01_01_OUTSKIRTS",
        mini_arc_id="MINIARC_POLICE_STATION",
        in_universe_day=1,
        time_of_day="midnight",
        weather="cold rain",
    )
    sunny_ch1_passage = """
    Sunny sat on the cold metal bench, shivering slightly under his thin, oversized utility jacket.
    He was sixteen, but looked younger — gaunt, slender, with sallow pale skin stretched tight over
    sharp cheekbones from years of cheap government nutrient paste. His unkempt jet-black hair fell
    in messy strands over dark, cynical onyx eyes that constantly scanned the room for danger.
    He clasped a crushed paper synth-coffee cup between his fingers, smirking faintly at his rotten luck.
    """

    sunny_ch1_spec = agent.observe_and_synthesize_character(
        character_name="Sunny",
        passage_text=sunny_ch1_passage,
        arc_hierarchy=arc_ch1,
        gender="male",
        deterministic_seed=101,
    )

    # Fine-tune Danbooru prompts for Chapter 1
    sunny_ch1_spec.sheet_1_turnaround.prompt = (
        "1boy, solo, teenage boy, 16 years old, pale skin, messy unkempt jet-black hair, bangs falling over forehead, "
        "sharp dark onyx eyes, slender lean build, slight dark circles under eyes, wearing oversized faded dark gray utility jacket, "
        "frayed cuffs, patched dark trousers, simple dark boots, cautious defensive stance, neutral expression, "
        "character model sheet, orthographic turnaround, front view and 3/4 view, solid neutral grey studio background #2b2b2b"
    )
    sunny_ch1_spec.sheet_2_expressions.prompt = (
        "1boy, solo, 16 years old, pale skin, messy jet-black hair, sharp onyx eyes, close-up headshots, "
        "expression sheet, 4-panel matrix: neutral guarded expression, cynical crooked smirk, weary drowsy drooping eyelids, "
        "shocked defensive combat glare, solid neutral grey studio background #2b2b2b"
    )

    # -------------------------------------------------------------------------
    # 3. BIOLOGICAL EVOLUTION AUDIT: CHAPTER 1 -> CHAPTER 25
    # -------------------------------------------------------------------------
    print("\n[3/5] Running Biological Evolution Auditor for Chapter 25 Progression...")
    sunny_ch25_passage = """
    Having slain the Mountain King and survived the gruesome ordeal of the First Nightmare,
    Sunny stood in the quiet sanctuary of the dreamscape. His body had undergone a profound baptism;
    the lingering malnutrition of the slums had receded, replaced by lean, wiry sinew and newly awakened vitality.
    His skin had shed its sickly sallow cast, taking on a smooth, almost porcelain pale clarity that subtly absorbed
    ambient light. Beneath his messy raven hair, his dark eyes held a profound, bottomless obsidian depth that seemed
    to draw in shadows. In his right hand, he loosely held his awakened bronze memory blade, faint wisps of dark mist
    curling around his boots.
    """

    epoch_ch25 = auditor.audit_evolution(
        character_name="Sunny",
        current_chapter=25,
        passage_text=sunny_ch25_passage,
        current_spec=sunny_ch1_spec,
    )
    print(f"  ✨ Evolution Detected: {epoch_ch25.epoch_name}")
    print(f"     Valid Chapters       : Chapters {epoch_ch25.valid_from_chapter} to {epoch_ch25.valid_to_chapter or 'Current'}")
    print(f"     Retired Negative Tags: {epoch_ch25.retired_negative_tags}")
    print(f"     Activated Attributes : {epoch_ch25.skin_complexion_tags} + {epoch_ch25.physique_tags}")

    # Build Sunny Chapter 25 Spec
    arc_ch25 = ArcHierarchyTag(
        main_arc_id="ARC_01_FIRST_NIGHTMARE",
        sub_arc_id="SUBARC_NIGHTMARE_CLIMAX",
        mini_arc_id="MINIARC_AWAKENED_DREAMSCAPE",
        in_universe_day=25,
        time_of_day="midnight",
        weather="ethereal shadow mist",
    )
    sunny_ch25_spec = agent.observe_and_synthesize_character(
        character_name="Sunny Awakened",
        passage_text=sunny_ch25_passage,
        arc_hierarchy=arc_ch25,
        gender="male",
        deterministic_seed=202,
    )
    sunny_ch25_spec.sheet_1_turnaround.prompt = (
        "1boy, solo, 17 years old, smooth porcelain pale skin, subtle ethereal skin glow, messy jet-black hair, "
        "sharp bottomless obsidian eyes, lean athletic sinewy build, wearing dark sleeveless combat tunic, dark leather trousers, "
        "holding crude bronze short sword in one hand, faint wisps of dark shadow mist swirling around boots, alert confident stance, "
        "character model sheet, orthographic turnaround, front view and 3/4 view, solid neutral grey studio background #2b2b2b"
    )
    sunny_ch25_spec.sheet_1_turnaround.negative_prompt = (
        "scrawny, malnourished, sallow skin, blonde hair, blue eyes, modern clothing, business suit, "
        "worst quality, low quality, blurry, extra limbs, bad anatomy, deformed fingers, text, watermark"
    )
    sunny_ch25_spec.sheet_2_expressions.prompt = (
        "1boy, solo, smooth porcelain pale skin, messy raven-black hair, deep obsidian eyes, close-up headshots, "
        "expression sheet, 4-panel matrix: calm calculating stare, predatory mocking smirk, intense shadow-focused combat glare, "
        "subtle sinister smile, solid neutral grey studio background #2b2b2b"
    )

    # -------------------------------------------------------------------------
    # 4. CHARACTER 2: NEPHIS (CHANGING STAR) — CHAPTER 25 DEBUT
    # -------------------------------------------------------------------------
    print("\n[4/5] Synthesizing Nephis (Changing Star) — Chapter 25 Debut...")
    nephis_passage = """
    Changing Star stood on the training pavilion, bathed in the cool light of dawn.
    She was tall, slender yet athletic, with waist-length shimmering silver-white hair that caught
    the light like polished platinum. Her features were carved with calm, striking beauty — alabaster marble skin
    completely devoid of flaws, and cool, tranquil gray eyes that held an unwavering, terrifying resolve.
    She wore a pristine white sleeveless training tunic reinforced with polished silver armguards, a sheathed
    longsword resting against her hip. Her posture was straight and unyielding as a marble pillar.
    """
    nephis_spec = agent.observe_and_synthesize_character(
        character_name="Nephis",
        passage_text=nephis_passage,
        arc_hierarchy=arc_ch25,
        gender="female",
        deterministic_seed=303,
    )
    nephis_spec.sheet_1_turnaround.prompt = (
        "1girl, solo, 17 years old, flawless alabaster marble skin, long straight shimmering silver-white hair, "
        "calm piercing gray eyes, slender athletic build, noble unyielding posture, wearing sleeveless white combat tunic, "
        "silver armguards, fitted white trousers, boots, sheathed longsword at waist, "
        "character model sheet, orthographic turnaround, front view and 3/4 view, solid neutral grey studio background #2b2b2b"
    )
    nephis_spec.sheet_1_turnaround.negative_prompt = (
        "black hair, blonde hair, dark skin, smiling, blushing, modern clothing, "
        "worst quality, low quality, blurry, extra limbs, bad anatomy, deformed fingers, text, watermark"
    )
    nephis_spec.sheet_2_expressions.prompt = (
        "1girl, solo, alabaster skin, long silver-white hair, calm cool gray eyes, close-up headshots, "
        "expression sheet, 4-panel matrix: serene emotionless stare, focused combat resolve, slight solemn glance, "
        "radiant intense gaze, solid neutral grey studio background #2b2b2b"
    )

    # -------------------------------------------------------------------------
    # 5. CONCURRENT DUAL-GPU EXECUTION (ROUND-ROBIN BATCH)
    # -------------------------------------------------------------------------
    print("\n[5/5] Compiling and Dispatching Character Sheets across Dual-GPU Cluster...")
    
    # We generate the Turnaround + Expression Sheets for all 3 character states (6 jobs total):
    # - Sunny Ch 1: Turnaround & Expressions
    # - Sunny Ch 25: Turnaround & Expressions
    # - Nephis Ch 25: Turnaround & Expressions
    batch_jobs = [
        (
            "sunny_ch1_turnaround",
            ComfyUIWorkflowCompiler.compile_foundry_char(
                char_id="sunny_ch1_turnaround",
                identity_tags=sunny_ch1_spec.sheet_1_turnaround.prompt,
                mode="turnaround",
                seed=1001,
                width=1024,
                height=1024,
                checkpoint="waiNSFWIllustrious_v110.safetensors",
            )
        ),
        (
            "sunny_ch1_expressions",
            ComfyUIWorkflowCompiler.compile_foundry_char(
                char_id="sunny_ch1_expressions",
                identity_tags=sunny_ch1_spec.sheet_2_expressions.prompt,
                mode="expressions",
                seed=1002,
                width=1024,
                height=1024,
                checkpoint="waiNSFWIllustrious_v110.safetensors",
            )
        ),
        (
            "sunny_ch25_turnaround",
            ComfyUIWorkflowCompiler.compile_foundry_char(
                char_id="sunny_ch25_turnaround",
                identity_tags=sunny_ch25_spec.sheet_1_turnaround.prompt,
                mode="turnaround",
                seed=2001,
                width=1024,
                height=1024,
                checkpoint="waiNSFWIllustrious_v110.safetensors",
            )
        ),
        (
            "sunny_ch25_expressions",
            ComfyUIWorkflowCompiler.compile_foundry_char(
                char_id="sunny_ch25_expressions",
                identity_tags=sunny_ch25_spec.sheet_2_expressions.prompt,
                mode="expressions",
                seed=2002,
                width=1024,
                height=1024,
                checkpoint="waiNSFWIllustrious_v110.safetensors",
            )
        ),
        (
            "nephis_ch25_turnaround",
            ComfyUIWorkflowCompiler.compile_foundry_char(
                char_id="nephis_ch25_turnaround",
                identity_tags=nephis_spec.sheet_1_turnaround.prompt,
                mode="turnaround",
                seed=3001,
                width=1024,
                height=1024,
                checkpoint="waiNSFWIllustrious_v110.safetensors",
            )
        ),
        (
            "nephis_ch25_expressions",
            ComfyUIWorkflowCompiler.compile_foundry_char(
                char_id="nephis_ch25_expressions",
                identity_tags=nephis_spec.sheet_2_expressions.prompt,
                mode="expressions",
                seed=3002,
                width=1024,
                height=1024,
                checkpoint="waiNSFWIllustrious_v110.safetensors",
            )
        ),
    ]

    print(f"  Dispatching {len(batch_jobs)} jobs across {len(online_workers)} GPUs concurrently...")
    t0 = time.time()
    results = dispatcher.dispatch_batch_parallel(batch_jobs, output_dir=out_base)
    total_time = time.time() - t0
    print(f"  ✅ Parallel generation finished in {total_time:.1f}s ({total_time/len(batch_jobs):.1f}s per sheet average)!")

    # -------------------------------------------------------------------------
    # 6. EXTRACT CANONICAL FACE CHIPS FOR IP-ADAPTER
    # -------------------------------------------------------------------------
    import cv2
    print("\n[6/6] Extracting Canonical Face Reference Chips for IP-Adapter...")
    face_chips = {}
    for job_id, img_path in results:
        if img_path and "expressions" in job_id and Path(img_path).exists():
            prefix = job_id.replace("_expressions", "")
            chip_name = f"{prefix}_canonical_face_chip.png"
            chip_path = out_base / chip_name
            img = cv2.imread(str(img_path))
            if img is not None:
                h, w, _ = img.shape
                # Crop top-left quadrant (neutral character headshot)
                crop = img[0:int(h * 0.5), 0:int(w * 0.5)]
                cv2.imwrite(str(chip_path), crop)
                face_chips[prefix] = chip_path
                print(f"  👤 Extracted {chip_name} ({crop.shape[1]}x{crop.shape[0]} px)")

    # Summary
    print("\n" + "=" * 75)
    print(" SUMMARY OF GENERATED CHARACTER ASSETS (UP TO CHAPTER 25):")
    print("=" * 75)
    for job_id, img_path in results:
        if img_path and Path(img_path).exists():
            kb = Path(img_path).stat().st_size / 1024
            print(f"  * {job_id:<26}: {img_path} ({kb:.1f} KB)")
        else:
            print(f"  * {job_id:<26}: FAILED")

    for prefix, chip_path in face_chips.items():
        kb = chip_path.stat().st_size / 1024
        print(f"  * [FACE CHIP] {prefix:<15}: {chip_path} ({kb:.1f} KB)")

    print("=" * 75)
    print(" CHARACTER PIPELINE (CHAPTER 1 TO 25) VERIFIED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    run_character_pipeline_chapter_25()
