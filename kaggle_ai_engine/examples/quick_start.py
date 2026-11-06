"""
Kaggle AI Engine — End-to-End Quickstart Example

Demonstrates:
1. Checking Kaggle $10/day Model Proxy quota
2. Generating Danbooru-locked character identity and outfit cards
3. Compiling an SDXL / Illustrious prompt block with deterministic seed
4. Running Multimodal Gate 3 QC on rendered frames
"""

import sys
from pathlib import Path

# Add workspace root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from kaggle_ai_engine import (
    KaggleAISettings,
    KaggleQuotaManager,
    VisualConsistencyArchitect,
    CharacterIdentityCard,
    OutfitCard,
    LocationPlateSpec,
    KaggleSemanticQCInspector,
)

def main():
    print("==================================================================")
    print(" NOVEL VISUALISER — KAGGLE $10/DAY AI ENGINE QUICKSTART")
    print("==================================================================\n")

    # 1. Quota Check
    print("[1] Checking Kaggle Model Proxy Quota...")
    qm = KaggleQuotaManager()
    print(qm.format_status_table())

    # 2. Character & Outfit Consistency Locks
    print("\n[2] Building Danbooru Consistency Locks for 'Sunny'...")
    vca = VisualConsistencyArchitect()

    # Pre-compiled or LLM-synthesized identity card
    sunny_identity = CharacterIdentityCard(
        character_name="Sunny",
        gender="male",
        age_appearance="16-year-old lean youth",
        danbooru_identity_tags=[
            "1boy",
            "pale gaunt skin",
            "messy black hair",
            "sharp dark onyx eyes",
            "short slender frame",
        ],
        distinguishing_features=["shadowy aura", "subtle grimace"],
        forbidden_drift_tags=["muscular", "blonde hair", "cheerful smile", "broad shoulders"],
    )

    sunny_ch1_outfit = OutfitCard(
        outfit_id="sunny_outfit_ch1",
        outfit_name="Outskirts Civilian Clothes",
        danbooru_clothing_tags=[
            "worn dark grey jacket",
            "faded black t-shirt",
            "tattered dark trousers",
            "worn combat boots",
        ],
        accessories=["lukewarm paper coffee cup"],
        color_palette=["#2b2b2b", "#1a1a1a", "#4a4a4a"],
    )

    police_station_plate = LocationPlateSpec(
        location_name="N25 Fortified Police Station Exterior",
        environment_type="urban dystopian street",
        danbooru_landscape_tags=[
            "dystopian cityscape",
            "rusty metal bench",
            "armored concrete police station facade",
            "cold streetlights",
            "no humans",
        ],
        lighting_baseline="dim gloomy twilight, heavy ambient shadows",
    )

    # 3. Deterministic Prompt Block Compilation
    print("\n[3] Compiling Deterministic Illustrious Prompt Block...")
    compiled_prompt = vca.compile_locked_prompt(
        identity=sunny_identity,
        outfit=sunny_ch1_outfit,
        location=police_station_plate,
        shot_framing="wide shot",
        camera_angle="low angle dramatic",
        action_pose="sitting on rusty bench, clutching paper cup with both hands",
        expression="bitter grimace, dark contemplative look",
        state_deltas=["smudge of dirt on right cheek"],
        lighting_vfx="cold neon street lamp glow, atmospheric mist",
    )

    seed = vca.compute_deterministic_seed("Shadow_Slave", chapter_num=1, shot_num=1)

    print(f"\n--- COMPILED POSITIVE PROMPT ---\n{compiled_prompt}\n")
    print(f"--- DETERMINISTIC SEED: {seed} ---\n")

    # 4. Multimodal Gate 3 QC (Demo)
    test_shot = Path("novel_visualiser/output/generated_images/chapter_01/CH01_Shot_01_wide.png")
    test_ref = Path("data/references/Sunny/Sunny_byDr._Nero.jpg")

    print("[4] Gate 3 Multimodal QC Architecture:")
    print(f"   Target Shot : {test_shot if test_shot.exists() else 'Shot 01 (Pending render)'}")
    print(f"   Reference   : {test_ref if test_ref.exists() else 'Canonical portrait (Pending)'}")
    print("   Inspector   : Ready to evaluate pixels once Kaggle Model Proxy token is active.")
    print("                 Run 'python -m kaggle_ai_engine.cli qc --image <path>' for live test.")

    print("\n==================================================================")
    print(" QUICKSTART FINISHED: Kaggle AI Engine is ready!")
    print("==================================================================")

if __name__ == "__main__":
    main()
