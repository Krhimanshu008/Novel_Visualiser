"""Novel Visualiser — Live Demonstration of Character Biological Evolution across Arcs.
Demonstrates the 3-tier hierarchy:
Tier 1: Permanent Archetypal DNA
Tier 2: Biological Evolutionary Epochs (Ch 1 Slum Orphan -> Ch 3 Slave Survivor -> Ch 25 Shadow Awakened)
Tier 3: Temporal Wardrobes & Poverty Wear
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.schemas.location_subagent import ArcHierarchyTag
from app.schemas.character_subagent import OutfitCard
from app.agents.character_subagent import CharacterContinuitySubagent
from app.agents.biological_evolution_auditor import BiologicalEvolutionAuditor

def main():
    print("=" * 75)
    print("  CHARACTER BIOLOGICAL EVOLUTION AUDIT (LIVE DEMONSTRATION)")
    print("=" * 75)

    agent = CharacterContinuitySubagent(use_ai_thinking=True)
    auditor = BiologicalEvolutionAuditor(use_ai_thinking=True)

    arc_tag = ArcHierarchyTag(
        main_arc_id="ARC_01_FIRST_NIGHTMARE",
        sub_arc_id="SUBARC_01_01_OUTSKIRTS",
        mini_arc_id="MINIARC_01_01_01_AWAKENING",
        parallel_thread_id=None,
        in_universe_day=1,
        time_of_day="midnight",
        weather="cold rain",
    )

    # --- 1. Chapter 1: Outskirts Slum Orphan ---
    print("\n[PHASE 1] Chapter 1: Ingesting Malnourished Slum Orphan...")
    ch1_text = """
    Sunny sat on the battered bench, shivering inside his oversized, faded charcoal utility jacket.
    He was an scrawny youth barely sixteen, his skin pale and sickly with sallow undertones from cheap nutrient paste.
    His jaw was sharp and delicate, and unruly raven-black hair fell in jagged bangs over hollow dark onyx eyes with insomnia circles.
    """
    spec_ch1 = agent.observe_and_synthesize_character("Sunny", ch1_text, arc_tag, gender="male", deterministic_seed=42)

    print(" • Active Epoch: Initial Outskirts Slum State")
    print(f" • Core Biology Tags: {spec_ch1.identity_lock.core_danbooru_tags}")
    print(f" • Negative Constraints: {spec_ch1.identity_lock.negative_identity_tags}")

    # --- 2. Chapter 3: First Nightmare Slave Survivor ---
    print("\n[PHASE 2] Chapter 3: Auditing First Nightmare Slave Caravan Milestone...")
    ch3_text = """
    Stripped of his clothes and forced into the slave caravan chained in heavy iron, Sunny struggled against the blizzard.
    The weeks of brutal labor and mountain combat had stripped away his weak adolescent softness, carving lean, sinewy muscle onto his frame.
    Whip scars and dark bruises crisscrossed his pale back. His dark onyx eyes burned with a vicious, desperate survivalist glare.
    He wore only tattered, bloodstained slave rags of rough coarse hemp.
    """
    epoch_ch3 = auditor.audit_evolution("Sunny", 3, ch3_text, spec_ch1)
    if not epoch_ch3:
        epoch_ch3 = auditor.audit_evolution("Sunny", 3, ch3_text, spec_ch1) # fallback

    spec_ch3 = auditor.apply_evolution_to_spec(
        spec_ch1.model_copy(deep=True),
        epoch_ch3,
        new_outfit=OutfitCard(
            outfit_id="outfit_slave_rags",
            outfit_name="Tattered Temple Slave Rags",
            danbooru_clothing_tags=["tattered slave rags", "rough hemp tunic", "barefoot", "iron slave collar"],
            poverty_wear_tags=["torn fabric", "bloodstains", "frost rimed"],
            negative_clothing_tags=["clean clothes", "armor", "utility jacket"],
        ),
    )
    print(f" • Detected New Epoch: {epoch_ch3.epoch_name}")
    print(f" • Updated Biology Tags: {spec_ch3.identity_lock.core_danbooru_tags}")
    print(f" • Retired Negative Tags: {spec_ch3.identity_lock.negative_identity_tags}")

    # --- 3. Chapter 25: Forgotten Shore Shadow Awakened ---
    print("\n[PHASE 3] Chapter 25: Auditing Forgotten Shore & Divine Shadow Aspect Awakening...")
    ch25_text = """
    Having survived the First Nightmare, claimed his True Name 'Lost from Light', and awakened his Divine Shadow Aspect,
    Sunny's physical body underwent a profound supernatural metamorphosis.
    His skin was no longer sallow or scarred, but had purified into smooth, flawless alabaster porcelain with a faint, ethereal glow.
    His pitch-black hair possessed a rich, silky texture, framing striking, delicate features with dark onyx eyes of obsidian depth.
    His scrawniness was gone, replaced by a tall, athletic, and agile frame that moved with preternatural shadow grace.
    """
    epoch_ch25 = auditor.audit_evolution("Sunny", 25, ch25_text, spec_ch3)
    spec_ch25 = auditor.apply_evolution_to_spec(spec_ch3.model_copy(deep=True), epoch_ch25)

    print(f" • Detected New Epoch: {epoch_ch25.epoch_name}")
    print(f" • Updated Biology Tags: {spec_ch25.identity_lock.core_danbooru_tags}")
    print(f" • Retired Negative Tags (CRITICAL): {spec_ch25.identity_lock.negative_identity_tags}")

    # Verify that scrawny and sallow skin are now FORBIDDEN
    assert "scrawny" in spec_ch25.identity_lock.negative_identity_tags
    assert "sallow skin" in spec_ch25.identity_lock.negative_identity_tags

    # Resolve a shot in Chapter 25
    shot_ch25 = agent.resolve_shot_character_anchor(
        spec_ch25,
        shot_action="standing on dark obsidian ruins, shadow tendrils coiling around feet",
        emotion="calm calculating smirk, obsidian shadow eyes",
        camera_framing="MEDIUM",
    )

    payload = {
        "epoch_1_slum_orphan": {
            "epoch_name": "Epoch 1: Outskirts Slum Orphan (Ch 1-2)",
            "core_biology": spec_ch1.identity_lock.core_danbooru_tags,
            "negatives": spec_ch1.identity_lock.negative_identity_tags,
            "outfit": spec_ch1.active_outfit.model_dump(),
        },
        "epoch_2_slave_survivor": {
            "epoch_name": epoch_ch3.epoch_name,
            "core_biology": spec_ch3.identity_lock.core_danbooru_tags,
            "negatives": spec_ch3.identity_lock.negative_identity_tags,
            "outfit": spec_ch3.active_outfit.model_dump(),
        },
        "epoch_3_shadow_awakened": {
            "epoch_name": epoch_ch25.epoch_name,
            "core_biology": spec_ch25.identity_lock.core_danbooru_tags,
            "negatives": spec_ch25.identity_lock.negative_identity_tags,
            "resolved_shot_prompt": shot_ch25.full_positive_prompt,
            "resolved_negative_prompt": shot_ch25.full_negative_prompt,
        },
    }

    out_file = Path("output/evolution_demonstration_live_output.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\n[COMPLETE] Evolution output successfully saved to:\n          {out_file}")

if __name__ == "__main__":
    main()
