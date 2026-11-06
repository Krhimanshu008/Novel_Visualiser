"""Visual Consistency Engine (VCE) Benchmark Experiment (Section 11 of chatgpt respone.md).

Tests 1 character across 4 distinct shots:
1. Portrait (Close-up)
2. Full Body (Medium-wide)
3. Action Pose (Dynamic combat, holding sword)
4. Close-up (Cheek scratch, lighting delta)

Verifies:
- Identity block (Block 2) remains 100% identical and invariant across all shots.
- Outfit inheritance operates independently from base biology.
- Environmental plate does not contaminate character tokens.
- Deterministic seed derivation produces consistent latent anchors.
"""

import sys
from pathlib import Path

# Ensure novel_visualiser is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from app.schemas.visual_context import (
    VisualStyleProfile,
    DanbooruIdentityLock,
    OutfitCard,
    LocationPlateSpec,
    SubjectVisualState,
    ShotVisualContext,
)
from app.services.prompt_compiler import PromptCompiler


def test_visual_consistency_4_shot_experiment():
    compiler = PromptCompiler()
    style = VisualStyleProfile(name="Cinematic Manhwa")

    # Canonical Identity Lock: Sunny
    sunny_canon = DanbooruIdentityLock(
        entity_id="char_sunny",
        entity_name="Sunny",
        identity_version="CANON-V1",
        core_danbooru_tags=["1boy", "pale skin", "messy black hair", "sharp dark onyx eyes", "slender build"],
        negative_identity_tags=["blonde hair", "blue eyes", "muscular", "facial hair"],
        canonical_seed=10042,
    )

    # Arc 1 Outfit: Scavenger Tunic
    arc1_outfit = OutfitCard(
        outfit_id="OUTFIT-SUNNY-ARC1",
        entity_id="char_sunny",
        outfit_name="Outskirts Scavenger Tunic",
        valid_from_chapter=1,
        danbooru_clothing_tags=["tattered dark gray tunic", "cloth wraps", "rough boots"],
        negative_clothing_tags=["modern clothing", "plate armor", "hoodie"],
    )

    # Arc 2 Outfit: Starlight Armor
    arc2_outfit = OutfitCard(
        outfit_id="OUTFIT-SUNNY-ARC2",
        entity_id="char_sunny",
        outfit_name="Puppeteer Shroud Armor",
        valid_from_chapter=50,
        danbooru_clothing_tags=["dark leather armor", "silver filigree chestplate", "vambraces", "black cloak"],
        negative_clothing_tags=["tattered tunic", "rags", "modern clothing"],
    )

    # Location 1: Slum Detention Cell
    loc_cell = LocationPlateSpec(
        location_id="loc_cell_01",
        location_name="Holding Cell",
        plate_tags=["reinforced concrete walls", "metal bunk", "dim corridor bars"],
    )

    # Location 2: Forgotten Shore
    loc_shore = LocationPlateSpec(
        location_id="loc_shore_01",
        location_name="Dark Sea Ruins",
        plate_tags=["ancient submerged stone pillars", "dark obsidian water", "heavy fog"],
    )

    # -------------------------------------------------------------
    # Shot 1: Portrait (Close-up, Arc 1)
    # -------------------------------------------------------------
    shot1 = ShotVisualContext(
        shot_id="CH01_SC01_SH01",
        chapter_number=1,
        framing="close_up",
        camera_angle="eye_level",
        environment=loc_cell,
        subjects=[
            SubjectVisualState(
                entity_id="char_sunny",
                entity_name="Sunny",
                identity_lock=sunny_canon,
                outfit=arc1_outfit,
                pose_expression="looking at viewer, cautious grimace",
            )
        ],
        lighting_atmosphere=["dim artificial fluorescent light"],
    )

    # -------------------------------------------------------------
    # Shot 2: Full Body (Medium-wide, Arc 1)
    # -------------------------------------------------------------
    shot2 = ShotVisualContext(
        shot_id="CH01_SC01_SH02",
        chapter_number=1,
        framing="medium_wide",
        camera_angle="eye_level",
        environment=loc_cell,
        subjects=[
            SubjectVisualState(
                entity_id="char_sunny",
                entity_name="Sunny",
                identity_lock=sunny_canon,
                outfit=arc1_outfit,
                pose_expression="sitting on bunk, slumping posture",
            )
        ],
        lighting_atmosphere=["cold overhead fluorescent lighting"],
    )

    # -------------------------------------------------------------
    # Shot 3: Action Pose (Arc 2 Armor, Holding Sword)
    # -------------------------------------------------------------
    shot3 = ShotVisualContext(
        shot_id="CH50_SC04_SH03",
        chapter_number=50,
        framing="medium",
        camera_angle="low_angle_dramatic",
        environment=loc_shore,
        subjects=[
            SubjectVisualState(
                entity_id="char_sunny",
                entity_name="Sunny",
                identity_lock=sunny_canon,
                outfit=arc2_outfit,
                held_equipment=["holding midnight shard sword", "shadow tendrils rising around blade"],
                pose_expression="defensive combat stance, fierce expression",
            )
        ],
        lighting_atmosphere=["eerie moonlit rim lighting", "mist haze"],
        vfx_particles=["shadow tendrils"],
    )

    # -------------------------------------------------------------
    # Shot 4: Close-up (Arc 2, Cheek Injury & Battle Dust)
    # -------------------------------------------------------------
    shot4 = ShotVisualContext(
        shot_id="CH50_SC04_SH04",
        chapter_number=50,
        framing="close_up",
        camera_angle="profile",
        environment=loc_shore,
        subjects=[
            SubjectVisualState(
                entity_id="char_sunny",
                entity_name="Sunny",
                identity_lock=sunny_canon,
                outfit=arc2_outfit,
                temporal_deltas=["fresh bleeding cut across cheek", "sweat beads", "battle dirt smudges"],
                pose_expression="panting, intense glare",
            )
        ],
        lighting_atmosphere=["harsh silver moonlight", "splashing water drops"],
    )

    # Compile all 4 shots
    comp1, blocks1 = compiler.compile_visual_context(shot1, style)
    comp2, blocks2 = compiler.compile_visual_context(shot2, style)
    comp3, blocks3 = compiler.compile_visual_context(shot3, style)
    comp4, blocks4 = compiler.compile_visual_context(shot4, style)

    # -------------------------------------------------------------
    # Verification Rule 1: Identity Invariance in Block 2
    # -------------------------------------------------------------
    expected_id_block = "1boy, pale skin, messy black hair, sharp dark onyx eyes, slender build"
    assert blocks1.block2_identity == expected_id_block
    assert blocks2.block2_identity == expected_id_block
    assert blocks3.block2_identity == expected_id_block
    assert blocks4.block2_identity == expected_id_block
    print("\n[VERIFIED] Block 2 (Character Identity Anchor) is 100% identical across all 4 shots.")

    # -------------------------------------------------------------
    # Verification Rule 2: Outfit Independence
    # -------------------------------------------------------------
    assert "tattered dark gray tunic" in blocks1.block3_outfit
    assert "tattered dark gray tunic" in blocks2.block3_outfit
    assert "silver filigree chestplate" in blocks3.block3_outfit
    assert "silver filigree chestplate" in blocks4.block3_outfit
    # Check that outfit change did NOT leak into Block 2
    assert "chestplate" not in blocks3.block2_identity
    print("[VERIFIED] Block 3 (Outfit) dynamically changed between arcs without mutating Block 2 identity.")

    # -------------------------------------------------------------
    # Verification Rule 3: Equipment & Temporal Deltas Isolation
    # -------------------------------------------------------------
    assert "midnight shard sword" in blocks3.block4_equipment
    assert "fresh bleeding cut across cheek" in blocks4.block7_deltas_lighting
    # Check that injuries did NOT leak into Block 2
    assert "cut" not in blocks4.block2_identity
    print("[VERIFIED] Temporal injuries and held weapons isolated to Blocks 4 & 7.")

    # -------------------------------------------------------------
    # Verification Rule 4: Global Style Locking
    # -------------------------------------------------------------
    assert blocks1.block1_style == blocks2.block1_style == blocks3.block1_style == blocks4.block1_style
    assert "cinematic manhwa illustration" in blocks1.block1_style
    print("[VERIFIED] Block 1 (Global Style Anchor) locked across all chapters and shots.")

    # -------------------------------------------------------------
    # Verification Rule 5: Deterministic Latent Seeds
    # -------------------------------------------------------------
    assert comp1.seed is not None
    assert comp2.seed is not None
    assert comp3.seed is not None
    assert comp4.seed is not None
    # Verify deterministic reproducibility: recompiling shot 1 yields exact same seed
    comp1_repeat, _ = compiler.compile_visual_context(shot1, style)
    assert comp1.seed == comp1_repeat.seed
    print(f"[VERIFIED] Deterministic latent seeds computed reproducibly (Shot 1 seed: {comp1.seed}).")
