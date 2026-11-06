"""
Novel Visualiser — Chapter 1 Full Animation Production Plan & Subagent Orchestrator

Calculates:
1. Exact total image count across Foundry (Locations, Characters, Creatures, Props)
   and Cinematic Keyframe Shots.
2. 3-GPU parallel dispatch routing across Colab (gcgpu), GPU 0 (gpu0), and GPU 1 (gpu1).
3. Qwen-Edit instruction-based repair integration for multi-gate QC inpainting.
4. Generates output/chapter_01_full_animation_spec.json for the frontend studio.
"""

import json
import os
import sys
from pathlib import Path

# Add project root
root_dir = Path(__file__).resolve().parent.parent.parent
novel_dir = root_dir / "novel_visualiser"
if str(novel_dir) not in sys.path:
    sys.path.insert(0, str(novel_dir))

from app.schemas.location_subagent import ArcHierarchyTag


def build_full_chapter1_animation_spec():
    print("=" * 80)
    print("🎬 NOVEL VISUALISER — CHAPTER 1 FULL ANIMATION MASTER PLAN")
    print("=" * 80)

    # 1. Foundry Assets (Foundational Identity Anchors)
    foundry_assets = {
        "locations": [
            {
                "id": "loc_outskirts_transit_bench_n24",
                "name": "Outskirts Transit Bench N24",
                "worker": "Colab (gcgpu.krhimanshu.in)",
                "workflow": "WF-FOUNDRY-LOC",
                "plates": [
                    {"type": "wide_establishing", "desc": "Wide vista with brutalist towers and toxic cyan twilight", "resolution": "1216x832"},
                    {"type": "depth_controlnet", "desc": "MiDaS/Depth-Anything depth map for ControlNet geometry", "resolution": "1216x832"},
                    {"type": "lineart_canny", "desc": "Canny edge structural boundary wireframe", "resolution": "1216x832"},
                    {"type": "reverse_reciprocal", "desc": "180-degree reverse shot facing the desolate street", "resolution": "1216x832"},
                    {"type": "macro_texture", "desc": "Rusted metal bench oxidation & acid-etched concrete", "resolution": "1024x1024"},
                ],
                "count": 5
            },
            {
                "id": "loc_fortified_precinct_hall",
                "name": "Fortified Police Station Reception",
                "worker": "Colab (gcgpu.krhimanshu.in)",
                "workflow": "WF-FOUNDRY-LOC",
                "plates": [
                    {"type": "wide_establishing", "desc": "Reinforced concrete bunker, blast doors, automated turrets", "resolution": "1216x832"},
                    {"type": "depth_controlnet", "desc": "Depth map of reception counter and interrogation booth", "resolution": "1216x832"},
                    {"type": "lineart_canny", "desc": "Industrial architectural lineart", "resolution": "1216x832"},
                    {"type": "medium_counter", "desc": "Desk interaction zone under harsh fluorescent tube light", "resolution": "1216x832"},
                ],
                "count": 4
            },
            {
                "id": "loc_concrete_detention_cell",
                "name": "Reinforced Concrete Detention Cell",
                "worker": "Colab (gcgpu.krhimanshu.in)",
                "workflow": "WF-FOUNDRY-LOC",
                "plates": [
                    {"type": "wide_establishing", "desc": "Cold steel bunk, steel bars, dim amber emergency beacon", "resolution": "1216x832"},
                    {"type": "depth_controlnet", "desc": "Depth map for bunk sitting perspective", "resolution": "1216x832"},
                    {"type": "medium_bunk", "desc": "Seated perspective on bunk with wall shadow", "resolution": "1216x832"},
                ],
                "count": 3
            },
            {
                "id": "loc_nightmare_spell_void",
                "name": "Nightmare Spell Cosmic Void",
                "worker": "Colab (gcgpu.krhimanshu.in)",
                "workflow": "WF-FOUNDRY-LOC",
                "plates": [
                    {"type": "wide_establishing", "desc": "Infinite dark ethereal void with celestial nebulae", "resolution": "1216x832"},
                    {"type": "runic_overlay", "desc": "Glowing golden ancient runic alphabet drifting in space", "resolution": "1024x1024"},
                ],
                "count": 2
            }
        ],
        "characters": [
            {
                "id": "char_sunny_epoch1",
                "name": "Sunny (Sunless) — Biological Epoch 1 (Sleeper Adolescent)",
                "worker": "GPU 0 (gpu0.krhimanshu.in)",
                "workflow": "WF-FOUNDRY-CHAR",
                "turnarounds": [
                    {"view": "front", "desc": "Front neutral pose in oversized frayed utility jacket"},
                    {"view": "side_90", "desc": "Side profile showing posture and slender scrawny build"},
                    {"view": "dynamic_three_quarter", "desc": "3/4 view with vigilant dark obsidian eyes"},
                    {"view": "back", "desc": "Rear construction of tattered dark cloak and hood"},
                ],
                "expressions": [
                    {"exp": "neutral_cautious", "desc": "Guarded, untrusting observant gaze"},
                    {"exp": "cynical_smirk", "desc": "Weary slum survival smirk"},
                    {"exp": "alert_fear", "desc": "Sudden shock at monster chitin screech"},
                    {"exp": "combat_focus", "desc": "Fingers coiled around Midnight Shard"},
                    {"exp": "drowsy_trance", "desc": "Eyelids heavy as Nightmare Spell induces slumber"},
                    {"exp": "spell_resonance", "desc": "Golden runes reflected in pupils"},
                ],
                "count": 10
            }
        ],
        "creatures": [
            {
                "id": "creature_mountain_centipede_larva",
                "name": "Mountain Centipede Larva (Awakened Beast)",
                "worker": "GPU 1 (gpu1.krhimanshu.in)",
                "workflow": "WF-FOUNDRY-CREATURE",
                "plates": [
                    {"view": "mandibles_front", "desc": "Chitinous segmented head, dripping venomous mandibles"},
                    {"view": "carapace_profile", "desc": "Multi-segmented armored body crawling along asphalt"},
                    {"view": "attack_lunge", "desc": "Rearing up with bladed predatory legs"},
                ],
                "count": 3
            }
        ],
        "props": [
            {
                "id": "prop_coffee_cup",
                "name": "Lukewarm Synthetic Coffee Cup",
                "worker": "GPU 1 (gpu1.krhimanshu.in)",
                "count": 1
            },
            {
                "id": "prop_midnight_shard",
                "name": "Midnight Shard (Obsidian Dagger)",
                "worker": "GPU 1 (gpu1.krhimanshu.in)",
                "count": 1
            }
        ]
    }

    total_foundry_images = (
        sum(loc["count"] for loc in foundry_assets["locations"])
        + sum(c["count"] for c in foundry_assets["characters"])
        + sum(cr["count"] for cr in foundry_assets["creatures"])
        + sum(p["count"] for p in foundry_assets["props"])
    )

    # 2. Cinematic Shots Breakdown (Keyframes & Motion Cuts)
    scenes = [
        {
            "scene_number": 1,
            "scene_title": "Scene 1: The Outskirts Slum & Cold Transit Bench",
            "duration_sec": 32.0,
            "shots": [
                {
                    "shot_id": "CH01_SC01_SH01",
                    "shot_type": "extreme_wide",
                    "camera_angle": "low_angle",
                    "duration": 5.0,
                    "worker": "Colab (gcgpu)",
                    "workflow": "WF-SHOT-ENV",
                    "description": "Desolate outskirts vista: Dilapidated concrete towers piercing dense toxic twilight smog.",
                    "audio": "Ambient toxic wind howling, distant industrial drone."
                },
                {
                    "shot_id": "CH01_SC01_SH02",
                    "shot_type": "wide",
                    "camera_angle": "eye_level",
                    "duration": 4.5,
                    "worker": "Colab (gcgpu)",
                    "workflow": "WF-SHOT-SOLO",
                    "description": "Rusted transit bench under flickering cyan streetlamp. A scrawny teenager sits alone.",
                    "audio": "Narration: Sunny sat shivering on the rusted transit bench in the outskirts."
                },
                {
                    "shot_id": "CH01_SC01_SH03",
                    "shot_type": "medium",
                    "camera_angle": "eye_level",
                    "duration": 4.5,
                    "worker": "GPU 0 (gpu0)",
                    "workflow": "WF-SHOT-SOLO",
                    "description": "Sunny pulls his oversized frayed utility jacket tighter, holding a cheap paper coffee cup.",
                    "audio": "Dialogue: 'Another ordinary day in the outskirts... or so I hoped.'"
                },
                {
                    "shot_id": "CH01_SC01_SH04",
                    "shot_type": "close_up",
                    "camera_angle": "low_angle",
                    "duration": 4.0,
                    "worker": "GPU 0 (gpu0)",
                    "workflow": "WF-SHOT-SOLO",
                    "description": "Close-up of Sunny's face: Pale sickly skin slick with acid mist, sharp onyx eyes shadowed by dark circles.",
                    "audio": "Sound of cold acid mist hissing softly against asphalt."
                }
            ]
        },
        {
            "scene_number": 2,
            "scene_title": "Scene 2: Sound of Scavenger & The Police Transport",
            "duration_sec": 28.0,
            "shots": [
                {
                    "shot_id": "CH01_SC02_SH05",
                    "shot_type": "medium_wide",
                    "camera_angle": "low_angle",
                    "duration": 4.0,
                    "worker": "GPU 1 (gpu1)",
                    "workflow": "WF-SHOT-ENV",
                    "description": "Dark alleyway adjacent to the bench. Wet concrete puddles reflect a sickening scraping sound.",
                    "audio": "SFX: Sickening screech of chitin scraping against cracked asphalt."
                },
                {
                    "shot_id": "CH01_SC02_SH06",
                    "shot_type": "two_subject_battle",
                    "camera_angle": "dutch_angle",
                    "duration": 5.0,
                    "worker": "GPU 1 (gpu1)",
                    "workflow": "WF-SHOT-GRP",
                    "description": "Multi-subject regional composition: Sunny on left defenseless, Centipede Larva rearing its venomous mandibles on right.",
                    "audio": "Narration: A monstrous centipede larva emerged, its segmented carapace gleaming in the dim light."
                },
                {
                    "shot_id": "CH01_SC02_SH07",
                    "shot_type": "extreme_close_up",
                    "camera_angle": "eye_level",
                    "duration": 3.5,
                    "worker": "GPU 0 (gpu0)",
                    "workflow": "WF-SHOT-SOLO",
                    "description": "Sunny draws the obsidian dagger, blade catching sharp purple rim highlights.",
                    "audio": "SFX: Sharp metallic unsheathing hiss."
                },
                {
                    "shot_id": "CH01_SC02_SH08",
                    "shot_type": "wide",
                    "camera_angle": "high_angle",
                    "duration": 4.5,
                    "worker": "Colab (gcgpu)",
                    "workflow": "WF-SHOT-ENV",
                    "description": "Blinding searchlights sweep the road: An armored municipal police transport arrives with sirens.",
                    "audio": "SFX: Heavy hydraulic brake hiss and siren wail."
                }
            ]
        },
        {
            "scene_number": 3,
            "scene_title": "Scene 3: Fortified Precinct & Interrogation Desk",
            "duration_sec": 26.0,
            "shots": [
                {
                    "shot_id": "CH01_SC03_SH09",
                    "shot_type": "wide",
                    "camera_angle": "eye_level",
                    "duration": 4.0,
                    "worker": "Colab (gcgpu)",
                    "workflow": "WF-SHOT-ENV",
                    "description": "Inside fortified police precinct: Heavy blast doors, automated ceiling turrets, harsh fluorescent lights.",
                    "audio": "BGM: Cold mechanical humming of police security systems."
                },
                {
                    "shot_id": "CH01_SC03_SH10",
                    "shot_type": "medium_shot",
                    "camera_angle": "eye_level",
                    "duration": 4.5,
                    "worker": "GPU 0 (gpu0)",
                    "workflow": "WF-SHOT-GRP",
                    "description": "Sunny sitting at metallic intake counter across from a weary armed officer.",
                    "audio": "Officer: 'You know what happens when you turn sixteen in this district, kid.'"
                },
                {
                    "shot_id": "CH01_SC03_SH11",
                    "shot_type": "close_up",
                    "camera_angle": "eye_level",
                    "duration": 4.0,
                    "worker": "GPU 0 (gpu0)",
                    "workflow": "WF-SHOT-SOLO",
                    "description": "Sunny's cynical smirk: Cold, detached resignation.",
                    "audio": "Sunny: 'Yeah. Lock me in a cell so I don't tear up the sidewalk when I fall asleep.'"
                }
            ]
        },
        {
            "scene_number": 4,
            "scene_title": "Scene 4: Concrete Detention Cell & The Onset of Slumber",
            "duration_sec": 25.0,
            "shots": [
                {
                    "shot_id": "CH01_SC04_SH12",
                    "shot_type": "wide",
                    "camera_angle": "high_angle",
                    "duration": 4.5,
                    "worker": "Colab (gcgpu)",
                    "workflow": "WF-SHOT-ENV",
                    "description": "Reinforced holding cell: Heavy steel bars, dim amber emergency beacon casting long gloomy shadows.",
                    "audio": "SFX: Heavy steel door slamming shut and locking with magnetic thud."
                },
                {
                    "shot_id": "CH01_SC04_SH13",
                    "shot_type": "medium",
                    "camera_angle": "eye_level",
                    "duration": 4.5,
                    "worker": "GPU 0 (gpu0)",
                    "workflow": "WF-SHOT-SOLO",
                    "description": "Sunny curled on cold steel bunk, head resting against concrete wall as exhaustion strikes.",
                    "audio": "Narration: An unnatural, suffocating drowsiness weighed upon his eyelids."
                },
                {
                    "shot_id": "CH01_SC04_SH14",
                    "shot_type": "close_up",
                    "camera_angle": "low_angle",
                    "duration": 3.5,
                    "worker": "GPU 0 (gpu0)",
                    "workflow": "WF-SHOT-SOLO",
                    "description": "Sunny's shadow on the floor begins to gently writhe and twist independently.",
                    "audio": "Eerie low-frequency resonance."
                }
            ]
        },
        {
            "scene_number": 5,
            "scene_title": "Scene 5: Awakening of the Nightmare Spell",
            "duration_sec": 30.0,
            "shots": [
                {
                    "shot_id": "CH01_SC05_SH15",
                    "shot_type": "wide",
                    "camera_angle": "eye_level",
                    "duration": 5.0,
                    "worker": "Colab (gcgpu)",
                    "workflow": "WF-SHOT-ENV",
                    "description": "The cell walls dissolve into infinite celestial darkness. Drifting wisps of obsidian and starlight.",
                    "audio": "BGM: Ethereal choral awakening crescendo."
                },
                {
                    "shot_id": "CH01_SC05_SH16",
                    "shot_type": "hero_medium",
                    "camera_angle": "eye_level",
                    "duration": 5.0,
                    "worker": "GPU 1 (gpu1)",
                    "workflow": "WF-SHOT-GRP",
                    "description": "Sunny floating suspended in cosmic space. Luminous golden runes slowly orbit his silhouette.",
                    "audio": "Spell Voice: 'Aspiring Sleeper! Welcome to the Nightmare Spell.'"
                },
                {
                    "shot_id": "CH01_SC05_SH17",
                    "shot_type": "macro_eye",
                    "camera_angle": "extreme_close_up",
                    "duration": 4.0,
                    "worker": "GPU 0 (gpu0)",
                    "workflow": "WF-SHOT-SOLO",
                    "description": "Extreme close-up of Sunny's dark obsidian pupil reflecting a sharp golden runic sigil.",
                    "audio": "Spell Voice: 'Prepare for your First Nightmare...'"
                }
            ]
        }
    ]

    total_shots = sum(len(s["shots"]) for s in scenes)
    total_scene_duration = sum(s["duration_sec"] for s in scenes)

    # 3. Qwen-Edit Models Integration Strategy
    qwen_edit_strategy = {
        "engine": "Qwen2.5-VL / Qwen-Image-Edit (Multimodal Instruction Inpainting)",
        "purpose": "Eliminate regeneration re-roll token waste by directing semantic fixes via text instructions.",
        "use_cases": [
            {
                "defect_type": "Hand & Finger Anatomical Glitches (Polydactyly)",
                "instruction": "Fix right hand on bench, curl fingers naturally around coffee cup with accurate 5-finger anatomy."
            },
            {
                "defect_type": "Wardrobe Damage & Weathering Deltas",
                "instruction": "Add frayed seams and tears along the right jacket sleeve, keeping face and lighting unchanged."
            },
            {
                "defect_type": "Prop Insertion & Blade Alignment",
                "instruction": "Inpaint the obsidian Midnight Shard dagger into Sunny's grip with dark glass reflection."
            },
            {
                "defect_type": "Spell Runic Glow Overlay",
                "instruction": "Add faint ethereal golden light emitting from the eyes and floating runic script."
            }
        ],
        "workflow_target": "WF-EDIT-INPAINT (GPU 1: gpu1.krhimanshu.in via InpaintModelConditioning + DifferentialDiffusion)"
    }

    # Summary Report
    manifest = {
        "project": "Shadow Slave",
        "chapter_number": 1,
        "chapter_title": "Chapter 1: Without a Home",
        "estimated_film_duration_sec": total_scene_duration,
        "total_scenes": len(scenes),
        "total_cinematic_shots": total_shots,
        "total_foundry_images": total_foundry_images,
        "grand_total_images_to_generate": total_foundry_images + total_shots,
        "gpu_workload_distribution": {
            "Colab (gcgpu)": "14 Foundry Location Plates + 7 Wide Environment Shots (21 images)",
            "Kaggle GPU 0 (gpu0)": "10 Character Turnaround Sheets + 7 Solo Character Shots (17 images)",
            "Kaggle GPU 1 (gpu1)": "5 Creature/Prop Anchors + 3 Group Shots + Inpaint Repair Loop (9+ images)"
        },
        "foundry_assets": foundry_assets,
        "scenes": scenes,
        "qwen_edit_strategy": qwen_edit_strategy
    }

    out_file = Path("output/chapter_01_full_animation_spec.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n📊 PRODUCTION METRICS:")
    print(f"   • Total Scenes                : {len(scenes)}")
    print(f"   • Total Keyframe Shots        : {total_shots} shots")
    print(f"   • Total Estimated Film Length : {total_scene_duration:.1f} seconds (~2.4 minutes)")
    print(f"   • Foundry Anchor Plates       : {total_foundry_images} images")
    print(f"   • Shot Keyframe Images        : {total_shots} images")
    print(f"   • GRAND TOTAL MASTER IMAGES   : {total_foundry_images + total_shots} images")
    print("=" * 80)
    print(f"✅ Specification saved to: {out_file}")

    return manifest


if __name__ == "__main__":
    build_full_chapter1_animation_spec()
