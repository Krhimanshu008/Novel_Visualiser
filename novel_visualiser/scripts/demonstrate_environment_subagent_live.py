"""Novel Visualiser — Demonstration of EnvironmentContinuitySubagent (Live Frontier AI Run).

Demonstrates:
1. Scene 1 (Ch 1): Outskirts Transit Bench at Twilight Smog (flickering sodium amber, cyan sky bounce, damp pavement).
2. Scene 2 (Ch 2): Outskirts Transit Bench at Midnight Acid Downpour (FROZEN GEOMETRY, new midnight/rain lighting rig).
3. Scene 3 (Ch 3): Mountain Altar at Freezing Rose-Gold Dawn (sub-zero blizzard, sharp low-angle sun, rime frost).
4. Decoupled 5-Dimension Observational Reasoning (Solar, Lighting Rig, Atmospheric Particles, Foliage, Surface Moisture).
5. Compiling prompt-ready positive & negative Danbooru token blocks.
6. SQLite canon persistence and historical temporal query.
7. Final Inpainting Compositing Prompt assembly (Location + Character + Environment).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database.models import Base, Entity, EntityType, Project
from app.schemas.location_subagent import ArcHierarchyTag
from app.agents.environment_subagent import EnvironmentContinuitySubagent


def main():
    print("=" * 75)
    print("  ENVIRONMENT CONTINUITY SUBAGENT — LIVE PRODUCTION RUN")
    print("=" * 75)

    # 1. Setup SQLite Database for Canon Persistence
    db_path = Path("output/novel_canon.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    project_id = "proj_shadow_slave_master"
    project = session.query(Project).filter(Project.id == project_id).first()
    if not project:
        project = Project(
            id=project_id,
            name="Shadow Slave Production",
            novel_title="Shadow Slave",
            author="Guiltythree",
        )
        session.add(project)
        session.commit()

    agent = EnvironmentContinuitySubagent(db=session, use_ai_thinking=True)

    # Define Scenes to Analyze
    scenarios = [
        {
            "title": "SCENE 1: Outskirts Transit Bench at Smog Twilight (Chapter 1)",
            "location_name": "Outskirts Transit Bench N24",
            "location_id": "loc_outskirts_transit_bench_n24",
            "chapter": 1,
            "arc": ArcHierarchyTag(
                main_arc_id="ARC_01_FIRST_NIGHTMARE",
                sub_arc_id="SUBARC_01_01_OUTSKIRTS",
                mini_arc_id="MINIARC_01_01_01_AWAKENING",
                time_of_day="twilight",
                weather="industrial smog",
            ),
            "passage": """
            The outskirts were smothered beneath a thick shroud of toxic industrial smog as twilight descended over the colossal concrete wall in the distance.
            High above the cracked pavement, an aged sodium vapor streetlamp flickered intermittently, buzzing with high voltage and casting a sickly amber glow onto the cold metal bench.
            Cold cyan light from the dying sky bounced off the damp asphalt, where oily puddles mirrored the streetlamp's beam.
            Through the deep fissures in the concrete curb, stubborn dead weeds and grey moss clung to life, rustling slightly in the chilly evening wind.
            """,
        },
        {
            "title": "SCENE 2: Outskirts Transit Bench at Midnight Acid Downpour (Chapter 2) [GEOMETRY FROZEN]",
            "location_name": "Outskirts Transit Bench N24",
            "location_id": "loc_outskirts_transit_bench_n24",
            "chapter": 2,
            "arc": ArcHierarchyTag(
                main_arc_id="ARC_01_FIRST_NIGHTMARE",
                sub_arc_id="SUBARC_01_01_OUTSKIRTS",
                mini_arc_id="MINIARC_01_01_02_CURFEW_DELUGE",
                time_of_day="midnight",
                weather="toxic acid rain",
            ),
            "passage": """
            By midnight, the temperature had plummeted, and the sky turned into a suffocating pitch-black void.
            A heavy downpour of toxic acid rain pounded against the cracked asphalt, sizzling faintly as puddles swelled into murky reflective pools.
            The sodium streetlamp had failed, leaving only the distant, eerie sweeping searchlights of the military wall patrol slicing through the torrential deluge.
            Cold rain lashed against the rusted bench armrests, and the dead weeds in the fissures were flattened by the violent gusts of wind.
            """,
        },
        {
            "title": "SCENE 3: Mountain Altar at Freezing Rose-Gold Dawn (Chapter 3)",
            "location_name": "Mountain Sacrificial Altar",
            "location_id": "loc_mountain_sacrificial_altar",
            "chapter": 3,
            "arc": ArcHierarchyTag(
                main_arc_id="ARC_01_FIRST_NIGHTMARE",
                sub_arc_id="SUBARC_01_02_MOUNTAIN_TEMPLE",
                mini_arc_id="MINIARC_01_02_01_ALTAR_BLOOD",
                time_of_day="dawn",
                weather="sub-zero blizzard",
            ),
            "passage": """
            Dawn broke over the treacherous mountain ridge, casting crisp, low-angle rose-gold light across the ancient stone ruins of the sacrificial altar.
            Violent sub-zero blizzards whipped whirlwinds of crystalline snow across the crumbling obsidian flagstones.
            Massive frozen pine branches hung heavy with icicles, groaning under the weight of the gale.
            The rising sun created blinding glints across the rime-frosted stone pillars, casting deep, sharp, elongated blue shadows westward into the abyss.
            """,
        },
    ]

    live_results = {}

    for scenario in scenarios:
        print("\n" + "-" * 75)
        print(f"  {scenario['title']}")
        print("-" * 75)

        # Ensure Location Entity exists in SQLite
        loc_ent = session.query(Entity).filter(Entity.id == scenario["location_id"]).first()
        if not loc_ent:
            loc_ent = Entity(
                id=scenario["location_id"],
                project_id=project_id,
                entity_type=EntityType.LOCATION,
                canonical_name=scenario["location_name"],
            )
            session.add(loc_ent)
            session.commit()

        # Execute 5-dimension AI observation and synthesis
        card = agent.observe_and_synthesize_environment(
            location_name=scenario["location_name"],
            passage_text=scenario["passage"],
            arc_hierarchy=scenario["arc"],
            location_id=scenario["location_id"],
            chapter_number=scenario["chapter"],
        )

        # Persist to SQLite
        agent.persist_environment_card(project_id=project_id, card=card)

        # Resolve Runtime Anchor
        anchor = agent.resolve_shot_environment_anchor(card)

        print(f" • Time of Day       : {card.time_of_day.value.upper()}")
        print(f" • Weather Condition : {card.weather_condition.value.upper()}")
        print(f" • Key Light Source  : {card.lighting_rig.key_light_source} ({card.lighting_rig.key_color_temperature_k}K)")
        print(f" • Key Direction     : {card.lighting_rig.key_light_direction}")
        print(f" • Rim Light Active  : {card.lighting_rig.rim_light_active} ({card.lighting_rig.rim_light_color})")
        print(f" • Shadow Hardness   : {card.lighting_rig.shadow_hardness}")
        print(f" • Surface Moisture  : {anchor.surface_moisture_level * 100:.0f}%")
        print(f" • Color Palette     : {card.color_palette_hex}")
        print(f" • Resolved Lighting : {anchor.resolved_lighting_block[:100]}...")
        print(f" • Negative Filter   : {anchor.resolved_negative_block[:80]}...")

        live_results[f"scene_{scenario['chapter']}"] = {
            "title": scenario["title"],
            "location_id": scenario["location_id"],
            "chapter": scenario["chapter"],
            "time_of_day": card.time_of_day.value,
            "weather": card.weather_condition.value,
            "observation": card.observation.model_dump(),
            "lighting_rig": card.lighting_rig.model_dump(),
            "color_palette_hex": card.color_palette_hex,
            "resolved_lighting_block": anchor.resolved_lighting_block,
            "resolved_negative_block": anchor.resolved_negative_block,
            "key_light_vector": anchor.key_light_vector,
            "shadow_direction": anchor.shadow_direction,
            "surface_moisture_level": anchor.surface_moisture_level,
        }

    # Historical Continuity Verification Query
    print("\n" + "=" * 75)
    print("  VERIFYING SQLite HISTORICAL CONTINUITY ACROSS CHAPTERS")
    print("=" * 75)

    cached_ch1 = agent.query_historical_environment(
        project_id=project_id,
        location_id="loc_outskirts_transit_bench_n24",
        chapter_number=1,
    )
    cached_ch2 = agent.query_historical_environment(
        project_id=project_id,
        location_id="loc_outskirts_transit_bench_n24",
        chapter_number=2,
    )

    print(f" [Ch 1 Lookup]: {cached_ch1.environment_name} -> {cached_ch1.time_of_day.value} ({cached_ch1.weather_condition.value})")
    print(f" [Ch 2 Lookup]: {cached_ch2.environment_name} -> {cached_ch2.time_of_day.value} ({cached_ch2.weather_condition.value})")
    assert cached_ch1.time_of_day != cached_ch2.time_of_day, "Temporal progression failed to differentiate!"

    # Save output JSON
    out_file = Path("output/environment_subagent_live_output.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(live_results, f, indent=2)

    print(f"\n[COMPLETE] Environment results saved to: {out_file}")


if __name__ == "__main__":
    main()

