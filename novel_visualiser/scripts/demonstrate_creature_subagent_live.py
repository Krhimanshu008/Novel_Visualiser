"""Novel Visualiser — Demonstration of CreatureMonsterSubagent (Live Frontier AI Run).

Demonstrates:
1. Nightmare Creature 1 (Awakened Beast): Mountain Centipede Larva (5m obsidian insectoid).
2. Nightmare Creature 2 (Fallen Tyrant): Mountain Devourer Tyrant (14m armored behemoth, 5 soul cores).
3. Hero Prop / Memory Weapon: Sunny's "Midnight Shard" (Obsidian Awakened weapon).
4. Power System Classification across the 2-Axis hierarchy (Rank: Awakened/Fallen, Class: Beast/Tyrant).
5. 4-Sheet Bestiary Reference Matrix synthesis & OpenCV human-to-monster scale comparison plate.
6. Auto-cropping predatory head chips for ComfyUI IP-Adapter Plus consistency.
7. SQLite canon persistence and historical memory query.
8. Runtime combat battle anchor resolution for scene composition.
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

from app.database.models import Base, Project
from app.schemas.location_subagent import ArcHierarchyTag
from app.agents.creature_subagent import CreatureMonsterSubagent


def main():
    print("=" * 75)
    print("  CREATURE, MONSTER & POWER SYSTEM SUBAGENT — LIVE PRODUCTION RUN")
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

    agent = CreatureMonsterSubagent(db=session, use_ai_thinking=True)
    out_dir = Path("output/bestiary_canon")
    out_dir.mkdir(parents=True, exist_ok=True)

    arc_tag = ArcHierarchyTag(
        main_arc_id="ARC_01_FIRST_NIGHTMARE",
        sub_arc_id="SUBARC_01_02_MOUNTAIN_TEMPLE",
        mini_arc_id="MINIARC_01_02_01_SHAMAN_ALTAR",
        time_of_day="dawn",
        weather="sub-zero blizzard",
    )

    scenarios = [
        {
            "name": "Mountain Centipede Larva",
            "type": "creature",
            "passage": """
            From the frozen mountain fissures crawled a gigantic centipede monstrosity, an Awakened Beast of the high peaks.
            Its five-meter-long body was encased in heavy, interlocking plates of segmented black obsidian chitin, glistening like polished volcanic glass.
            Dozens of barbed scythe-like legs skittered with lethal speed over the snow, leaving sizzling trails of green corrosive acid.
            A terrifying cluster of six crimson eyes burned above its triple-hinged serrated mandibles, leaking dark shadow miasma into the blizzard.
            """,
        },
        {
            "name": "Mountain Devourer Tyrant",
            "type": "creature",
            "passage": """
            From the subterranean abyssal depths of the mountain temple arose a colossal nightmare—a Fallen Tyrant known as the Mountain Devourer.
            Towering nearly fourteen meters in length and six meters high at the shoulder, its grotesque frame possessed five distinct soul cores 
            pulsing with malevolent crimson luminescence beneath thick, calcified bone armor plates.
            Along its swollen flanks were pulsating hive brood sacs, spawning swarms of acidic crawler minions.
            The tyrant's very presence exuded a crushing gravitational pressure that cracked the obsidian flagstones beneath its clawed gait.
            """,
        },
        {
            "name": "Midnight Shard",
            "type": "item",
            "passage": """
            A translucent notification materialized in Sunny's vision: [You have slain an Awakened Beast! You have received a Memory: Midnight Shard].
            Summoning the weapon, an elegant single-edged straight sword made of pitch-black obsidian materialized in his grip.
            The black blade was eighty-five centimeters long, so dark it seemed to absorb ambient light, its surface as reflective as dark water with faint shadowy mist rising from the razor edge.
            """,
        },
    ]

    live_results = {}

    for sc in scenarios:
        print("\n" + "-" * 75)
        print(f"  PROCESSING: {sc['name']} ({sc['type'].upper()})")
        print("-" * 75)

        if sc["type"] == "creature":
            card = agent.observe_and_synthesize_creature(
                creature_name=sc["name"],
                passage_text=sc["passage"],
                arc_hierarchy=arc_tag,
                chapter_number=3,
            )

            # Render 4-sheet plates and auto-crop predatory head chip
            rendered_card = agent.render_bestiary_matrix_and_head_chip(
                card=card,
                output_dir=out_dir,
                deterministic_seed=3000,
            )

            # Persist to SQLite
            agent.persist_creature_card(project_id=project_id, card=rendered_card)

            # Resolve Combat Battle Anchor
            anchor = agent.resolve_shot_creature_anchor(
                card=rendered_card,
                combat_action="lunging forward with razor mandibles and scythe legs striking",
                spatial_anchor="center_foreground",
            )

            print(f" • Soul Rank         : {rendered_card.rank.value.upper()}")
            print(f" • Class Hierarchy   : {rendered_card.creature_class.value.upper()} ({rendered_card.soul_core_count} Soul Cores)")
            print(f" • Taxonomy Clade    : {rendered_card.taxonomy.value.upper()}")
            print(f" • Scale Dimensions  : {rendered_card.scale_spec.estimated_length_meters:.1f}m length x {rendered_card.scale_spec.estimated_height_meters:.1f}m height")
            print(f" • Human Scale Ratio : {rendered_card.scale_spec.human_scale_ratio} human height")
            print(f" • Predatory Head Chip: {rendered_card.head_reference_chip_path}")
            print(f" • Scale Sheet Path  : {rendered_card.sheet_4_scale_comparison.image_path}")
            print(f" • Battle Prompt     : {anchor.resolved_creature_prompt[:100]}...")
            print(f" • Negative Filter   : {anchor.resolved_negative_prompt[:80]}...")

            live_results[rendered_card.creature_id] = {
                "creature_name": rendered_card.creature_name,
                "rank": rendered_card.rank.value,
                "creature_class": rendered_card.creature_class.value,
                "taxonomy": rendered_card.taxonomy.value,
                "soul_core_count": rendered_card.soul_core_count,
                "observation": rendered_card.observation.model_dump(),
                "scale_spec": rendered_card.scale_spec.model_dump(),
                "head_chip_path": rendered_card.head_reference_chip_path,
                "sheet_1_turnaround_path": rendered_card.sheet_1_turnaround.image_path,
                "sheet_4_scale_comparison_path": rendered_card.sheet_4_scale_comparison.image_path,
                "resolved_battle_prompt": anchor.resolved_creature_prompt,
                "resolved_negative_prompt": anchor.resolved_negative_prompt,
            }

        elif sc["type"] == "item":
            item_card = agent.observe_and_synthesize_item(
                item_name=sc["name"],
                passage_text=sc["passage"],
                arc_hierarchy=arc_tag,
                item_type="weapon",
                chapter_number=3,
            )

            # Persist to SQLite
            agent.persist_item_card(project_id=project_id, item_card=item_card, chapter_number=3)

            print(f" • Memory Item       : {item_card.item_name}")
            print(f" • Rank Tier         : {item_card.rank_tier}")
            print(f" • Material Physics  : {item_card.material_physics}")
            print(f" • Dimensions        : {item_card.dimensions}")
            print(f" • Danbooru Tokens   : {item_card.danbooru_item_tags}")

            live_results[item_card.item_id] = item_card.model_dump()

    # Historical Continuity Verification
    print("\n" + "=" * 75)
    print("  VERIFYING SQLite HISTORICAL BESTIARY LOOKUP")
    print("=" * 75)

    cached_centipede = agent.query_historical_creature(
        project_id=project_id,
        creature_name="Mountain Centipede Larva",
        chapter_number=3,
    )
    cached_tyrant = agent.query_historical_creature(
        project_id=project_id,
        creature_name="Mountain Devourer Tyrant",
        chapter_number=3,
    )
    cached_item = agent.query_historical_item(
        project_id=project_id,
        item_name="Midnight Shard",
        chapter_number=3,
    )

    print(f" [Creature 1 Lookup]: {cached_centipede.creature_name} -> {cached_centipede.rank.value} {cached_centipede.creature_class.value}")
    print(f" [Creature 2 Lookup]: {cached_tyrant.creature_name} -> {cached_tyrant.rank.value} {cached_tyrant.creature_class.value} ({cached_tyrant.soul_core_count} Cores)")
    print(f" [Memory Item Lookup]: {cached_item.item_name} -> {cached_item.rank_tier} ({cached_item.dimensions})")

    assert cached_centipede is not None
    assert cached_tyrant is not None
    assert cached_item is not None

    # Save output JSON
    out_file = Path("output/creature_subagent_live_output.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(live_results, f, indent=2)

    print(f"\n[COMPLETE] Bestiary & Prop output successfully saved to: {out_file}")


if __name__ == "__main__":
    main()

