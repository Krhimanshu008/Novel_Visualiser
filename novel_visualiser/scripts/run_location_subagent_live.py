"""Run LocationContinuitySubagent autonomously on Chapter 1 of Shadow Slave.
Prints the direct, unedited output of the subagent's reasoning, prompts, and camera anchors.
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.schemas.location_subagent import ArcHierarchyTag
from app.agents.location_subagent import LocationContinuitySubagent

def main():
    agent = LocationContinuitySubagent(use_ai_thinking=True)

    arc_context = ArcHierarchyTag(
        main_arc_id="ARC_01_FIRST_NIGHTMARE",
        sub_arc_id="SUBARC_01_01_OUTSKIRTS",
        mini_arc_id="MINIARC_01_01_01_AWAKENING",
        parallel_thread_id=None,
        in_universe_day=1,
        time_of_day="midnight",
        weather="cold rain, wet drizzle, mist",
    )

    locations = [
        (
            "Outskirts Transit Bench N24",
            """
            Sunny sat on the battered, rusted metal bench across from the police station, nursing a cup of cheap lukewarm synth-coffee.
            The dreary outskirts of the N24 district were drenched in cold rain. Leaning brutalist tenements loomed over the rain-slicked boulevard.
            Flickering holographic neon signs sputtered above cracked curbs with oil-streaked puddles.
            """,
        ),
        (
            "Fortified Police Station Reception",
            """
            Inside the fortress-like police precinct, thick reinforced blast doors closed behind him.
            Harsh cold cyan fluorescent tubes cast stark shadows across the industrial concrete lobby.
            Automated ceiling security turrets hummed behind wire mesh, aimed at the heavy steel turnstiles.
            A reinforced armored glass counter separated the officers from the public.
            """,
        ),
        (
            "Detention Holding Cell 07",
            """
            A narrow concrete cube with thick steel bars. A solitary cold iron bunk bolted to the damp wall.
            The floor was cold pitted stone. The only light filtered dimly through the iron bars from the corridor.
            Sunny sat huddled against the cold corner, exhaustion and supernatural drowsiness dragging him down as the darkness expanded.
            """,
        ),
    ]

    results = []
    for loc_name, text in locations:
        spec = agent.observe_and_synthesize_location(
            location_name=loc_name,
            passage_text=text,
            arc_hierarchy=arc_context,
            deterministic_seed=1000 + len(results) * 10,
        )
        results.append(spec)

    # Save to JSON for direct verification
    output_file = Path("output/subagent_live_output.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in results], f, indent=2, ensure_ascii=False)

    print(f"COMPLETE: Subagent output written to {output_file}")

if __name__ == "__main__":
    main()

