"""Novel Visualiser — Run CharacterContinuitySubagent autonomously on Chapter 1 (Sunny).
Executes live AI thinking via Kaggle Model Proxy (Gemini 3 Flash Preview).
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.schemas.location_subagent import ArcHierarchyTag
from app.agents.character_subagent import CharacterContinuitySubagent

def main():
    agent = CharacterContinuitySubagent(use_ai_thinking=True)

    arc_context = ArcHierarchyTag(
        main_arc_id="ARC_01_FIRST_NIGHTMARE",
        sub_arc_id="SUBARC_01_01_OUTSKIRTS",
        mini_arc_id="MINIARC_01_01_01_AWAKENING",
        parallel_thread_id=None,
        in_universe_day=1,
        time_of_day="midnight",
        weather="cold rain, wet drizzle, mist",
    )

    chapter_1_sunny_passage = """
    Sunny sat on the battered, rusted metal bench across from the fortified police station, nursing a cup of cheap lukewarm synth-coffee.
    He was an scrawny youth from the outskirts, barely sixteen, his skin pale and sickly with sallow undertones from a lifetime of surviving on government nutrient paste.
    His jaw was sharp and delicate, and unruly raven-black hair fell in jagged bangs across his forehead, partially obscuring hollow dark onyx eyes framed by deep purplish-grey circles of chronic insomnia.
    He shivered inside an oversized, faded charcoal utility jacket, its frayed cuff seams shedding grey insulation batting. The tarnished plastic zipper was missing several teeth, and the knees of his dark trousers were roughly patched with mismatched coarse thread.
    He took a sip of the bitter synth-coffee, grimacing with cynical weariness. Later, inside the concrete detention cell, he sat huddled in the corner by the cold steel bars, his head resting on his drawn-up knees as supernatural drowsiness dragged him into darkness.
    """

    print("=" * 75)
    print("  RUNNING CHARACTER CONTINUITY SUBAGENT (LIVE AI THINKING)")
    print("=" * 75)

    spec = agent.observe_and_synthesize_character(
        character_name="Sunny",
        passage_text=chapter_1_sunny_passage,
        arc_hierarchy=arc_context,
        gender="male",
        deterministic_seed=42,
    )

    # Resolve Inpainting Anchors for Chapter 1's Scenes
    shot_anchors = [
        agent.resolve_shot_character_anchor(
            spec,
            shot_action="sitting on curb by bench, holding paper coffee cup with both hands",
            emotion="bitter cynical grimace, weary sideways glance",
            camera_framing="MEDIUM",
        ),
        agent.resolve_shot_character_anchor(
            spec,
            shot_action="holding paper coffee cup near lips, grimacing at taste",
            emotion="sharp cynical dark eyes, grimace",
            camera_framing="CLOSE_UP",
        ),
        agent.resolve_shot_character_anchor(
            spec,
            shot_action="sitting huddled in cell corner by steel bars, knees drawn up to chest",
            emotion="drowsy drooping eyelids, head slumping in supernatural trance",
            camera_framing="MEDIUM",
        ),
    ]

    output_payload = {
        "character_spec": spec.model_dump(),
        "resolved_shot_anchors": [a.model_dump() for a in shot_anchors],
    }

    output_file = Path("output/character_subagent_live_output.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)

    print(f"\n[SUCCESS] Character subagent output written to:\n         {output_file}")

if __name__ == "__main__":
    main()

