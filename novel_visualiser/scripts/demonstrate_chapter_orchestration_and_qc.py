"""Novel Visualiser — Master Chapter Orchestration & Multi-Gate QC Demonstration.

Demonstrates:
1. Master Orchestrator (SceneDirectorAgent) decomposing Chapter 1 of 'Shadow Slave' into dramatic scenes.
2. Synchronized subagent delegation across all 5 autonomous subsystems:
   - LocationContinuitySubagent (Outskirts Transit Bench N24 - 4 plates, depth & canny wireframes)
   - EnvironmentContinuitySubagent (Twilight smog -> Midnight rain lighting rig, Kelvin temp, specular wetness)
   - CharacterContinuitySubagent & BiologicalEvolutionAuditor (Sunny's DNA, slum adolescent epoch, face chip)
   - CreatureMonsterSubagent (Mountain Centipede Larva Awakened Beast, Midnight Shard weapon)
   - CompositingSubagent (De-duplicated multi-layer prompt assembly, inpaint & controlnet specs)
3. Multi-Gate Visual Quality Assurance (QCContinuityAgent):
   - Gate 1: Deterministic Tag Audit (promoted negative tags, illegal contradictory traits, lighting harmony)
   - Gate 2: OpenCV Spatial Metrics (Laplacian sharpness >= 100, exposure clipping, bbox bounds)
   - Gate 3: Multimodal Vision Audit (Identity, lighting, wardrobe, anatomy)
4. Full SQLite canon persistence in output/novel_canon.db.
"""

from __future__ import annotations

import json
import logging
import os
import sys
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Setup workspace paths
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from app.database.models import Base, Chapter, Entity, Project, Scene, Shot
from app.schemas.location_subagent import ArcHierarchyTag
from app.agents.scene_director import SceneDirectorAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("novel_visualiser.live_orchestrator_demo")


def main():
    print("=" * 80)
    print("🌟 NOVEL VISUALISER — MASTER SCENE DIRECTOR & MULTI-GATE QC SYSTEM")
    print("=" * 80)

    # 1. Database Connection
    db_path = Path("output/novel_canon.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    # 2. Setup Arc Hierarchy Tag
    arc_tag = ArcHierarchyTag(
        main_arc_id="ARC_01_FIRST_NIGHTMARE",
        sub_arc_id="SUBARC_01_01_OUTSKIRTS",
        mini_arc_id="MINIARC_01_01_01_BENCH_WAITING",
    )

    # 3. Chapter 1 Raw Text
    chapter_1_title = "Chapter 1: Without a Home"
    chapter_1_text = (
        "Sunny sat shivering on the rusted transit bench in the outskirts. A toxic twilight smog "
        "hung heavy over the dilapidated concrete towers, casting long cyan shadows across the damp street. "
        "He pulled his oversized, frayed utility jacket tighter around his scrawny frame. His pale, sallow skin "
        "was slick with cold acid mist, and dark circles shadowed his obsidian onyx eyes.\n\n"
        "Suddenly, the sickening screech of chitin scraped against cracked asphalt. From the deep darkness of the alleyway, "
        "a monstrous centipede larva emerged. Its segmented carapace gleamed in the dim lamp light, and its mandibles dripped venom. "
        "Sunny gasped and drew the Midnight Shard, the dark obsidian blade gleaming faintly with sharp rim highlights."
    )

    # 4. Instantiate Master Orchestrator
    director = SceneDirectorAgent(
        db=db,
        use_ai_thinking=True,
        output_dir=Path("output/chapter_production_plans"),
    )

    # 5. Execute Chapter Orchestration
    print("\n[Phase 1] Orchestrating Chapter 1 Across All 5 Subagents...")
    plan = director.orchestrate_chapter(
        chapter_number=1,
        chapter_title=chapter_1_title,
        novel_text=chapter_1_text,
        arc_tag=arc_tag,
        project_id="shadow_slave",
    )

    print(f"\n✅ Production Plan Successfully Generated:")
    print(f"   - Novel Title: {plan.novel_title}")
    print(f"   - Chapter: {plan.chapter_number} ({plan.chapter_title})")
    print(f"   - Total Words: {plan.total_words}")
    print(f"   - Total Budgeted Shots: {plan.total_shots}")
    print(f"   - Estimated Duration: {plan.total_duration_sec:.1f} seconds")
    print(f"   - Scenes Orchestrated: {len(plan.scenes)}")

    for sc in plan.scenes:
        print(f"\n   🎬 Scene {sc.scene_number}: '{sc.scene_title}'")
        print(f"      Type: {sc.scene_type} | Pacing: {sc.pacing_curve} | Target Shots: {sc.target_shots}")
        print(f"      Location: {sc.location_matrix.canonical_name}")
        rig = sc.environment_card.lighting_rig
        kelvin = getattr(rig, "key_color_temperature_k", getattr(rig, "key_light_kelvin", 4000))
        fill_info = getattr(rig, "fill_light_source", getattr(rig, "fill_light_ratio", "ambient"))
        print(f"      Lighting Rig: {kelvin}K, Fill: {fill_info}")
        print(f"      Active Character: {sc.active_characters[0].entity_name}")
        if sc.active_creatures:
            print(f"      Creature in Battle: {sc.active_creatures[0].creature_card_id}")
        if sc.active_props:
            print(f"      Hero Item/Weapon: {sc.active_props[0].item_name}")

    # 6. Execute Render & Multi-Gate Visual QC
    print("\n[Phase 2] Executing Composited Shots & Multi-Gate Visual QA...")
    # Executing live compositing across Dual-GPU cluster with live VLM Gate 3 QA
    results = director.execute_and_audit_chapter(production_plan=plan, force_mock=False)

    print("\n[Phase 3] Multi-Gate Visual Quality Assurance Audit Ledger:")
    print("-" * 80)
    for item in results:
        shot_id = item["shot_id"]
        audit = item["qc_audit"]
        g1 = audit.gate_1
        g2 = audit.gate_2
        g3 = audit.gate_3

        status_icon = "🟢 PASS" if audit.verdict.value == "approved" else "🟡 PATCH"
        print(f"{status_icon} Shot: {shot_id}")
        print(f"      Verdict: {audit.verdict.value.upper()} | Composite Score: {audit.overall_score:.1f}/100")
        print(f"      Gate 1 (Prompt Contract): {'PASSED' if g1.passed else 'FAILED'} | Promoted Negatives: {g1.promoted_negative_compliant}")
        print(f"      Gate 2 (OpenCV Integrity): Sharpness={g2.laplacian_sharpness_score:.1f} (Blurry={g2.is_blurry}) | Clipped={g2.highlights_clipped_pct:.1f}%")
        if g3:
            print(f"      Gate 3 (Multimodal Vision): Identity={g3.identity_fidelity_score:.1f} | Lighting={g3.lighting_consistency_score:.1f} | Wardrobe={g3.wardrobe_fidelity_score:.1f} | Anatomy={g3.anatomical_integrity_score:.1f}")
        print(f"      Image Output: {item['artifact'].image_path}")
        print("-" * 80)

    # 7. Verify SQLite Database Records
    print("\n[Phase 4] Verifying SQLite Canon Persistence:")
    db_ch = db.query(Chapter).filter(Chapter.project_id == "shadow_slave", Chapter.chapter_number == 1).first()
    db_scenes = db.query(Scene).filter(Scene.chapter_id == db_ch.id).all()
    db_shots = db.query(Shot).filter(Shot.chapter_id == db_ch.id).all()
    print(f"   - Database Chapter: ID={db_ch.id}, Title='{db_ch.title}', Status={db_ch.status.value}")
    print(f"   - Database Scenes Count: {len(db_scenes)}")
    print(f"   - Database Shots Count: {len(db_shots)}")

    # 8. Save Comprehensive Output JSON
    output_summary = {
        "chapter_production_plan": plan.model_dump(),
        "qc_audits": [item["qc_audit"].model_dump() for item in results],
        "database_ledger": {
            "chapter_id": db_ch.id,
            "scenes": [s.id for s in db_scenes],
            "shots": [s.id for s in db_shots],
        },
    }
    summary_path = Path("output/chapter_1_orchestration_and_qc_live_manifest.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(output_summary, f, indent=2)

    print(f"\n🎉 Master Chapter Production Manifest Saved To: {summary_path.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
