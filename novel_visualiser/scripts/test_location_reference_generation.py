"""Novel Visualiser — Live Dual-GPU Location Reference Generation Test.

Tests 4-Plate Standardized Matrix generation for a highly detailed location with intricate items
and background architecture, dispatched across GPU 0 (https://gpu0.krhimanshu.in) and
GPU 1 (https://gpu1.krhimanshu.in).
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.location_subagent import LocationContinuitySubagent
from app.schemas.location_subagent import ArcHierarchyTag
from app.services.comfyui_client import ComfyUIWorkflowCompiler, DualGPUComfyUIDispatcher


def run_location_reference_test():
    print("=" * 70)
    print(" NOVEL VISUALISER — DUAL-GPU LOCATION REFERENCE GENERATION")
    print("=" * 70)

    # 1. Dual-GPU Cluster Configuration
    gpu_urls = [
        "https://gpu0.krhimanshu.in",
        "https://gpu1.krhimanshu.in",
    ]
    print(f"\n[1/5] Initializing Dual-GPU Cluster:")
    dispatcher = DualGPUComfyUIDispatcher(urls=gpu_urls)
    
    online_workers = dispatcher.get_online_clients()
    for worker in online_workers:
        health = worker.check_health()
        print(f"  🟢 {worker.base_url:<30} -> {health.get('gpu_name', 'GPU')} ({health.get('vram_free_gb', 0)} GB free VRAM)")

    if not online_workers:
        print("❌ No GPU workers are online! Exiting.")
        sys.exit(1)

    # 2. Rich Environmental Scene Description with Detailed Items & Background
    location_name = "Dark City Cathedral Relic Chamber"
    passage_text = """
    Deep inside the ruined gothic cathedral of the Dark City, ancient stone pillars carved
    with twisting serpents supported a cracked vaulted ceiling where pale luminescent moss
    clung in thick clusters. Below, scattered across rotting mahogany workbenches and heavy
    iron-bound chests, lay forgotten arcane instruments: brass astrolabes with verdigris tarnish,
    etched soul shards glowing with faint ethereal purple light, heaps of weathered parchment scrolls
    tied with frayed leather cords, and cracked glass alembics containing crystallised alchemical residue.
    Rain dripped through high arched clerestory windows, splashing rhythmically onto the broken marble
    floor where black reflective water pooled between sunken flagstones. A cold midnight draft stirred
    dust motes, illuminated by glowing runic engravings along the elevated stone altar dais.
    """

    arc_tag = ArcHierarchyTag(
        main_arc_id="ARC_01_FIRST_NIGHTMARE",
        sub_arc_id="SUBARC_DARK_CITY",
        mini_arc_id="MINIARC_CATHEDRAL_SANCTUM",
        in_universe_day=14,
        time_of_day="midnight",
        weather="rainstorm and cold mist",
    )

    # 3. Observational Reasoning & 4-Plate Standard Synthesis
    print(f"\n[2/5] Synthesizing 5-Dimension Observational Analysis for '{location_name}'...")
    agent = LocationContinuitySubagent(comfy_dispatcher=dispatcher, use_ai_thinking=False)
    
    # We craft highly expressive, detailed prompts for the 4-plate matrix
    neg_prompt = (
        "1boy, 1girl, character, person, human, face, crowd, silhouette, "
        "worst quality, low quality, blurry, deformed, watermark, signature, modern clothing"
    )

    matrix_spec = agent.observe_and_synthesize_location(
        location_name=location_name,
        passage_text=passage_text,
        arc_hierarchy=arc_tag,
        deterministic_seed=7770,
    )

    # Inject rich item details into the prompts
    matrix_spec.plate_1_vista.prompt = (
        "masterpiece, best quality, ultra-detailed anime background art, wide establishing vista shot, "
        "ruined gothic cathedral interior, towering stone pillars carved with serpents, high vaulted ceiling, "
        "cracked clerestory windows with cold rain pouring outside, dark puddles reflecting moonlight on fractured marble flagstones, "
        "glowing purple luminescent moss, volumetric shafts of pale cyan moonlight, moody atmospheric mist, empty sanctum, no humans"
    )
    matrix_spec.plate_1_vista.negative_prompt = neg_prompt

    matrix_spec.plate_2_stage.prompt = (
        "masterpiece, best quality, ultra-detailed anime background art, medium eye-level action stage shot, "
        "arcane relic workbench in gothic chamber, ancient rotting mahogany table cluttered with detailed items: "
        "tarnished brass astrolabe, glowing violet soul crystal shards, scattered parchment scrolls tied with leather cord, "
        "cracked glass alembic bottles with dried alchemical powder, heavy iron-bound chest, damp stone floor, "
        "dramatic rim lighting, ethereal purple glow, atmospheric dust motes, empty room, no humans"
    )
    matrix_spec.plate_2_stage.negative_prompt = neg_prompt

    matrix_spec.plate_3_reverse.prompt = (
        "masterpiece, best quality, ultra-detailed anime background art, 180-degree reciprocal reverse shot, "
        "looking toward the grand gothic archway exit, flooded marble nave, broken altar steps, fallen stone gargoyle statue, "
        "deep shadows, cold dark blue and purple ambient lighting, puddles of dark water, rain dripping from arches, no humans"
    )
    matrix_spec.plate_3_reverse.negative_prompt = neg_prompt

    matrix_spec.plate_4_macro.prompt = (
        "masterpiece, best quality, ultra-detailed anime background art, extreme close-up macro texture shot, "
        "intricate antique brass astrolabe on weathered dark wood grain, glowing violet crystal soul shard casting soft purple light, "
        "verdigris copper patina, finely engraved astronomical gears, water droplets on polished bronze, shallow depth of field, cinematic lighting"
    )
    matrix_spec.plate_4_macro.negative_prompt = neg_prompt

    print(f"  * Plate 1 (Vista)   : {matrix_spec.plate_1_vista.width}x{matrix_spec.plate_1_vista.height} - Wide Cathedral Vista")
    print(f"  * Plate 2 (Stage)   : {matrix_spec.plate_2_stage.width}x{matrix_spec.plate_2_stage.height} - Relic Workbench & Altar Zone")
    print(f"  * Plate 3 (Reverse) : {matrix_spec.plate_3_reverse.width}x{matrix_spec.plate_3_reverse.height} - 180° Reciprocal Exit View")
    print(f"  * Plate 4 (Macro)   : {matrix_spec.plate_4_macro.width}x{matrix_spec.plate_4_macro.height} - Brass Astrolabe & Soul Crystal")

    # 4. Dispatch Generation Across Dual-GPU Cluster Concurrently
    output_dir = Path("output/location_references/dark_city_relic_chamber")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[3/5] Dispatching 4-Plate Generation across Dual-GPU Cluster in Parallel...")
    t0 = time.time()
    
    # We use DreamShaperXL Turbo for rich cinematic background & prop detail
    updated_spec = agent.generate_location_matrix(
        matrix_spec=matrix_spec,
        output_dir=output_dir,
        checkpoint="DreamShaperXL_Turbo_v2_1.safetensors",
    )
    
    total_time = time.time() - t0
    print(f"\n[4/5] Multi-Plate Generation completed in {total_time:.1f}s across 2 GPUs!")

    # 5. Report Generated Files & Structural Guides
    print("\n[5/5] Generated Reference Plates & OpenCV Guides:")
    plates = [
        ("Plate 1: Vista", updated_spec.plate_1_vista),
        ("Plate 2: Stage", updated_spec.plate_2_stage),
        ("Plate 3: Reverse", updated_spec.plate_3_reverse),
        ("Plate 4: Macro", updated_spec.plate_4_macro),
    ]

    for title, pl in plates:
        print(f"\n  === {title} ===")
        print(f"    RGB Plate    : {pl.image_path}")
        print(f"    3D Depth Map : {pl.depth_map_path}")
        print(f"    Lineart Wire : {pl.lineart_map_path}")
        if pl.image_path and Path(pl.image_path).exists():
            size_kb = Path(pl.image_path).stat().st_size / 1024
            print(f"    File Size    : {size_kb:.1f} KB")

    print("\n" + "=" * 70)
    print(" ALL 4 LOCATION REFERENCE PLATES SUCCESSFULLY GENERATED!")
    print("=" * 70)


if __name__ == "__main__":
    run_location_reference_test()

