"""Unit and integration tests for CompositingSubagent.

Verifies:
1. Multi-subagent prompt assembly and de-duplication without token bleed.
2. Workflow compilation across 3 compositing modes:
   - Inpaint Differential (freezing location stage plate)
   - ControlNet + IP-Adapter Fusion (depth guidance + face chip)
   - Multi-Subject Battle (character vs. creature)
3. End-to-end shot execution with deterministic mock canvas fallback.
"""

from pathlib import Path
import pytest

from app.schemas.character_subagent import DanbooruIdentityLock, ShotCharacterAnchor
from app.schemas.compositing_subagent import (
    BoundingBox2D,
    CompositingModeEnum,
    RenderedShotArtifact,
    ShotCompositionSpec,
)
from app.schemas.creature_subagent import ShotCreatureAnchor
from app.schemas.environment_subagent import (
    ShotEnvironmentAnchor,
    TimeOfDayEnum,
    WeatherConditionEnum,
)
from app.schemas.location_subagent import ArcHierarchyTag, PlateRole, ShotLocationAnchor
from app.agents.compositing_subagent import CompositingSubagent


@pytest.fixture
def arc_tag():
    return ArcHierarchyTag(
        main_arc_id="ARC_01_FIRST_NIGHTMARE",
        sub_arc_id="SUBARC_01_01_OUTSKIRTS",
        mini_arc_id="MINIARC_01_01_01_AWAKENING",
    )


@pytest.fixture
def char_anchor():
    return ShotCharacterAnchor(
        entity_id="char_sunny",
        entity_name="Sunny",
        full_positive_prompt="1boy, short black hair, onyx eyes, pale skin, sallow skin, scrawny, oversized charcoal utility jacket, frayed seams, dirty clothes, shivering on bench",
        full_negative_prompt="blonde hair, blue eyes, muscular, beard, clean clothes, armor",
        face_reference_path=None,
        expression_tag="exhausted drooping eyelids with cynical smirk",
        pose_action_tag="shivering on bench inside oversized jacket",
        recommended_inpaint_box=[0.25, 0.25, 0.75, 0.90],
    )


@pytest.fixture
def env_anchor():
    return ShotEnvironmentAnchor(
        environment_card_id="env_outskirts_twilight",
        time_of_day=TimeOfDayEnum.TWILIGHT_BLUE_HOUR,
        weather_condition=WeatherConditionEnum.OVERCAST_SMOG,
        resolved_lighting_block="sodium vapor streetlamp, warm amber glow, flickering light, volumetric lighting, cyan ambient light, rim light, overcast, smog, wet asphalt, puddle reflection",
        resolved_negative_block="sunny blue sky, bright daylight, clear sky, lush green grass, clean dry ground",
        key_light_vector="top_left",
        shadow_direction="bottom_right",
        surface_moisture_level=0.6,
    )


@pytest.fixture
def loc_anchor():
    return ShotLocationAnchor(
        location_id="loc_outskirts_transit_bench_n24",
        canonical_name="Outskirts Transit Bench N24",
        resolved_role=PlateRole.ACTION_STAGE,
        selected_plate_image_path="location_matrix_samples/loc_outskirts_transit_bench_n24_action_stage.png",
        depth_map_path="location_matrix_samples/loc_outskirts_transit_bench_n24_action_stage_depth.png",
        lineart_map_path="location_matrix_samples/loc_outskirts_transit_bench_n24_action_stage_lineart.png",
        lighting_baseline="outskirts transit station, battered rusted metal bench, cracked concrete sidewalk, empty street",
        weather_condition="overcast smog",
        recommended_inpaint_box=[0.25, 0.25, 0.75, 0.90],
        temporal_delta_tags=["battered bench", "cracked curb"],
    )


def test_assemble_shot_prompts_deduplication(arc_tag, char_anchor, env_anchor, loc_anchor):
    """Test merging and de-duplication of prompt tokens across all subagents."""
    agent = CompositingSubagent(dispatcher=None)

    spec = ShotCompositionSpec(
        shot_id="shot_test_001",
        novel_arc=arc_tag,
        character_anchor=char_anchor,
        environment_anchor=env_anchor,
        location_anchor=loc_anchor,
    )

    pos, neg = agent.assemble_shot_prompts(spec)

    # 1. Verify Positive Prompt Assembly
    assert "masterpiece" in pos
    assert "1boy" in pos
    assert "sodium vapor streetlamp" in pos
    assert "battered rusted metal bench" in pos
    assert "shivering on bench" in pos

    # Verify no accidental token duplication (e.g. '1boy' appears once)
    tokens_lower = [t.strip().lower() for t in pos.split(",")]
    assert tokens_lower.count("1boy") == 1

    # 2. Verify Negative Prompt Assembly
    assert "worst quality" in neg
    assert "blonde hair" in neg
    assert "sunny blue sky" in neg


def test_compile_workflow_differential_inpaint(arc_tag, char_anchor, env_anchor, loc_anchor):
    """Test compiling WF-EDIT-INPAINT for freezing location plate and inpainting character."""
    agent = CompositingSubagent(dispatcher=None)

    spec = ShotCompositionSpec(
        shot_id="shot_inpaint_001",
        novel_arc=arc_tag,
        mode=CompositingModeEnum.INPAINT_DIFFERENTIAL,
        location_plate_path="location_matrix_samples/loc_outskirts_transit_bench_n24_action_stage.png",
        character_anchor=char_anchor,
        environment_anchor=env_anchor,
        location_anchor=loc_anchor,
        character_bbox=BoundingBox2D(x=300, y=200, width=500, height=500),
    )

    pos, neg = agent.assemble_shot_prompts(spec)
    workflow = agent.compile_workflow(spec, pos, neg)

    # Verify presence of DifferentialDiffusion & Inpainting nodes
    class_types = [node.get("class_type") for node in workflow.values()]
    assert "DifferentialDiffusion" in class_types
    assert "InpaintModelConditioning" in class_types
    assert "MaskComposite" in class_types
    assert "KSampler" in class_types


def test_compile_workflow_controlnet_ipadapter_fusion(arc_tag, char_anchor, env_anchor, loc_anchor):
    """Test compiling WF-SHOT-SOLO with ControlNet Depth map and IPAdapter Face chip."""
    agent = CompositingSubagent(dispatcher=None)

    char_with_face = char_anchor.model_copy()
    char_with_face.face_reference_path = "output/char_sunny_canonical_face_chip.png"

    spec = ShotCompositionSpec(
        shot_id="shot_fusion_001",
        novel_arc=arc_tag,
        mode=CompositingModeEnum.CONTROLNET_IPADAPTER_FUSION,
        location_depth_path="location_matrix_samples/loc_outskirts_transit_bench_n24_action_stage_depth.png",
        character_anchor=char_with_face,
        environment_anchor=env_anchor,
        location_anchor=loc_anchor,
    )

    pos, neg = agent.assemble_shot_prompts(spec)
    workflow = agent.compile_workflow(spec, pos, neg)

    class_types = [node.get("class_type") for node in workflow.values()]
    assert "IPAdapterAdvanced" in class_types
    assert "ControlNetApplyAdvanced" in class_types
    assert "FaceDetailer" in class_types


def test_compile_workflow_multi_subject_battle(arc_tag, char_anchor, env_anchor):
    """Test compiling WF-SHOT-MULTI for Character vs. Creature combat shot."""
    agent = CompositingSubagent(dispatcher=None)

    creature_anchor = ShotCreatureAnchor(
        creature_card_id="creature_mountain_centipede",
        resolved_creature_prompt="segmented black chitin, multiple legs, mandibles, cluster of red eyes, dark mist, lunging strike",
        resolved_negative_prompt="1boy, 1girl, human",
        head_chip_path="output/bestiary_canon/creature_head_chip.png",
        suggested_spatial_anchor="right_flank",
    )

    spec = ShotCompositionSpec(
        shot_id="shot_battle_001",
        novel_arc=arc_tag,
        mode=CompositingModeEnum.MULTI_SUBJECT_BATTLE,
        character_anchor=char_anchor,
        environment_anchor=env_anchor,
        creature_anchor=creature_anchor,
    )

    pos, neg = agent.assemble_shot_prompts(spec)
    workflow = agent.compile_workflow(spec, pos, neg)

    class_types = [node.get("class_type") for node in workflow.values()]
    assert "ConditioningSetMask" in class_types
    assert "MaskComposite" in class_types


def test_execute_composite_shot_offline_mock(arc_tag, char_anchor, env_anchor, loc_anchor, tmp_path):
    """Test executing composite shot with deterministic mock canvas fallback."""
    agent = CompositingSubagent(dispatcher=None, allow_offline_fallback=True)

    spec = ShotCompositionSpec(
        shot_id="shot_ch01_sc01_001",
        novel_arc=arc_tag,
        mode=CompositingModeEnum.CONTROLNET_IPADAPTER_FUSION,
        character_anchor=char_anchor,
        environment_anchor=env_anchor,
        location_anchor=loc_anchor,
    )

    artifact = agent.execute_composite_shot(spec=spec, output_dir=tmp_path, force_mock=True)

    assert isinstance(artifact, RenderedShotArtifact)
    assert artifact.shot_id == "shot_ch01_sc01_001"
    assert Path(artifact.image_path).exists()
    assert artifact.qc_passed is True
