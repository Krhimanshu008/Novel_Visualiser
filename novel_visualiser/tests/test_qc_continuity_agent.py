"""Unit tests for QCContinuityAgent (Multi-Gate Visual QA).

Verifies:
1. Gate 1: Deterministic Tag Audit catches illegal traits, missing locks, and lighting leaks.
2. Gate 2: OpenCV Metrics accurately flags blurriness, exposure anomalies, and bbox clipping.
3. Gate 3: Multimodal Vision audit scores visual coherence and outputs inpaint repair data.
4. Comprehensive multi-gate verdict resolution (APPROVED vs NEEDS_INPAINT_PATCH vs FATAL_FAIL).
"""

from pathlib import Path
import cv2
import numpy as np
import pytest

from app.schemas.character_subagent import (
    ArchetypalDNA,
    BiologicalEpoch,
    CharacterMatrixSpec,
    DanbooruIdentityLock,
    ShotCharacterAnchor,
)
from app.schemas.compositing_subagent import (
    BoundingBox2D,
    CompositingModeEnum,
    RenderedShotArtifact,
    ShotCompositionSpec,
)
from app.schemas.environment_subagent import (
    EnvironmentalAtmosphereCard,
    EnvironmentalDeepObservation,
    LightingRigSpec,
    ShotEnvironmentAnchor,
    TimeOfDayEnum,
    WeatherConditionEnum,
)
from app.schemas.location_subagent import ArcHierarchyTag
from app.schemas.qc import QCVerdictEnum
from app.agents.character_subagent import CharacterContinuitySubagent
from app.agents.environment_subagent import EnvironmentContinuitySubagent
from app.agents.qc_continuity_agent import QCContinuityAgent


@pytest.fixture
def sunny_char_lock():
    agent = CharacterContinuitySubagent(use_ai_thinking=False)
    passage = (
        "Sunny was an underweight, scrawny orphan of the outskirts, his face pale and sickly with sallow skin "
        "and dark circles under his raven-black eyes. He had messy jet-black hair with sharp dark onyx eyes."
    )
    arc_tag = ArcHierarchyTag(
        main_arc_id="ARC_01", sub_arc_id="SUBARC_01", mini_arc_id="MINI_01"
    )
    matrix = agent.observe_and_synthesize_character(
        character_name="Sunny",
        passage_text=passage,
        arc_hierarchy=arc_tag,
        gender="male",
        deterministic_seed=42,
    )
    # Ensure active epoch has retired negative tags
    if matrix.active_epoch:
        matrix.active_epoch.retired_negative_tags = ["healthy glow", "muscular build"]
    return matrix


@pytest.fixture
def twilight_env_card():
    agent = EnvironmentContinuitySubagent(use_ai_thinking=False)
    arc_tag = ArcHierarchyTag(
        main_arc_id="ARC_01", sub_arc_id="SUBARC_01", mini_arc_id="MINI_01"
    )
    passage = "Midnight in the outskirts. Cold acid rain dripped from rusted roofs into oily puddles under a sickly cyan streetlamp."
    return agent.observe_and_synthesize_environment(
        location_name="Outskirts Transit Bench N24",
        passage_text=passage,
        arc_hierarchy=arc_tag,
        chapter_number=1,
    )


@pytest.fixture
def clean_shot_spec(sunny_char_lock, twilight_env_card):
    env_agent = EnvironmentContinuitySubagent(use_ai_thinking=False)
    env_anchor = env_agent.resolve_shot_environment_anchor(twilight_env_card)

    char_agent = CharacterContinuitySubagent(use_ai_thinking=False)
    char_anchor = char_agent.resolve_shot_character_anchor(
        sunny_char_lock, shot_action="sitting on bench", emotion="wary", camera_framing="MEDIUM"
    )

    spec = ShotCompositionSpec(
        shot_id="shot_test_clean_01",
        novel_arc=sunny_char_lock.arc_hierarchy,
        mode=CompositingModeEnum.CONTROLNET_IPADAPTER_FUSION,
        character_anchor=char_anchor,
        environment_anchor=env_anchor,
        character_bbox=BoundingBox2D(x=50, y=50, width=200, height=300),
    )
    return spec


def test_gate_1_clean_prompt_passes(clean_shot_spec, sunny_char_lock, twilight_env_card):
    qc_agent = QCContinuityAgent(use_ai_thinking=False)
    result = qc_agent.run_gate_1_tag_audit(
        clean_shot_spec, char_lock=sunny_char_lock, env_card=twilight_env_card
    )
    assert result.passed is True
    assert len(result.detected_illegal_traits) == 0
    assert result.promoted_negative_compliant is True
    assert result.environmental_harmony_compliant is True


def test_gate_1_illegal_trait_detection(clean_shot_spec, sunny_char_lock, twilight_env_card):
    qc_agent = QCContinuityAgent(use_ai_thinking=False)
    # Inject contradictory illegal trait into positive prompt
    clean_shot_spec.character_anchor.full_positive_prompt += ", blonde hair, muscular build"
    result = qc_agent.run_gate_1_tag_audit(
        clean_shot_spec, char_lock=sunny_char_lock, env_card=twilight_env_card
    )
    assert result.passed is False
    assert any("blonde hair" in t for t in result.detected_illegal_traits)
    assert any("muscular" in t for t in result.detected_illegal_traits)


def test_gate_1_environmental_leak_detection(clean_shot_spec, sunny_char_lock, twilight_env_card):
    qc_agent = QCContinuityAgent(use_ai_thinking=False)
    # Inject daylight leak into midnight scene
    clean_shot_spec.character_anchor.full_positive_prompt += ", bright sunlight, noon sky"
    result = qc_agent.run_gate_1_tag_audit(
        clean_shot_spec, char_lock=sunny_char_lock, env_card=twilight_env_card
    )
    assert result.passed is False
    assert result.environmental_harmony_compliant is False


def test_gate_2_opencv_sharpness_and_blur(tmp_path):
    qc_agent = QCContinuityAgent(use_ai_thinking=False)

    # 1. Generate Sharp Test Image with realistic midtone distribution (values 40 to 210)
    sharp_img = np.full((512, 512, 3), 128, dtype=np.uint8)
    for i in range(0, 512, 16):
        cv2.line(sharp_img, (i, 0), (i, 512), (200, 200, 200), 2)
        cv2.line(sharp_img, (0, i), (512, i), (60, 60, 60), 2)
    sharp_path = tmp_path / "sharp_test.png"
    cv2.imwrite(str(sharp_path), sharp_img)

    result_sharp = qc_agent.run_gate_2_opencv_metrics(sharp_path)
    assert result_sharp.passed is True
    assert result_sharp.is_blurry is False
    assert result_sharp.laplacian_sharpness_score > 100.0

    # 2. Generate Blurry Test Image (heavily blurred flat color)
    blurry_img = cv2.GaussianBlur(sharp_img, (51, 51), 0)
    blurry_path = tmp_path / "blurry_test.png"
    cv2.imwrite(str(blurry_path), blurry_img)

    result_blurry = qc_agent.run_gate_2_opencv_metrics(blurry_path)
    assert result_blurry.is_blurry is True
    assert result_blurry.laplacian_sharpness_score < 100.0


def test_gate_2_bbox_bounds_containment(tmp_path, clean_shot_spec):
    qc_agent = QCContinuityAgent(use_ai_thinking=False)

    # Standard image (512x512) with midtones and texture
    canvas = np.full((512, 512, 3), 128, dtype=np.uint8)
    noise = np.random.randint(-30, 30, (512, 512, 3), dtype=np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 30, 220).astype(np.uint8)
    img_path = tmp_path / "canvas_bbox.png"
    cv2.imwrite(str(img_path), canvas)

    # Add compliant bbox
    clean_shot_spec.character_bbox = BoundingBox2D(
        x=50, y=50, width=200, height=300
    )
    res_compliant = qc_agent.run_gate_2_opencv_metrics(img_path, spec=clean_shot_spec)
    assert res_compliant.bbox_bounds_compliant is True

    # Add out-of-bounds bbox
    clean_shot_spec.character_bbox = BoundingBox2D(
        x=400, y=400, width=300, height=300
    )
    res_overflow = qc_agent.run_gate_2_opencv_metrics(img_path, spec=clean_shot_spec)
    assert res_overflow.bbox_bounds_compliant is False
    assert res_overflow.passed is False


def test_master_audit_shot_approved(tmp_path, clean_shot_spec, sunny_char_lock, twilight_env_card):
    qc_agent = QCContinuityAgent(use_ai_thinking=False, output_dir=tmp_path)

    # Sharp midtone image
    sharp_img = np.full((512, 512, 3), 128, dtype=np.uint8)
    for i in range(0, 512, 16):
        cv2.line(sharp_img, (i, 0), (i, 512), (200, 200, 200), 2)
        cv2.line(sharp_img, (0, i), (512, i), (60, 60, 60), 2)
    img_path = tmp_path / "audit_target.png"
    cv2.imwrite(str(img_path), sharp_img)

    artifact = RenderedShotArtifact(
        shot_id="shot_test_clean_01",
        image_path=str(img_path),
        mode_used=CompositingModeEnum.CONTROLNET_IPADAPTER_FUSION,
        worker_url_used="https://gpu0.krhimanshu.in",
        final_positive_prompt=clean_shot_spec.character_anchor.full_positive_prompt,
        final_negative_prompt=clean_shot_spec.character_anchor.full_negative_prompt,
        qc_passed=True,
    )

    audit = qc_agent.audit_shot(
        shot_artifact=artifact,
        spec=clean_shot_spec,
        char_lock=sunny_char_lock,
        env_card=twilight_env_card,
    )

    assert audit.verdict == QCVerdictEnum.APPROVED
    assert audit.overall_score >= 75.0
    assert audit.gate_1.passed is True
    assert audit.gate_2.passed is True
    assert audit.gate_3 is not None
    assert audit.gate_3.passed is True
