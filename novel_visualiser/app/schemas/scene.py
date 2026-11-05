"""Scene Direction & Production contracts.

Schemas for shot planning, prompt compilation, quality control evaluation,
narration scripting with dual subtitles, and film editing assembly. These
contracts drive the visual production pipeline from scene direction through
to final video output.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SubjectSpec(BaseModel):
    """A single entity's placement within a shot."""

    entity_id: str = Field(description="Canonical entity identifier")
    entity_name: str = Field(description="Display name of the entity")
    region_box: list[float] = Field(
        description="[x1, y1, x2, y2] normalised 0.0-1.0 bounding box for entity placement"
    )
    pose: str | None = Field(default=None, description="Body pose directive")
    expression: str | None = Field(
        default=None, description="Facial expression directive"
    )
    z_order: int = Field(
        default=0, description="Front-to-back layering order (lower is closer)"
    )


class ShotSpec(BaseModel):
    """Complete specification for one visual shot, output by Scene Director Agent."""

    shot_number: int = Field(description="Sequential shot number within the chapter")
    source_passages: list[str] = Field(
        description="Passage IDs that this shot visualises"
    )
    workflow_id: str = Field(description="ComfyUI workflow identifier, WF-001 to WF-008")
    shot_type: Literal[
        "close_up",
        "medium_close_up",
        "medium",
        "medium_wide",
        "wide",
        "extreme_wide",
        "two_shot",
        "group",
        "pov",
        "over_the_shoulder",
        "insert",
    ] = Field(description="Camera framing classification")
    camera_angle: Literal[
        "eye_level",
        "low_angle_dramatic",
        "high_angle",
        "dutch_angle",
        "birds_eye",
        "worms_eye",
        "profile",
    ] = Field(description="Camera angle for this shot")
    location_id: str | None = Field(
        default=None, description="Location entity ID for scene backdrop"
    )
    subjects: list[SubjectSpec] = Field(
        default_factory=list, description="Entities placed within this shot"
    )
    lighting_override: str | None = Field(
        default=None,
        description="Lighting directive, e.g. 'cold moonlight with harsh rim shadows'",
    )
    vfx_overlay_tag: str | None = Field(
        default=None,
        description="VFX overlay identifier, e.g. 'falling_snow_heavy', 'fire_embers'",
    )
    motion_tier: Literal["static_parallax", "i2v_climax"] = Field(
        default="static_parallax",
        description="Motion complexity tier for this shot",
    )
    camera_motion_preset: str | None = Field(
        default=None,
        description="Camera motion preset, e.g. 'slow_zoom_in', 'pan_left_to_right'",
    )
    narration_text: str | None = Field(
        default=None, description="The prose passage this shot covers"
    )
    mood: str | None = Field(
        default=None, description="Emotional tone for this shot"
    )


class CompiledPrompt(BaseModel):
    """The final ComfyUI-ready prompt payload, output by the Jinja2 Prompt Compiler."""

    shot_id: str = Field(description="Identifier linking back to the ShotSpec")
    workflow_id: str = Field(description="ComfyUI workflow identifier")
    positive_prompt: str = Field(description="Positive generation prompt")
    negative_prompt: str = Field(description="Negative generation prompt")
    width: int = Field(default=1024, description="Output image width in pixels")
    height: int = Field(default=768, description="Output image height in pixels")
    seed: int | None = Field(
        default=None, description="Random seed for reproducible generation"
    )
    steps: int = Field(default=30, description="Number of diffusion steps")
    cfg_scale: float = Field(
        default=7.0, description="Classifier-free guidance scale"
    )
    sampler: str = Field(default="euler", description="Sampler algorithm name")
    scheduler: str = Field(default="normal", description="Scheduler type")
    checkpoint: str | None = Field(
        default=None, description="Checkpoint model name or path"
    )
    # Per-subject regional conditioning
    regional_prompts: list[dict] | None = Field(
        default=None,
        description="Per-subject regional conditioning: [{region_box, prompt, ip_adapter_path, lora}]",
    )
    # Reference images
    ip_adapter_images: list[str] = Field(
        default_factory=list, description="Paths to IP-Adapter reference images"
    )
    controlnet_images: list[str] = Field(
        default_factory=list, description="Paths to ControlNet input images"
    )
    # Full ComfyUI node parameter payload
    comfyui_workflow_params: dict | None = Field(
        default=None,
        description="Complete ComfyUI workflow node parameters for direct API submission",
    )


class QCResult(BaseModel):
    """Quality control evaluation result for a generated image."""

    job_id: str = Field(description="Generation job identifier")
    shot_id: str = Field(description="Shot identifier being evaluated")
    # Gate 1: Technical (OpenCV)
    gate1_passed: bool = Field(description="Whether Gate 1 (technical quality) passed")
    gate1_blur_score: float | None = Field(
        default=None, description="Laplacian variance blur score"
    )
    gate1_exposure_ok: bool | None = Field(
        default=None, description="Whether exposure is within acceptable range"
    )
    gate1_subject_count: int | None = Field(
        default=None, description="Number of subjects detected in the image"
    )
    gate1_expected_subjects: int | None = Field(
        default=None, description="Expected number of subjects from the shot spec"
    )
    # Gate 2: Identity (InsightFace/CLIP)
    gate2_passed: bool | None = Field(
        default=None, description="Whether Gate 2 (identity consistency) passed"
    )
    gate2_face_similarity: float | None = Field(
        default=None, description="Cosine similarity of face embeddings"
    )
    gate2_style_similarity: float | None = Field(
        default=None, description="CLIP style similarity score"
    )
    gate2_threshold: float = Field(
        default=0.6, description="Minimum similarity threshold for Gate 2"
    )
    # Gate 3: Semantic (VLM)
    gate3_passed: bool | None = Field(
        default=None, description="Whether Gate 3 (semantic accuracy) passed"
    )
    gate3_checklist_results: dict | None = Field(
        default=None,
        description="Semantic checklist results, e.g. {'wearing_armor': true, 'holding_sword': false}",
    )
    gate3_vlm_response: str | None = Field(
        default=None, description="Raw VLM evaluation response"
    )
    # Overall
    overall_passed: bool = Field(description="Whether all applicable gates passed")
    overall_score: float = Field(
        ge=0.0, le=1.0, description="Aggregate quality score from 0.0 to 1.0"
    )
    failure_reason: str | None = Field(
        default=None, description="Human-readable failure explanation if any gate failed"
    )
    recommended_action: (
        Literal[
            "approve",
            "retry_new_seed",
            "retry_higher_weight",
            "manual_review",
            "inpaint_face",
        ]
        | None
    ) = Field(default=None, description="Suggested remediation action")


class NarrationSegment(BaseModel):
    """One narration block linked to a shot, with TTS script and dual subtitles."""

    shot_id: str = Field(description="Shot this narration segment accompanies")
    segment_index: int = Field(
        description="Sequential index within the chapter narration plan"
    )
    hinglish_tts_script: str = Field(
        description="Hinglish TTS script, e.g. 'Andhere mein uski saansein ruk gayi...'"
    )
    subtitle_en: str = Field(
        description="Clean English subtitle, ideally under 42 characters"
    )
    subtitle_hi: str = Field(description="Devanagari Hindi subtitle")
    prosody_tag: (
        Literal[
            "calm_narration",
            "tense_whisper",
            "dramatic_rise",
            "intense_action",
            "gentle_reflection",
        ]
        | None
    ) = Field(default=None, description="TTS prosody/emotion tag")
    pause_before_ms: int = Field(
        default=0, description="Silence in milliseconds before this segment"
    )
    pause_after_ms: int = Field(
        default=200, description="Silence in milliseconds after this segment"
    )


class ChapterNarrationPlan(BaseModel):
    """Complete narration plan for an entire chapter."""

    chapter_number: int = Field(description="Chapter number this plan covers")
    total_segments: int = Field(description="Total number of narration segments")
    segments: list[NarrationSegment] = Field(
        description="Ordered list of narration segments"
    )
    estimated_duration_seconds: float = Field(
        description="Estimated total audio duration in seconds"
    )


class FilmEditSpec(BaseModel):
    """Editing specification for assembling one chapter's final video."""

    chapter_number: int = Field(description="Chapter number being assembled")
    shots: list[dict] = Field(
        description=(
            "Ordered shot list: [{shot_id, rgb_path, depth_path, audio_path, "
            "duration_ms, motion_tier, camera_motion, vfx_overlay}]"
        )
    )
    background_music_track: str | None = Field(
        default=None, description="Path to mood-based background music file"
    )
    sfx_cues: list[dict] = Field(
        default_factory=list,
        description="Sound effect cues: [{shot_id, sfx_file, volume_db}]",
    )
    subtitle_en_path: str | None = Field(
        default=None, description="Path to English subtitle file (SRT/ASS)"
    )
    subtitle_hi_path: str | None = Field(
        default=None, description="Path to Hindi subtitle file (SRT/ASS)"
    )
    output_path: str = Field(description="Output video file path")
    resolution: tuple[int, int] = Field(
        default=(1920, 1080), description="Output video resolution (width, height)"
    )
    fps: int = Field(default=60, description="Output video frame rate")
    audio_duck_db: float = Field(
        default=-12.0,
        description="Music ducking level in dB when narration is active",
    )
