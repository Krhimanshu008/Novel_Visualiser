"""
Novel Visualiser — Phase 5 Verification Test Suite
Tests Audio Engine, Hinglish Script Adapter, TTS, Dual Subtitles, and FFmpeg Film Assembly.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import imageio_ffmpeg
import numpy as np
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.database.models import (
    Base,
    Chapter,
    ChapterStatus,
    GenerationJob,
    JobStatus,
    MotionTier,
    NarrationSegment,
    Passage,
    PassageType,
    Project,
    Scene,
    Shot,
    ShotStatus,
    SubtitleFormat,
    SubtitleLanguage,
    SubtitleTrack,
)
from app.orchestrator.graph import generate_narration, render_film
from app.orchestrator.state import PipelineStage, ProductionState
from app.schemas.scene import FilmEditSpec
from app.services.audio_engine import (
    HinglishScriptAdapter,
    PhoneticLoreEnforcer,
    SubtitleGenerator,
    TTSEngine,
)
from app.services.film_editor import (
    CameraMotionPresets,
    FilmEditor,
    VFXOverlayPresets,
)


def get_test_env():
    """Create isolated temp directory and DB."""
    temp_dir = tempfile.mkdtemp(prefix="nv_phase5_test_")
    db_path = Path(temp_dir) / "test_p5.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    return {
        "dir": Path(temp_dir),
        "db_path": db_path,
        "engine": engine,
        "session_factory": SessionLocal,
    }


# =====================================================================
# Test 1: Phonetic Lore Enforcer
# =====================================================================
def test_01_phonetic_lore_enforcer():
    """Verify pronunciation shielding of fantasy lore terms and Hindi transliterations."""
    enforcer = PhoneticLoreEnforcer()

    raw_text = "Sunny met Nephis as an Aspirant before entering the Nightmare Spell."
    shielded = enforcer.shield_text(raw_text)

    assert "Neh-fis" in shielded, f"Failed Nephis shielding: {shielded}"
    assert "As-pi-rant" in shielded, f"Failed Aspirant shielding: {shielded}"
    assert "Night-mare Spell" in shielded, f"Failed Nightmare Spell shielding: {shielded}"

    assert enforcer.get_hindi_term("Sunny") == "सन्नी"
    assert enforcer.get_hindi_term("Nephis") == "नेफिस"
    assert enforcer.get_hindi_term("Nightmare Spell") == "नाइटमेयर स्पेल"
    print(" [PASS] Test 1: Phonetic lore pronunciation shielding verified.")


# =====================================================================
# Test 2: Hinglish Script Adapter
# =====================================================================
def test_02_hinglish_script_adapter():
    """Verify script adapter produces natural Hinglish, clean English subs, and Hindi subs."""
    adapter = HinglishScriptAdapter()

    passage = "Sunny sat on a rusty bench outside the police station, holding a cup of real coffee."
    segment = adapter.adapt_passage_to_segment(
        shot_id="shot_001",
        segment_index=0,
        passage_text=passage,
        mood="tense_whisper",
    )

    assert segment.shot_id == "shot_001"
    assert len(segment.hinglish_tts_script) > 10
    assert len(segment.subtitle_en) <= 60
    assert "सन्नी" in segment.subtitle_hi or len(segment.subtitle_hi) > 5
    assert segment.prosody_tag in ["tense_whisper", "calm_narration", "intense_action"]
    print(" [PASS] Test 2: Hinglish script adaptation & dual subtitle generation verified.")


# =====================================================================
# Test 3: Edge-TTS Speech Synthesis & Duration Parsing
# =====================================================================
def test_03_edge_tts_synthesis_and_duration(env):
    """Verify neural TTS speech generation and FFmpeg duration extraction."""
    tts = TTSEngine(voice="hi-IN-MadhurNeural")
    tts.output_dir = env["dir"] / "audio"
    tts.output_dir.mkdir(parents=True, exist_ok=True)

    text = "सन्नी पुलिस स्टेशन के सामने एक ज़ंग लगी बेंच पर बैठा था।"
    audio_file = tts.synthesize(text, "test_narration.mp3")

    assert audio_file.exists()
    assert audio_file.stat().st_size > 1000

    duration_ms = tts.get_audio_duration_ms(audio_file)
    assert duration_ms > 1000
    print(f" [PASS] Test 3: Synthesized {audio_file.name} (duration = {duration_ms} ms).")


# =====================================================================
# Test 4: Dual Subtitle Generation (.ass & .srt)
# =====================================================================
def test_04_subtitle_generation(env):
    """Verify ASS and SRT file creation with dual styles and timestamps."""
    segments = [
        {
            "subtitle_en": "Sunny sat on a rusty bench holding coffee.",
            "subtitle_hi": "सन्नी पुलिस स्टेशन के सामने बेंच पर बैठा था।",
            "duration_ms": 3000,
        },
        {
            "subtitle_en": "Everything went pitch black.",
            "subtitle_hi": "चारों तरफ गहरा अँधेरा छा गया।",
            "duration_ms": 2500,
        },
    ]

    ass_path = env["dir"] / "subtitles.ass"
    srt_path = env["dir"] / "subtitles.srt"

    SubtitleGenerator.generate_dual_ass(segments, ass_path)
    SubtitleGenerator.generate_srt(segments, srt_path, lang="en")

    assert ass_path.exists()
    assert srt_path.exists()

    ass_content = ass_path.read_text(encoding="utf-8")
    assert "Style: HindiLine" in ass_content
    assert "Style: EnglishLine" in ass_content
    assert "Dialogue: 0,0:00:00.00,0:00:03.00,HindiLine" in ass_content
    assert "Dialogue: 0,0:00:00.00,0:00:03.00,EnglishLine" in ass_content

    srt_content = srt_path.read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:03,000" in srt_content
    assert "Sunny sat on a rusty bench" in srt_content
    print(" [PASS] Test 4: Dual ASS and SRT subtitles generated successfully.")


# =====================================================================
# Test 5: FFmpeg Film Editor Assembly (Motion + Audio + Subtitles)
# =====================================================================
def test_05_film_editor_assembly(env):
    """Verify single shot clip generation and full multi-shot film assembly with FFmpeg."""
    editor = FilmEditor()
    ffmpeg_exe = editor.ffmpeg_exe

    import cv2
    img1_path = env["dir"] / "shot1.png"
    img2_path = env["dir"] / "shot2.png"

    canvas1 = np.zeros((720, 1280, 3), dtype=np.uint8)
    canvas1[:] = (30, 20, 50)
    cv2.putText(canvas1, "Shot 1: Sunny on Bench", (100, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.imwrite(str(img1_path), canvas1)

    canvas2 = np.zeros((720, 1280, 3), dtype=np.uint8)
    canvas2[:] = (50, 30, 20)
    cv2.putText(canvas2, "Shot 2: Dark Station", (100, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.imwrite(str(img2_path), canvas2)

    aud1_path = env["dir"] / "aud1.mp3"
    aud2_path = env["dir"] / "aud2.mp3"
    for ap in [aud1_path, aud2_path]:
        subprocess.run(
            [ffmpeg_exe, "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", "1.5", "-c:a", "libmp3lame", str(ap)],
            capture_output=True,
            check=True,
        )

    ass_path = env["dir"] / "film_subs.ass"
    SubtitleGenerator.generate_dual_ass(
        [
            {"subtitle_en": "Sunny sat in silence.", "subtitle_hi": "सन्नी खामोशी में बैठा था।", "duration_ms": 1500},
            {"subtitle_en": "The station loomed ahead.", "subtitle_hi": "स्टेशन सामने खड़ा था।", "duration_ms": 1500},
        ],
        ass_path,
    )

    clip1 = editor.generate_shot_clip(
        image_path=img1_path,
        duration_ms=1500,
        camera_motion="slow_zoom_in",
        vfx_overlay="vignette_darken",
        output_path=env["dir"] / "clip1.mp4",
        resolution=(1280, 720),
        fps=30,
    )
    assert clip1.exists()
    assert clip1.stat().st_size > 5000

    master_mp4 = env["dir"] / "master_test_film.mp4"
    spec = FilmEditSpec(
        chapter_number=1,
        shots=[
            {
                "shot_id": "s1",
                "rgb_path": str(img1_path),
                "audio_path": str(aud1_path),
                "duration_ms": 1500,
                "camera_motion": "slow_zoom_in",
                "vfx_overlay": "vignette_darken",
            },
            {
                "shot_id": "s2",
                "rgb_path": str(img2_path),
                "audio_path": str(aud2_path),
                "duration_ms": 1500,
                "camera_motion": "pan_left_to_right",
                "vfx_overlay": "snow_blizzard",
            },
        ],
        background_music_track=None,
        subtitle_hi_path=str(ass_path),
        output_path=str(master_mp4),
        resolution=(1280, 720),
        fps=30,
    )

    assembled = editor.assemble_film(spec)
    assert assembled.exists()
    assert assembled.stat().st_size > 20000
    print(f" [PASS] Test 5: Assembled master film {assembled.name} ({assembled.stat().st_size} bytes).")


# =====================================================================
# Test 6: LangGraph Phase 5 Orchestrator Node Integration
# =====================================================================
def test_06_orchestrator_phase5_nodes(env):
    """Verify generate_narration and render_film nodes execute within the LangGraph state machine."""
    import cv2
    from contextlib import contextmanager
    from app.config import get_settings
    import app.database.session as session_module

    settings = get_settings()
    settings.project.media_root = str(env["dir"])

    SessionLocal = env["session_factory"]

    with SessionLocal() as session:
        proj = Project(
            id=str(uuid.uuid4()),
            name="Shadow Slave Test",
            novel_title="Shadow Slave",
            author="Guiltythree",
        )
        session.add(proj)
        session.flush()

        ch = Chapter(
            id=str(uuid.uuid4()),
            project_id=proj.id,
            chapter_number=1,
            title="Nightmare Begins",
            raw_text="Sunny sat on a bench outside the police station.",
            status=ChapterStatus.PARSED,
        )
        session.add(ch)
        session.flush()

        sc = Scene(
            id=str(uuid.uuid4()),
            chapter_id=ch.id,
            scene_number=1,
            start_passage_id=1,
            end_passage_id=1,
            mood="tense cold night",
        )
        session.add(sc)
        session.flush()

        sh = Shot(
            id=str(uuid.uuid4()),
            scene_id=sc.id,
            chapter_id=ch.id,
            shot_number=1,
            workflow_id="WF-001",
            shot_type="wide",
            camera_angle="eye_level",
            camera_motion_preset="slow_zoom_in",
            motion_tier=MotionTier.STATIC_PARALLAX,
            status=ShotStatus.APPROVED,
        )
        session.add(sh)
        session.flush()

        img_p = env["dir"] / "test_rgb.png"
        cv2.imwrite(str(img_p), np.zeros((360, 640, 3), dtype=np.uint8))

        job = GenerationJob(
            id=str(uuid.uuid4()),
            shot_id=sh.id,
            workflow_id="WF-001",
            prompt_text="Sunny sitting on a bench",
            seed=42,
            output_rgb_path=str(img_p),
            status=JobStatus.COMPLETED,
        )
        session.add(job)
        session.commit()

        project_id = proj.id

    @contextmanager
    def mock_get_session():
        sess = SessionLocal()
        try:
            yield sess
        finally:
            sess.close()

    orig_get_session = session_module.get_session
    session_module.get_session = mock_get_session

    try:
        state = ProductionState(
            project_id=project_id,
            current_chapter=1,
            chapters_in_batch=[1],
            current_stage=PipelineStage.NARRATION,
        )

        state = generate_narration(state)
        assert 1 in state.chapters_narrated
        assert state.current_stage == PipelineStage.FILM_EDITING

        with SessionLocal() as session:
            segs = session.execute(select(NarrationSegment)).scalars().all()
            assert len(segs) >= 1
            assert segs[0].audio_path is not None
            assert segs[0].audio_duration_ms > 0

            tracks = session.execute(select(SubtitleTrack)).scalars().all()
            assert len(tracks) >= 2

        state = render_film(state)
        assert 1 in state.chapters_rendered
        assert state.current_stage == PipelineStage.COMPLETE

        expected_mp4 = env["dir"] / "rendered_films" / "CH01_Master.mp4"
        assert expected_mp4.exists()
        assert expected_mp4.stat().st_size > 10000
        print(f" [PASS] Test 6: Full LangGraph Phase 5 orchestrator node pipeline verified ({expected_mp4.stat().st_size} bytes).")

    finally:
        session_module.get_session = orig_get_session


def run_all_tests():
    print("\n=======================================================")
    print(" NOVEL VISUALISER — PHASE 5 VERIFICATION SUITE")
    print("=======================================================\n")

    env = get_test_env()
    try:
        test_01_phonetic_lore_enforcer()
        test_02_hinglish_script_adapter()
        test_03_edge_tts_synthesis_and_duration(env)
        test_04_subtitle_generation(env)
        test_05_film_editor_assembly(env)
        test_06_orchestrator_phase5_nodes(env)

        print("\n=======================================================")
        print(" ALL 6/6 PHASE 5 VERIFICATION TESTS PASSED SUCCESSFULLY! ")
        print("=======================================================\n")
    finally:
        shutil.rmtree(env["dir"], ignore_errors=True)


if __name__ == "__main__":
    run_all_tests()
