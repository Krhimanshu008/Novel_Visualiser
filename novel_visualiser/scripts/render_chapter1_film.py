"""
Novel Visualiser — Render Chapter 1 Master Film (Phase 5 Production Script)

Assembles the 5 GPU-rendered shots from Chapter 1 of Shadow Slave:
1. Generates neural Hinglish narration (edge-tts hi-IN-MadhurNeural)
2. Parses exact millisecond durations
3. Generates dual ASS & SRT subtitles (Gold Devanagari Hindi + White English)
4. Renders camera motion (zoompan) & atmospheric VFX filters
5. Mixes audio & burns dual subtitles into CH01_Master.mp4
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.schemas.scene import FilmEditSpec
from app.services.audio_engine import HinglishScriptAdapter, SubtitleGenerator, TTSEngine
from app.services.film_editor import FilmEditor


def main():
    print("=================================================================")
    print(" NOVEL VISUALISER — CHAPTER 1 FILM PRODUCTION RUN (PHASE 5)")
    print("=================================================================\n")

    base_dir = Path(__file__).resolve().parent.parent
    images_dir = base_dir / "output" / "generated_images" / "chapter_01"
    audio_dir = base_dir / "output" / "audio"
    subs_dir = base_dir / "output" / "subtitles"
    films_dir = base_dir / "output" / "rendered_films"

    audio_dir.mkdir(parents=True, exist_ok=True)
    subs_dir.mkdir(parents=True, exist_ok=True)
    films_dir.mkdir(parents=True, exist_ok=True)

    tts = TTSEngine(voice="hi-IN-MadhurNeural")
    tts.output_dir = audio_dir
    adapter = HinglishScriptAdapter()
    editor = FilmEditor()

    # The 5 key shots from Chapter 1 storyboard
    shots_plan = [
        {
            "shot_number": 1,
            "shot_type": "wide",
            "camera_motion": "slow_zoom_in",
            "vfx_overlay": "vignette_darken",
            "image_filename": "CH01_Shot_01_wide.png",
            "prose": "Sunny sat on a rusty bench outside the police station, clutching a lukewarm cup of real coffee.",
            "mood": "calm_narration",
        },
        {
            "shot_number": 2,
            "shot_type": "close_up",
            "camera_motion": "pan_left_to_right",
            "vfx_overlay": "dark_atmosphere",
            "image_filename": "CH01_Shot_02_close_up.png",
            "prose": "He took a small sip and grimaced. Ah, so bitter! But considering how much credits it had cost, he forced himself to swallow.",
            "mood": "tense_whisper",
        },
        {
            "shot_number": 16,
            "shot_type": "two_shot",
            "camera_motion": "slow_zoom_out",
            "vfx_overlay": "vignette_darken",
            "image_filename": "CH01_Shot_16_two_shot.png",
            "prose": "The police station was heavily fortified, armor plates lining the walls. A tired desk officer glared suspiciously at Sunny.",
            "mood": "tense_whisper",
        },
        {
            "shot_number": 31,
            "shot_type": "close_up",
            "camera_motion": "fast_push_in",
            "vfx_overlay": "dark_atmosphere",
            "image_filename": "CH01_Shot_31_close_up.png",
            "prose": "Inside the dim detention cell, Sunny sat alone on the cold bunk, waiting for the inevitable nightmare to claim his mind.",
            "mood": "dramatic_rise",
        },
        {
            "shot_number": 49,
            "shot_type": "over_the_shoulder",
            "camera_motion": "slow_zoom_in",
            "vfx_overlay": "snow_blizzard",
            "image_filename": "CH01_Shot_49_over_the_shoulder.png",
            "prose": "Suddenly darkness engulfed him completely. A chilling bell chimed: Aspirant, welcome to the Nightmare Spell. Prepare for your trial.",
            "mood": "intense_action",
        },
    ]

    assembled_shots = []
    subtitles_data = []

    print("[1/4] Synthesizing Neural Hinglish Narration via Edge-TTS...")
    for item in shots_plan:
        img_path = images_dir / item["image_filename"]
        if not img_path.exists():
            print(f"Warning: {img_path} not found, skipping!")
            continue

        shot_id = f"CH01_Shot_{item['shot_number']:02d}"
        segment = adapter.adapt_passage_to_segment(
            shot_id=shot_id,
            segment_index=item["shot_number"],
            passage_text=item["prose"],
            mood=item["mood"],
        )

        audio_filename = f"{shot_id}_narration.mp3"
        audio_path = tts.synthesize(segment.hinglish_tts_script, audio_filename)
        duration_ms = tts.get_audio_duration_ms(audio_path)

        print(f"  - Shot {item['shot_number']:02d} ({item['shot_type']}):")
        print(f"    TTS Script: \"{segment.hinglish_tts_script}\"")
        print(f"    Duration: {duration_ms} ms ({duration_ms / 1000.0:.2f}s)")
        print(f"    Sub (EN): \"{segment.subtitle_en}\"")
        print(f"    Sub (HI): \"{segment.subtitle_hi}\"")

        subtitles_data.append({
            "subtitle_en": segment.subtitle_en,
            "subtitle_hi": segment.subtitle_hi,
            "duration_ms": duration_ms,
        })

        assembled_shots.append({
            "shot_id": shot_id,
            "rgb_path": str(img_path),
            "audio_path": str(audio_path),
            "duration_ms": duration_ms,
            "camera_motion": item["camera_motion"],
            "vfx_overlay": item["vfx_overlay"],
        })

    print(f"\n[2/4] Generating Synchronized Dual Subtitles (.ass & .srt)...")
    ass_path = subs_dir / "CH01_dual.ass"
    srt_path = subs_dir / "CH01_en.srt"
    SubtitleGenerator.generate_dual_ass(subtitles_data, ass_path)
    SubtitleGenerator.generate_srt(subtitles_data, srt_path, lang="en")
    print(f"  - Dual ASS Subtitles: {ass_path} ({ass_path.stat().st_size} bytes)")
    print(f"  - English SRT Subtitles: {srt_path} ({srt_path.stat().st_size} bytes)")

    print(f"\n[3/4] Assembling Cinematic Film with FFmpeg...")
    master_mp4 = films_dir / "CH01_Master.mp4"
    spec = FilmEditSpec(
        chapter_number=1,
        shots=assembled_shots,
        background_music_track=None,
        subtitle_hi_path=str(ass_path),
        output_path=str(master_mp4),
        resolution=(1920, 1080),
        fps=30,
    )

    result_file = editor.assemble_film(spec)

    total_duration_s = sum(s["duration_ms"] for s in assembled_shots) / 1000.0
    filesize_mb = result_file.stat().st_size / (1024 * 1024)

    print(f"\n[4/4] Film Assembly Completed Successfully!")
    print(f"  - Output Film: {result_file}")
    print(f"  - Total Duration: {total_duration_s:.2f} seconds ({len(assembled_shots)} shots)")
    print(f"  - Resolution: 1920x1080 @ 30 FPS")
    print(f"  - Video Codec: H.264 (libx264, yuv420p, crf=19)")
    print(f"  - Audio Codec: AAC 192kbps (hi-IN-MadhurNeural)")
    print(f"  - Subtitles: Baked-in Dual ASS (Gold Hindi + White English)")
    print(f"  - File Size: {filesize_mb:.2f} MB")
    print("=================================================================\n")


if __name__ == "__main__":
    main()
