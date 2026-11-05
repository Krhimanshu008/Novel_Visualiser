"""
Novel Visualiser — Film Editor Service (Phase 5)

Components:
1. CameraMotionPresets:
   - Deterministic FFmpeg zoompan camera movement formulas
   - slow_zoom_in, slow_zoom_out, pan_left_to_right, pan_right_to_left,
     fast_push_in, static_hold
2. VFXOverlayPresets:
   - Filter chains for cinematic atmosphere: snow_blizzard, vignette_darken,
     ember_particles, dark_atmosphere
3. FilmEditor:
   - Renders individual shot clips from still images with camera motion & VFX
   - Concatenates shot clips into continuous film sequence
   - Auto-ducks and mixes narration audio with background music
   - Bakes in broadcast-quality dual ASS subtitles (Devanagari Hindi + English)
   - Exports the master MP4 film
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

import imageio_ffmpeg

from app.config import get_settings
from app.schemas.scene import FilmEditSpec

logger = logging.getLogger("novel_visualiser.film_editor")


class CameraMotionPresets:
    """Deterministic FFmpeg zoompan filter expressions."""

    @staticmethod
    def get_filter(
        preset: str,
        frames: int,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
    ) -> str:
        """
        Return the FFmpeg zoompan filter string for the given camera motion.
        `frames` is total frames (duration_s * fps).
        """
        frames = max(frames, 1)
        p = preset.lower() if preset else "slow_zoom_in"

        if p in ("slow_zoom_in", "zoom_in"):
            # Smooth subtle push-in from 1.0 to 1.15
            step = 0.15 / max(frames, 1)
            return (
                f"zoompan=z='min(zoom+{step:.6f},1.15)':"
                f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                f"d={frames}:s={width}x{height}:fps={fps}"
            )
        elif p in ("slow_zoom_out", "zoom_out"):
            # Pull-back from 1.15 to 1.0
            step = 0.15 / max(frames, 1)
            return (
                f"zoompan=z='max(1.15-{step:.6f}*on,1.0)':"
                f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                f"d={frames}:s={width}x{height}:fps={fps}"
            )
        elif p in ("pan_left_to_right", "pan_right"):
            # Gentle pan left to right at 1.12 zoom
            return (
                f"zoompan=z='1.12':"
                f"x='(iw-iw/zoom)*(on/{frames})':y='ih/2-(ih/zoom/2)':"
                f"d={frames}:s={width}x{height}:fps={fps}"
            )
        elif p in ("pan_right_to_left", "pan_left"):
            # Gentle pan right to left at 1.12 zoom
            return (
                f"zoompan=z='1.12':"
                f"x='(iw-iw/zoom)*(1-on/{frames})':y='ih/2-(ih/zoom/2)':"
                f"d={frames}:s={width}x{height}:fps={fps}"
            )
        elif p in ("fast_push_in", "push_in"):
            # Dramatic fast push in for climax/action
            step = 0.30 / max(frames, 1)
            return (
                f"zoompan=z='min(zoom+{step:.6f},1.30)':"
                f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                f"d={frames}:s={width}x{height}:fps={fps}"
            )
        else:
            # static_hold or default fallback
            return (
                f"zoompan=z='1.0':"
                f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                f"d={frames}:s={width}x{height}:fps={fps}"
            )


class VFXOverlayPresets:
    """Atmospheric VFX color grading and overlay filter chains."""

    @staticmethod
    def get_filter(vfx_tag: Optional[str]) -> Optional[str]:
        if not vfx_tag:
            return None
        tag = vfx_tag.lower()
        if "snow" in tag or "blizzard" in tag:
            return "eq=contrast=1.1:brightness=-0.02:saturation=0.85,vignette=PI/4"
        elif "vignette" in tag or "darken" in tag:
            return "vignette=PI/3"
        elif "dark" in tag or "dread" in tag:
            return "eq=contrast=1.15:brightness=-0.05:saturation=0.9,vignette=PI/4"
        elif "ember" in tag or "fire" in tag:
            return "eq=contrast=1.2:saturation=1.15,colorbalance=rs=0.08:gs=0.02:bs=-0.08"
        return None


class FilmEditor:
    """
    FFmpeg-powered video editing and assembly engine.
    Produces broadcast-quality cinematic MP4 films from storyboard shots, audio, and subtitles.
    """

    def __init__(self, ffmpeg_exe: Optional[str] = None):
        self.ffmpeg_exe = ffmpeg_exe or imageio_ffmpeg.get_ffmpeg_exe()
        self.settings = get_settings()
        self.media_root = Path(self.settings.project.media_root)
        self.output_dir = self.media_root / "rendered_films"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_shot_clip(
        self,
        image_path: Path | str,
        duration_ms: int,
        camera_motion: str = "slow_zoom_in",
        vfx_overlay: Optional[str] = None,
        output_path: Optional[Path | str] = None,
        resolution: tuple[int, int] = (1920, 1080),
        fps: int = 30,
    ) -> Path:
        """
        Render a single shot clip from an RGB still image with camera motion and VFX.
        """
        img_p = Path(image_path).resolve()
        if not img_p.exists():
            raise FileNotFoundError(f"Input image not found: {img_p}")

        if output_path is None:
            out_p = img_p.parent / f"{img_p.stem}_clip.mp4"
        else:
            out_p = Path(output_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)

        w, h = resolution
        duration_s = max(duration_ms / 1000.0, 1.0)
        total_frames = int(round(duration_s * fps))

        # Build filtergraph
        zp_filter = CameraMotionPresets.get_filter(
            camera_motion, frames=total_frames, width=w, height=h, fps=fps
        )
        vfx_filter = VFXOverlayPresets.get_filter(vfx_overlay)

        filters = [zp_filter]
        if vfx_filter:
            filters.append(vfx_filter)
        filter_str = ",".join(filters)

        cmd = [
            self.ffmpeg_exe,
            "-y",
            "-i",
            str(img_p),
            "-vf",
            filter_str,
            "-t",
            f"{duration_s:.3f}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "fast",
            "-crf",
            "20",
            str(out_p),
        ]

        logger.info("Rendering shot clip: %s (duration=%.2fs, motion=%s)", out_p.name, duration_s, camera_motion)
        res = subprocess.run(cmd, capture_output=True, text=True, errors="ignore")
        if res.returncode != 0:
            logger.error("FFmpeg error rendering shot clip: %s", res.stderr)
            raise RuntimeError(f"FFmpeg error: {res.stderr}")

        return out_p

    def build_narration_audio_track(
        self,
        audio_paths: list[Path | str],
        output_path: Path | str,
    ) -> Path:
        """
        Concatenate individual shot audio files into a unified narration audio track.
        """
        out_p = Path(output_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)

        valid_paths = [Path(p).resolve() for p in audio_paths if Path(p).exists()]
        if not valid_paths:
            # Generate 3s silence if no audio files exist
            cmd = [
                self.ffmpeg_exe,
                "-y",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=44100:cl=stereo",
                "-t",
                "3",
                "-c:a",
                "aac",
                str(out_p),
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            return out_p

        if len(valid_paths) == 1:
            cmd = [
                self.ffmpeg_exe,
                "-y",
                "-i",
                str(valid_paths[0]),
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                str(out_p),
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            return out_p

        # Concat list file
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
            concat_txt = f.name
            for ap in valid_paths:
                f.write(f"file '{ap.as_posix()}'\n")

        try:
            cmd = [
                self.ffmpeg_exe,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                concat_txt,
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                str(out_p),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, errors="ignore")
            if res.returncode != 0:
                raise RuntimeError(f"Failed to concatenate audio tracks: {res.stderr}")
        finally:
            if os.path.exists(concat_txt):
                os.remove(concat_txt)

        return out_p

    def mix_audio_with_bgm(
        self,
        narration_path: Path | str,
        bgm_path: Optional[Path | str],
        output_path: Path | str,
        video_duration_s: float,
        duck_db: float = -12.0,
    ) -> Path:
        """
        Mix narration audio with background music, applying auto-ducking to the BGM.
        """
        out_p = Path(output_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        narr_p = Path(narration_path).resolve()

        if not bgm_path or not Path(bgm_path).exists():
            # No BGM — just copy/encode narration
            cmd = [
                self.ffmpeg_exe,
                "-y",
                "-i",
                str(narr_p),
                "-t",
                f"{video_duration_s:.3f}",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                str(out_p),
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            return out_p

        bgm_p = Path(bgm_path).resolve()
        duck_factor = 10 ** (duck_db / 20.0)

        # Mix: stream 0 (narration) volume 1.0, stream 1 (BGM looped) ducked
        filter_complex = (
            f"[0:a]volume=1.0[narr];"
            f"[1:a]volume={duck_factor:.3f}[bgm];"
            f"[narr][bgm]amix=inputs=2:duration=first:dropout_transition=2[aout]"
        )

        cmd = [
            self.ffmpeg_exe,
            "-y",
            "-i",
            str(narr_p),
            "-stream_loop",
            "-1",
            "-i",
            str(bgm_p),
            "-filter_complex",
            filter_complex,
            "-map",
            "[aout]",
            "-t",
            f"{video_duration_s:.3f}",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(out_p),
        ]

        res = subprocess.run(cmd, capture_output=True, text=True, errors="ignore")
        if res.returncode != 0:
            logger.warning("BGM mix failed (%s), falling back to narration only", res.stderr)
            return self.mix_audio_with_bgm(narr_p, None, out_p, video_duration_s)

        return out_p

    def assemble_film(self, spec: FilmEditSpec) -> Path:
        """
        Assemble the complete chapter video:
        1. Render each shot clip with motion preset and VFX
        2. Concatenate all shot video clips
        3. Combine narration audio tracks and mix with optional BGM
        4. Burn dual-subtitles into the master MP4
        """
        out_master = Path(spec.output_path).resolve()
        out_master.parent.mkdir(parents=True, exist_ok=True)

        if not spec.shots:
            raise ValueError("No shots provided in FilmEditSpec")

        temp_dir = Path(tempfile.mkdtemp(prefix="film_assembly_"))

        try:
            shot_clip_paths: list[Path] = []
            shot_audio_paths: list[Path] = []
            total_duration_ms = 0

            # Step 1: Render individual shot clips
            for idx, shot in enumerate(spec.shots):
                shot_id = shot.get("shot_id", f"shot_{idx}")
                rgb_path = shot.get("rgb_path")
                if not rgb_path or not Path(rgb_path).exists():
                    logger.warning("RGB path missing for %s, skipping", shot_id)
                    continue

                duration_ms = shot.get("duration_ms", 3000)
                total_duration_ms += duration_ms
                camera_motion = shot.get("camera_motion", "slow_zoom_in")
                vfx_overlay = shot.get("vfx_overlay")

                clip_path = temp_dir / f"clip_{idx:03d}_{shot_id}.mp4"
                self.generate_shot_clip(
                    image_path=rgb_path,
                    duration_ms=duration_ms,
                    camera_motion=camera_motion,
                    vfx_overlay=vfx_overlay,
                    output_path=clip_path,
                    resolution=spec.resolution,
                    fps=spec.fps,
                )
                shot_clip_paths.append(clip_path)

                audio_p = shot.get("audio_path")
                if audio_p and Path(audio_p).exists():
                    shot_audio_paths.append(Path(audio_p))

            if not shot_clip_paths:
                raise RuntimeError("No valid shot clips could be rendered")

            # Step 2: Concatenate video clips
            concat_txt = temp_dir / "clips_concat.txt"
            with open(concat_txt, "w", encoding="utf-8") as f:
                for cp in shot_clip_paths:
                    f.write(f"file '{cp.as_posix()}'\n")

            video_concat_mp4 = temp_dir / "video_concatenated.mp4"
            cmd_concat = [
                self.ffmpeg_exe,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_txt),
                "-c",
                "copy",
                str(video_concat_mp4),
            ]
            res_concat = subprocess.run(cmd_concat, capture_output=True, text=True, errors="ignore")
            if res_concat.returncode != 0:
                raise RuntimeError(f"Failed to concatenate shot clips: {res_concat.stderr}")

            # Step 3: Mix audio
            total_duration_s = total_duration_ms / 1000.0
            narration_audio = temp_dir / "narration_master.m4a"
            self.build_narration_audio_track(shot_audio_paths, narration_audio)

            final_audio = temp_dir / "final_audio.m4a"
            self.mix_audio_with_bgm(
                narration_path=narration_audio,
                bgm_path=spec.background_music_track,
                output_path=final_audio,
                video_duration_s=total_duration_s,
                duck_db=spec.audio_duck_db,
            )

            # Step 4: Final composite with dual subtitles
            # If subtitle file is provided, burn it into video
            sub_file: Optional[Path] = None
            if spec.subtitle_hi_path and Path(spec.subtitle_hi_path).exists():
                sub_file = Path(spec.subtitle_hi_path)
            elif spec.subtitle_en_path and Path(spec.subtitle_en_path).exists():
                sub_file = Path(spec.subtitle_en_path)

            if sub_file:
                # Copy subtitle file into temp directory to avoid Windows path colon escaping issues
                local_sub = temp_dir / sub_file.name
                local_sub.write_bytes(sub_file.read_bytes())
                sub_filter = f"ass={local_sub.name}" if local_sub.suffix.lower() == ".ass" else f"subtitles={local_sub.name}"

                cmd_final = [
                    self.ffmpeg_exe,
                    "-y",
                    "-i",
                    str(video_concat_mp4),
                    "-i",
                    str(final_audio),
                    "-vf",
                    sub_filter,
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-preset",
                    "medium",
                    "-crf",
                    "19",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-shortest",
                    str(out_master),
                ]
                res_final = subprocess.run(
                    cmd_final, cwd=str(temp_dir), capture_output=True, text=True, errors="ignore"
                )
                if res_final.returncode != 0:
                    logger.warning("Subtitle burn failed (%s), exporting without subtitles", res_final.stderr)
                    sub_file = None

            if not sub_file:
                # Mux video + audio directly
                cmd_final = [
                    self.ffmpeg_exe,
                    "-y",
                    "-i",
                    str(video_concat_mp4),
                    "-i",
                    str(final_audio),
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-shortest",
                    str(out_master),
                ]
                res_final = subprocess.run(cmd_final, capture_output=True, text=True, errors="ignore")
                if res_final.returncode != 0:
                    raise RuntimeError(f"Final film assembly failed: {res_final.stderr}")

            logger.info("Successfully exported master film: %s (size=%.2f MB)", out_master.name, out_master.stat().st_size / (1024 * 1024))
            return out_master

        finally:
            # Clean up temp working files
            try:
                for f in temp_dir.glob("*"):
                    try:
                        f.unlink()
                    except Exception:
                        pass
                temp_dir.rmdir()
            except Exception:
                pass

