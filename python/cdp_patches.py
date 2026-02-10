#!/usr/bin/env python

from __future__ import annotations

# Small, repo-local monkeypatches used by GitHub Actions workflows.
# Keep this dependency-light and defensive: it runs in CI containers.

from typing import Any, Optional, Tuple


def patch_thumbnails(
    *,
    num_frames: int = 10,
    gif_duration_seconds: float = 10.0,
) -> None:
    """
    Patch CDP Backend thumbnail generation:

    - Static thumbnail: pick a frame around the middle of the (trimmed) video.
    - Hover preview: deterministically sample N frames across the video and
      play them evenly over `gif_duration_seconds`.

    This avoids "dead air" at the start producing ugly thumbnails/previews.
    """
    import math

    from cdp_backend.utils import file_utils as cdp_file_utils

    def _parse_fps(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            if isinstance(value, (int, float)):
                return float(value)
            s = str(value)
            if "/" in s:
                num, den = s.split("/", 1)
                num = float(num)
                den = float(den)
                return (num / den) if den else None
            return float(s)
        except Exception:
            return None

    def _probe_duration_and_fps(video_path: str) -> Tuple[Optional[float], Optional[float]]:
        duration = None
        fps = None
        try:
            import ffmpeg  # ffmpeg-python; depends on ffprobe being present in PATH

            info = ffmpeg.probe(video_path)
            streams = info.get("streams") or []
            video_stream = None
            for s in streams:
                if s.get("codec_type") == "video":
                    video_stream = s
                    break
            if video_stream:
                if video_stream.get("duration") is not None:
                    duration = float(video_stream["duration"])
                fps = _parse_fps(video_stream.get("avg_frame_rate"))

            if duration is None:
                fmt = info.get("format") or {}
                if fmt.get("duration") is not None:
                    duration = float(fmt["duration"])
        except Exception:
            pass
        return duration, fps

    def _reader_meta_duration_and_fps(reader: Any) -> Tuple[Optional[float], Optional[float]]:
        duration = None
        fps = None
        try:
            meta = reader.get_meta_data() or {}
            duration = meta.get("duration")
            fps = meta.get("fps")
        except Exception:
            pass
        try:
            if duration is not None:
                duration = float(duration)
        except Exception:
            duration = None
        fps = _parse_fps(fps)
        return duration, fps

    def get_static_thumbnail(video_path: str, session_content_hash: str, seconds: int = 30) -> str:
        import imageio
        from PIL import Image

        png_path = f"{session_content_hash}-static-thumbnail.png"
        reader = imageio.get_reader(video_path)

        duration, fps = _probe_duration_and_fps(video_path)
        if duration is None or fps is None:
            d2, f2 = _reader_meta_duration_and_fps(reader)
            duration = duration if duration is not None else d2
            fps = fps if fps is not None else f2
        if fps is None or fps <= 0:
            fps = 30.0

        frame_idx = None
        try:
            length = reader.get_length()
            if isinstance(length, int) and length > 0:
                if duration and duration > 0:
                    t = max(0.0, min(duration * 0.5, max(duration - 0.1, 0.0)))
                    frame_idx = int(fps * t)
                else:
                    frame_idx = length // 2
                frame_idx = max(0, min(frame_idx, length - 1))
        except Exception:
            frame_idx = None

        if frame_idx is None:
            # Fallback to CDP backend default behavior: fixed offset from the start.
            frame_idx = int(fps * max(0, seconds))

        try:
            image = reader.get_data(frame_idx)
        except Exception:
            try:
                image = reader.get_data(0)
            except Exception:
                reader = imageio.get_reader(video_path)
                image = reader.get_data(0)

        final_ratio = cdp_file_utils.find_proper_resize_ratio(image.shape[0], image.shape[1])
        if final_ratio < 1:
            image = Image.fromarray(image).resize(
                (
                    math.floor(image.shape[1] * final_ratio),
                    math.floor(image.shape[0] * final_ratio),
                )
            )

        imageio.imwrite(png_path, image)
        return png_path

    def get_hover_thumbnail(
        video_path: str,
        session_content_hash: str,
        _num_frames: int = 10,
        _duration: float = 6.0,
    ) -> str:
        # Note: we ignore upstream-provided defaults and use our pinned, repo-wide
        # values via closure (`num_frames` + `gif_duration_seconds`).
        import imageio
        import numpy as np
        from PIL import Image

        gif_path = f"{session_content_hash}-hover-thumbnail.gif"
        reader = imageio.get_reader(video_path)

        probe_duration, probe_fps = _probe_duration_and_fps(video_path)
        meta_duration, meta_fps = _reader_meta_duration_and_fps(reader)
        total_duration = probe_duration if probe_duration is not None else meta_duration
        fps = probe_fps if probe_fps is not None else meta_fps
        if fps is None or fps <= 0:
            fps = 30.0

        sample = reader.get_data(0)
        height = sample.shape[0]
        width = sample.shape[1]
        final_ratio = cdp_file_utils.find_proper_resize_ratio(height, width)

        indices = []
        try:
            length = reader.get_length()
            if isinstance(length, int) and length > 0:
                if total_duration and total_duration > 0:
                    for i in range(num_frames):
                        # Midpoints across the duration avoids hitting the exact first/last frame.
                        t = (i + 0.5) * (total_duration / num_frames)
                        idx = int(fps * t)
                        indices.append(max(0, min(idx, length - 1)))
                else:
                    for i in range(num_frames):
                        idx = int((i + 0.5) * (length / num_frames))
                        indices.append(max(0, min(idx, length - 1)))
        except Exception:
            indices = []

        if not indices:
            # Fallback: take the first N frames by index.
            indices = list(range(num_frames))

        # Even playback: N frames over total duration => fixed seconds per frame.
        per_frame_seconds = max(0.1, float(gif_duration_seconds) / float(num_frames))
        with imageio.get_writer(gif_path, mode="I", duration=per_frame_seconds) as writer:
            for idx in indices:
                try:
                    frame = reader.get_data(idx)
                except Exception:
                    continue
                image = Image.fromarray(frame)
                if final_ratio < 1:
                    image = image.resize(
                        (
                            math.floor(width * final_ratio),
                            math.floor(height * final_ratio),
                        )
                    )
                writer.append_data(np.asarray(image).astype(np.uint8))

        return gif_path

    cdp_file_utils.get_static_thumbnail = get_static_thumbnail
    cdp_file_utils.get_hover_thumbnail = get_hover_thumbnail

