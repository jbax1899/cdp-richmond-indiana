#!/usr/bin/env python

from __future__ import annotations

# Small, repo-local monkeypatches used by GitHub Actions workflows.
# Keep this dependency-light and defensive: it runs in CI containers.

from typing import Any, Optional, Tuple


def patch_thumbnails(
    *,
    num_frames: int = 10,
    gif_duration_seconds: float = 10.0,
    clip_duration_seconds: float = 2.0,
) -> None:
    """
    Patch CDP Backend thumbnail generation:

    - Static thumbnail: use the first frame when it has visual content;
      otherwise seek forward until visual content appears.
    - Hover preview: deterministically sample N evenly spaced clips across the
      source and concatenate them into a short MP4 preview.

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
        # Fallback path when ffmpeg-python isn't installed or probing fails.
        if duration is None or fps is None:
            try:
                import json
                import shutil
                import subprocess

                ffprobe_bin = shutil.which("ffprobe")
                if ffprobe_bin:
                    raw = subprocess.check_output(
                        [
                            ffprobe_bin,
                            "-v",
                            "error",
                            "-show_entries",
                            "stream=codec_type,avg_frame_rate,duration:format=duration",
                            "-of",
                            "json",
                            video_path,
                        ],
                        text=True,
                    )
                    payload = json.loads(raw)
                    streams = payload.get("streams") or []
                    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
                    if video_stream:
                        if duration is None and video_stream.get("duration") is not None:
                            duration = float(video_stream["duration"])
                        if fps is None:
                            fps = _parse_fps(video_stream.get("avg_frame_rate"))
                    if duration is None:
                        fmt = payload.get("format") or {}
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

    def _is_remote_video_source(video_path: str) -> bool:
        s = str(video_path).lower()
        return s.startswith("http://") or s.startswith("https://") or s.startswith("gs://")

    def get_static_thumbnail(video_path: str, session_content_hash: str, seconds: int = 30) -> str:
        import imageio
        import shutil
        import subprocess
        import tempfile
        from PIL import Image, ImageFilter, ImageStat

        png_path = f"{session_content_hash}-static-thumbnail.png"

        def _clamp_frame_idx(idx: int) -> int:
            idx = max(0, int(idx))
            if length is not None:
                idx = min(idx, length - 1)
            return idx

        def _read_frame(idx: int) -> Any:
            return reader.get_data(_clamp_frame_idx(idx))

        def _is_visually_blank(frame: Any) -> bool:
            # Treat low-detail lead-in slates as blank regardless of brightness.
            # This catches "gray screen" starts while preserving real scene content.
            try:
                gray = Image.fromarray(frame).convert("L").resize((64, 36))
                lo, hi = gray.getextrema()
                stats = ImageStat.Stat(gray)
                mean = float(stats.mean[0])
                stddev = float(stats.var[0]) ** 0.5
                dynamic = float(hi - lo)
                edge_stats = ImageStat.Stat(gray.filter(ImageFilter.FIND_EDGES))
                edge_mean = float(edge_stats.mean[0])

                if stddev < 18.0 and dynamic < 60.0 and edge_mean < 24.0:
                    return True
                if mean < 12.0 and dynamic < 24.0:
                    return True
                return False
            except Exception:
                return False

        def _write_image(image: Any) -> str:
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

        duration, fps = _probe_duration_and_fps(video_path)

        if _is_remote_video_source(video_path):
            if fps is None or fps <= 0:
                fps = 30.0
            ffmpeg_bin = shutil.which("ffmpeg")
            if not ffmpeg_bin:
                raise RuntimeError(
                    "ffmpeg not found on PATH for remote static thumbnail generation "
                    f"(source: {video_path})"
                )

            with tempfile.TemporaryDirectory() as td:
                frame_num = 0

                def _read_remote_frame_at_seconds(seconds_from_start: float) -> Any:
                    nonlocal frame_num
                    frame_num += 1
                    probe_png = f"{td}/probe-{frame_num}.png"
                    subprocess.run(
                        [
                            ffmpeg_bin,
                            "-y",
                            "-v",
                            "error",
                            "-ss",
                            f"{max(0.0, float(seconds_from_start)):.6f}",
                            "-i",
                            str(video_path),
                            "-frames:v",
                            "1",
                            probe_png,
                        ],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    return imageio.imread(probe_png)

                first_frame = _read_remote_frame_at_seconds(0.0)
                if not _is_visually_blank(first_frame):
                    return _write_image(first_frame)

                image = first_frame
                prev_blank_t = 0.0
                first_content_t = None
                first_content_frame = None
                max_t = max(0.0, float(duration or 0.0) - 0.1)

                if max_t > 0:
                    t = min(1.0, max_t)
                    while True:
                        candidate = _read_remote_frame_at_seconds(t)
                        if not _is_visually_blank(candidate):
                            first_content_t = t
                            first_content_frame = candidate
                            break
                        prev_blank_t = t
                        if t >= max_t:
                            break
                        next_t = min(max_t, max(t * 2.0, t + 0.1))
                        if next_t <= t:
                            break
                        t = next_t
                else:
                    # Duration probing can fail for some remote streams.
                    # In that case, perform bounded sparse probing.
                    for t in [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]:
                        candidate = _read_remote_frame_at_seconds(t)
                        if not _is_visually_blank(candidate):
                            first_content_t = t
                            first_content_frame = candidate
                            break

                if first_content_t is not None and first_content_frame is not None:
                    lo = prev_blank_t
                    hi = first_content_t
                    chosen = first_content_frame
                    for _ in range(4):
                        if hi - lo <= 0.25:
                            break
                        mid = (lo + hi) / 2.0
                        candidate = _read_remote_frame_at_seconds(mid)
                        if not _is_visually_blank(candidate):
                            hi = mid
                            chosen = candidate
                        else:
                            lo = mid
                    image = chosen

                return _write_image(image)

        reader = imageio.get_reader(video_path)
        if duration is None or fps is None:
            d2, f2 = _reader_meta_duration_and_fps(reader)
            duration = duration if duration is not None else d2
            fps = fps if fps is not None else f2
        if fps is None or fps <= 0:
            fps = 30.0

        length = None
        try:
            raw_length = reader.get_length()
            if isinstance(raw_length, int) and raw_length > 0:
                length = raw_length
        except Exception:
            length = None

        image = None
        try:
            first_frame = _read_frame(0)
            if not _is_visually_blank(first_frame):
                image = first_frame
            elif duration and duration > 0:
                max_t = max(0.0, duration - 0.1)
                prev_blank_t = 0.0
                first_content_t = None
                first_content_frame = None

                if max_t > 0:
                    t = min(1.0, max_t)
                    while True:
                        candidate = _read_frame(int(fps * max(0.0, t)))
                        if not _is_visually_blank(candidate):
                            first_content_t = t
                            first_content_frame = candidate
                            break
                        prev_blank_t = t
                        if t >= max_t:
                            break
                        next_t = min(max_t, max(t * 2.0, t + 0.1))
                        if next_t <= t:
                            break
                        t = next_t

                if first_content_t is not None and first_content_frame is not None:
                    lo = prev_blank_t
                    hi = first_content_t
                    chosen = first_content_frame
                    for _ in range(4):
                        if hi - lo <= 0.25:
                            break
                        mid = (lo + hi) / 2.0
                        candidate = _read_frame(int(fps * max(0.0, mid)))
                        if not _is_visually_blank(candidate):
                            hi = mid
                            chosen = candidate
                        else:
                            lo = mid
                    image = chosen
                else:
                    image = first_frame
        except Exception:
            image = None

        frame_idx = None
        try:
            if image is None and isinstance(length, int) and length > 0:
                if duration and duration > 0:
                    t = max(0.0, min(duration * 0.5, max(duration - 0.1, 0.0)))
                    frame_idx = int(fps * t)
                else:
                    frame_idx = length // 2
                frame_idx = _clamp_frame_idx(frame_idx)
        except Exception:
            frame_idx = None

        if image is None and frame_idx is None:
            # Fallback to CDP backend default behavior: fixed offset from the start.
            frame_idx = int(fps * max(0, seconds))

        if image is None and frame_idx is not None:
            image = _read_frame(frame_idx)

        if image is None:
            image = _read_frame(0)

        _write_image(image)
        try:
            reader.close()
        except Exception:
            pass
        return png_path

    def get_hover_thumbnail(
        video_path: str,
        session_content_hash: str,
        _num_frames: int = 10,
        _duration: float = 6.0,
    ) -> str:
        # Note: we ignore upstream-provided defaults and use pinned repo-wide values.
        import shutil
        import subprocess

        preview_path = f"{session_content_hash}-hover-preview.mp4"
        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            raise RuntimeError("ffmpeg not found on PATH")

        total_duration, _ = _probe_duration_and_fps(video_path)
        if total_duration is None or total_duration <= 0:
            raise RuntimeError(f"Unable to determine video duration for hover preview: {video_path}")

        clip_seconds = max(0.25, float(clip_duration_seconds))
        fps = 8
        max_width = 320
        crf = 31
        preset = "medium"

        # Generate N evenly spaced clip starts (centered in each span), clamped to legal range.
        # total preview duration = num_frames * clip_seconds
        starts = []
        for i in range(num_frames):
            center_t = (i + 0.5) * (total_duration / float(num_frames))
            start_t = center_t - (clip_seconds / 2.0)
            max_start = max(0.0, total_duration - clip_seconds)
            starts.append(max(0.0, min(start_t, max_start)))

        chains = []
        labels = []
        for i, start_t in enumerate(starts):
            label = f"v{i}"
            labels.append(f"[{label}]")
            chains.append(
                f"[0:v]trim=start={start_t:.6f}:duration={clip_seconds:.6f},"
                f"setpts=PTS-STARTPTS,fps={fps},scale={max_width}:-2[{label}]"
            )

        # Concat all short clips into one preview stream.
        filter_complex = ";".join(chains + [f"{''.join(labels)}concat=n={num_frames}:v=1:a=0,format=yuv420p[vout]"])

        subprocess.run(
            [
                ffmpeg_bin,
                "-y",
                "-v",
                "error",
                "-i",
                video_path,
                "-filter_complex",
                filter_complex,
                "-map",
                "[vout]",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-crf",
                str(crf),
                "-preset",
                preset,
                preview_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        return preview_path

    cdp_file_utils.get_static_thumbnail = get_static_thumbnail
    cdp_file_utils.get_hover_thumbnail = get_hover_thumbnail
