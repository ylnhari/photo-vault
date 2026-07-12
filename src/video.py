"""One home for every ffmpeg operation, so the rest of the app stays
decode-agnostic (the way platformfs.py owns OS branching and faces.py owns
InsightFace). Nothing else in the codebase shells out to ffmpeg.

Frame extraction and metadata are the only genuinely video-specific problems;
everything else about videos is plumbing a media_type flag through the catalog,
timeline, grid, lightbox and search.

The ffmpeg binary ships with the `imageio-ffmpeg` dependency
(`imageio_ffmpeg.get_ffmpeg_exe()`), so there is no user setup and it is the
same static build on every OS. imageio-ffmpeg bundles ffmpeg only (no ffprobe),
so metadata is parsed from `ffmpeg -i` stderr rather than ffprobe JSON.
"""
import os
import re
import subprocess
import tempfile

import json
import shutil

_FFMPEG_EXE: str | None = None
_FFPROBE_EXE: str | None = None
_FFPROBE_CHECKED = False

# Long enough for a real extraction on a slow disk, short enough that a wedged
# ffmpeg on a corrupt file can't hang a job worker forever.
_TIMEOUT = 120

# Never let a console window flash on Windows when a job worker shells out.
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def ffmpeg_exe() -> str:
    """Path to the bundled ffmpeg binary. Cached. Raises a clear, actionable
    error if imageio-ffmpeg isn't installed (the one video dependency)."""
    global _FFMPEG_EXE
    if _FFMPEG_EXE is None:
        try:
            import imageio_ffmpeg
        except ImportError as e:
            raise RuntimeError(
                "video support needs imageio-ffmpeg — run `uv sync` "
                "(it is a declared dependency)."
            ) from e
        _FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
    return _FFMPEG_EXE


def ffprobe_exe() -> str | None:
    """Path to an ffprobe binary, or None if none is available.

    imageio-ffmpeg ships ffmpeg but NOT ffprobe, so real structured metadata
    (per-stream codec/fps/rotation, audio-stream presence) needs ffprobe from
    elsewhere. Preference: the `static-ffmpeg` package (fetches a static build,
    zero system setup — an opt-in dependency), then any ffprobe on PATH. Returns
    None when neither exists, so probe() degrades to the ffmpeg -i stderr parse
    instead of failing. Cached (including the negative result)."""
    global _FFPROBE_EXE, _FFPROBE_CHECKED
    if _FFPROBE_CHECKED:
        return _FFPROBE_EXE
    _FFPROBE_CHECKED = True
    try:
        from static_ffmpeg import run as _sf_run
        _, probe = _sf_run.get_or_fetch_platform_executables_else_raise()
        if probe and os.path.exists(probe):
            _FFPROBE_EXE = probe
            return _FFPROBE_EXE
    except Exception:
        pass
    _FFPROBE_EXE = shutil.which("ffprobe")
    return _FFPROBE_EXE


def _probe_ffprobe(path: str) -> dict | None:
    """Structured metadata via `ffprobe -print_format json`. Returns the full
    schema dict (duration_s, width, height, codec, capture_time, fps, rotation,
    has_audio) or None if ffprobe is unavailable or the file has no video
    stream. Far more robust than regex-scraping ffmpeg stderr — the production
    path when ffprobe is installed."""
    probe = ffprobe_exe()
    if not probe:
        return None
    try:
        proc = subprocess.run(
            [probe, "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", path],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=_TIMEOUT, creationflags=_NO_WINDOW,
        )
        data = json.loads(proc.stdout.decode("utf-8", "replace") or "{}")
    except Exception as e:
        print(f"[video] ffprobe failed for {path}: {e}")
        return None

    streams = data.get("streams", [])
    vstreams = [s for s in streams if s.get("codec_type") == "video"]
    if not vstreams:
        return None  # audio-only / image / not a video
    v = vstreams[0]
    fmt = data.get("format", {})

    def _num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    # Duration can live on the stream or the container.
    duration = _num(v.get("duration")) or _num(fmt.get("duration"))
    # r_frame_rate is "num/den".
    fps = None
    rate = v.get("avg_frame_rate") or v.get("r_frame_rate") or ""
    if "/" in str(rate):
        n, _, d = str(rate).partition("/")
        try:
            fps = round(float(n) / float(d), 3) if float(d) else None
        except (ValueError, ZeroDivisionError):
            fps = None
    # Rotation: modern ffprobe puts it in side_data_list (Display Matrix) or the
    # legacy tags:rotate. Normalize to degrees so callers can orient frames.
    rotation = None
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            try:
                rotation = int(sd["rotation"]) % 360
            except (TypeError, ValueError):
                pass
    if rotation is None:
        try:
            rotation = int(v.get("tags", {}).get("rotate")) % 360
        except (TypeError, ValueError):
            rotation = None

    creation = (v.get("tags", {}).get("creation_time")
                or fmt.get("tags", {}).get("creation_time"))
    return {
        "duration_s": round(duration, 3) if duration else None,
        "width": v.get("width"),
        "height": v.get("height"),
        "codec": v.get("codec_name"),
        "capture_time": creation,
        "fps": fps,
        "rotation": rotation,
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
    }


def _run(args: list[str], capture_stdout: bool) -> subprocess.CompletedProcess:
    return subprocess.run(
        [ffmpeg_exe(), "-hide_banner", *args],
        stdout=subprocess.PIPE if capture_stdout else subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        timeout=_TIMEOUT,
        creationflags=_NO_WINDOW,
    )


# ── metadata ──────────────────────────────────────────────────────────────────

_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d\d):(\d\d(?:\.\d+)?)")
_VIDEO_STREAM_RE = re.compile(
    r"Stream #\d+:\d+.*?:\s*Video:\s*([\w0-9]+).*?(\b\d{2,5})x(\d{2,5})\b",
    re.DOTALL,
)
_CREATION_RE = re.compile(r"creation_time\s*:\s*([0-9T:\-.]+Z?)")


def probe(path: str) -> dict | None:
    """Video metadata. Uses ffprobe JSON when available (structured, robust —
    the production path), falling back to parsing `ffmpeg -i` stderr with regex
    when ffprobe isn't installed. Returns
    {duration_s, width, height, codec, capture_time, fps, rotation, has_audio}
    or None if the file isn't a decodable video.

    The fallback can't reliably determine fps/rotation/audio from stderr, so
    those come back None/False there — callers must treat them as "unknown",
    not "absent" (e.g. don't skip ASR just because has_audio is False under the
    regex fallback; only trust has_audio=False from the ffprobe path)."""
    meta = _probe_ffprobe(path)
    if meta is not None:
        return meta

    # Fallback: `ffmpeg -i` with no output exits non-zero by design and prints
    # the stream info we want to stderr — that's expected, not an error.
    try:
        proc = _run(["-i", path], capture_stdout=False)
    except Exception as e:
        print(f"[video] probe failed for {path}: {e}")
        return None
    info = proc.stderr.decode("utf-8", "replace")

    out: dict = {"duration_s": None, "width": None, "height": None,
                 "codec": None, "capture_time": None,
                 "fps": None, "rotation": None,
                 # Unknown under the regex fallback (no ffprobe). None (not
                 # False) so ASR doesn't wrongly skip a video that does have
                 # audio — see the transcribe path, which probes with ffprobe.
                 "has_audio": None}

    m = _DURATION_RE.search(info)
    if m:
        h, mn, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
        out["duration_s"] = round(h * 3600 + mn * 60 + s, 3)

    m = _VIDEO_STREAM_RE.search(info)
    if not m:
        # No video stream found — audio file, image, or unreadable. Not a video.
        return None
    out["codec"] = m.group(1)
    out["width"] = int(m.group(2))
    out["height"] = int(m.group(3))

    m = _CREATION_RE.search(info)
    if m:
        out["capture_time"] = m.group(1)
    # Best-effort audio detection from stderr for the fallback path.
    if re.search(r"Stream #\d+:\d+.*?:\s*Audio:", info):
        out["has_audio"] = True
    return out


def _poster_timestamp(duration_s: float | None) -> float:
    """Seek target for a representative still: 10% in (skips the black/blurry
    lead-in most clips open with), capped at 3s so short clips don't overshoot,
    floored at 0 for unknown/zero-length durations."""
    if not duration_s or duration_s <= 0:
        return 0.0
    return min(max(duration_s * 0.1, 0.0), 3.0)


# ── frame extraction ──────────────────────────────────────────────────────────

def poster_frame(path: str, out_path: str, at: float | None = None) -> bool:
    """Write a single representative JPEG poster frame for `path` to `out_path`.
    Returns False (never raises) on any failure so a callers's thumbnail step
    degrades to a placeholder instead of crashing a job."""
    if at is None:
        meta = probe(path)
        at = _poster_timestamp(meta.get("duration_s") if meta else None)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    try:
        # -ss before -i = fast (keyframe) seek; -frames:v 1 = one frame.
        proc = _run(["-y", "-ss", f"{at:.3f}", "-i", path,
                     "-frames:v", "1", "-q:v", "3", out_path],
                    capture_stdout=False)
        if proc.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path):
            return True
        # Fast seek can miss on some containers — retry from the very start.
        proc = _run(["-y", "-i", path, "-frames:v", "1", "-q:v", "3", out_path],
                    capture_stdout=False)
        return (proc.returncode == 0 and os.path.exists(out_path)
                and os.path.getsize(out_path) > 0)
    except Exception as e:
        print(f"[video] poster_frame failed for {path}: {e}")
        return False


def _extract_at_timestamps(path: str, timestamps: list[float]) -> list[bytes]:
    """Extract one JPEG per timestamp, returned as in-memory bytes in order.
    Frames that fail are skipped rather than aborting the set."""
    frames: list[bytes] = []
    with tempfile.TemporaryDirectory(prefix="pv_frames_") as tmp:
        for i, ts in enumerate(timestamps):
            out = os.path.join(tmp, f"f{i}.jpg")
            try:
                proc = _run(["-y", "-ss", f"{ts:.3f}", "-i", path,
                             "-frames:v", "1", "-q:v", "3", out],
                            capture_stdout=False)
                if proc.returncode == 0 and os.path.exists(out):
                    with open(out, "rb") as f:
                        data = f.read()
                    if data:
                        frames.append(data)
            except Exception as e:
                print(f"[video] frame {i} @ {ts}s failed for {path}: {e}")
    return frames


def _uniform_timestamps(duration: float, count: int) -> list[float]:
    """Evenly spaced seek targets, avoiding the exact start/end (often black)."""
    if duration <= 0:
        return [0.0]
    step = duration / (count + 1)
    return [round(step * (i + 1), 3) for i in range(count)]


def extract_frames(path: str, count: int = 4) -> list[bytes]:
    """Sample up to `count` JPEG keyframes evenly across the clip, returned as
    in-memory bytes (for the vision pipeline). Frames that fail to extract are
    skipped rather than aborting the set. Empty list means nothing decodable."""
    count = max(1, count)
    duration = (probe(path) or {}).get("duration_s") or 0.0
    return _extract_at_timestamps(path, _uniform_timestamps(duration, count))


def _scene_timestamps(path: str, threshold: float = 0.3, limit: int = 60) -> list[float]:
    """Timestamps (s) of scene changes, via ffmpeg's scene-detection filter
    (`select='gt(scene,threshold)'` + showinfo, whose pts_time we parse from
    stderr). Empty list on failure or a static clip with no cuts. This is the
    content-adaptive signal production systems use instead of blind uniform
    sampling — a fast-cut montage yields many, a static clip yields none."""
    try:
        proc = _run(["-i", path, "-vf",
                     f"select='gt(scene,{threshold})',showinfo",
                     "-fps_mode", "vfr", "-f", "null", "-"],
                    capture_stdout=False)
    except Exception as e:
        print(f"[video] scene detect failed for {path}: {e}")
        return []
    info = proc.stderr.decode("utf-8", "replace")
    times = []
    for m in re.finditer(r"pts_time:([0-9.]+)", info):
        try:
            times.append(round(float(m.group(1)), 3))
        except ValueError:
            pass
    return sorted(set(times))[:limit]


def extract_keyframes(path: str, max_frames: int = 8,
                      scene_threshold: float = 0.3) -> list[bytes]:
    """Content-adaptive keyframes: frames at detected scene changes, blended
    with a few evenly-spaced anchors so even a static clip still yields coverage.
    Returns up to `max_frames` JPEG blobs in temporal order. Degrades to uniform
    extract_frames on any failure (unknown duration, no ffmpeg scene support).

    This is the production-style 'representative frame per shot' selection rather
    than 'every duration/N'. `max_frames` caps cost; scene_threshold ~0.3 is a
    reasonable cut sensitivity (higher = only hard cuts)."""
    max_frames = max(1, max_frames)
    duration = (probe(path) or {}).get("duration_s") or 0.0
    if duration <= 0:
        return extract_frames(path, count=max_frames)

    scenes = _scene_timestamps(path, scene_threshold)
    # Always include a few anchors so a static clip (no cuts) is still covered,
    # and the very start/end (often black) are avoided.
    anchors = [round(duration * f, 3) for f in (0.1, 0.5, 0.9)]
    cand = sorted({round(t, 1) for t in (scenes + anchors) if 0 < t < duration})
    if not cand:
        return extract_frames(path, count=max_frames)
    # Thin evenly to <= max_frames while preserving temporal spread.
    if len(cand) > max_frames:
        step = len(cand) / max_frames
        cand = [cand[min(len(cand) - 1, int(i * step))] for i in range(max_frames)]
    frames = _extract_at_timestamps(path, cand)
    return frames or extract_frames(path, count=max_frames)
