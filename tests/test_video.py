"""video.py — the one home for ffmpeg. The binary is always mocked (no real
ffmpeg runs), so probe-parsing, frame extraction and failure handling are
exercised on any host with no external dependency."""
from types import SimpleNamespace

import pytest

import video


# A realistic `ffmpeg -i` stderr dump (the info ffmpeg prints when asked to
# decode a file with no output specified — exit code 1, but the metadata we
# parse is all here).
_FFMPEG_INFO = b"""
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'clip.mp4':
  Metadata:
    creation_time   : 2022-05-17T10:30:00.000000Z
  Duration: 00:01:23.45, start: 0.000000, bitrate: 12000 kb/s
    Stream #0:0(und): Video: h264 (High) (avc1 / 0x31637661), yuv420p, 1920x1080 [SAR 1:1 DAR 16:9], 12000 kb/s, 30 fps
    Stream #0:1(und): Audio: aac (LC), 48000 Hz, stereo, fltp, 128 kb/s
"""

_AUDIO_ONLY = b"""
Input #0, mp3, from 'song.mp3':
  Duration: 00:03:10.00, start: 0.000000, bitrate: 320 kb/s
    Stream #0:0: Audio: mp3, 44100 Hz, stereo, fltp, 320 kb/s
"""


# ── probe ─────────────────────────────────────────────────────────────────────

def test_probe_parses_duration_dims_codec_and_time(monkeypatch):
    monkeypatch.setattr(video, "_run",
                        lambda args, capture_stdout: SimpleNamespace(stderr=_FFMPEG_INFO))
    info = video.probe("clip.mp4")
    assert info is not None
    assert info["duration_s"] == pytest.approx(83.45)
    assert info["width"] == 1920 and info["height"] == 1080
    assert info["codec"] == "h264"
    assert info["capture_time"].startswith("2022-05-17T10:30:00")


def test_probe_returns_none_for_no_video_stream(monkeypatch):
    # An audio-only file has a Duration but no Video stream — not a video.
    monkeypatch.setattr(video, "_run",
                        lambda args, capture_stdout: SimpleNamespace(stderr=_AUDIO_ONLY))
    assert video.probe("song.mp3") is None


def test_probe_returns_none_on_ffmpeg_error(monkeypatch):
    def boom(args, capture_stdout):
        raise OSError("ffmpeg missing")
    monkeypatch.setattr(video, "_run", boom)
    assert video.probe("clip.mp4") is None


def test_poster_timestamp_skips_lead_in_and_caps():
    assert video._poster_timestamp(None) == 0.0
    assert video._poster_timestamp(0) == 0.0
    assert video._poster_timestamp(10) == pytest.approx(1.0)   # 10% of 10s
    assert video._poster_timestamp(100) == 3.0                 # capped at 3s


# ── poster_frame / extract_frames (mock the ffmpeg subprocess) ────────────────

def _fake_ffmpeg_writing(out_bytes=b"\xff\xd8jpegdata"):
    """A subprocess.run stand-in: writes the requested output file (last argv
    token) and reports success, so poster/frame extraction can be tested without
    a real ffmpeg."""
    def run(argv, **kw):
        out = argv[-1]
        if out.endswith(".jpg"):
            with open(out, "wb") as f:
                f.write(out_bytes)
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
    return run


def test_poster_frame_writes_jpeg_and_uses_frame_args(tmp_path, monkeypatch):
    monkeypatch.setattr(video, "ffmpeg_exe", lambda: "ffmpeg")
    seen = {}
    def run(argv, **kw):
        seen["argv"] = argv
        with open(argv[-1], "wb") as f:
            f.write(b"\xff\xd8jpeg")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
    monkeypatch.setattr(video.subprocess, "run", run)
    out = tmp_path / "poster.jpg"
    assert video.poster_frame("clip.mp4", str(out), at=1.5) is True
    assert out.read_bytes().startswith(b"\xff\xd8")
    assert "-frames:v" in seen["argv"] and "1" in seen["argv"]
    assert "1.500" in seen["argv"]  # the requested seek timestamp


def test_poster_frame_returns_false_when_ffmpeg_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(video, "ffmpeg_exe", lambda: "ffmpeg")
    monkeypatch.setattr(video.subprocess, "run",
                        lambda argv, **kw: SimpleNamespace(returncode=1, stdout=b"", stderr=b"err"))
    out = tmp_path / "poster.jpg"
    assert video.poster_frame("bad.mp4", str(out)) is False


def test_extract_frames_samples_evenly_and_returns_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(video, "ffmpeg_exe", lambda: "ffmpeg")
    monkeypatch.setattr(video, "probe", lambda p: {"duration_s": 40.0})
    monkeypatch.setattr(video.subprocess, "run", _fake_ffmpeg_writing())
    frames = video.extract_frames("clip.mp4", count=4)
    assert len(frames) == 4
    assert all(f.startswith(b"\xff\xd8") for f in frames)


def test_extract_frames_empty_when_nothing_decodes(tmp_path, monkeypatch):
    monkeypatch.setattr(video, "ffmpeg_exe", lambda: "ffmpeg")
    monkeypatch.setattr(video, "probe", lambda p: {"duration_s": 40.0})
    # ffmpeg "succeeds" but writes nothing → no frames harvested.
    monkeypatch.setattr(video.subprocess, "run",
                        lambda argv, **kw: SimpleNamespace(returncode=1, stdout=b"", stderr=b""))
    assert video.extract_frames("clip.mp4", count=3) == []


# ── video transcript folding (ASR → searchable caption) ─────────────────────

def test_attach_transcript_folds_into_caption_and_stores_field():
    import json
    from indexer import _attach_transcript
    cj = json.dumps({"caption": "A person speaks to the camera.", "mood": "happy"})
    out = json.loads(_attach_transcript(cj, "hello there friends", language="en"))
    assert out["transcript"] == "hello there friends"
    assert out["transcript_language"] == "en"
    # speech rides along in caption (what gets embedded) so it's searchable
    assert "hello there friends" in out["caption"]
    assert out["mood"] == "happy"  # other fields untouched


def test_attach_transcript_noop_when_empty():
    import json
    from indexer import _attach_transcript
    cj = json.dumps({"caption": "silent clip"})
    assert _attach_transcript(cj, "") == cj
    assert json.loads(_attach_transcript(cj, "   ".strip()))["caption"] == "silent clip"
