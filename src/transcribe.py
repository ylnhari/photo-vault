"""Local speech-to-text for videos — production-grade ASR via Whisper
(faster-whisper / CTranslate2). This is the same capability that makes spoken
words in videos searchable in Google Photos / Apple Photos; we run it locally so
no audio leaves the machine and there's no API cost.

Optional by design: if faster-whisper isn't installed, every entry point returns
an empty transcript and video captioning still works — ASR is enrichment, not a
hard dependency. The model is lazy-loaded once and cached; a short personal clip
transcribes in a few seconds on CPU (int8). Never raises — a transcription
failure degrades to "no transcript", never a job failure.
"""
import os
import threading

_model = None
_model_lock = threading.Lock()

# base = the speed/quality sweet spot for short personal clips on CPU. Override
# with PHOTO_VAULT_WHISPER_MODEL (tiny/base/small/medium/large-v3).
_MODEL_SIZE = os.environ.get("PHOTO_VAULT_WHISPER_MODEL", "base")
# CPU int8 is portable and fast enough; a CUDA box can set device via env.
_DEVICE = os.environ.get("PHOTO_VAULT_WHISPER_DEVICE", "cpu")
_COMPUTE = os.environ.get("PHOTO_VAULT_WHISPER_COMPUTE", "int8")


def available() -> bool:
    """True when the optional ASR dependency is installed."""
    try:
        import faster_whisper  # noqa: F401
        return True
    except Exception:
        return False


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from faster_whisper import WhisperModel
                _model = WhisperModel(_MODEL_SIZE, device=_DEVICE,
                                      compute_type=_COMPUTE)
    return _model


def transcribe_video(path: str, has_audio: bool | None = None) -> dict:
    """Transcribe a video's speech. Returns {"text": str, "language": str|None}.

    Returns an empty transcript (never raises) when: ASR isn't installed, the
    container has no audio stream, there's no detectable speech, or decoding
    fails. `has_audio` short-circuits the no-audio case — pass the value from
    video.probe(); only an explicit False is trusted (None = unknown, so we
    still attempt and let the decoder tell us)."""
    empty = {"text": "", "language": None}
    if not available():
        return empty
    if has_audio is None:
        try:
            import video
            has_audio = (video.probe(path) or {}).get("has_audio")
        except Exception:
            has_audio = None
    if has_audio is False:
        return empty
    try:
        model = _get_model()
        # vad_filter drops silence/non-speech so we don't hallucinate captions
        # over music or ambience.
        segments, info = model.transcribe(path, vad_filter=True)
        text = " ".join(s.text.strip() for s in segments).strip()
        return {"text": text, "language": getattr(info, "language", None)}
    except Exception as e:
        # A video with no audio track raises inside PyAV (IndexError); anything
        # else is an unexpected decode error. Either way ASR is best-effort.
        print(f"[transcribe] ASR unavailable for {path}: {type(e).__name__}: {e}")
        return empty
