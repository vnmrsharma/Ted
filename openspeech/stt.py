# stt.py
from faster_whisper import WhisperModel

# Load once at import (downloads on first use).
# Options: tiny, base, small, medium, large-v3
# CPU tip: int8 is fastest; GPU tip: compute_type="float16"
_whisper = WhisperModel("small", compute_type="int8")

def _ts(seconds: float) -> str:
    ms = int((seconds - int(seconds)) * 1000)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def _to_srt(segments) -> str:
    lines = []
    for i, s in enumerate(segments, 1):
        lines += [str(i), f"{_ts(s.start)} --> {_ts(s.end)}", s.text.strip(), ""]
    return "\n".join(lines)

def transcribe(path: str, language: str | None = None, beam_size: int = 1) -> dict:
    seg_gen, info = _whisper.transcribe(path, language=language, beam_size=beam_size)
    segs = list(seg_gen)
    text = "".join(s.text for s in segs).strip()
    return {
        "text": text,
        "language": info.language,
        "duration": info.duration,
        "srt": _to_srt(segs),
    }
