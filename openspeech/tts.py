# tts.py
import subprocess, sys, shutil
from pathlib import Path

VOICE = Path("models/tts/en_US-libritts_r-medium.onnx")
VOICE_CFG = Path("models/tts/en_US-libritts_r-medium.onnx.json")

def _piper_cmd():
    exe = shutil.which("piper")
    return [exe] if exe else [sys.executable, "-m", "piper"]

def synthesize(text: str, out_wav: str = "speech.wav") -> str:
    cmd = _piper_cmd() + [
        "--model", str(VOICE),
        "--config", str(VOICE_CFG),
        "--output_file", out_wav,
    ]
    subprocess.run(cmd, input=text.encode("utf-8"), check=True)
    return out_wav
