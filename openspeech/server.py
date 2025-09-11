# server.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import tempfile, os
from stt import transcribe
from tts import synthesize

app = FastAPI(title="Open Speech Stack")

@app.post("/stt")
async def stt_endpoint(file: UploadFile = File(...), language: str | None = Form(None)):
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(await file.read())
        path = f.name
    try:
        result = transcribe(path, language=language)
        return JSONResponse(result)
    finally:
        os.remove(path)

@app.post("/tts")
async def tts_endpoint(text: str = Form(...)):
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    synthesize(text, out_path)
    return FileResponse(out_path, media_type="audio/wav", filename="speech.wav")

# Serve the simple web client
app.mount("/", StaticFiles(directory="web", html=True), name="web")
