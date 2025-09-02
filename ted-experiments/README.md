# Experimental Code Directory

This directory contains various code experiments and attempts we made while trying to create Ted. Not all the code here is correct or working - these are our experiments with different approaches and objectives.

## What You Need to Download/Install

1. **Python Dependencies**: `pip install -r requirements.txt`
2. **Ollama**: Download from [ollama.com](https://ollama.com) 
3. **GPT-OSS Model**: Run `ollama pull gpt-oss:20b` (13GB download)
4. **YOLO Model**: Downloads automatically when you run the scripts
 
Note: you can further offload the gpt-oss:20b and run it directly, but would require a lot of virtual memeory allocation.

## File Descriptions

### Working Files
- `smart-vision-yolo-enhanced.py` - **THE BEST ONE** - YOLO + GPT-OSS with rich visual understanding for clothing/pose questions
- `main.py` - Simple YOLO webcam detection with distance estimation and memory monitoring
- `gpt-oos.py` - Clean GPT-OSS chat interface using Ollama (no more complex setup!)

### Failed Experiments
- `smart-vision.py` - Early attempt at YOLO + AI integration
- `smart-vision-fixed.py` - Tried to fix the above, still had issues
- `smart-vision-working.py` - Got basic functionality working
- `smart-vision-final.py` - Thought this was final, but wasn't
- `smart-vision-final-working.py` - Another "final" attempt
- `smart-vision-optimized.py` - Performance improvements attempt
- `smart-vision-minimal-oss.py` - Stripped down version
- `smart-vision-visual-oss.py` - Tried to make GPT-OSS see images (impossible)
- `smart-vision-gpt-oss-fixed.py` - More fixes that didn't quite work

### Integration Attempts
- `integrated-oos-yolo.py` - Early YOLO + GPT-OSS combination
- `integrated-ollama-yolo.py` - Using Ollama for easier model management
- `gui-transcription.py` - Attempted GUI interface
- `clean-transcription.py` - Clean terminal output version

### Debug Scripts
- `debug-gpt.py` - Testing GPT-OSS responses
- `debug-gpt-oss-deep.py` - Deep debugging of model issues
- `test-ollama.py` - Testing Ollama connection

## Quick Start

1. Install stuff: `pip install -r requirements.txt`
2. Get Ollama and the model: `ollama pull gpt-oss:20b`
3. Run the good one: `python3 smart-vision-yolo-enhanced.py`
4. Or test GPT-OSS alone: `python3 gpt-oos.py`

## Notes

- Most of these files are failed attempts - we kept them to show the learning process
- The `.gitignore` prevents uploading the huge model files
- Memory usage info helps determine if it runs on Raspberry Pi



