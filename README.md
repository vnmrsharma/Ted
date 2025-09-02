
## Vision Directory

The `vision/` folder contains a high-performance YOLOv11 face emotion detection system:

### Files:
- `main.py` - Ultra-fast face emotion detection with 29.8 FPS performance
- `requirements.txt` - Dependencies for the vision system
- `setup.py` - Setup script for downloading models and configuring environment
- `yolo11n.pt` - YOLOv11 nano model for object detection

### Features:
- **High Performance**: 29.8 FPS real-time detection
- **Face-Only Detection**: No false positives from background objects
- **Advanced Emotion Classification**: Multi-feature analysis (eye/mouth regions, symmetry, contrast)
- **Lightweight**: No heavy dependencies, optimized for speed

### Usage:
```bash
cd vision
python3 setup.py  # One-time setup
python3 main.py   # Run emotion detection
```

### Controls:
- **SPACE**: Take screenshot
- **R**: Reset emotion history
- **+/-**: Adjust frame skip for speed vs accuracy
- **Q**: Quit
