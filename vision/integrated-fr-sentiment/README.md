# Integrated Face Recognition + Emotion Detection

This system combines the best of both worlds: **face recognition** from the DeepFace system and **real-time emotion detection** from the trained DeepFace models.

## 🎯 What It Does

- **Face Recognition**: Learns and remembers people's faces from images
- **Emotion Detection**: Analyzes emotions in real-time (7 emotion classes)
- **Real-time Processing**: Live webcam feed with instant results
- **Persistent Memory**: Saves learned faces to disk for future use

## 🚀 Features

### Face Recognition
- Learn new faces from images or live camera frames
- Recognize known people with confidence scores
- Persistent storage of face embeddings
- Adjustable recognition threshold

### Emotion Detection
- Real-time emotion analysis
- 7 emotion classes: angry, disgust, fear, happy, sad, surprise, neutral
- Confidence scores for each emotion
- Emotion history tracking

### User Interface
- Live webcam feed with real-time overlays
- Color-coded emotion display
- Recognition status indicators
- FPS counter and performance metrics

## 📋 Requirements

- Python 3.8+
- Webcam
- Sufficient RAM (4GB+ recommended)

## 🛠️ Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify DeepFace models** (first run will download automatically):
   ```bash
   python -c "from deepface import DeepFace; print('DeepFace ready')"
   ```

## 🚀 Quick Start

### Get Running in 2 Minutes!
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the interface
python gradio_interface.py

# 3. Open your browser at: http://localhost:7860
```

### What You'll See:
- **📚 Learn Face Tab**: Upload images to teach the system new faces
- **📹 Real-time Recognition Tab**: Live webcam with face recognition + emotion detection
- **🌐 Web-based**: Accessible from any device on your network
- **🎨 User-friendly**: Clean, intuitive interface with real-time updates

## 🎮 Usage

### Launch the Interface
```bash
python gradio_interface.py
```

The interface will open at `http://localhost:7860` and provide:
- **📚 Learn Face Tab**: Upload images to teach the system new faces
- **📹 Real-time Recognition Tab**: Live webcam with face recognition + emotion detection
- **🌐 Web-based**: Accessible from any device on your network
- **🎨 User-friendly**: Clean, intuitive interface with real-time updates

### Interactive Controls
- **'q'**: Quit the application
- **'l'**: Learn a new face from current camera frame
- **'e'**: Force emotion analysis on current frame

### Learning New Faces

#### Method 1: Live Camera Learning
1. Run the system: `python main.py`
2. Position the person in front of the camera
3. Press **'l'** to enter learning mode
4. Enter the person's name when prompted
5. The system will learn their face from the current frame

#### Method 2: Image File Learning
```python
from main import IntegratedFaceEmotionSystem

system = IntegratedFaceEmotionSystem()
success, message = system.learn_face_from_image("path/to/image.jpg", "John")
print(message)
```

## 🔧 Configuration

### Recognition Threshold
Adjust the sensitivity of face recognition:
```python
system.recognition_threshold = 0.6  # Default: 0.6 (higher = stricter)
```

### Camera Settings
```python
system = IntegratedFaceEmotionSystem(
    camera_id=0,        # Camera device ID
    frame_width=640,    # Frame width
    frame_height=480    # Frame height
)
```

## 📊 Performance

- **Face Recognition**: ~30-60ms per frame
- **Emotion Detection**: ~100-200ms per frame
- **Overall FPS**: 5-15 FPS (depending on hardware)
- **Memory Usage**: ~500MB-1GB

## 🗂️ File Structure

```
integrated-fr-sentiment/
├── gradio_interface.py  # Main Gradio web interface
├── requirements.txt     # Dependencies
├── README.md           # Documentation and usage guide
└── known_faces.json    # Learned face database (auto-generated)
```

## 🔍 How It Works

### Face Recognition Pipeline
1. **Face Detection**: OpenCV detects faces in each frame
2. **Feature Extraction**: DeepFace extracts 2622-dimensional embeddings
3. **Comparison**: Euclidean distance between current and known embeddings
4. **Matching**: Threshold-based recognition decision

### Emotion Detection Pipeline
1. **Face Detection**: Locate faces in the frame
2. **Emotion Analysis**: DeepFace analyzes facial expressions
3. **Classification**: 7-class emotion classification
4. **Confidence Scoring**: Probability scores for each emotion

## 🐛 Troubleshooting

### Common Issues

**"No face detected"**
- Ensure good lighting
- Face should be clearly visible
- Check camera positioning

**Low recognition accuracy**
- Adjust `recognition_threshold`
- Retrain faces in better lighting
- Use multiple training images per person

**Emotion detection not working**
- Press 'e' to force emotion analysis
- Check console for error messages
- Run `python test_emotion.py` to test DeepFace
- Ensure good lighting and clear face visibility
- Emotion analysis runs every 10 frames by default (adjustable)

**Slow performance**
- Reduce frame resolution
- Close other applications
- Check GPU availability

### Error Messages

- **"DeepFace initialization failed"**: Check TensorFlow installation
- **"Failed to open camera"**: Verify camera permissions and device ID
- **"No face detected in image"**: Ensure image contains clear face

## 🔬 Technical Details

### Models Used
- **Face Recognition**: VGG-Face (DeepFace backend)
- **Face Detection**: OpenCV Haar Cascades
- **Emotion Detection**: DeepFace emotion classifier

### Data Storage
- Face embeddings stored as JSON
- 2622-dimensional vectors per face
- Timestamp tracking for each learned face

### Performance Optimization
- Frame skipping for emotion analysis
- Efficient embedding comparison
- Memory-efficient face storage

## 📈 Future Enhancements

- [ ] Multiple face recognition per frame
- [ ] Age and gender detection
- [ ] Expression intensity measurement
- [ ] Cloud-based face database
- [ ] Mobile app integration
- [ ] Real-time alerts and notifications

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is part of the larger vision system workspace.
