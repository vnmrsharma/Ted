# 🎯 Face Recognition System with DeepFace

A simple, user-friendly face recognition system that learns one person's face and recognizes them in real-time using webcam or uploaded images.

## ✨ Features

- **📚 Learn Face**: Upload an image and teach the system who someone is
- **🔍 Recognize Face**: Check if uploaded images match the learned face
- **📹 Webcam Recognition**: Real-time face recognition using your webcam
- **⚙️ Adjustable Settings**: Fine-tune recognition sensitivity
- **💾 Persistent Storage**: Remembers faces between sessions
- **🎨 Beautiful UI**: Clean Gradio interface with tabs and visual feedback

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python3 main.py
```

### 3. Open Your Browser
Navigate to `http://localhost:7860`

## 📖 How to Use

### **Step 1: Learn a Face**
1. Go to the **"📚 Learn Face"** tab
2. Upload a clear image of the person's face
3. Enter their name
4. Click **"🎯 Learn Face"**
5. The system will extract facial features and save them

### **Step 2: Recognize Faces**
1. Go to the **"🔍 Recognize Face"** tab
2. Upload an image to check
3. Click **"🔍 Check Face"**
4. See if it matches the learned person

### **Step 3: Real-time Webcam Recognition**
1. Go to the **"📹 Webcam Recognition"** tab
2. Click **"🚀 Start Recognition"**
3. The webcam will continuously scan for the known face
4. Click **"🛑 Stop Recognition"** when done

### **Step 4: Adjust Settings**
1. Go to the **"⚙️ Settings"** tab
2. Adjust recognition threshold (lower = stricter matching)
3. Clear known faces if needed
4. Check system status

## 🔧 Technical Details

### **Recognition Threshold**
- **Lower values (0.1-0.3)**: Very strict matching, fewer false positives
- **Medium values (0.4-0.6)**: Balanced accuracy and sensitivity
- **Higher values (0.7-0.9)**: More lenient matching, more false positives

### **How It Works**
1. **Face Detection**: Uses OpenCV to find faces in images
2. **Feature Extraction**: DeepFace extracts 2622-dimensional face embeddings
3. **Similarity Calculation**: Compares embeddings using cosine similarity
4. **Matching**: Determines if faces are the same person based on threshold

### **Supported Image Formats**
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)

## 📁 File Structure

```
Face-recorgnisation-deepface/
├── main.py              # Main application file
├── requirements.txt     # Python dependencies
├── README.md           # This documentation
└── known_face.json     # Saved face data (created automatically)
```

## 🎮 Controls

### **Keyboard Shortcuts**
- **Tab Navigation**: Switch between different functions
- **Button Clicks**: Primary interaction method
- **Slider Adjustments**: Fine-tune recognition sensitivity

### **System Status**
- **✅ Green**: Known face loaded, system ready
- **❌ Red**: No known face, need to learn one first
- **🟢 Running**: Webcam recognition active
- **🔴 Stopped**: Webcam recognition inactive

## 🚨 Troubleshooting

### **Common Issues**

1. **"No face detected"**
   - Ensure the image contains a clear, front-facing face
   - Try different lighting conditions
   - Make sure the face is not too small or blurry

2. **"Failed to open webcam"**
   - Check if another application is using the webcam
   - Ensure webcam permissions are granted
   - Try refreshing the page

3. **"Recognition not working"**
   - Adjust the threshold in Settings tab
   - Try relearning the face with a different image
   - Ensure good lighting conditions

4. **"Slow performance"**
   - Close other applications using the camera
   - Reduce image resolution if possible
   - Check system resources

### **Performance Tips**

- **Good Lighting**: Bright, even lighting improves accuracy
- **Clear Images**: High-quality, focused images work best
- **Front-facing**: Face should be looking directly at camera
- **No Obstructions**: Avoid sunglasses, masks, or extreme angles

## 🔒 Privacy & Security

- **Local Processing**: All face recognition happens on your device
- **No Cloud Upload**: Images are not sent to external servers
- **Temporary Storage**: Face embeddings are stored locally only
- **Easy Deletion**: Clear learned faces anytime from Settings

## 🎯 Use Cases

- **Home Security**: Recognize family members
- **Access Control**: Simple door/gate authentication
- **Personal Assistant**: Identify who's in the room
- **Learning Tool**: Understand face recognition technology
- **Research**: Test and compare different recognition methods

## 🔮 Future Enhancements

- [ ] **Multiple Faces**: Remember multiple people
- [ ] **Face Database**: Store multiple images per person
- [ ] **Advanced Analytics**: Recognition confidence over time
- [ ] **Export Results**: Save recognition logs and statistics
- [ ] **Mobile Support**: Optimize for mobile devices
- [ ] **API Endpoints**: REST API for integration

## 📚 Technical Requirements

### **Hardware**
- **Camera**: Webcam or USB camera
- **RAM**: 4GB+ recommended
- **Storage**: 1GB+ free space
- **CPU**: Modern multi-core processor

### **Software**
- **Python**: 3.8+
- **OS**: Windows 10+, macOS 10.14+, Ubuntu 18.04+
- **Browser**: Chrome, Firefox, Safari, Edge

## 🤝 Contributing

Feel free to:
- Report bugs and issues
- Suggest new features
- Improve documentation
- Optimize performance
- Add new recognition models

## 📄 License

This project uses DeepFace which is licensed under the MIT License. See the [DeepFace repository](https://github.com/serengil/deepface) for more details.

## 🙏 Acknowledgments

- **DeepFace**: Core face recognition functionality
- **Gradio**: Beautiful web interface framework
- **OpenCV**: Computer vision capabilities
- **TensorFlow**: Deep learning backend

---

## 🎉 **Ready to Recognize Faces!**

**Start Learning**: Upload an image and teach the system  
**Test Recognition**: Check if images match your learned face  
**Go Live**: Use webcam for real-time recognition  

Your personal face recognition system is ready to go! 🚀
