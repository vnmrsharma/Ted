#!/usr/bin/env python3
"""
Setup script for YOLOv11 Emotion Detection
Downloads required models and sets up the environment
"""

import os
import sys
import subprocess
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("�� Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def download_models():
    """Download YOLOv11 models"""
    print("🔄 Downloading YOLOv11 models...")
    
    # Import after requirements are installed
    try:
        from ultralytics import YOLO
        
        # Download YOLOv11n (base model)
        print("📥 Downloading YOLOv11n...")
        model = YOLO("yolo11n.pt")
        print("✅ YOLOv11n downloaded")
        
        # Try to download face-specific model (if available)
        try:
            print("📥 Attempting to download YOLOv11 face model...")
            face_model = YOLO("yolo11n-face.pt")
            print("✅ YOLOv11 face model downloaded")
        except:
            print("⚠️  YOLOv11 face model not available, will use standard model")
        
        return True
        
    except ImportError:
        print("❌ Ultralytics not installed properly")
        return False
    except Exception as e:
        print(f"❌ Failed to download models: {e}")
        return False

def test_camera():
    """Test camera availability"""
    print("📷 Testing camera...")
    try:
        import cv2
        
        for i in range(4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret:
                    print(f"✅ Camera {i} detected and working")
                    return True
        
        print("❌ No working camera found")
        return False
        
    except ImportError:
        print("❌ OpenCV not installed")
        return False

def main():
    """Main setup function"""
    print("🎯 YOLOv11 Emotion Detection Setup")
    print("=" * 40)
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Download models
    if not download_models():
        return False
    
    # Test camera
    if not test_camera():
        print("⚠️  Camera test failed, but you can still use video files")
    
    print("\n✅ Setup completed successfully!")
    print("\n🚀 To run emotion detection:")
    print("   python3 main.py")
    print("\n📋 Available options:")
    print("   --model MODEL_PATH    (custom model)")
    print("   --source SOURCE       (0 for webcam, path for video)")
    print("   --conf CONFIDENCE     (detection confidence)")
    print("   --size WIDTH HEIGHT   (window size)")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
