#!/usr/bin/env python3
"""
Test script for Face Recognition System
Verifies that all components are working correctly
"""

import sys
import os
import numpy as np

def test_imports():
    """Test if all required packages can be imported"""
    print("🧪 Testing package imports...")
    
    try:
        import cv2
        print("✅ OpenCV imported successfully")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import gradio as gr
        print("✅ Gradio imported successfully")
    except ImportError as e:
        print(f"❌ Gradio import failed: {e}")
        return False
    
    try:
        from deepface import DeepFace
        print("✅ DeepFace imported successfully")
    except ImportError as e:
        print(f"❌ DeepFace import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print("✅ PIL imported successfully")
    except ImportError as e:
        print(f"❌ PIL import failed: {e}")
        return False
    
    return True

def test_deepface_models():
    """Test if DeepFace models are available"""
    print("\n🧪 Testing DeepFace models...")
    
    try:
        from deepface import DeepFace
        
        # Test if VGG-Face model is available
        print("📥 Checking VGG-Face model availability...")
        
        # This will download the model if not available
        models = DeepFace.build_model("VGG-Face")
        print("✅ VGG-Face model loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ DeepFace model test failed: {e}")
        print("💡 This might take some time on first run as models download automatically")
        return False

def test_camera_access():
    """Test if camera can be accessed"""
    print("\n🧪 Testing camera access...")
    
    try:
        import cv2
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Camera access failed - no camera available")
            return False
        
        ret, frame = cap.read()
        if not ret:
            print("❌ Camera access failed - cannot read frames")
            cap.release()
            return False
        
        print(f"✅ Camera accessed successfully - Frame size: {frame.shape}")
        cap.release()
        return True
        
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def test_face_detection():
    """Test basic face detection functionality"""
    print("\n🧪 Testing face detection...")
    
    try:
        from deepface import DeepFace
        import cv2
        
        # Create a simple test image (no face)
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Test face detection
        result = DeepFace.extract_faces(
            test_image,
            detector_backend="opencv",
            enforce_detection=False
        )
        
        print("✅ Face detection test completed")
        return True
        
    except Exception as e:
        print(f"❌ Face detection test failed: {e}")
        return False

def test_system_components():
    """Test main system components"""
    print("\n🧪 Testing system components...")
    
    try:
        # Import the main system
        sys.path.append('.')
        from main import FaceRecognitionSystem
        
        # Create instance
        system = FaceRecognitionSystem()
        print("✅ FaceRecognitionSystem created successfully")
        
        # Test status
        status = system.get_status()
        print(f"✅ System status: {status}")
        
        return True
        
    except Exception as e:
        print(f"❌ System component test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Face Recognition System - System Test")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("DeepFace Models", test_deepface_models),
        ("Camera Access", test_camera_access),
        ("Face Detection", test_face_detection),
        ("System Components", test_system_components)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} - PASSED")
            else:
                print(f"❌ {test_name} - FAILED")
        except Exception as e:
            print(f"❌ {test_name} - ERROR: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("\n🚀 To start the application:")
        print("   python3 main.py")
        print("   Then open http://localhost:7860 in your browser")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        print("\n💡 Common solutions:")
        print("   - Install missing packages: pip install -r requirements.txt")
        print("   - Check camera permissions")
        print("   - Ensure internet connection for model downloads")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
