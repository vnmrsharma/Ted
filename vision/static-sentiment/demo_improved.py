#!/usr/bin/env python3
"""
Demo script for the improved emotion detection system
Shows the system capabilities without requiring a webcam
"""

import cv2
import numpy as np
from deepface import DeepFace
import time

def create_test_faces():
    """Create synthetic test faces for demonstration"""
    faces = []
    
    # Happy face (bright, wide)
    happy = np.ones((200, 200, 3), dtype=np.uint8) * 180
    happy[80:120, 60:140] = [255, 255, 255]  # Eyes
    happy[140:180, 70:130] = [255, 255, 255]  # Mouth
    faces.append(("Happy", happy))
    
    # Sad face (darker, narrow)
    sad = np.ones((200, 200, 3), dtype=np.uint8) * 100
    sad[80:120, 60:140] = [150, 150, 150]  # Eyes
    sad[140:180, 70:130] = [150, 150, 150]  # Mouth
    faces.append(("Sad", sad))
    
    # Angry face (high contrast)
    angry = np.ones((200, 200, 3), dtype=np.uint8) * 120
    angry[80:120, 60:140] = [50, 50, 50]   # Dark eyes
    angry[140:180, 70:130] = [50, 50, 50]   # Dark mouth
    faces.append(("Angry", angry))
    
    # Neutral face (medium)
    neutral = np.ones((200, 200, 3), dtype=np.uint8) * 128
    neutral[80:120, 60:140] = [100, 100, 100]  # Medium eyes
    neutral[140:180, 70:130] = [100, 100, 100]  # Medium mouth
    faces.append(("Neutral", neutral))
    
    return faces

def analyze_emotion_deepface(face_img):
    """Analyze emotion using DeepFace"""
    try:
        # Convert BGR to RGB
        rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Analyze emotion
        result = DeepFace.analyze(
            rgb_face,
            actions=['emotion'],
            enforce_detection=False,
            silent=True
        )
        
        if isinstance(result, list) and len(result) > 0:
            emotion_data = result[0]
            if 'emotion' in emotion_data:
                emotions = emotion_data['emotion']
                return emotions
        
        return {}
        
    except Exception as e:
        return {}

def demo_improved_system():
    """Demonstrate the improved emotion detection system"""
    print("🎭 Improved Emotion Detection System Demo")
    print("=" * 50)
    print("🔍 Testing DeepFace emotion recognition on synthetic faces")
    print()
    
    # Create test faces
    test_faces = create_test_faces()
    
    print("📊 Analysis Results:")
    print("-" * 50)
    
    for face_name, face_img in test_faces:
        print(f"\n🎭 {face_name} Face:")
        
        # Analyze with DeepFace
        start_time = time.time()
        emotions = analyze_emotion_deepface(face_img)
        analysis_time = time.time() - start_time
        
        if emotions:
            # Sort emotions by confidence
            sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
            
            print(f"   ⏱️  Analysis time: {analysis_time:.3f}s")
            print(f"   🎯 Top emotions:")
            
            for i, (emotion, confidence) in enumerate(sorted_emotions[:3]):
                emoji = {
                    'angry': '😠', 'disgust': '🤢', 'fear': '😨',
                    'happy': '😊', 'sad': '😢', 'surprise': '😲', 'neutral': '😐'
                }.get(emotion, '❓')
                
                print(f"      {i+1}. {emoji} {emotion}: {confidence:.1f}%")
        else:
            print("   ❌ No emotion detected")
        
        # Display the face
        cv2.imshow(f"{face_name} Face", face_img)
        cv2.waitKey(1000)  # Show for 1 second
    
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 50)
    print("🚀 Demo completed!")
    print("💡 Run 'python main_improved.py' for real-time detection")

def main():
    """Main function"""
    try:
        demo_improved_system()
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        print("💡 Make sure DeepFace is installed: pip install deepface")

if __name__ == "__main__":
    main()
