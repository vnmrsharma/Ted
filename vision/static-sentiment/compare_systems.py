#!/usr/bin/env python3
"""
Comparison script for emotion detection systems
Shows the differences between rule-based and DeepFace-based approaches
"""

import cv2
import numpy as np
import time
from deepface import DeepFace

def test_rule_based_emotion(face_roi):
    """Test the old rule-based emotion detection"""
    try:
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        h, w = gray_face.shape
        
        # Simple rule-based features (from old system)
        eye_region = gray_face[:h//2, :]
        mouth_region = gray_face[h//2:, :]
        
        eye_mean = np.mean(eye_region)
        mouth_mean = np.mean(mouth_region)
        mouth_eye_ratio = mouth_mean / (eye_mean + 1e-6)
        
        brightness = np.mean(gray_face)
        contrast = np.std(gray_face)
        
        # Rule-based classification
        if mouth_eye_ratio > 1.05 and brightness > 110:
            return "Happy"
        elif mouth_eye_ratio < 0.95 and brightness < 100:
            return "Sad"
        elif contrast > 45 and brightness < 120:
            return "Angry"
        elif contrast > 40 and mouth_eye_ratio > 1.1:
            return "Surprise"
        elif contrast > 50 and brightness < 90:
            return "Fear"
        elif mouth_eye_ratio < 0.9 and contrast > 35:
            return "Disgust"
        else:
            return "Neutral"
            
    except:
        return "Neutral"

def test_deepface_emotion(face_roi):
    """Test the new DeepFace emotion detection"""
    try:
        # Convert BGR to RGB for DeepFace
        rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        
        # Analyze with DeepFace
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
                dominant_emotion = max(emotions.items(), key=lambda x: x[1])
                return dominant_emotion[0], dominant_emotion[1]
        
        return "neutral", 0.0
        
    except Exception as e:
        return "neutral", 0.0

def compare_emotion_detection():
    """Compare both emotion detection methods"""
    print("🔍 Emotion Detection System Comparison")
    print("=" * 50)
    print("📊 Testing rule-based vs DeepFace approaches")
    print()
    
    # Test with different synthetic face patterns
    test_cases = [
        ("Happy Face", np.ones((224, 224, 3), dtype=np.uint8) * 150),  # Bright
        ("Sad Face", np.ones((224, 224, 3), dtype=np.uint8) * 80),    # Dark
        ("High Contrast", np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)),  # Random
        ("Neutral", np.ones((224, 224, 3), dtype=np.uint8) * 128),    # Medium
    ]
    
    print("📋 Test Results:")
    print("-" * 50)
    
    for name, test_image in test_cases:
        print(f"\n🎭 {name}:")
        
        # Test rule-based
        rule_result = test_rule_based_emotion(test_image)
        print(f"   Rule-based: {rule_result}")
        
        # Test DeepFace
        deepface_result, confidence = test_deepface_emotion(test_image)
        print(f"   DeepFace: {deepface_result} (conf: {confidence:.1f}%)")
    
    print("\n" + "=" * 50)
    print("📈 Key Differences:")
    print("✅ Rule-based: Fast, lightweight, but limited accuracy")
    print("✅ DeepFace: High accuracy, pre-trained models, but slower")
    print("✅ Improved system: Combines both for best performance")

def main():
    """Main function"""
    try:
        compare_emotion_detection()
    except Exception as e:
        print(f"❌ Error during comparison: {e}")
        print("💡 Make sure DeepFace is installed: pip install deepface")

if __name__ == "__main__":
    main()
