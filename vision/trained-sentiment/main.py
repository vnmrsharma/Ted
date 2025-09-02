#!/usr/bin/env python3
"""
Real-time Emotion Detection System
Uses the trained neural network model for live webcam emotion analysis
"""

import torch
import torch.nn as nn
from torchvision import models
import cv2
import numpy as np
from pathlib import Path
import time
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings('ignore')

class RealTimeEmotionDetector:
    """Real-time emotion detection using trained model"""
    
    def __init__(self, model_path='best_emotion_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.emotion_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.emotion_colors = {
            'angry': (0, 0, 255),      # Red
            'disgust': (0, 165, 255),   # Orange
            'fear': (255, 0, 255),      # Magenta
            'happy': (0, 255, 0),       # Green
            'sad': (255, 0, 0),         # Blue
            'surprise': (0, 255, 255),  # Yellow
            'neutral': (128, 128, 128)  # Gray
        }
        
        # Load trained model
        self.model = self.load_model(model_path)
        self.model.eval()
        
        # Image transforms (same as training)
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.fps = 0
        self.frame_count = 0
        
        # Emotion history for smoothing
        self.emotion_history = []
        self.history_size = 5
        
        print(f"✅ Trained emotion model loaded from {model_path}")
        print(f"🚀 Using device: {self.device}")
        print(f"🎯 Detecting emotions: {', '.join(self.emotion_names)}")
    
    def load_model(self, model_path):
        """Load the trained emotion detection model"""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create the same model architecture used in training
        class AdvancedEmotionClassifier(nn.Module):
            def __init__(self, num_classes=7, pretrained=False):
                super(AdvancedEmotionClassifier, self).__init__()
                
                # Use EfficientNet-B2 as backbone
                self.backbone = models.efficientnet_b2(pretrained=pretrained)
                
                # Get the number of features from the backbone
                # EfficientNet-B2 has 1408 features in the last layer
                num_features = 1408
                
                # Create a proper classifier that works with the backbone
                self.classifier = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Dropout(p=0.4),
                    nn.Linear(num_features, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.3),
                    nn.Linear(1024, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.2),
                    nn.Linear(512, num_classes)
                )
            
            def forward(self, x):
                # Pass through backbone (excluding the original classifier)
                x = self.backbone.features(x)
                
                # Apply our custom classifier
                x = self.classifier(x)
                
                return x
        
        # Create model instance
        model = AdvancedEmotionClassifier(num_classes=7, pretrained=False)
        
        # Load trained weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        return model
    
    def detect_faces(self, frame):
        """Detect faces in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Enhanced face detection parameters
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.05,  # More sensitive
            minNeighbors=3,    # Less strict
            minSize=(60, 60),  # Larger minimum size
            maxSize=(300, 300) # Maximum size limit
        )
        
        return faces
    
    def predict_emotion(self, face_roi):
        """Predict emotion from face region"""
        try:
            # Preprocess face image
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            transformed = self.transform(image=face_rgb)
            face_tensor = transformed['image'].unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(face_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            emotion = self.emotion_names[predicted_class]
            return emotion, confidence, probabilities[0].cpu().numpy()
            
        except Exception as e:
            return "unknown", 0.0, np.zeros(7)
    
    def smooth_emotion(self, emotion, confidence):
        """Apply temporal smoothing to emotion predictions"""
        self.emotion_history.append((emotion, confidence))
        
        # Keep only recent history
        if len(self.emotion_history) > self.history_size:
            self.emotion_history.pop(0)
        
        # If we have enough history, apply smoothing
        if len(self.emotion_history) >= 3:
            # Get most common emotion in recent history
            recent_emotions = [e for e, _ in self.emotion_history[-3:]]
            emotion_counts = {}
            for e in recent_emotions:
                emotion_counts[e] = emotion_counts.get(e, 0) + 1
            
            # Find most common emotion
            most_common = max(emotion_counts.items(), key=lambda x: x[1])
            
            # Only change if we have strong consensus (2+ same emotion)
            if most_common[1] >= 2:
                return most_common[0], confidence
        
        return emotion, confidence
    
    def draw_emotion_info(self, frame, face, emotion, confidence, all_probs):
        """Draw emotion detection results on frame"""
        x, y, w, h = face
        
        # Draw face rectangle with emotion color
        color = self.emotion_colors.get(emotion, (255, 255, 255))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        
        # Draw emotion label with background
        label = f"{emotion.upper()}: {confidence:.1%}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        
        # Background rectangle for label
        cv2.rectangle(frame, (x, y - label_size[1] - 10), 
                     (x + label_size[0] + 10, y), color, -1)
        
        # Label text
        cv2.putText(frame, label, (x + 5, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw confidence bars for all emotions
        bar_width = 120
        bar_height = 12
        y_offset = y + h + 20
        
        for i, (emotion_name, prob) in enumerate(zip(self.emotion_names, all_probs)):
            bar_x = x + 10
            bar_y = y_offset + i * (bar_height + 3)
            
            # Background bar
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                         (50, 50, 50), -1)
            
            # Probability bar
            prob_width = int(prob * bar_width)
            bar_color = self.emotion_colors.get(emotion_name, (128, 128, 128))
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + prob_width, bar_y + bar_height), 
                         bar_color, -1)
            
            # Text
            text = f"{emotion_name}: {prob:.1%}"
            cv2.putText(frame, text, (bar_x + bar_width + 10, bar_y + bar_height - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def draw_ui(self, frame):
        """Draw UI elements on frame"""
        # FPS counter
        cv2.putText(frame, f"FPS: {self.fps}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Frame counter
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Model info
        cv2.putText(frame, "Trained Model: EfficientNet-B2", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit, 's' to save frame, 'r' to reset", 
                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def run_detection(self, camera_id=0, save_frames=False):
        """Run real-time emotion detection"""
        print("🎥 Starting real-time emotion detection...")
        print("📋 Controls:")
        print("   - Press 'q' to quit")
        print("   - Press 's' to save current frame")
        print("   - Press 'r' to reset FPS counter")
        print("   - Press 'f' to toggle fullscreen")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"❌ Could not open camera {camera_id}")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
        
        # Performance tracking
        processing_times = []
        fullscreen = False
        
        try:
            while True:
                start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to read frame")
                    break
                
                # Detect faces
                faces = self.detect_faces(frame)
                
                # Process each face
                for face in faces:
                    x, y, w, h = face
                    
                    # Ensure face region is valid
                    if w < 60 or h < 60:
                        continue
                    
                    # Extract face ROI
                    face_roi = frame[y:y+h, x:x+w]
                    
                    # Predict emotion
                    emotion, confidence, all_probs = self.predict_emotion(face_roi)
                    
                    # Apply temporal smoothing
                    smoothed_emotion, smoothed_confidence = self.smooth_emotion(emotion, confidence)
                    
                    # Draw results
                    self.draw_emotion_info(frame, face, smoothed_emotion, smoothed_confidence, all_probs)
                
                # Update FPS
                self.update_fps()
                
                # Draw UI
                self.draw_ui(frame)
                
                # Calculate processing time
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                
                # Keep only recent processing times
                if len(processing_times) > 30:
                    processing_times.pop(0)
                
                # Display average processing time
                avg_processing = np.mean(processing_times)
                cv2.putText(frame, f"Processing: {avg_processing*1000:.1f}ms", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display frame
                window_name = 'Real-time Emotion Detection'
                if fullscreen:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                
                cv2.imshow(window_name, frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f"emotion_frame_{self.frame_count:04d}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"💾 Frame saved as {filename}")
                elif key == ord('r'):
                    self.fps_counter = 0
                    self.fps_start_time = time.time()
                    processing_times.clear()
                    print("🔄 Counters reset")
                elif key == ord('f'):
                    fullscreen = not fullscreen
                    print(f"🖥️  Fullscreen: {'ON' if fullscreen else 'OFF'}")
                
                self.frame_count += 1
                
        except KeyboardInterrupt:
            print("\n⏹️  Detection interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Print final statistics
            print(f"\n✅ Detection completed!")
            print(f"📊 Total frames processed: {self.frame_count}")
            print(f"📊 Average FPS: {self.fps:.1f}")
            if processing_times:
                print(f"📊 Average processing time: {np.mean(processing_times)*1000:.1f}ms")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Real-time Emotion Detection System")
    parser.add_argument("--model", default="best_emotion_model.pth", 
                       help="Path to trained model file")
    parser.add_argument("--camera", type=int, default=0, 
                       help="Camera device ID")
    parser.add_argument("--save", action="store_true", 
                       help="Enable frame saving")
    
    args = parser.parse_args()
    
    print("🎯 Real-time Emotion Detection System")
    print("=" * 50)
    
    try:
        # Initialize detector
        detector = RealTimeEmotionDetector(args.model)
        
        # Start detection
        detector.run_detection(args.camera, args.save)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
