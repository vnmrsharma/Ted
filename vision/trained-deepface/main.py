#!/usr/bin/env python3
"""
Working Real-time Emotion Analysis using DeepFace
This version ensures emotion detection actually works and displays results
"""

import cv2
import numpy as np
import time
from deepface import DeepFace
from collections import deque
import threading
import queue
import argparse

class WorkingEmotionAnalyzer:
    """Working real-time emotion analyzer"""
    
    def __init__(self, camera_id=0, frame_width=640, frame_height=480):
        self.camera_id = camera_id
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Emotion tracking
        self.current_emotion = None
        self.emotion_confidence = 0.0
        self.emotion_history = deque(maxlen=20)
        
        # Analysis tracking
        self.analysis_attempts = 0
        self.last_analysis_time = 0
        self.is_running = False
        
        # Camera
        self.cap = None
        
        # Analysis state
        self.analyzing = False
        self.last_frame = None
        
        # Colors for emotions
        self.emotion_colors = {
            'angry': (0, 0, 255),      # Red
            'disgust': (0, 255, 0),    # Green
            'fear': (255, 0, 255),     # Magenta
            'happy': (0, 255, 255),    # Yellow
            'sad': (255, 0, 0),        # Blue
            'surprise': (255, 255, 0), # Cyan
            'neutral': (128, 128, 128) # Gray
        }
        
        print("🔧 Initializing DeepFace...")
        self._test_deepface()
    
    def _test_deepface(self):
        """Test DeepFace initialization"""
        try:
            # Test with a simple image
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            _ = DeepFace.analyze(
                test_img, 
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            print("✅ DeepFace initialized successfully")
        except Exception as e:
            print(f"❌ DeepFace initialization failed: {e}")
            raise
    
    def start_camera(self):
        """Start camera capture"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        print(f"📹 Camera {self.camera_id} started")
    
    def _analyze_emotion_sync(self, frame):
        """Analyze emotion synchronously (blocking)"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Analyze with DeepFace
            result = DeepFace.analyze(
                rgb_frame,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            
            if isinstance(result, list) and len(result) > 0:
                emotion_data = result[0]
                if 'emotion' in emotion_data:
                    emotions = emotion_data['emotion']
                    dominant_emotion = max(emotions.items(), key=lambda x: x[1])
                    
                    return {
                        'success': True,
                        'emotion': dominant_emotion[0],
                        'confidence': dominant_emotion[1],
                        'all_emotions': emotions
                    }
            
            return {'success': False, 'error': 'No emotion detected'}
            
        except Exception as e:
            print(f"❌ Analysis error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _draw_emotion_info(self, frame, emotion_data):
        """Draw emotion information on frame"""
        if not emotion_data.get('success', False):
            return frame
        
        emotion = emotion_data['emotion']
        confidence = emotion_data['confidence']
        
        # Draw emotion box
        box_color = self.emotion_colors.get(emotion, (255, 255, 255))
        cv2.rectangle(frame, (10, 10), (350, 120), (0, 0, 0), -1)  # Black background
        cv2.rectangle(frame, (10, 10), (350, 120), box_color, 3)    # Colored border
        
        # Draw emotion text
        cv2.putText(frame, f"EMOTION: {emotion.upper()}", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"CONFIDENCE: {confidence:.1f}%", 
                   (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"ATTEMPTS: {self.analysis_attempts}", 
                   (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def _draw_status_info(self, frame):
        """Draw status information"""
        # Status box
        cv2.rectangle(frame, (10, 10), (400, 100), (0, 0, 0), -1)  # Black background
        cv2.rectangle(frame, (10, 10), (400, 100), (0, 255, 255), 2)  # Cyan border
        
        if self.current_emotion:
            cv2.putText(frame, "EMOTION DETECTED!", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Current: {self.current_emotion.upper()}", 
                       (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        elif self.analyzing:
            cv2.putText(frame, "ANALYZING EMOTIONS...", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"Attempts: {self.analysis_attempts}", 
                       (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "READY TO ANALYZE", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, f"Press SPACE to analyze", 
                       (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Face detection indicator
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            cv2.circle(frame, (frame.shape[1] - 30, 30), 15, (0, 255, 0), -1)
            cv2.putText(frame, "FACE", (frame.shape[1] - 80, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.circle(frame, (frame.shape[1] - 30, 30), 15, (0, 0, 255), -1)
            cv2.putText(frame, "NO FACE", (frame.shape[1] - 90, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    def run(self):
        """Main run loop"""
        try:
            self.start_camera()
            self.is_running = True
            
            print("🚀 Starting emotion analysis...")
            print("   Press 'q' to quit, 's' to save frame, SPACE to analyze")
            
            last_analysis_time = 0
            analysis_interval = 2.0  # Analyze every 2 seconds
            
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                self.last_frame = frame.copy()
                
                current_time = time.time()
                
                # Auto-analyze every few seconds
                if current_time - last_analysis_time > analysis_interval and not self.analyzing:
                    self.analyzing = True
                    self.analysis_attempts += 1
                    print(f"🔍 Analyzing frame (attempt {self.analysis_attempts})...")
                    
                    # Analyze in a separate thread to avoid blocking
                    analysis_thread = threading.Thread(target=self._analyze_frame_async)
                    analysis_thread.daemon = True
                    analysis_thread.start()
                    
                    last_analysis_time = current_time
                
                # Draw information on frame
                if self.current_emotion:
                    frame = self._draw_emotion_info(frame, {
                        'success': True,
                        'emotion': self.current_emotion,
                        'confidence': self.emotion_confidence
                    })
                else:
                    self._draw_status_info(frame)
                
                # Display frame
                cv2.imshow('Real-time Emotion Analysis', frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self._save_frame(frame)
                elif key == ord(' '):  # Spacebar
                    # Manual analysis
                    if not self.analyzing:
                        self.analyzing = True
                        self.analysis_attempts += 1
                        print(f"🔍 Manual analysis (attempt {self.analysis_attempts})...")
                        
                        analysis_thread = threading.Thread(target=self._analyze_frame_async)
                        analysis_thread.daemon = True
                        analysis_thread.start()
                
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            self.cleanup()
    
    def _analyze_frame_async(self):
        """Analyze frame asynchronously"""
        try:
            if self.last_frame is not None:
                result = self._analyze_emotion_sync(self.last_frame)
                
                if result.get('success', False):
                    self.current_emotion = result['emotion']
                    self.emotion_confidence = result['confidence']
                    self.emotion_history.append(self.current_emotion)
                    self.last_analysis_time = time.time()
                    print(f"✅ Emotion detected: {self.current_emotion} ({self.emotion_confidence:.1f}%)")
                else:
                    print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
                
                self.analyzing = False
        except Exception as e:
            print(f"❌ Async analysis error: {e}")
            self.analyzing = False
    
    def _save_frame(self, frame):
        """Save current frame"""
        if self.current_emotion:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"emotion_{timestamp}_{self.current_emotion}.jpg"
            cv2.imwrite(filename, frame)
            print(f"💾 Saved: {filename}")
        else:
            print("⚠️  No emotion detected to save")
    
    def cleanup(self):
        """Clean up resources"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("🧹 Cleanup completed")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Working Real-time Emotion Analysis')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    parser.add_argument('--width', type=int, default=640, help='Frame width')
    parser.add_argument('--height', type=int, default=480, help='Frame height')
    
    args = parser.parse_args()
    
    try:
        analyzer = WorkingEmotionAnalyzer(
            camera_id=args.camera,
            frame_width=args.width,
            frame_height=args.height
        )
        analyzer.run()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
