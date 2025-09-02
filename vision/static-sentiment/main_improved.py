#!/usr/bin/env python3
"""
Improved High-Performance Face Emotion Detection System
- Uses DeepFace for accurate emotion classification
- Optimized OpenCV for fast face detection
- Multi-threaded processing for smooth performance
- Advanced face tracking and emotion smoothing
"""

import cv2
import numpy as np
import time
import argparse
from collections import deque, Counter
import threading
import queue
import warnings
from deepface import DeepFace
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

class ImprovedEmotionDetector:
    """High-performance emotion detection system with DeepFace integration"""
    
    def __init__(self, confidence=0.7, max_faces=5):
        self.confidence = confidence
        self.max_faces = max_faces
        
        # Performance optimization
        self.frame_skip = 1
        self.frame_counter = 0
        self.last_detections = []
        
        # Face detection
        self.face_cascade = self._load_face_detector()
        
        # Emotion tracking
        self.emotion_history = {}
        self.face_tracker = {}
        self.emotion_cache = {}
        
        # Emotion labels (DeepFace compatible)
        self.emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
        
        # Colors for each emotion (BGR format)
        self.emotion_colors = {
            "angry": (0, 0, 255),      # Red
            "disgust": (0, 255, 0),    # Green
            "fear": (255, 0, 255),     # Magenta
            "happy": (0, 255, 255),    # Yellow
            "sad": (255, 0, 0),        # Blue
            "surprise": (255, 255, 0), # Cyan
            "neutral": (128, 128, 128) # Gray
        }
        
        # Performance tracking
        self.fps_counter = deque(maxlen=30)
        self.analysis_times = deque(maxlen=30)
        
        # Face filtering parameters
        self.min_face_size = 80
        self.face_confidence_threshold = 0.3
        
        # Multi-threading
        self.analysis_queue = queue.Queue(maxsize=3)
        self.result_queue = queue.Queue(maxsize=3)
        self.analysis_thread = None
        self.is_running = False
        
        # Initialize DeepFace
        self._initialize_deepface()
        
        print("✅ Improved emotion detector ready with DeepFace integration!")
        
    def _initialize_deepface(self):
        """Initialize DeepFace models"""
        print("🔧 Initializing DeepFace models...")
        try:
            # Test with a dummy image to ensure models are loaded
            dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
            _ = DeepFace.analyze(
                dummy_img, 
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            print("✅ DeepFace models initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing DeepFace: {e}")
            raise
    
    def _load_face_detector(self):
        """Load optimized face detector"""
        try:
            # Use Haar cascade for fast face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if face_cascade.empty():
                raise RuntimeError("Failed to load Haar cascade")
            
            print("🎯 Using optimized Haar cascade face detector")
            return face_cascade
            
        except Exception as e:
            print(f"⚠️  Face detector error: {e}")
            return None
    
    def _find_working_camera(self):
        """Find the first working camera"""
        for i in range(4):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"📷 Camera {i} detected")
                        return cap, i
                cap.release()
            except:
                continue
        return None, None
    
    def start_analysis_thread(self):
        """Start emotion analysis thread"""
        self.analysis_thread = threading.Thread(target=self._analysis_worker, daemon=True)
        self.analysis_thread.start()
        self.is_running = True
        print("🧠 Emotion analysis thread started")
    
    def _analysis_worker(self):
        """Worker thread for emotion analysis"""
        while self.is_running:
            try:
                if not self.analysis_queue.empty():
                    analysis_data = self.analysis_queue.get(timeout=0.1)
                    face_id, face_roi = analysis_data
                    
                    # Analyze emotion using DeepFace
                    emotion_result = self._analyze_emotion_deepface(face_roi)
                    
                    # Put result in queue
                    if not self.result_queue.full():
                        self.result_queue.put((face_id, emotion_result))
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Analysis error: {e}")
                continue
        
        print("🧠 Analysis thread stopped")
    
    def _analyze_emotion_deepface(self, face_roi):
        """Analyze emotion using DeepFace"""
        try:
            # Convert BGR to RGB for DeepFace
            rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            
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
                    
                    # Find dominant emotion
                    dominant_emotion = max(emotions.items(), key=lambda x: x[1])
                    
                    return {
                        'emotion': dominant_emotion[0],
                        'confidence': dominant_emotion[1] / 100.0,  # Convert to 0-1 range
                        'all_emotions': emotions
                    }
            
            return {'emotion': 'neutral', 'confidence': 0.5, 'all_emotions': {}}
            
        except Exception as e:
            return {'emotion': 'neutral', 'confidence': 0.5, 'all_emotions': {}}
    
    def _detect_faces_improved(self, frame):
        """Improved face detection with tracking"""
        # Skip frames for performance
        self.frame_counter += 1
        if self.frame_counter % self.frame_skip != 0:
            return self.last_detections
        
        faces = []
        
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            if self.face_cascade is not None:
                face_rects = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(self.min_face_size, self.min_face_size),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                # Process detected faces
                for (x, y, w, h) in face_rects[:self.max_faces]:
                    # Filter out very small faces
                    if w < self.min_face_size or h < self.min_face_size:
                        continue
                    
                    # Extract face region
                    face_roi = frame[y:y+h, x:x+w]
                    
                    # Generate face ID based on position and size
                    face_id = f"{x//20}_{y//20}_{w//20}_{h//20}"
                    
                    # Calculate face quality
                    face_confidence = self._calculate_face_quality(face_roi)
                    
                    # Skip low-quality faces
                    if face_confidence < self.face_confidence_threshold:
                        continue
                    
                    # Add to analysis queue for emotion detection
                    if not self.analysis_queue.full():
                        self.analysis_queue.put((face_id, face_roi))
                    
                    # Get cached emotion result
                    emotion_result = self.emotion_cache.get(face_id, {
                        'emotion': 'neutral',
                        'confidence': 0.5
                    })
                    
                    # Smooth emotion detection
                    stable_emotion = self._smooth_emotion(face_id, emotion_result['emotion'])
                    
                    faces.append({
                        'bbox': (x, y, x+w, y+h),
                        'emotion': stable_emotion,
                        'confidence': emotion_result['confidence'],
                        'face_id': face_id,
                        'size': (w, h),
                        'quality': face_confidence
                    })
            
            # Cache for skipped frames
            self.last_detections = faces
            
        except Exception as e:
            print(f"Face detection error: {e}")
            faces = []
        
        return faces
    
    def _calculate_face_quality(self, face_roi):
        """Calculate face quality score"""
        if face_roi.size == 0:
            return 0.0
        
        try:
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 500.0, 1.0)
            
            # Brightness consistency
            mean_brightness = np.mean(gray_face)
            brightness_score = 1.0 - abs(mean_brightness - 128) / 128.0
            
            # Size score
            area = face_roi.shape[0] * face_roi.shape[1]
            size_score = min(area / (120 * 120), 1.0)
            
            # Combined quality score
            quality = (sharpness_score * 0.4 + brightness_score * 0.3 + size_score * 0.3)
            return min(max(quality, 0.0), 1.0)
            
        except:
            return 0.5
    
    def _smooth_emotion(self, face_id, emotion):
        """Smooth emotion detection per face"""
        if face_id not in self.emotion_history:
            self.emotion_history[face_id] = deque(maxlen=5)
        
        self.emotion_history[face_id].append(emotion)
        
        # Use majority voting with minimum history
        if len(self.emotion_history[face_id]) >= 3:
            emotion_counts = Counter(self.emotion_history[face_id])
            most_common = emotion_counts.most_common(1)[0][0]
            return most_common
        
        return emotion
    
    def _process_analysis_results(self):
        """Process results from analysis thread"""
        while not self.result_queue.empty():
            try:
                face_id, emotion_result = self.result_queue.get_nowait()
                self.emotion_cache[face_id] = emotion_result
            except queue.Empty:
                break
    
    def _draw_improved_ui(self, frame, detections):
        """Draw improved UI with emotion information"""
        overlay = frame.copy()
        
        # Count emotions
        emotion_counts = Counter([d['emotion'] for d in detections])
        total_confidence = sum(d['confidence'] for d in detections)
        avg_confidence = total_confidence / len(detections) if detections else 0
        
        # Performance panel
        panel_height = 120
        cv2.rectangle(overlay, (10, 10), (600, panel_height), (0, 0, 0), -1)
        cv2.rectangle(overlay, (10, 10), (600, panel_height), (0, 255, 0), 2)
        
        # Title
        cv2.putText(overlay, "Improved Face Emotion Detection (DeepFace)", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Performance metrics
        if self.fps_counter:
            fps = 1.0 / np.mean(self.fps_counter) if np.mean(self.fps_counter) > 0 else 0
            perf_text = f"FPS: {fps:.1f} | Faces: {len(detections)} | Avg Conf: {avg_confidence:.2f}"
            cv2.putText(overlay, perf_text, (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Emotion summary
        emotions_text = " | ".join([f"{e}: {c}" for e, c in emotion_counts.items()])
        if emotions_text:
            cv2.putText(overlay, emotions_text, (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Controls info
        cv2.putText(overlay, "SPACE: Screenshot | R: Reset | +/-: Frame Skip | Q: Quit", (20, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Draw face detections
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            emotion = detection['emotion']
            confidence = detection['confidence']
            face_size = detection['size']
            quality = detection['quality']
            
            # Get color
            color = self.emotion_colors.get(emotion, (255, 255, 255))
            
            # Draw bounding box (thickness based on confidence)
            thickness = int(2 + confidence * 3)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
            
            # Draw emotion label
            label = f"{emotion.upper()} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Label background
            cv2.rectangle(overlay, (x1, y1 - 30), 
                         (x1 + label_size[0] + 10, y1), color, -1)
            
            # Label text
            cv2.putText(overlay, label, (x1 + 5, y1 - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Quality indicator
            quality_text = f"Q:{quality:.2f}"
            cv2.putText(overlay, quality_text, (x1, y2 + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Face size
            size_text = f"{face_size[0]}x{face_size[1]}"
            cv2.putText(overlay, size_text, (x1, y2 + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Blend overlay
        alpha = 0.85
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    def run_detection(self, source=0, window_size=(1280, 720)):
        """Run improved emotion detection"""
        print("🚀 Starting Improved Face Emotion Detection...")
        print("=" * 70)
        print("🎯 Features: DeepFace integration, multi-threading, face tracking")
        print("Controls:")
        print("  SPACE - Screenshot")
        print("  R - Reset emotion history")
        print("  + - Increase frame skip (faster, less accurate)")
        print("  - - Decrease frame skip (slower, more accurate)")
        print("  Q - Quit")
        print("=" * 70)
        
        # Initialize camera
        cap, camera_id = self._find_working_camera()
        if cap is None:
            print("❌ No working camera found!")
            return False
        
        # Optimize camera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, window_size[0])
        cap.set(cv2.CAP_PROP_PROP_FRAME_HEIGHT, window_size[1])
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Start analysis thread
        self.start_analysis_thread()
        
        # Setup window
        window_name = "Improved Face Emotion Detection (DeepFace)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, window_size[0], window_size[1])
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                frame_start = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Process analysis results
                self._process_analysis_results()
                
                # Detect faces and emotions
                detections = self._detect_faces_improved(frame)
                
                # Draw UI
                annotated_frame = self._draw_improved_ui(frame, detections)
                
                # Calculate FPS
                frame_time = time.time() - frame_start
                self.fps_counter.append(frame_time)
                
                # Display
                cv2.imshow(window_name, annotated_frame)
                
                # Handle input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:
                    break
                elif key == ord(' '):
                    screenshot_name = f"improved_emotion_{int(time.time())}.jpg"
                    cv2.imwrite(screenshot_name, annotated_frame)
                    print(f"📸 Screenshot: {screenshot_name}")
                elif key == ord('r'):
                    self.emotion_history.clear()
                    self.emotion_cache.clear()
                    print("🔄 Emotion history and cache reset")
                elif key == ord('+') or key == ord('='):
                    self.frame_skip = min(5, self.frame_skip + 1)
                    print(f"⚡ Frame skip: {self.frame_skip}")
                elif key == ord('-'):
                    self.frame_skip = max(1, self.frame_skip - 1)
                    print(f"🎯 Frame skip: {self.frame_skip}")
                
                frame_count += 1
                
                # Stats every 60 frames
                if frame_count % 60 == 0:
                    elapsed = time.time() - start_time
                    avg_fps = frame_count / elapsed
                    print(f"📊 Frames: {frame_count} | FPS: {avg_fps:.1f} | Faces: {len(detections)}")
        
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user")
        finally:
            self.is_running = False
            cap.release()
            cv2.destroyAllWindows()
            
            total_time = time.time() - start_time
            print(f"\n📈 Performance Report:")
            print(f"   Total frames: {frame_count}")
            print(f"   Runtime: {total_time:.1f}s")
            print(f"   Average FPS: {frame_count/total_time:.1f}")
            print("✅ Improved detection completed!")
        
        return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Improved Face Emotion Detection with DeepFace")
    parser.add_argument("--source", default=0, help="Video source")
    parser.add_argument("--conf", type=float, default=0.7, help="Confidence threshold")
    parser.add_argument("--size", nargs=2, type=int, default=[1280, 720], help="Window size")
    parser.add_argument("--skip", type=int, default=1, help="Frame skip")
    parser.add_argument("--max-faces", type=int, default=5, help="Maximum faces to detect")
    
    args = parser.parse_args()
    
    print("🚀 Improved Face Emotion Detection System with DeepFace")
    print("=" * 60)
    print(f"🎯 Resolution: {args.size[0]}x{args.size[1]}")
    print(f"🔥 Confidence: {args.conf}")
    print(f"⚡ Frame skip: {args.skip}")
    print(f"👥 Max faces: {args.max_faces}")
    print("=" * 60)
    
    detector = ImprovedEmotionDetector(
        confidence=args.conf,
        max_faces=args.max_faces
    )
    detector.frame_skip = args.skip
    
    success = detector.run_detection(
        source=args.source,
        window_size=tuple(args.size)
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
