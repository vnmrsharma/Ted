#!/usr/bin/env python3
"""
YOLO11-Based Face Emotion Detection System
- Uses YOLO11n for person detection and face region extraction
- Advanced emotion classification on detected faces
- Optimized for accuracy with YOLO's superior detection capabilities
"""

import cv2
import numpy as np
import time
import argparse
from collections import deque, Counter
import threading
import queue
import warnings
from ultralytics import YOLO
import torch

# Suppress warnings
warnings.filterwarnings('ignore')

class YOLOEmotionDetector:
    """YOLO11-based emotion detection system with enhanced accuracy"""
    
    def __init__(self, model_path="yolo11n.pt", confidence=0.5):
        self.confidence = confidence
        self.model_path = model_path
        
        # Load YOLO model
        print("🤖 Loading YOLO11n model...")
        self.yolo_model = YOLO(model_path)
        print("✅ YOLO11n model loaded successfully!")
        
        # Check if CUDA is available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🔧 Using device: {self.device}")
        
        # Face detection model (for emotion analysis on YOLO-detected persons)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Performance optimization
        self.frame_skip = 1
        self.frame_counter = 0
        self.last_detections = []
        
        # Emotion labels
        self.emotions = ["Neutral", "Happy", "Sad", "Angry", "Surprise", "Fear", "Disgust"]
        
        # Colors for each emotion (BGR format)
        self.emotion_colors = {
            "Neutral": (128, 128, 128),    # Gray
            "Happy": (0, 255, 0),          # Green
            "Sad": (255, 0, 0),            # Blue
            "Angry": (0, 0, 255),          # Red
            "Surprise": (0, 255, 255),     # Yellow
            "Fear": (128, 0, 128),         # Purple
            "Disgust": (0, 128, 255)       # Orange
        }
        
        # Performance tracking
        self.fps_counter = deque(maxlen=30)
        self.emotion_history = {}
        
        # Detection parameters
        self.min_face_size = 60
        self.max_faces_per_person = 2
        
        print("✅ YOLO emotion detector ready!")
        
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
    
    def _detect_persons_yolo(self, frame):
        """Use YOLO11n to detect persons in the frame"""
        try:
            # Run YOLO inference
            results = self.yolo_model(frame, conf=self.confidence, verbose=False)
            
            persons = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Check if detected object is a person (class 0 in COCO)
                        if int(box.cls) == 0:  # Person class
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            
                            # Filter by confidence
                            if conf >= self.confidence:
                                persons.append({
                                    'bbox': (x1, y1, x2, y2),
                                    'confidence': conf,
                                    'area': (x2 - x1) * (y2 - y1)
                                })
            
            # Sort by confidence and area (larger, more confident detections first)
            persons.sort(key=lambda x: (x['confidence'], x['area']), reverse=True)
            return persons[:5]  # Limit to top 5 detections
            
        except Exception as e:
            print(f"YOLO detection error: {e}")
            return []
    
    def _extract_faces_from_person(self, frame, person_bbox):
        """Extract faces from detected person region"""
        x1, y1, x2, y2 = person_bbox
        
        # Extract person region with some padding
        padding = 20
        person_region = frame[max(0, y1-padding):min(frame.shape[0], y2+padding), 
                             max(0, x1-padding):min(frame.shape[1], x2+padding)]
        
        if person_region.size == 0:
            return []
        
        # Convert to grayscale for face detection
        gray_person = cv2.cvtColor(person_region, cv2.COLOR_BGR2GRAY)
        
        # Detect faces within person region
        faces = self.face_cascade.detectMultiScale(
            gray_person,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(self.min_face_size, self.min_face_size),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        face_detections = []
        for (fx, fy, fw, fh) in faces[:self.max_faces_per_person]:
            # Convert face coordinates back to original frame coordinates
            abs_fx = max(0, x1 - padding) + fx
            abs_fy = max(0, y1 - padding) + fy
            
            # Extract face region from original frame
            face_roi = frame[abs_fy:abs_fy+fh, abs_fx:abs_fx+fw]
            
            if face_roi.size > 0:
                # Classify emotion
                emotion = self._classify_emotion_advanced(face_roi)
                
                # Calculate face quality
                face_confidence = self._calculate_face_confidence(face_roi)
                
                # Smooth emotion
                face_id = f"{abs_fx}_{abs_fy}"
                stable_emotion = self._smooth_emotion(face_id, emotion)
                
                face_detections.append({
                    'bbox': (abs_fx, abs_fy, abs_fx+fw, abs_fy+fh),
                    'emotion': stable_emotion,
                    'confidence': face_confidence,
                    'face_id': face_id,
                    'size': (fw, fh)
                })
        
        return face_detections
    
    def _detect_emotions_yolo(self, frame):
        """Main detection pipeline using YOLO + emotion analysis"""
        # Skip frames for performance
        self.frame_counter += 1
        if self.frame_counter % self.frame_skip != 0:
            return self.last_detections
        
        try:
            # Step 1: Detect persons using YOLO
            persons = self._detect_persons_yolo(frame)
            
            all_faces = []
            
            # Step 2: Extract and analyze faces from each detected person
            for person in persons:
                person_faces = self._extract_faces_from_person(frame, person['bbox'])
                
                # Add person context to face detections
                for face in person_faces:
                    face['person_bbox'] = person['bbox']
                    face['person_confidence'] = person['confidence']
                    all_faces.append(face)
            
            # Cache for skipped frames
            self.last_detections = {
                'faces': all_faces,
                'persons': persons
            }
            
            return self.last_detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            return {'faces': [], 'persons': []}
    
    def _calculate_face_confidence(self, face_roi):
        """Calculate face quality/confidence score"""
        if face_roi.size == 0:
            return 0.0
        
        try:
            # Calculate various quality metrics
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Variance of Laplacian (sharpness)
            laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 500.0, 1.0)
            
            # Brightness consistency
            mean_brightness = np.mean(gray_face)
            brightness_score = 1.0 - abs(mean_brightness - 128) / 128.0
            
            # Size score
            area = face_roi.shape[0] * face_roi.shape[1]
            size_score = min(area / (80 * 80), 1.0)
            
            # Combined confidence
            confidence = (sharpness_score * 0.4 + brightness_score * 0.3 + size_score * 0.3)
            return min(max(confidence, 0.0), 1.0)
            
        except:
            return 0.5
    
    def _classify_emotion_advanced(self, face_roi):
        """Advanced emotion classification using multiple features"""
        if face_roi.size == 0:
            return "Neutral"
        
        try:
            # Convert to grayscale
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            h, w = gray_face.shape
            
            # Facial region analysis
            eye_region = gray_face[:h//2, :]
            mouth_region = gray_face[h//2:, :]
            
            # Feature extraction
            features = {}
            
            # 1. Eye region analysis
            eye_mean = np.mean(eye_region)
            eye_std = np.std(eye_region)
            features['eye_brightness'] = eye_mean
            features['eye_contrast'] = eye_std
            
            # 2. Mouth region analysis
            mouth_mean = np.mean(mouth_region)
            mouth_std = np.std(mouth_region)
            features['mouth_brightness'] = mouth_mean
            features['mouth_contrast'] = mouth_std
            
            # 3. Overall face analysis
            features['face_brightness'] = np.mean(gray_face)
            features['face_contrast'] = np.std(gray_face)
            
            # 4. Brightness ratios
            features['mouth_eye_ratio'] = mouth_mean / (eye_mean + 1e-6)
            
            # 5. Edge detection for facial structure
            edges = cv2.Canny(gray_face, 50, 150)
            features['edge_density'] = np.sum(edges > 0) / edges.size
            
            # 6. Symmetry analysis
            left_half = gray_face[:, :w//2]
            right_half = cv2.flip(gray_face[:, w//2:], 1)
            min_width = min(left_half.shape[1], right_half.shape[1])
            if min_width > 0:
                left_resized = left_half[:, :min_width]
                right_resized = right_half[:, :min_width]
                symmetry = 1.0 - np.mean(np.abs(left_resized.astype(float) - right_resized.astype(float))) / 255.0
                features['symmetry'] = symmetry
            else:
                features['symmetry'] = 0.5
            
            # Enhanced feature: Gradient analysis
            grad_x = cv2.Sobel(gray_face, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_face, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            features['gradient_intensity'] = np.mean(gradient_magnitude)
            
            # Emotion classification based on features
            return self._classify_from_features(features)
            
        except Exception as e:
            return "Neutral"
    
    def _classify_from_features(self, features):
        """Enhanced emotion classification from extracted features"""
        try:
            # Rule-based classification with multiple criteria
            brightness = features.get('face_brightness', 128)
            contrast = features.get('face_contrast', 30)
            mouth_eye_ratio = features.get('mouth_eye_ratio', 1.0)
            edge_density = features.get('edge_density', 0.1)
            symmetry = features.get('symmetry', 0.5)
            gradient_intensity = features.get('gradient_intensity', 50)
            
            # Enhanced Happy detection
            if (mouth_eye_ratio > 1.08 and brightness > 115 and 
                edge_density < 0.12 and symmetry > 0.45 and gradient_intensity < 60):
                return "Happy"
            
            # Enhanced Sad detection
            elif (mouth_eye_ratio < 0.92 and brightness < 105 and 
                  contrast < 32 and gradient_intensity < 45):
                return "Sad"
            
            # Enhanced Angry detection
            elif (contrast > 48 and edge_density > 0.22 and 
                  brightness < 125 and symmetry < 0.55 and gradient_intensity > 70):
                return "Angry"
            
            # Enhanced Surprise detection
            elif (edge_density > 0.28 and contrast > 42 and 
                  mouth_eye_ratio > 1.15 and gradient_intensity > 65):
                return "Surprise"
            
            # Enhanced Fear detection
            elif (contrast > 52 and brightness < 95 and 
                  edge_density > 0.32 and symmetry < 0.45):
                return "Fear"
            
            # Enhanced Disgust detection
            elif (mouth_eye_ratio < 0.88 and contrast > 38 and 
                  brightness > 95 and brightness < 135 and edge_density > 0.18):
                return "Disgust"
            
            # Default to neutral
            else:
                return "Neutral"
                
        except:
            return "Neutral"
    
    def _smooth_emotion(self, face_id, emotion):
        """Smooth emotion detection per face with longer history"""
        if face_id not in self.emotion_history:
            self.emotion_history[face_id] = deque(maxlen=6)  # Longer history for stability
        
        self.emotion_history[face_id].append(emotion)
        
        # Use majority voting with minimum samples
        if len(self.emotion_history[face_id]) >= 4:
            emotion_counts = Counter(self.emotion_history[face_id])
            most_common = emotion_counts.most_common(1)[0][0]
            return most_common
        
        return emotion
    
    def _draw_yolo_ui(self, frame, detections):
        """Enhanced UI drawing for YOLO-based detection"""
        overlay = frame.copy()
        
        faces = detections.get('faces', [])
        persons = detections.get('persons', [])
        
        # Count emotions
        emotion_counts = {}
        total_confidence = 0
        for face in faces:
            emotion = face['emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            total_confidence += face['confidence']
        
        avg_confidence = total_confidence / len(faces) if faces else 0
        
        # Enhanced performance panel
        panel_height = 110
        cv2.rectangle(overlay, (10, 10), (600, panel_height), (20, 20, 20), -1)
        cv2.rectangle(overlay, (10, 10), (600, panel_height), (0, 255, 255), 3)
        
        # Title with YOLO branding
        cv2.putText(overlay, "YOLO11n + Face Emotion Detection", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Performance metrics
        if self.fps_counter:
            fps = 1.0 / np.mean(self.fps_counter) if np.mean(self.fps_counter) > 0 else 0
            perf_text = f"FPS: {fps:.1f} | Persons: {len(persons)} | Faces: {len(faces)} | Device: {self.device}"
            cv2.putText(overlay, perf_text, (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Detection confidence
        conf_text = f"Avg Face Conf: {avg_confidence:.2f} | YOLO Conf: {self.confidence}"
        cv2.putText(overlay, conf_text, (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Emotion summary
        emotions_text = " | ".join([f"{e}: {c}" for e, c in emotion_counts.items()])
        if emotions_text:
            cv2.putText(overlay, emotions_text, (20, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        # Draw person bounding boxes (lighter)
        for person in persons:
            x1, y1, x2, y2 = person['bbox']
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (100, 100, 255), 2)
            cv2.putText(overlay, f"Person {person['confidence']:.2f}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
        
        # Draw face detections with emotions
        for face in faces:
            x1, y1, x2, y2 = face['bbox']
            emotion = face['emotion']
            confidence = face['confidence']
            face_size = face['size']
            
            # Get color
            color = self.emotion_colors.get(emotion, (255, 255, 255))
            
            # Draw bounding box (thickness based on confidence)
            thickness = int(3 + confidence * 4)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
            
            # Draw emotion label with enhanced styling
            label = f"{emotion} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            
            # Label background with rounded effect
            cv2.rectangle(overlay, (x1, y1 - 35), 
                         (x1 + label_size[0] + 15, y1), color, -1)
            cv2.rectangle(overlay, (x1, y1 - 35), 
                         (x1 + label_size[0] + 15, y1), (255, 255, 255), 2)
            
            # Label text
            cv2.putText(overlay, label, (x1 + 8, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Enhanced confidence indicator
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            radius = int(4 + confidence * 8)
            cv2.circle(overlay, (center_x, center_y), radius, color, -1)
            cv2.circle(overlay, (center_x, center_y), radius, (255, 255, 255), 2)
            
            # Face size and person link indicator
            cv2.putText(overlay, f"{face_size[0]}x{face_size[1]}", 
                       (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Blend with higher alpha for better visibility
        alpha = 0.88
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    def run_detection(self, source=0, window_size=(1280, 720)):
        """Run YOLO-based emotion detection"""
        print("🚀 Starting YOLO11n Face Emotion Detection...")
        print("=" * 80)
        print("🤖 YOLO11n + Advanced Emotion Classification")
        print("🎯 Features: YOLO person detection, Haar face detection, custom emotion analysis")
        print(f"🔧 Device: {self.device}")
        print("Controls:")
        print("  SPACE - Screenshot")
        print("  R - Reset emotion history")
        print("  + - Increase confidence threshold")
        print("  - - Decrease confidence threshold")
        print("  S - Toggle frame skip")
        print("  Q - Quit")
        print("=" * 80)
        
        # Initialize camera
        cap, camera_id = self._find_working_camera()
        if cap is None:
            print("❌ No working camera found!")
            return False
        
        # Optimize camera for performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, window_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, window_size[1])
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Setup window
        window_name = "YOLO11n Face Emotion Detection"
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
                
                # Detect using YOLO + emotion analysis
                detections = self._detect_emotions_yolo(frame)
                
                # Draw enhanced UI
                annotated_frame = self._draw_yolo_ui(frame, detections)
                
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
                    screenshot_name = f"yolo_emotion_detection_{int(time.time())}.jpg"
                    cv2.imwrite(screenshot_name, annotated_frame)
                    print(f"📸 Screenshot: {screenshot_name}")
                elif key == ord('r'):
                    self.emotion_history.clear()
                    print("🔄 Emotion history reset")
                elif key == ord('+') or key == ord('='):
                    self.confidence = min(0.9, self.confidence + 0.05)
                    print(f"📈 YOLO confidence: {self.confidence:.2f}")
                elif key == ord('-'):
                    self.confidence = max(0.1, self.confidence - 0.05)
                    print(f"📉 YOLO confidence: {self.confidence:.2f}")
                elif key == ord('s'):
                    self.frame_skip = 2 if self.frame_skip == 1 else 1
                    print(f"⚡ Frame skip: {self.frame_skip}")
                
                frame_count += 1
                
                # Stats every 60 frames
                if frame_count % 60 == 0:
                    elapsed = time.time() - start_time
                    avg_fps = frame_count / elapsed
                    faces_count = len(detections.get('faces', []))
                    persons_count = len(detections.get('persons', []))
                    print(f"📊 Frames: {frame_count} | FPS: {avg_fps:.1f} | Persons: {persons_count} | Faces: {faces_count}")
        
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            total_time = time.time() - start_time
            print(f"\n📈 YOLO Performance Report:")
            print(f"   Total frames: {frame_count}")
            print(f"   Runtime: {total_time:.1f}s")
            print(f"   Average FPS: {frame_count/total_time:.1f}")
            print(f"   Device used: {self.device}")
            print("✅ YOLO-based detection completed!")
        
        return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="YOLO11n Face Emotion Detection")
    parser.add_argument("--source", default=0, help="Video source")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.5, help="YOLO confidence threshold")
    parser.add_argument("--size", nargs=2, type=int, default=[1280, 720], help="Window size")
    parser.add_argument("--skip", type=int, default=1, help="Frame skip")
    
    args = parser.parse_args()
    
    print("🤖 YOLO11n Face Emotion Detection System")
    print("=" * 60)
    print(f"🎯 Model: {args.model}")
    print(f"🎯 Resolution: {args.size[0]}x{args.size[1]}")
    print(f"🔥 YOLO Confidence: {args.conf}")
    print(f"⚡ Frame skip: {args.skip}")
    print("=" * 60)
    
    detector = YOLOEmotionDetector(model_path=args.model, confidence=args.conf)
    detector.frame_skip = args.skip
    
    success = detector.run_detection(
        source=args.source,
        window_size=tuple(args.size)
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
