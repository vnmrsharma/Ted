#!/usr/bin/env python3
"""
Gradio Interface for Integrated Face Recognition + Emotion Detection System
Combines image upload for face learning with real-time webcam analysis
"""

import gradio as gr
import cv2
import numpy as np
import time
import os
import json
import threading
import queue
from deepface import DeepFace
from collections import deque
import tempfile
from PIL import Image, ImageDraw, ImageFont
import base64

class GradioIntegratedSystem:
    """Gradio interface for the integrated face recognition + emotion system"""
    
    def __init__(self):
        # Face recognition
        self.known_faces = {}  # name -> embedding
        self.face_db_file = "known_faces.json"
        self.recognition_threshold = 0.6
        
        # Emotion detection
        self.current_emotion = None
        self.emotion_confidence = 0.0
        self.emotion_history = deque(maxlen=20)
        
        # Webcam processing
        self.is_webcam_running = False
        self.webcam_thread = None
        self.frame_queue = queue.Queue(maxsize=10)
        self.latest_results = {
            'recognized_name': None,
            'confidence': 0.0,
            'emotion': None,
            'emotion_conf': 0.0,
            'frame': None
        }
        
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
        
        # Load existing known faces
        self.load_known_faces()
        
        print("🔧 Initializing Gradio Integrated Face Recognition + Emotion System...")
        self._test_deepface()
    
    def _test_deepface(self):
        """Test DeepFace initialization"""
        try:
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
    
    def load_known_faces(self):
        """Load previously saved face embeddings"""
        if os.path.exists(self.face_db_file):
            try:
                with open(self.face_db_file, 'r') as f:
                    data = json.load(f)
                    for name, face_data in data.items():
                        self.known_faces[name] = np.array(face_data['embedding'])
                    print(f"✅ Loaded {len(self.known_faces)} known faces")
            except Exception as e:
                print(f"⚠️ Could not load saved faces: {e}")
    
    def save_known_faces(self):
        """Save face embeddings to file"""
        try:
            data = {}
            for name, embedding in self.known_faces.items():
                data[name] = {
                    'embedding': embedding.tolist(),
                    'timestamp': time.time()
                }
            
            with open(self.face_db_file, 'w') as f:
                json.dump(data, f)
            print(f"💾 Saved {len(self.known_faces)} face embeddings")
            return True
        except Exception as e:
            print(f"❌ Failed to save faces: {e}")
            return False
    
    def extract_face_embedding(self, image):
        """Extract face embedding from image"""
        try:
            # Convert PIL image to numpy array if needed
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Ensure image is in RGB format
            if len(image.shape) == 3 and image.shape[2] == 3:
                if image.dtype == np.uint8:
                    # Convert BGR to RGB if needed
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Extract face embedding
            embedding = DeepFace.represent(
                image,
                model_name="VGG-Face",
                detector_backend="opencv",
                enforce_detection=False
            )
            
            if embedding and len(embedding) > 0:
                return np.array(embedding[0]['embedding'])
            else:
                return None
                
        except Exception as e:
            print(f"❌ Error extracting face embedding: {e}")
            return None
    
    def learn_face_from_image(self, image, name):
        """Learn a new face from uploaded image"""
        if not name or name.strip() == "":
            return "❌ Please enter a name for the person", None
        
        if image is None:
            return "❌ Please upload an image", None
        
        try:
            # Extract face embedding
            embedding = self.extract_face_embedding(image)
            if embedding is None:
                return "❌ No face detected in the image", None
            
            # Save the face
            self.known_faces[name] = embedding
            self.save_known_faces()
            
            return f"✅ Successfully learned face for {name}", self.get_known_faces_list()
            
        except Exception as e:
            return f"❌ Error learning face: {e}", None
    
    def get_known_faces_list(self):
        """Get list of known faces for display"""
        if not self.known_faces:
            return "No faces learned yet"
        
        faces_list = []
        for name in self.known_faces.keys():
            faces_list.append(f"👤 {name}")
        
        return "\n".join(faces_list)
    
    def recognize_face(self, frame):
        """Recognize faces in the frame"""
        if not self.known_faces:
            return None, 0.0
        
        try:
            # Extract face embedding from current frame
            embedding = self.extract_face_embedding(frame)
            if embedding is None:
                return None, 0.0
            
            # Compare with known faces
            best_match = None
            best_distance = float('inf')
            
            for name, known_embedding in self.known_faces.items():
                distance = np.linalg.norm(embedding - known_embedding)
                if distance < best_distance:
                    best_distance = distance
                    best_match = name
            
            # Convert distance to similarity score (0-1)
            similarity = 1.0 / (1.0 + best_distance)
            
            if similarity > self.recognition_threshold:
                return best_match, similarity
            
            return None, similarity
            
        except Exception as e:
            print(f"❌ Error in face recognition: {e}")
            return None, 0.0
    
    def analyze_emotion(self, frame):
        """Analyze emotion in the frame"""
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
                    
                    self.current_emotion = dominant_emotion[0]
                    self.emotion_confidence = dominant_emotion[1] / 100.0
                    
                    # Add to history
                    self.emotion_history.append({
                        'emotion': self.current_emotion,
                        'confidence': self.emotion_confidence,
                        'timestamp': time.time()
                    })
                    
                    return self.current_emotion, self.emotion_confidence
            
            return None, 0.0
            
        except Exception as e:
            print(f"❌ Error in emotion analysis: {e}")
            return None, 0.0
    
    def process_webcam_frame(self, frame):
        """Process a single webcam frame"""
        try:
            # Face recognition
            recognized_name, confidence = self.recognize_face(frame)
            
            # Emotion detection
            emotion, emotion_conf = self.analyze_emotion(frame)
            
            # Store results
            self.latest_results = {
                'recognized_name': recognized_name,
                'confidence': confidence,
                'emotion': emotion,
                'emotion_conf': emotion_conf,
                'frame': frame
            }
            
        except Exception as e:
            print(f"❌ Error processing frame: {e}")
    
    def webcam_thread_func(self):
        """Webcam processing thread"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Failed to open camera")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        
        print("📹 Webcam thread started")
        
        while self.is_webcam_running:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Process frame
            self.process_webcam_frame(frame)
            
            # Small delay to prevent overwhelming
            time.sleep(0.1)
        
        cap.release()
        print("📹 Webcam thread stopped")
    
    def start_webcam(self):
        """Start webcam processing"""
        if self.is_webcam_running:
            return "⚠️ Webcam is already running"
        
        self.is_webcam_running = True
        self.webcam_thread = threading.Thread(target=self.webcam_thread_func)
        self.webcam_thread.start()
        
        return "🚀 Started webcam recognition and emotion detection"
    
    def stop_webcam(self):
        """Stop webcam processing"""
        if not self.is_webcam_running:
            return "⚠️ Webcam is not running"
        
        self.is_webcam_running = False
        if self.webcam_thread:
            self.webcam_thread.join(timeout=2)
        
        return "🛑 Stopped webcam processing"
    
    def get_webcam_results(self):
        """Get current webcam results for display"""
        if not self.is_webcam_running:
            return "Webcam not running", None
        
        results = self.latest_results
        
        # Create status text
        if results['recognized_name']:
            status = f"✅ Recognized: {results['recognized_name']} (Conf: {results['confidence']:.2f})"
        else:
            status = f"❓ Unknown person (Conf: {results['confidence']:.2f})"
        
        if results['emotion']:
            emotion_color = self.emotion_colors.get(results['emotion'], (128, 128, 128))
            status += f"\n🎭 Emotion: {results['emotion'].title()} (Conf: {results['emotion_conf']:.2f})"
        else:
            status += "\n🎭 Emotion: Analyzing..."
        
        # Convert frame to PIL image for Gradio
        if results['frame'] is not None:
            # Draw results on frame
            frame_with_results = self.draw_results_on_frame(results['frame'], results)
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame_with_results, cv2.COLOR_BGR2RGB)
            # Convert to PIL
            pil_image = Image.fromarray(rgb_frame)
        else:
            pil_image = None
        
        return status, pil_image
    
    def draw_results_on_frame(self, frame, results):
        """Draw recognition and emotion results on frame"""
        height, width = frame.shape[:2]
        
        # Draw face recognition info
        if results['recognized_name']:
            # Green box for recognized face
            cv2.rectangle(frame, (10, 10), (width-10, height-10), (0, 255, 0), 3)
            
            # Recognition text
            recognition_text = f"Person: {results['recognized_name']} ({results['confidence']:.2f})"
            cv2.putText(frame, recognition_text, (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            # Red box for unknown face
            cv2.rectangle(frame, (10, 10), (width-10, height-10), (0, 0, 255), 3)
            cv2.putText(frame, "Unknown Person", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Draw emotion info
        if results['emotion']:
            emotion_color = self.emotion_colors.get(results['emotion'], (255, 255, 255))
            emotion_text = f"Emotion: {results['emotion'].title()} ({results['emotion_conf']:.2f})"
            cv2.putText(frame, emotion_text, (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, emotion_color, 2)
        else:
            emotion_text = "Emotion: Analyzing..."
            cv2.putText(frame, emotion_text, (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
        
        return frame
    
    def delete_face(self, name):
        """Delete a learned face"""
        if name in self.known_faces:
            del self.known_faces[name]
            self.save_known_faces()
            return f"🗑️ Deleted face: {name}", self.get_known_faces_list()
        else:
            return f"❌ Face not found: {name}", self.get_known_faces_list()

def create_gradio_interface():
    """Create the Gradio interface"""
    system = GradioIntegratedSystem()
    
    with gr.Blocks(title="Integrated Face Recognition + Emotion Detection", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🎭 Integrated Face Recognition + Emotion Detection")
        gr.Markdown("Learn faces from images and perform real-time recognition with emotion analysis")
        
        with gr.Tabs():
            # Tab 1: Learn Face
            with gr.Tab("📚 Learn Face"):
                gr.Markdown("## Upload an image to teach the system a new face")
                
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(label="Upload Face Image", type="pil")
                        name_input = gr.Textbox(label="Person's Name", placeholder="Enter the person's name")
                        
                        with gr.Row():
                            learn_btn = gr.Button("🎓 Learn Face", variant="primary")
                            clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                        
                        learn_output = gr.Textbox(label="Learning Result", interactive=False)
                    
                    with gr.Column():
                        gr.Markdown("### Currently Learned Faces")
                        known_faces_display = gr.Textbox(
                            value=system.get_known_faces_list(),
                            label="Known Faces",
                            interactive=False,
                            lines=10
                        )
                        
                        with gr.Row():
                            delete_name_input = gr.Textbox(label="Name to Delete", placeholder="Enter name to remove")
                            delete_btn = gr.Button("🗑️ Delete Face", variant="stop")
                        
                        delete_output = gr.Textbox(label="Delete Result", interactive=False)
            
            # Tab 2: Real-time Recognition
            with gr.Tab("📹 Real-time Recognition"):
                gr.Markdown("## Live webcam face recognition and emotion detection")
                
                with gr.Row():
                    with gr.Column():
                        webcam_status = gr.Textbox(
                            value="Webcam not started",
                            label="Webcam Status",
                            interactive=False
                        )
                        
                        with gr.Row():
                            start_webcam_btn = gr.Button("🚀 Start Webcam", variant="primary")
                            stop_webcam_btn = gr.Button("🛑 Stop Webcam", variant="stop")
                        
                        refresh_btn = gr.Button("🔄 Refresh Results", variant="secondary")
                    
                    with gr.Column():
                        gr.Markdown("### Live Results")
                        results_display = gr.Textbox(
                            label="Recognition Results",
                            interactive=False,
                            lines=5
                        )
                
                gr.Markdown("### Live Webcam Feed")
                webcam_output = gr.Image(label="Webcam Feed", interactive=False)
                
                gr.Markdown("""
                **Instructions:**
                1. Click "Start Webcam" to begin recognition
                2. Position your face in front of the camera
                3. The system will recognize known faces and detect emotions
                4. Click "Refresh Results" to see current status
                5. Click "Stop Webcam" when done
                """)
        
        # Event handlers
        learn_btn.click(
            fn=system.learn_face_from_image,
            inputs=[image_input, name_input],
            outputs=[learn_output, known_faces_display]
        )
        
        clear_btn.click(
            fn=lambda: (None, ""),
            outputs=[image_input, name_input]
        )
        
        delete_btn.click(
            fn=system.delete_face,
            inputs=[delete_name_input],
            outputs=[delete_output, known_faces_display]
        )
        
        start_webcam_btn.click(
            fn=system.start_webcam,
            outputs=[webcam_status]
        )
        
        stop_webcam_btn.click(
            fn=system.stop_webcam,
            outputs=[webcam_status]
        )
        
        refresh_btn.click(
            fn=system.get_webcam_results,
            outputs=[results_display, webcam_output]
        )
        
        # Auto-refresh webcam results
        interface.load(
            fn=lambda: ("Webcam not started", None),
            outputs=[results_display, webcam_output]
        )
    
    return interface

if __name__ == "__main__":
    # Create and launch the interface
    interface = create_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    )
