#!/usr/bin/env python3
"""
Simple Face Recognition System using DeepFace with Gradio Interface
Remembers one person and identifies them in real-time webcam feed
"""

import gradio as gr
import cv2
import numpy as np
import time
import os
import json
from deepface import DeepFace
import threading
import queue
from PIL import Image, ImageDraw, ImageFont
import tempfile

class FaceRecognitionSystem:
    """Face recognition system with DeepFace"""
    
    def __init__(self):
        self.known_face_embedding = None
        self.known_name = None
        self.face_db_file = "known_face.json"
        self.recognition_threshold = 0.6
        self.is_recording = False
        self.cap = None
        self.recognition_thread = None
        self.stop_recognition = False
        self.current_frame = None
        self.frame_queue = queue.Queue(maxsize=10)
        self.latest_frame = None
        
        # Load existing known face if available
        self.load_known_face()
    
    def load_known_face(self):
        """Load previously saved face embedding"""
        if os.path.exists(self.face_db_file):
            try:
                with open(self.face_db_file, 'r') as f:
                    data = json.load(f)
                    self.known_face_embedding = np.array(data['embedding'])
                    self.known_name = data['name']
                    print(f"✅ Loaded known face: {self.known_name}")
            except Exception as e:
                print(f"⚠️ Could not load saved face: {e}")
    
    def save_known_face(self, embedding, name):
        """Save face embedding to file"""
        try:
            data = {
                'name': name,
                'embedding': embedding.tolist(),
                'timestamp': time.time()
            }
            with open(self.face_db_file, 'w') as f:
                json.dump(data, f)
            print(f"💾 Saved face embedding for {name}")
            return True
        except Exception as e:
            print(f"❌ Failed to save face: {e}")
            return False
    
    def extract_face_embedding(self, image):
        """Extract face embedding from image"""
        try:
            # Convert PIL image to numpy array if needed
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Ensure image is in BGR format for OpenCV
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Convert RGB to BGR if needed
                if image.dtype == np.uint8:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
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
        """Learn face from uploaded image"""
        if not name or name.strip() == "":
            return "❌ Please enter a name for the person", None
        
        if image is None:
            return "❌ Please upload an image", None
        
        try:
            # Extract face embedding
            embedding = self.extract_face_embedding(image)
            
            if embedding is None:
                return "❌ No face detected in the image. Please try another image.", None
            
            # Save the face
            if self.save_known_face(embedding, name.strip()):
                self.known_face_embedding = embedding
                self.known_name = name.strip()
                return f"✅ Successfully learned face for {name.strip()}!", image
            else:
                return "❌ Failed to save face embedding", None
                
        except Exception as e:
            return f"❌ Error learning face: {str(e)}", None
    
    def recognize_face_in_image(self, image):
        """Recognize face in uploaded image"""
        if self.known_face_embedding is None:
            return "❌ No known face loaded. Please learn a face first.", None
        
        if image is None:
            return "❌ Please upload an image", None
        
        try:
            # Extract face embedding from current image
            current_embedding = self.extract_face_embedding(image)
            
            if current_embedding is None:
                return "❌ No face detected in the image", None
            
            # Calculate cosine similarity
            similarity = np.dot(self.known_face_embedding, current_embedding) / (
                np.linalg.norm(self.known_face_embedding) * np.linalg.norm(current_embedding)
            )
            
            # Convert to distance (0 = identical, 2 = completely different)
            distance = 1 - similarity
            
            # Check if it's the same person
            if distance < self.recognition_threshold:
                confidence = similarity
                result_text = f"✅ MATCH! This is {self.known_name}\nConfidence: {confidence:.3f}"
                return result_text, image
            else:
                result_text = f"❌ NOT A MATCH\nDistance: {distance:.3f}\nThreshold: {self.recognition_threshold}"
                return result_text, image
                
        except Exception as e:
            return f"❌ Error recognizing face: {str(e)}", None
    
    def start_webcam_recognition(self):
        """Start real-time webcam recognition"""
        if self.known_face_embedding is None:
            return "❌ No known face loaded. Please learn a face first."
        
        if self.is_recording:
            return "⚠️ Recognition is already running"
        
        self.is_recording = True
        self.stop_recognition = False
        
        # Start recognition in background thread
        self.recognition_thread = threading.Thread(target=self._webcam_recognition_loop)
        self.recognition_thread.daemon = True
        self.recognition_thread.start()
        
        return f"🚀 Started recognizing {self.known_name} in webcam"
    
    def stop_webcam_recognition(self):
        """Stop real-time webcam recognition"""
        if not self.is_recording:
            return "⚠️ Recognition is not running"
        
        self.stop_recognition = True
        self.is_recording = False
        
        if self.recognition_thread:
            self.recognition_thread.join(timeout=2)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        return "🛑 Stopped webcam recognition"
    
    def _webcam_recognition_loop(self):
        """Background thread for webcam recognition"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("❌ Failed to open webcam")
                return
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 15)  # Limit FPS for better performance
            
            while self.is_recording and not self.stop_recognition:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                # Recognize face
                name, confidence, face_bbox = self._recognize_face_in_frame(frame)
                
                # Draw results on frame
                annotated_frame = self._draw_recognition_results(frame, name, confidence, face_bbox)
                
                # Convert BGR to RGB for PIL
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image
                pil_image = Image.fromarray(rgb_frame)
                
                # Store latest frame
                self.latest_frame = pil_image
                
                # Put frame in queue for Gradio to display
                try:
                    if not self.frame_queue.full():
                        self.frame_queue.put(pil_image)
                    else:
                        # Remove old frame and add new one
                        self.frame_queue.get()
                        self.frame_queue.put(pil_image)
                except:
                    pass
                
                time.sleep(0.1)  # Limit processing rate
                
        except Exception as e:
            print(f"❌ Webcam recognition error: {e}")
        finally:
            if self.cap:
                self.cap.release()
                self.cap = None
    
    def _recognize_face_in_frame(self, frame):
        """Recognize face in a single frame"""
        try:
            # First detect faces using OpenCV
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return None, 0.0, None
            
            # Get the largest face (most prominent)
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            
            # Extract face region
            face_roi = frame[y:y+h, x:x+w]
            
            # Get face embedding
            current_embedding = DeepFace.represent(
                face_roi,
                model_name="VGG-Face",
                detector_backend="opencv",
                enforce_detection=False
            )
            
            if not current_embedding:
                return None, 0.0, (x, y, x+w, y+h)
            
            current_embedding = np.array(current_embedding[0]['embedding'])
            
            # Calculate cosine similarity
            similarity = np.dot(self.known_face_embedding, current_embedding) / (
                np.linalg.norm(self.known_face_embedding) * np.linalg.norm(current_embedding)
            )
            
            # Convert to distance
            distance = 1 - similarity
            
            # Check if it's the same person
            if distance < self.recognition_threshold:
                return self.known_name, similarity, (x, y, x+w, y+h)
            else:
                return None, similarity, (x, y, x+w, y+h)
                
        except Exception as e:
            return None, 0.0, None
    
    def _draw_recognition_results(self, frame, name, confidence, face_bbox):
        """Draw recognition results on frame"""
        overlay = frame.copy()
        
        if face_bbox:
            x1, y1, x2, y2 = face_bbox
            
            if name:
                # Known person detected - draw green box
                color = (0, 255, 0)  # Green
                thickness = 3
                
                # Draw bounding box
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
                
                # Draw name label background
                label = f"{name}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                
                # Label background
                cv2.rectangle(overlay, (x1, y1 - 35), 
                             (x1 + label_size[0] + 10, y1), color, -1)
                
                # Label text
                cv2.putText(overlay, label, (x1 + 5, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                
                # Draw confidence
                conf_text = f"Conf: {confidence:.3f}"
                conf_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                cv2.rectangle(overlay, (x1, y2), 
                             (x1 + conf_size[0] + 10, y2 + 25), color, -1)
                cv2.putText(overlay, conf_text, (x1 + 5, y2 + 18), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                
            else:
                # Unknown person - draw red box
                color = (0, 0, 255)  # Red
                thickness = 2
                
                # Draw bounding box
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
                
                # Draw "Unknown" label
                label = "Unknown"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                
                # Label background
                cv2.rectangle(overlay, (x1, y1 - 35), 
                             (x1 + label_size[0] + 10, y1), color, -1)
                
                # Label text
                cv2.putText(overlay, label, (x1 + 5, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add status overlay
        status_color = (0, 255, 0) if name else (0, 0, 255)
        cv2.rectangle(overlay, (10, 10), (300, 60), (0, 0, 0), -1)
        cv2.rectangle(overlay, (10, 10), (300, 60), status_color, 2)
        
        if name:
            status_text = f"Recognized: {name}"
        else:
            status_text = "No match found"
        
        cv2.putText(overlay, status_text, (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add FPS counter
        fps_text = f"FPS: {int(1/0.1)}"
        cv2.putText(overlay, fps_text, (20, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return overlay
    
    def get_webcam_frame(self):
        """Get current webcam frame for Gradio display"""
        if not self.is_recording or self.latest_frame is None:
            return None
        
        return self.latest_frame
    
    def get_status(self):
        """Get current system status"""
        if self.known_face_embedding is not None:
            return f"✅ Known Face: {self.known_name}\n🔧 Recognition Threshold: {self.recognition_threshold}\n📹 Webcam: {'🟢 Running' if self.is_recording else '🔴 Stopped'}"
        else:
            return "❌ No known face loaded\n📝 Please learn a face first"
    
    def adjust_threshold(self, new_threshold):
        """Adjust recognition threshold"""
        try:
            threshold = float(new_threshold)
            if 0.1 <= threshold <= 0.9:
                self.recognition_threshold = threshold
                return f"✅ Threshold updated to {threshold}"
            else:
                return "❌ Threshold must be between 0.1 and 0.9"
        except ValueError:
            return "❌ Invalid threshold value"
    
    def clear_known_face(self):
        """Clear the known face"""
        try:
            if os.path.exists(self.face_db_file):
                os.remove(self.face_db_file)
            
            self.known_face_embedding = None
            self.known_name = None
            
            return "🗑️ Cleared known face"
        except Exception as e:
            return f"❌ Error clearing face: {str(e)}"

def create_gradio_interface():
    """Create Gradio interface"""
    # Initialize face recognition system
    face_system = FaceRecognitionSystem()
    
    with gr.Blocks(title="Face Recognition System", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🎯 Face Recognition System")
        gr.Markdown("Learn one person's face and recognize them in images or webcam")
        
        with gr.Tabs():
            # Tab 1: Learn Face
            with gr.Tab("📚 Learn Face"):
                gr.Markdown("### Upload an image and learn a person's face")
                
                with gr.Row():
                    with gr.Column():
                        learn_image = gr.Image(label="Upload Image with Face", type="pil")
                        learn_name = gr.Textbox(label="Person's Name", placeholder="Enter the person's name")
                        learn_btn = gr.Button("🎯 Learn Face", variant="primary")
                    
                    with gr.Column():
                        learn_output = gr.Textbox(label="Status", lines=3)
                        learn_preview = gr.Image(label="Learned Face")
                
                learn_btn.click(
                    face_system.learn_face_from_image,
                    inputs=[learn_image, learn_name],
                    outputs=[learn_output, learn_preview]
                )
            
            # Tab 2: Recognize Face
            with gr.Tab("🔍 Recognize Face"):
                gr.Markdown("### Upload an image to check if it matches the learned face")
                
                with gr.Row():
                    with gr.Column():
                        recognize_image = gr.Image(label="Upload Image to Check", type="pil")
                        recognize_btn = gr.Button("🔍 Check Face", variant="primary")
                    
                    with gr.Column():
                        recognize_output = gr.Textbox(label="Recognition Result", lines=4)
                        recognize_preview = gr.Image(label="Checked Image")
                
                recognize_btn.click(
                    face_system.recognize_face_in_image,
                    inputs=[recognize_image],
                    outputs=[recognize_output, recognize_preview]
                )
            
            # Tab 3: Webcam Recognition
            with gr.Tab("📹 Webcam Recognition"):
                gr.Markdown("### Real-time face recognition using webcam")
                
                with gr.Row():
                    webcam_start_btn = gr.Button("🚀 Start Recognition", variant="primary")
                    webcam_stop_btn = gr.Button("🛑 Stop Recognition", variant="secondary")
                
                webcam_status = gr.Textbox(label="Webcam Status", lines=2)
                
                # Live webcam display
                webcam_display = gr.Image(label="Live Webcam Feed")
                
                # Refresh button for manual updates
                refresh_btn = gr.Button("🔄 Refresh Frame", variant="secondary")
                
                # Instructions
                gr.Markdown("**💡 Tip**: Click '🔄 Refresh Frame' to see the latest webcam feed with face recognition results")
                
                webcam_start_btn.click(
                    face_system.start_webcam_recognition,
                    outputs=[webcam_status]
                )
                
                webcam_stop_btn.click(
                    face_system.stop_webcam_recognition,
                    outputs=[webcam_status]
                )
                
                # Manual refresh button
                refresh_btn.click(
                    face_system.get_webcam_frame,
                    outputs=[webcam_display]
                )
            
            # Tab 4: Settings
            with gr.Tab("⚙️ Settings"):
                gr.Markdown("### System configuration and status")
                
                with gr.Row():
                    with gr.Column():
                        threshold_slider = gr.Slider(
                            minimum=0.1, 
                            maximum=0.9, 
                            value=0.6, 
                            step=0.05,
                            label="Recognition Threshold (Lower = Stricter)"
                        )
                        threshold_btn = gr.Button("🔧 Update Threshold")
                    
                    with gr.Column():
                        clear_btn = gr.Button("🗑️ Clear Known Face", variant="stop")
                        status_output = gr.Textbox(label="System Status", lines=5)
                
                # Update threshold
                threshold_btn.click(
                    face_system.adjust_threshold,
                    inputs=[threshold_slider],
                    outputs=[status_output]
                )
                
                # Clear face
                clear_btn.click(
                    face_system.clear_known_face,
                    outputs=[status_output]
                )
                
                # Show status
                gr.Button("📊 Refresh Status").click(
                    face_system.get_status,
                    outputs=[status_output]
                )
        
        # Footer
        gr.Markdown("---")
        gr.Markdown("**Built with DeepFace and Gradio** | Lower threshold = stricter matching")
        
        # Initialize status
        interface.load(lambda: face_system.get_status(), outputs=[status_output])
    
    return interface

if __name__ == "__main__":
    # Create and launch interface
    interface = create_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
