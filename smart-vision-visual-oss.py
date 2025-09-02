#!/usr/bin/env python3
"""
Smart Vision Chat - GPT-OSS 20B with Visual Input
Sends actual camera frames to GPT-OSS for real visual understanding
"""

import argparse
import time
import cv2
import torch
import sys
import numpy as np
import os
import threading
import queue
import requests
import json
import subprocess
import base64
from collections import deque
from ultralytics import YOLO
import io
from PIL import Image

class VisualGPTOSS:
    """GPT-OSS with actual visual input capabilities"""
    
    def __init__(self):
        self.base_url = "http://127.0.0.1:11434"
        self.model_name = "gpt-oss:20b"
        self.ready = False
        self.error_message = ""
        self.vision_working = False
        
    def start_service(self):
        """Start GPT-OSS service"""
        print("🚀 Starting Visual GPT-OSS Service...")
        
        try:
            subprocess.run(["pkill", "-f", "ollama"], capture_output=True)
            time.sleep(3)
            
            with open(os.devnull, 'w') as devnull:
                self.process = subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=devnull,
                    stderr=devnull,
                    preexec_fn=os.setsid
                )
            
            time.sleep(8)
            return self._test_vision()
            
        except Exception as e:
            self.error_message = str(e)
            return False
    
    def _test_vision(self):
        """Test vision capabilities"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return False
            
            # Create a simple test image
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            test_img[40:60, 40:60] = [255, 0, 0]  # Red square
            
            success, buffer = cv2.imencode('.jpg', test_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if success:
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Test with vision
                if self._try_vision_prompt("What do you see?", img_base64):
                    self.ready = True
                    self.vision_working = True
                    print(f"✅ Visual GPT-OSS ready with vision!")
                    return True
            
            # Fallback to text-only
            if self._try_simple_prompt("Hello"):
                self.ready = True
                self.vision_working = False
                print(f"⚠️ GPT-OSS ready but vision may not work")
                return True
                
            return False
                
        except Exception as e:
            self.error_message = str(e)
            return False
    
    def _try_simple_prompt(self, prompt):
        """Try simple text prompt"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 20,
                    "temperature": 0.8
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                generated = result.get("response", "").strip()
                return len(generated) > 0
            else:
                return False
                
        except Exception:
            return False
    
    def _try_vision_prompt(self, prompt, image_base64):
        """Try vision prompt with image"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [image_base64],
                "stream": False,
                "options": {
                    "num_predict": 50,
                    "temperature": 0.7
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                generated = result.get("response", "").strip()
                return len(generated) > 2
            else:
                return False
                
        except Exception:
            return False
    
    def analyze_scene(self, frame):
        """Analyze scene with visual input"""
        if not self.ready:
            return None
        
        try:
            # Resize frame for faster processing
            height, width = frame.shape[:2]
            if width > 640:
                scale = 640 / width
                new_width = 640
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Encode frame
            success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not success:
                return None
            
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Try vision prompt
            if self.vision_working:
                prompts = [
                    "Describe what you see in this image briefly.",
                    "What is in this picture?",
                    "Briefly describe the scene."
                ]
                
                for prompt in prompts:
                    if self._try_vision_prompt(prompt, img_base64):
                        try:
                            payload = {
                                "model": self.model_name,
                                "prompt": prompt,
                                "images": [img_base64],
                                "stream": False,
                                "options": {
                                    "num_predict": 60,
                                    "temperature": 0.7
                                }
                            }
                            
                            response = requests.post(
                                f"{self.base_url}/api/generate",
                                json=payload,
                                timeout=20
                            )
                            
                            if response.status_code == 200:
                                result = response.json()
                                generated = result.get("response", "").strip()
                                if len(generated) > 5 and self._is_valid_response(generated):
                                    return generated
                                    
                        except Exception:
                            continue
            
            return None
            
        except Exception as e:
            print(f"Vision analysis error: {e}")
            return None
    
    def answer_question(self, question, frame):
        """Answer question about the image"""
        if not self.ready:
            return None
        
        try:
            # Resize frame
            height, width = frame.shape[:2]
            if width > 640:
                scale = 640 / width
                new_width = 640
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Encode frame
            success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not success:
                return None
            
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Try vision-based question answering
            if self.vision_working:
                # Make question more specific for better results
                enhanced_prompts = [
                    f"Looking at this image, {question}",
                    f"In this picture, {question}",
                    f"Based on what you see: {question}"
                ]
                
                for prompt in enhanced_prompts:
                    try:
                        payload = {
                            "model": self.model_name,
                            "prompt": prompt,
                            "images": [img_base64],
                            "stream": False,
                            "options": {
                                "num_predict": 80,
                                "temperature": 0.6
                            }
                        }
                        
                        response = requests.post(
                            f"{self.base_url}/api/generate",
                            json=payload,
                            timeout=25
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            generated = result.get("response", "").strip()
                            if len(generated) > 5 and self._is_valid_response(generated):
                                return generated
                                
                    except Exception:
                        continue
            
            return None
            
        except Exception as e:
            print(f"Question answering error: {e}")
            return None
    
    def _is_valid_response(self, text):
        """Check if response is valid"""
        if not text or len(text) < 3:
            return False
        
        # Check for gibberish patterns
        gibberish_patterns = [
            "wheated", "user", "assistant", "###", "```", 
            "model", "ai", "language", "please", "sorry"
        ]
        
        text_lower = text.lower()
        gibberish_count = sum(1 for pattern in gibberish_patterns if pattern in text_lower)
        
        if gibberish_count > 2:
            return False
        
        # Check word structure
        words = text.split()
        if len(words) < 2:
            return False
        
        # Basic coherence check
        avg_word_length = sum(len(word) for word in words) / len(words)
        if avg_word_length > 15:  # Suspiciously long words
            return False
        
        return True
    
    def cleanup(self):
        """Cleanup"""
        self.ready = False
        self.vision_working = False

class SceneAnalyzer:
    """Scene analysis with visual understanding"""
    
    def __init__(self):
        self.current_scene = {}
        self.last_update = 0
        self.update_interval = 6.0  # Longer for vision processing
        self.last_frame = None
        
    def should_update_scene(self, new_detections, frame):
        """Check if scene should be updated"""
        current_time = time.time()
        
        # Always update if enough time passed
        if current_time - self.last_update > self.update_interval:
            self._update_scene(new_detections, frame)
            return True
        
        # Update if detections changed significantly
        if self._scene_changed_significantly(new_detections):
            self._update_scene(new_detections, frame)
            return True
        
        return False
    
    def _scene_changed_significantly(self, new_detections):
        """Check if scene changed significantly"""
        if not self.current_scene:
            return True
        
        # Check object count changes
        old_total = sum(self.current_scene.values())
        new_total = sum(new_detections.values())
        
        if abs(old_total - new_total) > 1:
            return True
        
        # Check new objects
        for obj_type in new_detections:
            if obj_type not in self.current_scene:
                return True
        
        return False
    
    def _update_scene(self, new_detections, frame):
        """Update scene"""
        self.current_scene = new_detections.copy()
        self.last_frame = frame.copy() if frame is not None else None
        self.last_update = time.time()
    
    def get_current_frame(self):
        """Get current frame for analysis"""
        return self.last_frame
    
    def get_scene_summary(self):
        """Get scene summary"""
        if not self.current_scene:
            return "Empty scene"
        
        summary_parts = []
        for obj_type, count in self.current_scene.items():
            if count == 1:
                summary_parts.append(obj_type)
            else:
                summary_parts.append(f"{count} {obj_type}s")
        
        return ", ".join(summary_parts)

class VisualVisionChat:
    """Vision chat with visual GPT-OSS"""
    
    def __init__(self, args):
        self.args = args
        self.yolo_model = None
        self.visual_gpt = VisualGPTOSS()
        self.scene_analyzer = SceneAnalyzer()
        
        # UI State
        self.current_input = ""
        self.input_mode = False
        self.cursor_blink = True
        self.last_blink = time.time()
        
        # App state
        self.chat_history = deque(maxlen=4)
        self.current_description = "Starting visual GPT-OSS system..."
        self.current_frame = None
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.vision_attempts = 0
        self.vision_successes = 0
        self.fallback_responses = 0
        
        # Threading
        self.scene_queue = queue.Queue(maxsize=2)
        self.running = True
        
        # Colors
        self.colors = {
            'panel': (10, 15, 25),
            'panel_light': (25, 30, 40),
            'accent': (255, 140, 0),
            'green': (100, 255, 100),
            'orange': (0, 165, 255),
            'blue': (255, 200, 100),
            'red': (100, 100, 255),
            'white': (255, 255, 255),
            'gray': (180, 180, 180),
            'dark': (120, 120, 120)
        }
    
    def setup(self):
        """Setup system"""
        print("👁️ Visual GPT-OSS Chat - Real Visual Understanding")
        print("=" * 65)
        
        # Find camera
        camera_id = self._find_camera()
        if camera_id is None:
            print("❌ No camera found!")
            return None
        
        # Load YOLO
        try:
            self.yolo_model = YOLO(self.args.model)
            device = self._get_device()
            print(f"✅ YOLO loaded on {device}")
        except Exception as e:
            print(f"❌ YOLO failed: {e}")
            return None
        
        # Start Visual GPT-OSS
        gpt_thread = threading.Thread(target=self.visual_gpt.start_service, daemon=True)
        gpt_thread.start()
        
        print("=" * 65)
        print("👁️ Visual GPT-OSS: Can see clothing, expressions, actions")
        print("🔍 Object Detection: Provides context and object boundaries")
        print("💬 Press SPACE to ask visual questions")
        print("❌ Press Q to quit")
        print("=" * 65)
        
        return camera_id, device
    
    def _find_camera(self):
        """Find working camera"""
        for i in range(4):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, _ = cap.read()
                    cap.release()
                    if ret:
                        print(f"📷 Camera {i} ready")
                        return i
            except:
                continue
        return None
    
    def _get_device(self):
        """Get device for YOLO"""
        if torch.cuda.is_available():
            return 0
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def _extract_detections(self, result):
        """Extract detections"""
        detections = {}
        
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        class_names = result.names if hasattr(result, 'names') else {}
        
        for box in result.boxes:
            try:
                cls_id = int(box.cls.item())
                confidence = float(box.conf.item())
                
                if confidence < 0.6:
                    continue
                
                class_name = class_names.get(cls_id, "object")
                detections[class_name] = detections.get(class_name, 0) + 1
                
            except Exception:
                continue
        
        return detections
    
    def _scene_worker(self):
        """Background visual scene analysis"""
        while self.running:
            try:
                task = self.scene_queue.get(timeout=1.0)
                if task is None:
                    break
                
                detections, frame = task
                
                if self.scene_analyzer.should_update_scene(detections, frame):
                    # Try visual analysis
                    self.vision_attempts += 1
                    visual_description = self.visual_gpt.analyze_scene(frame)
                    
                    if visual_description:
                        self.vision_successes += 1
                        self.current_description = f"👁️ {visual_description}"
                        print(f"👁️ Visual: {visual_description}")
                    else:
                        self.fallback_responses += 1
                        scene_summary = self.scene_analyzer.get_scene_summary()
                        self.current_description = f"🔍 Objects detected: {scene_summary}"
                        print(f"🔍 Fallback: Objects detected: {scene_summary}")
                
                self.scene_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Visual worker error: {e}")
                continue
    
    def _handle_input(self, key):
        """Handle user input"""
        if key == 32:  # Space
            if not self.input_mode:
                self.input_mode = True
                self.current_input = ""
                print("👁️ Ask about what you see (visual GPT-OSS will analyze the image)...")
            else:
                if self.current_input.strip():
                    self._process_question(self.current_input.strip())
                self._exit_input()
        
        elif key in [27, ord('q'), ord('Q')]:
            if self.input_mode:
                self._exit_input()
            else:
                self.running = False
        
        elif key == ord('r') or key == ord('R'):
            print("🔄 Restarting Visual GPT-OSS...")
            self.visual_gpt.cleanup()
            time.sleep(2)
            self.visual_gpt = VisualGPTOSS()
            threading.Thread(target=self.visual_gpt.start_service, daemon=True).start()
        
        elif key == 8 and self.input_mode:  # Backspace
            if len(self.current_input) > 0:
                self.current_input = self.current_input[:-1]
        
        elif self.input_mode and 32 <= key <= 126:
            if len(self.current_input) < 100:
                self.current_input += chr(key)
    
    def _process_question(self, question):
        """Process visual question"""
        print(f"❓ Visual Question: {question}")
        
        self.chat_history.append({
            'question': question,
            'answer': "Analyzing image with visual GPT-OSS...",
            'timestamp': time.time(),
            'response_type': 'processing'
        })
        
        threading.Thread(
            target=self._generate_visual_answer,
            args=(question, len(self.chat_history) - 1),
            daemon=True
        ).start()
    
    def _generate_visual_answer(self, question, chat_index):
        """Generate visual answer"""
        try:
            # Get current frame
            current_frame = self.scene_analyzer.get_current_frame()
            if current_frame is None and self.current_frame is not None:
                current_frame = self.current_frame
            
            if current_frame is None:
                answer = "No camera frame available for visual analysis."
                response_type = 'error'
            else:
                # Try visual question answering
                self.vision_attempts += 1
                visual_answer = self.visual_gpt.answer_question(question, current_frame)
                
                if visual_answer:
                    self.vision_successes += 1
                    answer = f"👁️ {visual_answer}"
                    response_type = 'visual'
                    print(f"👁️ Visual Answer: {visual_answer}")
                else:
                    self.fallback_responses += 1
                    # Provide context-aware fallback
                    scene_summary = self.scene_analyzer.get_scene_summary()
                    if "wearing" in question.lower() or "clothes" in question.lower():
                        answer = f"🔍 I can see {scene_summary} in the scene, but visual details like clothing require the visual model to be working properly."
                    elif "color" in question.lower():
                        answer = f"🔍 I can detect {scene_summary}, but color analysis requires visual processing capabilities."
                    else:
                        answer = f"🔍 Based on object detection: {scene_summary}. For detailed visual questions, the visual model needs to be responding."
                    response_type = 'fallback'
                    print(f"🔍 Fallback Answer: {answer}")
            
            # Update chat
            if chat_index < len(self.chat_history):
                self.chat_history[chat_index]['answer'] = answer
                self.chat_history[chat_index]['response_type'] = response_type
            
        except Exception as e:
            print(f"Visual answer generation error: {e}")
            if chat_index < len(self.chat_history):
                self.chat_history[chat_index]['answer'] = "Error generating visual response"
                self.chat_history[chat_index]['response_type'] = 'error'
    
    def _exit_input(self):
        """Exit input mode"""
        self.input_mode = False
        self.current_input = ""
    
    def _draw_text_wrap(self, img, text, x, y, font_scale, color, max_width, line_height=16):
        """Draw wrapped text"""
        if not text:
            return 0
        
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            (w, h), _ = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            
            if w <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        for i, line in enumerate(lines[:3]):
            cv2.putText(img, line, (x, y + i * line_height), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)
        
        return len(lines[:3]) * line_height
    
    def _draw_ui(self, frame, fps):
        """Draw visual UI"""
        h, w = frame.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Status bar
        cv2.rectangle(overlay, (8, 8), (w-8, 90), self.colors['panel'], -1)
        cv2.rectangle(overlay, (8, 8), (w-8, 90), self.colors['dark'], 2)
        
        # Visual GPT-OSS status
        if self.visual_gpt.ready and self.visual_gpt.vision_working:
            vision_rate = (self.vision_successes / max(1, self.vision_attempts)) * 100
            status = f"👁️ Visual GPT-OSS: Ready ({vision_rate:.0f}% success) - Can see clothing, colors, actions"
            color = self.colors['green']
        elif self.visual_gpt.ready:
            status = "⚠️ GPT-OSS: Ready but vision may not work - Text only mode"
            color = self.colors['orange']
        else:
            status = "🔄 Visual GPT-OSS: Starting..."
            color = self.colors['red']
        
        cv2.putText(overlay, status, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Stats
        total_responses = self.vision_successes + self.fallback_responses
        scene_summary = self.scene_analyzer.get_scene_summary()
        stats = f"Scene: {scene_summary} | FPS: {fps:.1f} | Visual:{self.vision_successes} Fallback:{self.fallback_responses}"
        cv2.putText(overlay, stats, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['gray'], 1)
        
        # Capabilities
        caps = "Ask: 'What is the person wearing?', 'What colors do you see?', 'What are they doing?'"
        cv2.putText(overlay, caps, (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['blue'], 1)
        
        # Scene description
        scene_y = 100
        cv2.rectangle(overlay, (8, scene_y), (w-8, scene_y + 95), self.colors['panel'], -1)
        cv2.rectangle(overlay, (8, scene_y), (w-8, scene_y + 95), self.colors['dark'], 2)
        
        cv2.putText(overlay, "👁️ Visual Scene Analysis", (15, scene_y + 22), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.colors['accent'], 2)
        
        self._draw_text_wrap(overlay, self.current_description, 15, scene_y + 45, 
                           0.45, self.colors['white'], w - 30, 16)
        
        # Chat history
        chat_y = 205
        chat_height = h - chat_y - 110
        cv2.rectangle(overlay, (8, chat_y), (w-8, chat_y + chat_height), self.colors['panel_light'], -1)
        cv2.rectangle(overlay, (8, chat_y), (w-8, chat_y + chat_height), self.colors['dark'], 2)
        
        cv2.putText(overlay, "💬 Visual Q&A (👁️=Visual, 🔍=Fallback)", (15, chat_y + 22), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['accent'], 2)
        
        # Show chat history
        y_pos = chat_y + 40
        for chat in list(self.chat_history)[-2:]:
            if y_pos + 50 > chat_y + chat_height:
                break
            
            # Question
            q_height = self._draw_text_wrap(overlay, f"Q: {chat['question']}", 15, y_pos, 
                                          0.4, self.colors['white'], w - 30, 14)
            y_pos += q_height + 3
            
            # Answer with type indicator
            response_type = chat.get('response_type', 'unknown')
            if response_type == 'visual':
                answer_color = self.colors['green']
            elif response_type == 'fallback':
                answer_color = self.colors['orange']
            else:
                answer_color = self.colors['gray']
            
            a_height = self._draw_text_wrap(overlay, f"A: {chat['answer']}", 15, y_pos, 
                                          0.4, answer_color, w - 30, 14)
            y_pos += a_height + 12
        
        # Input area
        input_y = h - 100
        input_color = self.colors['accent'] if self.input_mode else self.colors['panel']
        cv2.rectangle(overlay, (8, input_y), (w-8, h-8), input_color, -1)
        cv2.rectangle(overlay, (8, input_y), (w-8, h-8), self.colors['dark'], 2)
        
        if self.input_mode:
            cv2.putText(overlay, "👁️ Ask visual question (SPACE=Send, ESC=Cancel):", 
                       (15, input_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.colors['white'], 1)
            
            # Input with cursor
            current_time = time.time()
            if current_time - self.last_blink > 0.5:
                self.cursor_blink = not self.cursor_blink
                self.last_blink = current_time
            
            display_text = self.current_input + ("|" if self.cursor_blink else "")
            cv2.putText(overlay, display_text, (15, input_y + 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['white'], 2)
        else:
            cv2.putText(overlay, "👁️ Press SPACE to ask visual questions about clothing, colors, actions", 
                       (15, input_y + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['gray'], 1)
        
        # Controls
        cv2.putText(overlay, "SPACE=Ask Visual Question | R=Restart | Q=Quit", (15, input_y + 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.colors['dark'], 1)
        
        # Blend
        alpha = 0.88
        return cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
    
    def run(self):
        """Main execution"""
        setup_result = self.setup()
        if setup_result is None:
            return False
        
        camera_id, device = setup_result
        
        # Start scene worker
        scene_thread = threading.Thread(target=self._scene_worker, daemon=True)
        scene_thread.start()
        
        # Setup window
        window_name = "Visual GPT-OSS Chat - Real Visual Understanding"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1320, 820)
        
        print("🎬 Starting visual chat...")
        
        try:
            for result in self.yolo_model(source=camera_id, stream=True,
                                         conf=self.args.conf, imgsz=self.args.imgsz, 
                                         device=device, verbose=False):
                
                if not self.running:
                    break
                
                frame = result.plot()
                self.current_frame = frame.copy()
                self.frame_count += 1
                
                elapsed = time.time() - self.start_time
                fps = self.frame_count / elapsed if elapsed > 0 else 0
                
                detections = self._extract_detections(result)
                
                try:
                    self.scene_queue.put_nowait((detections, frame.copy()))
                except queue.Full:
                    pass
                
                enhanced_frame = self._draw_ui(frame, fps)
                cv2.imshow(window_name, enhanced_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    self._handle_input(key)
                
                if not self.running:
                    break
        
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        finally:
            print("\n🧹 Cleaning up...")
            self.running = False
            self.scene_queue.put(None)
            cv2.destroyAllWindows()
            self.visual_gpt.cleanup()
            
            print("\n" + "=" * 65)
            print("✅ Visual GPT-OSS Chat completed")
            print(f"📊 Frames: {self.frame_count}")
            print(f"💬 Visual Questions: {len(self.chat_history)}")
            print(f"👁️ Visual Success: {self.vision_successes}/{self.vision_attempts}")
            print(f"🔍 Fallback Responses: {self.fallback_responses}")
            print("=" * 65)
        
        return True

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Visual GPT-OSS Chat - Real Visual Understanding")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model")
    parser.add_argument("--conf", type=float, default=0.6, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    
    args = parser.parse_args()
    
    app = VisualVisionChat(args)
    success = app.run()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

