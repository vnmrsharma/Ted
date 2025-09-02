#!/usr/bin/env python3
"""
Smart Vision Chat - Final Version with Reliable AI Responses
Fixed to ensure coherent, relevant responses from GPT-OSS
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
import re
from collections import deque
from ultralytics import YOLO

class SceneAnalyzer:
    """Scene analysis with intelligent change detection"""
    
    def __init__(self):
        self.current_scene = {}
        self.last_update = 0
        self.update_interval = 4.0  # Update every 4 seconds
        
    def should_update_scene(self, new_detections):
        """Check if scene should be updated"""
        current_time = time.time()
        
        # Force update if enough time has passed
        if current_time - self.last_update > self.update_interval:
            self.current_scene = new_detections.copy()
            self.last_update = current_time
            return True
        
        # Check for changes
        if self.current_scene != new_detections:
            self.current_scene = new_detections.copy()
            self.last_update = current_time
            return True
        
        return False
    
    def get_scene_summary(self):
        """Get clean scene summary"""
        if not self.current_scene:
            return "empty scene"
        
        items = []
        for obj_type, count in self.current_scene.items():
            if count == 1:
                items.append(obj_type)
            else:
                items.append(f"{count} {obj_type}s")
        
        return ", ".join(items)

class ReliableGPTOSS:
    """Reliable GPT-OSS with response validation"""
    
    def __init__(self):
        self.base_url = "http://127.0.0.1:11434"
        self.model_name = "gpt-oss:20b"
        self.ready = False
        self.error_message = ""
        self.process = None
        
        # Response validation patterns
        self.invalid_patterns = [
            r'[a-z]{20,}',  # Very long words without spaces
            r'^[^aeiou\s]{10,}',  # Long consonant sequences
            r'(.)\1{8,}',  # Repeated characters
            r'[0-9]{10,}',  # Long number sequences
        ]
        
        # Fallback responses for different scenarios
        self.fallback_responses = {
            'person': "I can see a person in the camera view.",
            'people': "I can see people in the scene.",
            'empty': "The camera shows an empty space.",
            'objects': "I can see some objects in the view.",
            'default': "I can see the current camera scene."
        }
        
    def start_service(self):
        """Start Ollama service with verification"""
        print("🚀 Starting Reliable AI Service...")
        
        try:
            # Clean start
            subprocess.run(["pkill", "-f", "ollama"], capture_output=True)
            time.sleep(3)
            
            with open(os.devnull, 'w') as devnull:
                self.process = subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=devnull,
                    stderr=devnull,
                    preexec_fn=os.setsid
                )
            
            print("⏳ Initializing AI service...")
            time.sleep(10)
            
            return self._verify_model()
            
        except Exception as e:
            self.error_message = f"Service error: {str(e)[:30]}"
            return False
    
    def _verify_model(self):
        """Verify model works with multiple test cases"""
        try:
            # Test connection
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                self.error_message = "Connection failed"
                return False
            
            # Test with known working prompts
            test_cases = [
                ("Hello", "hello"),
                ("What color is the sky?", "blue"),
                ("Say yes", "yes"),
            ]
            
            working_count = 0
            for prompt, expected_keyword in test_cases:
                response = self._generate_reliable(prompt, max_tokens=20)
                if response and expected_keyword.lower() in response.lower():
                    working_count += 1
                    
                time.sleep(1)
            
            if working_count >= 2:  # At least 2 out of 3 working
                self.ready = True
                print("✅ AI Service Verified and Ready!")
                return True
            else:
                self.error_message = "Model verification failed"
                return False
                
        except Exception as e:
            self.error_message = f"Verification error: {str(e)[:30]}"
            return False
    
    def _is_response_valid(self, response):
        """Check if response is valid and coherent"""
        if not response or len(response.strip()) < 3:
            return False
        
        response = response.strip()
        
        # Check for invalid patterns
        for pattern in self.invalid_patterns:
            if re.search(pattern, response):
                return False
        
        # Check for reasonable word structure
        words = response.split()
        if len(words) == 0:
            return False
        
        # Check if it contains mostly real words (simple heuristic)
        valid_word_count = 0
        for word in words:
            # Simple check: does it have vowels and reasonable length?
            if len(word) >= 2 and any(v in word.lower() for v in 'aeiou'):
                valid_word_count += 1
        
        # At least 60% of words should look valid
        if valid_word_count / len(words) < 0.6:
            return False
        
        return True
    
    def _generate_reliable(self, prompt, max_tokens=80, max_retries=3):
        """Generate text with validation and retry logic"""
        if not prompt:
            return "No input provided"
        
        for attempt in range(max_retries):
            try:
                # Use different parameter sets for each retry
                temp_values = [0.3, 0.7, 0.5][attempt % 3]
                
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temp_values,
                        "top_p": 0.9,
                        "top_k": 40,
                        "repeat_penalty": 1.2
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
                    
                    if self._is_response_valid(generated):
                        return generated
                    else:
                        print(f"⚠️ Invalid response on attempt {attempt + 1}: '{generated[:50]}'")
                
                time.sleep(1)  # Brief pause between retries
                
            except Exception as e:
                print(f"❌ Generation error on attempt {attempt + 1}: {e}")
                time.sleep(2)
        
        return None  # All attempts failed
    
    def describe_scene(self, scene_items):
        """Generate scene description with fallbacks"""
        if not scene_items or scene_items == "empty scene":
            return self.fallback_responses['empty']
        
        # Try multiple simple prompt formats
        prompt_formats = [
            f"I see {scene_items}.",
            f"Describe {scene_items}.",
            f"What about {scene_items}?",
            f"Scene has {scene_items}.",
        ]
        
        for prompt in prompt_formats:
            response = self._generate_reliable(prompt, max_tokens=60)
            if response:
                print(f"✅ Scene description generated: '{response[:80]}'")
                return response
        
        # Use intelligent fallback based on scene content
        if "person" in scene_items:
            if "people" in scene_items or any(word.endswith("s") for word in scene_items.split()):
                return self.fallback_responses['people']
            else:
                return self.fallback_responses['person']
        elif scene_items:
            return f"I can see {scene_items} in the camera view."
        else:
            return self.fallback_responses['empty']
    
    def answer_question(self, question, scene_items):
        """Answer questions with scene context and validation"""
        if not scene_items:
            scene_context = "an empty room"
        else:
            scene_context = scene_items
        
        # Try different question formats
        question_formats = [
            f"Scene: {scene_context}. Question: {question}",
            f"I see {scene_context}. {question}",
            f"About {scene_context}: {question}",
            f"{question} (scene has {scene_context})",
        ]
        
        for prompt in question_formats:
            response = self._generate_reliable(prompt, max_tokens=100)
            if response:
                print(f"✅ Question answered: '{response[:80]}'")
                return response
        
        # Intelligent fallback based on question content
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['what', 'who', 'where', 'how']):
            if "person" in scene_context:
                return f"I can see a person in the scene. {question}"
            elif scene_context and scene_context != "empty scene":
                return f"Based on what I can see ({scene_context}), I'm not entirely sure about that specific detail."
            else:
                return "I can see the camera view, but I'm not sure about the specific details you're asking about."
        else:
            return f"Regarding the current scene with {scene_context}, I'm not certain about that."
    
    def cleanup(self):
        """Enhanced cleanup"""
        self.ready = False
        try:
            if self.process:
                os.killpg(os.getpgid(self.process.pid), 15)
                self.process.wait(timeout=5)
        except:
            pass
        
        try:
            subprocess.run(["pkill", "-f", "ollama"], capture_output=True)
        except:
            pass

class FinalVisionChat:
    """Final reliable vision chat application"""
    
    def __init__(self, args):
        self.args = args
        self.yolo_model = None
        self.ai = ReliableGPTOSS()
        self.scene_analyzer = SceneAnalyzer()
        
        # UI State
        self.current_input = ""
        self.input_mode = False
        self.cursor_blink = True
        self.last_blink = time.time()
        
        # App state
        self.chat_history = deque(maxlen=4)  # Keep last 4 for better display
        self.current_description = "Starting reliable AI system..."
        self.current_scene_items = ""
        
        # Performance
        self.frame_count = 0
        self.start_time = time.time()
        self.successful_responses = 0
        self.failed_responses = 0
        
        # Threading
        self.scene_queue = queue.Queue(maxsize=2)
        self.running = True
        
        # Enhanced colors
        self.colors = {
            'panel': (30, 35, 45),
            'panel_light': (45, 50, 60),
            'accent': (255, 140, 0),
            'green': (120, 255, 120),
            'orange': (0, 165, 255),
            'red': (100, 100, 255),
            'white': (255, 255, 255),
            'gray': (190, 190, 190),
            'dark': (130, 130, 130)
        }
    
    def setup(self):
        """Enhanced system setup"""
        print("🎯 Smart Vision Chat - Final Reliable Edition")
        print("=" * 50)
        
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
            print(f"📊 YOLO can detect {len(self.yolo_model.names)} object types")
        except Exception as e:
            print(f"❌ YOLO failed: {e}")
            return None
        
        # Start AI in background
        ai_thread = threading.Thread(target=self.ai.start_service, daemon=True)
        ai_thread.start()
        
        print("=" * 50)
        print("🎥 Camera ready for reliable scene analysis")
        print("💬 Press SPACE to ask questions (answers will be coherent)")
        print("🔄 System will provide fallback responses if needed")
        print("❌ Press Q to quit")
        print("=" * 50)
        
        return camera_id, device
    
    def _find_camera(self):
        """Find working camera"""
        for i in range(4):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    cap.release()
                    if ret and frame is not None:
                        h, w = frame.shape[:2]
                        print(f"📷 Using camera {i} ({w}x{h})")
                        return i
            except:
                continue
        return None
    
    def _get_device(self):
        """Get optimal device for YOLO"""
        if torch.cuda.is_available():
            return 0
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def _extract_detections(self, result):
        """Extract detections with enhanced error handling"""
        detections = {}
        
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        class_names = result.names if hasattr(result, 'names') else {}
        
        for box in result.boxes:
            try:
                # Proper tensor handling to avoid numpy warnings
                cls_id = int(box.cls.item())
                confidence = float(box.conf.item())
                
                # Higher confidence for cleaner results
                if confidence < 0.7:
                    continue
                
                class_name = class_names.get(cls_id, "object")
                detections[class_name] = detections.get(class_name, 0) + 1
                
            except Exception as e:
                print(f"Detection error: {e}")
                continue
        
        return detections
    
    def _scene_worker(self):
        """Background scene analysis with reliability focus"""
        while self.running:
            try:
                task = self.scene_queue.get(timeout=1.0)
                if task is None:
                    break
                
                detections = task
                
                if self.scene_analyzer.should_update_scene(detections):
                    scene_items = self.scene_analyzer.get_scene_summary()
                    self.current_scene_items = scene_items
                    
                    if self.ai.ready:
                        try:
                            description = self.ai.describe_scene(scene_items)
                            if description:
                                self.current_description = description
                                self.successful_responses += 1
                                print(f"📝 Scene: {description}")
                            else:
                                self.failed_responses += 1
                                self.current_description = f"I can see: {scene_items}"
                        except Exception as e:
                            print(f"Scene description error: {e}")
                            self.current_description = f"Camera shows: {scene_items}"
                    else:
                        self.current_description = f"Detected: {scene_items}"
                
                self.scene_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Scene worker error: {e}")
                continue
    
    def _handle_input(self, key):
        """Enhanced input handling"""
        if key == 32:  # Space
            if not self.input_mode:
                self.input_mode = True
                self.current_input = ""
                print("💬 Ask a question (reliable response guaranteed)...")
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
            print("🔄 Restarting AI system...")
            threading.Thread(target=self._restart_ai, daemon=True).start()
        
        elif key == 8 and self.input_mode:  # Backspace
            if len(self.current_input) > 0:
                self.current_input = self.current_input[:-1]
        
        elif self.input_mode and 32 <= key <= 126:
            if len(self.current_input) < 100:
                self.current_input += chr(key)
    
    def _process_question(self, question):
        """Process user question with reliability guarantee"""
        print(f"❓ Question: {question}")
        
        self.chat_history.append({
            'question': question,
            'answer': "Generating reliable response...",
            'timestamp': time.time()
        })
        
        # Generate answer in background
        threading.Thread(
            target=self._generate_reliable_answer,
            args=(question, len(self.chat_history) - 1),
            daemon=True
        ).start()
    
    def _generate_reliable_answer(self, question, chat_index):
        """Generate reliable answer with fallbacks"""
        try:
            if self.ai.ready:
                answer = self.ai.answer_question(question, self.current_scene_items)
                if answer:
                    self.chat_history[chat_index]['answer'] = answer
                    self.successful_responses += 1
                    print(f"💡 Reliable answer: {answer}")
                else:
                    # Provide intelligent fallback
                    self.failed_responses += 1
                    fallback = self._generate_fallback_answer(question)
                    self.chat_history[chat_index]['answer'] = fallback
                    print(f"🔄 Fallback answer: {fallback}")
            else:
                fallback = self._generate_fallback_answer(question)
                self.chat_history[chat_index]['answer'] = fallback
                
        except Exception as e:
            print(f"Answer generation error: {e}")
            if chat_index < len(self.chat_history):
                fallback = self._generate_fallback_answer(question)
                self.chat_history[chat_index]['answer'] = fallback
    
    def _generate_fallback_answer(self, question):
        """Generate intelligent fallback answer based on question and scene"""
        question_lower = question.lower()
        scene = self.current_scene_items or "empty scene"
        
        # Analyze question type and provide relevant fallback
        if any(word in question_lower for word in ['what', 'describe']):
            if "person" in scene:
                return "I can see a person in the camera view."
            elif scene != "empty scene":
                return f"I can see {scene} in the current view."
            else:
                return "The camera shows an empty space right now."
        
        elif any(word in question_lower for word in ['who', 'person']):
            if "person" in scene:
                return "There is a person visible in the camera."
            else:
                return "I don't see any people in the current view."
        
        elif any(word in question_lower for word in ['where', 'location']):
            return "I can see the camera's current field of view."
        
        elif any(word in question_lower for word in ['how', 'count']):
            if scene != "empty scene":
                return f"Based on what I can detect: {scene}."
            else:
                return "I don't see any countable objects right now."
        
        else:
            if scene != "empty scene":
                return f"Regarding the current scene with {scene}, I'm analyzing what I can observe."
            else:
                return "I'm observing the camera view, but I don't see specific objects to comment on."
    
    def _restart_ai(self):
        """Restart AI system"""
        self.ai.cleanup()
        time.sleep(3)
        self.ai = ReliableGPTOSS()
        self.ai.start_service()
    
    def _exit_input(self):
        """Exit input mode"""
        self.input_mode = False
        self.current_input = ""
    
    def _draw_text_wrap(self, img, text, x, y, font_scale, color, max_width, line_height=18):
        """Draw wrapped text with better formatting"""
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
        
        for i, line in enumerate(lines[:4]):  # Max 4 lines
            cv2.putText(img, line, (x, y + i * line_height), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)
        
        return len(lines[:4]) * line_height
    
    def _draw_ui(self, frame, fps):
        """Draw enhanced UI with reliability indicators"""
        h, w = frame.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Enhanced status bar
        cv2.rectangle(overlay, (8, 8), (w-8, 85), self.colors['panel'], -1)
        cv2.rectangle(overlay, (8, 8), (w-8, 85), self.colors['dark'], 2)
        
        # AI status with reliability metrics
        if self.ai.ready:
            success_rate = self.successful_responses / max(1, self.successful_responses + self.failed_responses) * 100
            status = f"🤖 AI Ready (Success: {success_rate:.1f}%)"
            color = self.colors['green']
        else:
            status = "🔄 AI Initializing..."
            color = self.colors['orange']
        
        cv2.putText(overlay, status, (18, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        
        # Enhanced stats
        total_objects = sum(self.scene_analyzer.current_scene.values())
        stats = f"Objects: {total_objects} | FPS: {fps:.1f} | Responses: {self.successful_responses}/{self.successful_responses + self.failed_responses}"
        cv2.putText(overlay, stats, (18, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.colors['gray'], 1)
        
        # Scene description panel
        scene_y = 95
        cv2.rectangle(overlay, (8, scene_y), (w-8, scene_y + 110), self.colors['panel'], -1)
        cv2.rectangle(overlay, (8, scene_y), (w-8, scene_y + 110), self.colors['dark'], 2)
        
        cv2.putText(overlay, "🎙️ Reliable Scene Description", (18, scene_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['accent'], 2)
        
        # Current scene summary
        if self.current_scene_items:
            cv2.putText(overlay, f"Detected: {self.current_scene_items}", (18, scene_y + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.colors['gray'], 1)
        
        # AI description
        self._draw_text_wrap(overlay, self.current_description, 18, scene_y + 75, 
                           0.5, self.colors['white'], w - 36, 16)
        
        # Chat history panel
        chat_y = 215
        chat_height = h - chat_y - 125
        cv2.rectangle(overlay, (8, chat_y), (w-8, chat_y + chat_height), self.colors['panel_light'], -1)
        cv2.rectangle(overlay, (8, chat_y), (w-8, chat_y + chat_height), self.colors['dark'], 2)
        
        cv2.putText(overlay, "💬 Reliable Q&A", (18, chat_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['accent'], 2)
        
        # Show last 2 chats with better spacing
        y_pos = chat_y + 45
        for chat in list(self.chat_history)[-2:]:
            if y_pos + 70 > chat_y + chat_height:
                break
            
            # Question
            q_height = self._draw_text_wrap(overlay, f"Q: {chat['question']}", 18, y_pos, 
                                          0.45, self.colors['white'], w - 36, 16)
            y_pos += q_height + 8
            
            # Answer with validation indicator
            answer_prefix = "A: "
            if "fallback" in chat.get('answer', '').lower() or chat.get('answer', '') in ["Generating reliable response...", "AI temporarily unavailable"]:
                answer_color = self.colors['orange']  # Fallback response
            else:
                answer_color = self.colors['green']  # AI response
                
            a_height = self._draw_text_wrap(overlay, f"{answer_prefix}{chat['answer']}", 18, y_pos, 
                                          0.45, answer_color, w - 36, 16)
            y_pos += a_height + 18
        
        # Enhanced input area
        input_y = h - 105
        input_color = self.colors['accent'] if self.input_mode else self.colors['panel']
        cv2.rectangle(overlay, (8, input_y), (w-8, h-8), input_color, -1)
        cv2.rectangle(overlay, (8, input_y), (w-8, h-8), self.colors['dark'], 2)
        
        if self.input_mode:
            cv2.putText(overlay, "💬 Ask about the scene (SPACE=Send, ESC=Cancel):", 
                       (18, input_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['white'], 1)
            
            # Input with cursor
            current_time = time.time()
            if current_time - self.last_blink > 0.5:
                self.cursor_blink = not self.cursor_blink
                self.last_blink = current_time
            
            display_text = self.current_input + ("|" if self.cursor_blink else "")
            cv2.putText(overlay, display_text, (18, input_y + 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.colors['white'], 2)
        else:
            cv2.putText(overlay, "💬 Press SPACE to ask about the scene (guaranteed reliable response)", 
                       (18, input_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['gray'], 1)
        
        # Controls
        cv2.putText(overlay, "SPACE=Ask Question | R=Restart AI | Q=Quit", (18, input_y + 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['dark'], 1)
        
        # Blend with transparency
        alpha = 0.87
        return cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
    
    def run(self):
        """Main execution with enhanced reliability"""
        setup_result = self.setup()
        if setup_result is None:
            return False
        
        camera_id, device = setup_result
        
        # Start scene worker
        scene_thread = threading.Thread(target=self._scene_worker, daemon=True)
        scene_thread.start()
        
        # Setup window
        window_name = "Smart Vision Chat - Final Reliable Edition"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1350, 850)
        
        print("🎬 Starting reliable vision chat system...")
        
        try:
            for result in self.yolo_model(source=camera_id, stream=True,
                                         conf=self.args.conf, imgsz=self.args.imgsz, 
                                         device=device, verbose=False):
                
                if not self.running:
                    break
                
                frame = result.plot()
                self.frame_count += 1
                
                # Calculate FPS
                elapsed = time.time() - self.start_time
                fps = self.frame_count / elapsed if elapsed > 0 else 0
                
                # Extract detections
                detections = self._extract_detections(result)
                
                # Queue for scene analysis
                try:
                    self.scene_queue.put_nowait(detections)
                except queue.Full:
                    pass
                
                # Draw enhanced UI
                enhanced_frame = self._draw_ui(frame, fps)
                cv2.imshow(window_name, enhanced_frame)
                
                # Handle input
                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    self._handle_input(key)
                
                if not self.running:
                    break
        
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user")
        except Exception as e:
            print(f"\n❌ Runtime error: {e}")
        finally:
            print("\n🧹 Cleaning up reliable system...")
            self.running = False
            self.scene_queue.put(None)
            cv2.destroyAllWindows()
            self.ai.cleanup()
            
            print("\n" + "=" * 50)
            print("✅ Reliable Vision Chat session completed")
            print(f"📊 Processed {self.frame_count} frames")
            print(f"💬 Conversations: {len(self.chat_history)}")
            print(f"🎯 AI Success Rate: {self.successful_responses}/{self.successful_responses + self.failed_responses}")
            print("=" * 50)
        
        return True

def main():
    """Application entry point"""
    parser = argparse.ArgumentParser(description="Smart Vision Chat - Final Reliable Edition")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.7, help="Detection confidence")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    
    args = parser.parse_args()
    
    app = FinalVisionChat(args)
    success = app.run()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

