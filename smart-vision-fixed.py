#!/usr/bin/env python3
"""
Smart Vision Chat - Fixed Version with Robust GPT-OSS Integration
Real-time scene understanding with reliable AI responses
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
from collections import deque
from ultralytics import YOLO

class SceneAnalyzer:
    """Intelligent scene analysis and change detection"""
    
    def __init__(self):
        self.current_scene = {}
        self.scene_history = deque(maxlen=10)
        self.last_significant_change = 0
        self.change_threshold = 0.2  # Lower threshold for more updates
        
    def analyze_scene_change(self, new_detections):
        """Detect if scene has changed significantly"""
        if not self.current_scene:
            self.current_scene = new_detections.copy()
            return True, "initial_scene"
        
        # Check for any changes
        old_types = set(self.current_scene.keys())
        new_types = set(new_detections.keys())
        
        added_types = new_types - old_types
        removed_types = old_types - new_types
        
        # Check for quantity changes
        quantity_changes = []
        for obj_type in old_types & new_types:
            old_count = self.current_scene[obj_type]
            new_count = new_detections[obj_type]
            if old_count != new_count:
                quantity_changes.append((obj_type, old_count, new_count))
        
        # Determine change significance
        change_reasons = []
        
        if added_types:
            change_reasons.append(f"Added: {', '.join(added_types)}")
        if removed_types:
            change_reasons.append(f"Removed: {', '.join(removed_types)}")
        if quantity_changes:
            for obj_type, old, new in quantity_changes:
                change_reasons.append(f"{obj_type}: {old}→{new}")
        
        # Check if change is significant
        is_significant = (
            len(added_types) > 0 or 
            len(removed_types) > 0 or 
            len(quantity_changes) > 0
        )
        
        if is_significant:
            self.current_scene = new_detections.copy()
            self.last_significant_change = time.time()
            change_description = "; ".join(change_reasons) if change_reasons else "scene_update"
            return True, change_description
        
        return False, ""
    
    def get_scene_context(self):
        """Get current scene context for AI"""
        if not self.current_scene:
            return "No objects detected in the camera view"
        
        context_parts = []
        for obj_type, count in self.current_scene.items():
            if count == 1:
                context_parts.append(f"1 {obj_type}")
            else:
                context_parts.append(f"{count} {obj_type}s")
        
        return f"Objects visible: {', '.join(context_parts)}"

class ImprovedGPTOSS:
    """Enhanced GPT-OSS with better prompting and error handling"""
    
    def __init__(self):
        self.base_url = "http://127.0.0.1:11434"
        self.model_name = "gpt-oss:20b"
        self.ready = False
        self.loading = False
        self.error_message = ""
        self.process = None
        self.response_cache = {}
        self.last_successful_response = time.time()
        
    def start_service(self):
        """Start Ollama service with enhanced initialization"""
        print("🚀 Starting Enhanced AI Vision System...")
        
        try:
            # Clean shutdown of existing processes
            subprocess.run(["pkill", "-f", "ollama"], capture_output=True, text=True)
            time.sleep(2)
            
            # Start fresh service
            with open(os.devnull, 'w') as devnull:
                self.process = subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=devnull,
                    stderr=devnull,
                    preexec_fn=os.setsid
                )
            
            print("⏳ Initializing AI service (this may take a moment)...")
            time.sleep(10)  # Give more time for startup
            
            return self._verify_and_setup_model()
            
        except FileNotFoundError:
            self.error_message = "Ollama not installed"
            return False
        except Exception as e:
            self.error_message = f"Service error: {str(e)[:30]}"
            return False
    
    def _verify_and_setup_model(self):
        """Verify service and ensure model works properly"""
        max_retries = 20
        
        for attempt in range(max_retries):
            try:
                print(f"🔍 Testing connection (attempt {attempt + 1}/{max_retries})...")
                
                # Test basic connection
                response = requests.get(f"{self.base_url}/api/tags", timeout=5)
                if response.status_code != 200:
                    time.sleep(2)
                    continue
                
                # Verify model exists
                models = response.json().get("models", [])
                model_exists = any(self.model_name in model.get("name", "") for model in models)
                
                if not model_exists:
                    print(f"📥 Model not found, downloading {self.model_name}...")
                    self.loading = True
                    
                    pull_response = requests.post(
                        f"{self.base_url}/api/pull",
                        json={"name": self.model_name},
                        timeout=900  # 15 minutes for download
                    )
                    
                    if pull_response.status_code != 200:
                        self.error_message = "Model download failed"
                        return False
                
                # Test text generation with multiple attempts
                test_prompts = [
                    "Say hello.",
                    "Describe a simple scene.",
                    "What do you see?"
                ]
                
                for test_prompt in test_prompts:
                    print(f"🧪 Testing generation: '{test_prompt}'")
                    test_result = self._generate_with_retry(test_prompt, max_tokens=20, retries=3)
                    
                    if test_result and len(test_result.strip()) > 0 and "error" not in test_result.lower():
                        self.ready = True
                        self.loading = False
                        print(f"✅ AI System Ready! Test response: '{test_result[:50]}'")
                        return True
                
                print(f"❌ Generation test failed, retrying...")
                time.sleep(3)
                
            except requests.exceptions.RequestException as e:
                print(f"Connection issue: {e}")
                time.sleep(3)
                continue
            except Exception as e:
                print(f"Test error: {e}")
                time.sleep(3)
                continue
        
        self.error_message = "Model verification failed after all attempts"
        return False
    
    def _generate_with_retry(self, prompt, max_tokens=100, timeout=30, retries=3):
        """Generate text with multiple retry attempts"""
        for attempt in range(retries):
            try:
                # Enhanced payload with better parameters
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "top_k": 40,
                        "repeat_penalty": 1.1,
                        "stop": ["\n\n", "Human:", "User:", "Q:"]
                    }
                }
                
                print(f"🔄 Generating response (attempt {attempt + 1}/{retries})...")
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    generated_text = result.get("response", "").strip()
                    
                    if generated_text and len(generated_text) > 0:
                        self.last_successful_response = time.time()
                        print(f"✅ Generated: '{generated_text[:100]}'")
                        return generated_text
                    else:
                        print(f"⚠️ Empty response on attempt {attempt + 1}")
                else:
                    print(f"❌ API error {response.status_code} on attempt {attempt + 1}")
                
                if attempt < retries - 1:
                    time.sleep(2 * (attempt + 1))  # Exponential backoff
                    
            except requests.exceptions.Timeout:
                print(f"⏰ Timeout on attempt {attempt + 1}")
                if attempt < retries - 1:
                    time.sleep(5)
            except Exception as e:
                print(f"❌ Error on attempt {attempt + 1}: {e}")
                if attempt < retries - 1:
                    time.sleep(2)
        
        return f"Failed to generate response after {retries} attempts"
    
    def analyze_scene_continuously(self, scene_context, change_description=""):
        """Generate scene description with improved prompting"""
        if not self.ready:
            return "AI system is initializing..."
        
        if not scene_context or "No objects" in scene_context:
            return "I can see an empty scene with no objects detected."
        
        # Create better prompts
        if change_description and change_description != "initial_scene":
            prompt = f"""Looking at a camera feed, I can see: {scene_context}

There was a recent change: {change_description}

Please describe what you observe in 1-2 natural sentences, mentioning the current objects and any notable changes."""
        else:
            prompt = f"""Looking at a camera feed, I can see: {scene_context}

Please describe this scene in 1-2 natural, conversational sentences."""
        
        return self._generate_with_retry(prompt, max_tokens=100, timeout=25)
    
    def answer_contextual_question(self, question, current_scene_context):
        """Answer questions with enhanced context awareness"""
        if not self.ready:
            return "AI system is not ready yet. Please wait a moment."
        
        # Create comprehensive prompt
        prompt = f"""I'm looking at a live camera feed that currently shows: {current_scene_context}

The user asks: {question}

Please answer this question based on what's currently visible in the camera view. Be specific and helpful."""
        
        return self._generate_with_retry(prompt, max_tokens=150, timeout=30)
    
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

class EnhancedVisionChat:
    """Main application with enhanced reliability"""
    
    def __init__(self, args):
        self.args = args
        self.yolo_model = None
        self.ai = ImprovedGPTOSS()
        self.scene_analyzer = SceneAnalyzer()
        
        # UI State
        self.current_input = ""
        self.input_mode = False
        self.cursor_blink = True
        self.last_blink = time.time()
        
        # Scene State
        self.chat_history = deque(maxlen=6)
        self.current_description = "Starting camera and AI system..."
        self.current_scene_context = ""
        self.last_update_time = 0
        
        # Performance
        self.frame_count = 0
        self.start_time = time.time()
        self.detection_count = 0
        
        # Threading
        self.scene_queue = queue.Queue(maxsize=2)
        self.running = True
        
        # Enhanced colors
        self.colors = {
            'bg_dark': (20, 25, 30),
            'panel_dark': (35, 40, 50),
            'panel_light': (50, 55, 65),
            'accent_blue': (255, 140, 0),
            'accent_green': (100, 255, 100),
            'accent_orange': (0, 165, 255),
            'accent_red': (100, 100, 255),
            'text_bright': (255, 255, 255),
            'text_dim': (190, 190, 190),
            'text_dark': (130, 130, 130)
        }
    
    def setup(self):
        """Enhanced system setup"""
        print("🎯 Smart Vision Chat v2.1 - Enhanced Edition")
        print("=" * 55)
        
        # Setup camera
        camera_id = self._find_working_camera()
        if camera_id is None:
            print("❌ No camera available!")
            return None
        
        # Setup YOLO with verbose output
        try:
            print("🔧 Loading YOLO model...")
            self.yolo_model = YOLO(self.args.model)
            device = self._get_optimal_device()
            print(f"✅ YOLO loaded successfully on {device}")
            print(f"📊 Model classes: {len(self.yolo_model.names)} object types")
        except Exception as e:
            print(f"❌ YOLO failed: {e}")
            return None
        
        # Start AI system
        ai_thread = threading.Thread(target=self._initialize_ai_system, daemon=True)
        ai_thread.start()
        
        print("=" * 55)
        print("🎥 Camera ready - Enhanced scene analysis active")
        print("💬 Press SPACE to ask questions about the scene")
        print("🔄 Press R to restart AI | Q/ESC to quit")
        print("=" * 55)
        
        return camera_id, device
    
    def _find_working_camera(self):
        """Enhanced camera detection"""
        print("🔍 Searching for cameras...")
        for i in range(8):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    cap.release()
                    if ret and frame is not None:
                        h, w = frame.shape[:2]
                        print(f"📷 Camera {i} found: {w}x{h}")
                        return i
                    else:
                        print(f"❌ Camera {i} failed to read")
                else:
                    print(f"❌ Camera {i} not accessible")
            except Exception as e:
                print(f"❌ Camera {i} error: {e}")
        return None
    
    def _get_optimal_device(self):
        """Enhanced device selection"""
        if torch.cuda.is_available():
            return 0
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def _initialize_ai_system(self):
        """Enhanced AI initialization"""
        success = self.ai.start_service()
        if success:
            print("🤖 AI system fully operational!")
            # Test with a scene description
            test_response = self.ai.analyze_scene_continuously("1 person")
            print(f"🧪 AI test response: {test_response[:100]}")
        else:
            print(f"⚠️ AI system issue: {self.ai.error_message}")
    
    def _extract_detections_enhanced(self, result):
        """Enhanced detection extraction with better error handling"""
        detections = {}
        
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        class_names = result.names if hasattr(result, 'names') else {}
        
        for i, box in enumerate(result.boxes):
            try:
                # Enhanced array handling
                cls_tensor = box.cls
                conf_tensor = box.conf
                
                if cls_tensor.numel() == 0 or conf_tensor.numel() == 0:
                    continue
                
                # Proper scalar extraction
                cls_id = int(cls_tensor.item())
                confidence = float(conf_tensor.item())
                
                # Enhanced confidence filtering
                if confidence < 0.6:
                    continue
                
                class_name = class_names.get(cls_id, f"object_{cls_id}")
                detections[class_name] = detections.get(class_name, 0) + 1
                self.detection_count += 1
                
            except Exception as e:
                print(f"Detection error: {e}")
                continue
        
        return detections
    
    def _scene_analysis_worker(self):
        """Enhanced scene analysis worker"""
        consecutive_failures = 0
        
        while self.running:
            try:
                task = self.scene_queue.get(timeout=1.0)
                if task is None:
                    break
                
                detections, timestamp = task
                
                # Analyze changes
                scene_changed, change_desc = self.scene_analyzer.analyze_scene_change(detections)
                
                if scene_changed or (time.time() - self.last_update_time > 10):  # Force update every 10s
                    self.current_scene_context = self.scene_analyzer.get_scene_context()
                    
                    # Generate description with retry
                    try:
                        new_description = self.ai.analyze_scene_continuously(
                            self.current_scene_context, 
                            change_desc
                        )
                        
                        if new_description and len(new_description.strip()) > 0:
                            self.current_description = new_description
                            self.last_update_time = timestamp
                            consecutive_failures = 0
                            print(f"📝 Scene update: {new_description[:80]}...")
                        else:
                            consecutive_failures += 1
                            print(f"⚠️ Empty scene description (failure {consecutive_failures})")
                            
                    except Exception as e:
                        consecutive_failures += 1
                        print(f"❌ Scene analysis error: {e}")
                        
                    # Restart AI if too many failures
                    if consecutive_failures >= 3:
                        print("🔄 Too many failures, restarting AI...")
                        threading.Thread(target=self._restart_ai, daemon=True).start()
                        consecutive_failures = 0
                
                self.scene_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker error: {e}")
                continue
    
    def _handle_user_input(self, key):
        """Enhanced input handling"""
        if key == 32:  # Space
            if not self.input_mode:
                self.input_mode = True
                self.current_input = ""
                print("💬 Ask about the current scene...")
            else:
                if self.current_input.strip():
                    print(f"❓ User question: {self.current_input}")
                    self._process_user_question(self.current_input.strip())
                self._exit_input_mode()
        
        elif key in [27, ord('q'), ord('Q')]:
            if self.input_mode:
                self._exit_input_mode()
            else:
                self.running = False
        
        elif key == ord('r') or key == ord('R'):
            print("🔄 Manually restarting AI system...")
            threading.Thread(target=self._restart_ai, daemon=True).start()
        
        elif key == 8 and self.input_mode:
            if len(self.current_input) > 0:
                self.current_input = self.current_input[:-1]
        
        elif self.input_mode and 32 <= key <= 126:
            if len(self.current_input) < 120:
                self.current_input += chr(key)
    
    def _process_user_question(self, question):
        """Enhanced question processing"""
        self.chat_history.append({
            'question': question,
            'answer': "Thinking about your question...",
            'timestamp': time.time(),
            'scene_context': self.current_scene_context
        })
        
        threading.Thread(
            target=self._generate_enhanced_answer,
            args=(question, len(self.chat_history) - 1),
            daemon=True
        ).start()
    
    def _generate_enhanced_answer(self, question, chat_index):
        """Enhanced answer generation"""
        try:
            print(f"🤔 Generating answer for: {question}")
            answer = self.ai.answer_contextual_question(question, self.current_scene_context)
            
            if answer and len(answer.strip()) > 0:
                self.chat_history[chat_index]['answer'] = answer
                print(f"💡 Answer: {answer[:80]}...")
            else:
                self.chat_history[chat_index]['answer'] = "I'm having trouble generating a response right now."
                
        except Exception as e:
            print(f"❌ Answer generation error: {e}")
            if chat_index < len(self.chat_history):
                self.chat_history[chat_index]['answer'] = f"Error generating response: {str(e)[:50]}"
    
    def _restart_ai(self):
        """Enhanced AI restart"""
        print("🔄 Restarting AI system...")
        self.ai.cleanup()
        time.sleep(3)
        self.ai = ImprovedGPTOSS()
        self._initialize_ai_system()
    
    def _exit_input_mode(self):
        """Exit input mode"""
        self.input_mode = False
        self.current_input = ""
    
    def _draw_enhanced_panel(self, img, x1, y1, x2, y2, color, border_color=None):
        """Enhanced panel drawing"""
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        if border_color:
            cv2.rectangle(img, (x1, y1), (x2, y2), border_color, 2)
    
    def _draw_text_wrapped_enhanced(self, img, text, x, y, font_scale, color, max_width, line_height=22):
        """Enhanced text wrapping"""
        if not text:
            return 0
            
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            (text_width, _), _ = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            
            if text_width <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                    current_line = word
                else:
                    lines.append(word)
        
        if current_line:
            lines.append(current_line)
        
        for i, line in enumerate(lines[:4]):  # Limit to 4 lines
            cv2.putText(img, line, (x, y + i * line_height), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)
        
        return len(lines[:4]) * line_height
    
    def _draw_enhanced_ui(self, frame, fps):
        """Enhanced UI with better information display"""
        h, w = frame.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Enhanced status bar
        self._draw_enhanced_panel(overlay, 12, 12, w-12, 90, self.colors['panel_dark'], self.colors['text_dark'])
        
        # AI status with more detail
        if self.ai.loading:
            status_text = "🔄 AI Loading..."
            status_color = self.colors['accent_orange']
        elif self.ai.ready:
            last_response_ago = time.time() - self.ai.last_successful_response
            if last_response_ago < 60:
                status_text = "🤖 AI Online & Responsive"
                status_color = self.colors['accent_green']
            else:
                status_text = f"🤖 AI Online (last response {last_response_ago:.0f}s ago)"
                status_color = self.colors['accent_orange']
        else:
            status_text = f"❌ AI Issue: {self.ai.error_message[:20]}"
            status_color = self.colors['accent_red']
        
        cv2.putText(overlay, status_text, (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
        
        # Enhanced stats
        total_objects = sum(self.scene_analyzer.current_scene.values())
        stats = f"Objects: {total_objects} | Detections: {self.detection_count} | FPS: {fps:.1f}"
        cv2.putText(overlay, stats, (25, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text_dim'], 1)
        
        # Enhanced scene panel
        scene_y = 100
        scene_height = 130
        self._draw_enhanced_panel(overlay, 12, scene_y, w-12, scene_y + scene_height, self.colors['panel_dark'])
        
        cv2.putText(overlay, "🎙️ Live Scene Analysis", (25, scene_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['accent_blue'], 2)
        
        # Scene context
        cv2.putText(overlay, self.current_scene_context, (25, scene_y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.colors['text_dim'], 1)
        
        # Scene description
        if self.current_description:
            self._draw_text_wrapped_enhanced(overlay, self.current_description, 25, scene_y + 75, 
                                           0.5, self.colors['text_bright'], w - 50, 20)
        
        # Enhanced chat panel
        chat_y = scene_y + scene_height + 15
        chat_height = h - chat_y - 140
        self._draw_enhanced_panel(overlay, 12, chat_y, w-12, chat_y + chat_height, self.colors['panel_light'])
        
        cv2.putText(overlay, "💬 Scene Q&A", (25, chat_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['accent_blue'], 2)
        
        # Chat history with better formatting
        y_pos = chat_y + 50
        for chat in list(self.chat_history)[-2:]:  # Show last 2 for better readability
            if y_pos + 90 > chat_y + chat_height:
                break
            
            # Question
            q_height = self._draw_text_wrapped_enhanced(overlay, f"Q: {chat['question']}", 25, y_pos, 
                                                      0.45, self.colors['text_bright'], w - 50, 18)
            y_pos += q_height + 8
            
            # Answer
            a_height = self._draw_text_wrapped_enhanced(overlay, f"A: {chat['answer']}", 25, y_pos, 
                                                      0.45, self.colors['accent_green'], w - 50, 18)
            y_pos += a_height + 20
        
        # Enhanced input panel
        input_y = h - 120
        input_color = self.colors['accent_blue'] if self.input_mode else self.colors['panel_dark']
        self._draw_enhanced_panel(overlay, 12, input_y, w-12, h-12, input_color)
        
        if self.input_mode:
            cv2.putText(overlay, "💬 Ask about the scene (SPACE=Send, ESC=Cancel):", 
                       (25, input_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text_bright'], 1)
            
            # Input with cursor
            current_time = time.time()
            if current_time - self.last_blink > 0.5:
                self.cursor_blink = not self.cursor_blink
                self.last_blink = current_time
            
            display_text = self.current_input + ("|" if self.cursor_blink else "")
            cv2.putText(overlay, display_text, (25, input_y + 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.colors['text_bright'], 2)
        else:
            cv2.putText(overlay, "💬 Press SPACE to ask about what you see in the camera", 
                       (25, input_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.colors['text_dim'], 1)
        
        # Controls
        controls = "SPACE=Ask Question | R=Restart AI | Q=Quit"
        cv2.putText(overlay, controls, (25, input_y + 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text_dark'], 1)
        
        # Blend
        alpha = 0.85
        result = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
        return result
    
    def run(self):
        """Enhanced main execution"""
        setup_result = self.setup()
        if setup_result is None:
            return False
        
        camera_id, device = setup_result
        
        # Start workers
        scene_thread = threading.Thread(target=self._scene_analysis_worker, daemon=True)
        scene_thread.start()
        
        # Setup window
        window_name = "Smart Vision Chat v2.1 - Enhanced Scene Analysis"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1400, 900)
        
        print("🎬 Enhanced scene analysis starting...")
        
        try:
            for result in self.yolo_model(source=camera_id, stream=True,
                                         conf=self.args.conf, imgsz=self.args.imgsz, 
                                         device=device, verbose=False):
                
                if not self.running:
                    break
                
                frame = result.plot()
                self.frame_count += 1
                elapsed_time = time.time() - self.start_time
                fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
                
                # Enhanced detection extraction
                detections = self._extract_detections_enhanced(result)
                
                # Queue for analysis
                try:
                    self.scene_queue.put_nowait((detections, time.time()))
                except queue.Full:
                    pass
                
                # Enhanced UI
                enhanced_frame = self._draw_enhanced_ui(frame, fps)
                cv2.imshow(window_name, enhanced_frame)
                
                # Input handling
                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    self._handle_user_input(key)
                
                if not self.running:
                    break
        
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        except Exception as e:
            print(f"\n❌ Runtime error: {e}")
        finally:
            print("\n🧹 Cleaning up enhanced system...")
            self.running = False
            self.scene_queue.put(None)
            cv2.destroyAllWindows()
            self.ai.cleanup()
            
            print("\n" + "=" * 55)
            print("✅ Enhanced Smart Vision Chat session completed")
            print(f"📊 Processed {self.frame_count} frames")
            print(f"🔍 Made {self.detection_count} object detections")
            print(f"💬 Handled {len(self.chat_history)} conversations")
            print("=" * 55)
        
        return True

def main():
    """Enhanced application entry point"""
    parser = argparse.ArgumentParser(
        description="Smart Vision Chat v2.1 - Enhanced Scene Analysis"
    )
    parser.add_argument("--model", default="yolov8n.pt", 
                       help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.6, 
                       help="Detection confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, 
                       help="Input image size")
    
    args = parser.parse_args()
    
    app = EnhancedVisionChat(args)
    success = app.run()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

