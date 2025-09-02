#!/usr/bin/env python3
"""
Smart Vision Chat - Minimal GPT-OSS Usage
Uses GPT-OSS 20B only for very simple tasks, with intelligent rule-based responses
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
    """Scene analysis with detailed object tracking"""
    
    def __init__(self):
        self.current_scene = {}
        self.last_update = 0
        self.update_interval = 4.0
        self.scene_history = deque(maxlen=5)
        
    def should_update_scene(self, new_detections):
        """Check if scene should be updated"""
        current_time = time.time()
        
        if current_time - self.last_update > self.update_interval:
            self._update_scene(new_detections)
            return True
        
        if self.current_scene != new_detections:
            self._update_scene(new_detections)
            return True
        
        return False
    
    def _update_scene(self, new_detections):
        """Update scene with history tracking"""
        self.scene_history.append({
            'time': time.time(),
            'objects': self.current_scene.copy()
        })
        self.current_scene = new_detections.copy()
        self.last_update = time.time()
    
    def get_scene_details(self):
        """Get detailed scene information"""
        if not self.current_scene:
            return {
                'summary': 'empty scene',
                'object_count': 0,
                'main_object': None,
                'details': 'No objects detected'
            }
        
        total_objects = sum(self.current_scene.values())
        main_object = max(self.current_scene.items(), key=lambda x: x[1])
        
        summary_parts = []
        for obj_type, count in self.current_scene.items():
            if count == 1:
                summary_parts.append(obj_type)
            else:
                summary_parts.append(f"{count} {obj_type}s")
        
        return {
            'summary': ", ".join(summary_parts),
            'object_count': total_objects,
            'main_object': main_object,
            'details': f"Total: {total_objects} objects",
            'objects': self.current_scene
        }

class MinimalGPTOSS:
    """Minimal GPT-OSS usage with smart fallbacks"""
    
    def __init__(self):
        self.base_url = "http://127.0.0.1:11434"
        self.model_name = "gpt-oss:20b"
        self.ready = False
        self.error_message = ""
        self.last_working_prompt = None
        
    def start_service(self):
        """Start GPT-OSS service"""
        print("🚀 Starting Minimal GPT-OSS Service...")
        
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
            return self._test_minimal()
            
        except Exception as e:
            self.error_message = str(e)
            return False
    
    def _test_minimal(self):
        """Test with absolute minimal prompt"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return False
            
            # Test with the simplest prompt possible
            test_result = self._try_simple_prompt("Hello")
            if test_result:
                self.ready = True
                self.last_working_prompt = "Hello"
                print(f"✅ Minimal GPT-OSS ready!")
                return True
            else:
                return False
                
        except Exception as e:
            self.error_message = str(e)
            return False
    
    def _try_simple_prompt(self, prompt):
        """Try the simplest possible prompt"""
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
    
    def get_simple_response(self, simple_word):
        """Get very simple response from GPT-OSS"""
        if not self.ready:
            return None
        
        # Only try with single words or very simple phrases
        simple_prompts = [
            simple_word,
            f"Yes {simple_word}",
            f"I see {simple_word}",
        ]
        
        for prompt in simple_prompts:
            result = self._try_simple_prompt(prompt)
            if result:
                try:
                    payload = {
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "num_predict": 15,
                            "temperature": 0.8
                        }
                    }
                    
                    response = requests.post(
                        f"{self.base_url}/api/generate",
                        json=payload,
                        timeout=8
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        generated = result.get("response", "").strip()
                        if len(generated) > 2:
                            return generated
                            
                except Exception:
                    continue
        
        return None
    
    def cleanup(self):
        """Cleanup"""
        self.ready = False

class SmartRuleBasedResponder:
    """Smart rule-based responses for when GPT-OSS fails"""
    
    def __init__(self):
        self.response_templates = {
            'scene_description': {
                'person': [
                    "I can see a person in the camera view.",
                    "There's someone visible in the scene.",
                    "A person is present in the camera frame.",
                ],
                'people': [
                    "I can see multiple people in the view.",
                    "Several people are visible in the scene.",
                    "There are people in the camera frame.",
                ],
                'empty': [
                    "The camera shows an empty space.",
                    "No objects are currently detected.",
                    "The view appears to be clear.",
                ],
                'objects': [
                    "I can see some objects in the scene.",
                    "Various items are visible in the camera view.",
                    "Multiple objects are detected in the frame.",
                ]
            },
            'questions': {
                'what_wearing': {
                    'person': "I can detect a person but cannot determine specific clothing details from object detection alone.",
                    'no_person': "No person is visible in the current view to assess clothing."
                },
                'who_person': {
                    'person': "I can detect that a person is present, but I cannot identify specific individuals.",
                    'no_person': "No people are currently detected in the camera view."
                },
                'where_objects': {
                    'has_objects': "The detected objects are positioned within the camera's field of view: {objects}.",
                    'no_objects': "No objects are currently detected to locate."
                },
                'how_many': {
                    'has_objects': "Based on object detection, I can count: {details}.",
                    'no_objects': "No objects are currently detected to count."
                },
                'what_see': {
                    'has_objects': "In the current view, I can detect: {objects}.",
                    'no_objects': "The camera currently shows an empty scene with no detected objects."
                }
            }
        }
    
    def generate_scene_description(self, scene_details, gpt_oss_response=None):
        """Generate intelligent scene description"""
        if gpt_oss_response and len(gpt_oss_response) > 5:
            # Use GPT-OSS response if it looks good
            return f"🤖 {gpt_oss_response}"
        
        # Use rule-based response
        if scene_details['object_count'] == 0:
            return self._random_choice(self.response_templates['scene_description']['empty'])
        elif 'person' in scene_details['objects']:
            person_count = scene_details['objects']['person']
            if person_count > 1:
                return self._random_choice(self.response_templates['scene_description']['people'])
            else:
                return self._random_choice(self.response_templates['scene_description']['person'])
        else:
            return self._random_choice(self.response_templates['scene_description']['objects'])
    
    def answer_question(self, question, scene_details, gpt_oss_response=None):
        """Generate intelligent answers"""
        if gpt_oss_response and len(gpt_oss_response) > 5:
            return f"🤖 {gpt_oss_response}"
        
        question_lower = question.lower()
        
        # Analyze question type and scene
        if "what" in question_lower and ("wearing" in question_lower or "clothes" in question_lower):
            if 'person' in scene_details['objects']:
                return self.response_templates['questions']['what_wearing']['person']
            else:
                return self.response_templates['questions']['what_wearing']['no_person']
        
        elif "who" in question_lower:
            if 'person' in scene_details['objects']:
                return self.response_templates['questions']['who_person']['person']
            else:
                return self.response_templates['questions']['who_person']['no_person']
        
        elif "where" in question_lower:
            if scene_details['object_count'] > 0:
                return self.response_templates['questions']['where_objects']['has_objects'].format(
                    objects=scene_details['summary']
                )
            else:
                return self.response_templates['questions']['where_objects']['no_objects']
        
        elif "how many" in question_lower or "count" in question_lower:
            if scene_details['object_count'] > 0:
                return self.response_templates['questions']['how_many']['has_objects'].format(
                    details=scene_details['details']
                )
            else:
                return self.response_templates['questions']['how_many']['no_objects']
        
        elif "what" in question_lower and ("see" in question_lower or "detect" in question_lower):
            if scene_details['object_count'] > 0:
                return self.response_templates['questions']['what_see']['has_objects'].format(
                    objects=scene_details['summary']
                )
            else:
                return self.response_templates['questions']['what_see']['no_objects']
        
        else:
            # General question
            if scene_details['object_count'] > 0:
                return f"Based on the current scene with {scene_details['summary']}, I can provide object detection information but may not have specific details for that question."
            else:
                return "The camera currently shows an empty scene, so I don't have specific objects to comment on."
    
    def _random_choice(self, options):
        """Choose response based on time (pseudo-random)"""
        import time
        index = int(time.time()) % len(options)
        return options[index]

class MinimalVisionChat:
    """Vision chat with minimal GPT-OSS and smart rules"""
    
    def __init__(self, args):
        self.args = args
        self.yolo_model = None
        self.gpt_oss = MinimalGPTOSS()
        self.rule_responder = SmartRuleBasedResponder()
        self.scene_analyzer = SceneAnalyzer()
        
        # UI State
        self.current_input = ""
        self.input_mode = False
        self.cursor_blink = True
        self.last_blink = time.time()
        
        # App state
        self.chat_history = deque(maxlen=5)
        self.current_description = "Starting intelligent vision system..."
        self.current_scene_details = {}
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.gpt_oss_attempts = 0
        self.gpt_oss_successes = 0
        self.rule_based_responses = 0
        
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
        print("🎯 Smart Vision Chat - Minimal GPT-OSS + Intelligent Rules")
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
        
        # Start GPT-OSS
        gpt_thread = threading.Thread(target=self.gpt_oss.start_service, daemon=True)
        gpt_thread.start()
        
        print("=" * 65)
        print("🎥 Camera ready with hybrid AI system")
        print("🤖 GPT-OSS 20B: Minimal usage for simple tasks")
        print("🧠 Smart Rules: Intelligent responses for complex questions")
        print("💬 Press SPACE to ask questions")
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
        """Extract detections with confidence filtering"""
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
        """Background scene analysis with hybrid approach"""
        while self.running:
            try:
                task = self.scene_queue.get(timeout=1.0)
                if task is None:
                    break
                
                detections = task
                
                if self.scene_analyzer.should_update_scene(detections):
                    scene_details = self.scene_analyzer.get_scene_details()
                    self.current_scene_details = scene_details
                    
                    # Try GPT-OSS first with minimal prompt
                    gpt_response = None
                    if self.gpt_oss.ready and scene_details['main_object']:
                        self.gpt_oss_attempts += 1
                        main_obj = scene_details['main_object'][0]  # Just the object name
                        gpt_response = self.gpt_oss.get_simple_response(main_obj)
                        
                        if gpt_response:
                            self.gpt_oss_successes += 1
                            print(f"🤖 GPT-OSS: {gpt_response}")
                        else:
                            self.rule_based_responses += 1
                    else:
                        self.rule_based_responses += 1
                    
                    # Generate description (hybrid approach)
                    description = self.rule_responder.generate_scene_description(
                        scene_details, gpt_response
                    )
                    
                    self.current_description = description
                    print(f"📝 Scene: {description}")
                
                self.scene_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Scene worker error: {e}")
                continue
    
    def _handle_input(self, key):
        """Handle user input"""
        if key == 32:  # Space
            if not self.input_mode:
                self.input_mode = True
                self.current_input = ""
                print("💬 Ask about the scene (hybrid AI will respond)...")
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
            print("🔄 Restarting GPT-OSS...")
            self.gpt_oss.cleanup()
            time.sleep(2)
            self.gpt_oss = MinimalGPTOSS()
            threading.Thread(target=self.gpt_oss.start_service, daemon=True).start()
        
        elif key == 8 and self.input_mode:  # Backspace
            if len(self.current_input) > 0:
                self.current_input = self.current_input[:-1]
        
        elif self.input_mode and 32 <= key <= 126:
            if len(self.current_input) < 100:
                self.current_input += chr(key)
    
    def _process_question(self, question):
        """Process user question"""
        print(f"❓ Question: {question}")
        
        self.chat_history.append({
            'question': question,
            'answer': "Analyzing with hybrid AI...",
            'timestamp': time.time(),
            'response_type': 'processing'
        })
        
        threading.Thread(
            target=self._generate_hybrid_answer,
            args=(question, len(self.chat_history) - 1),
            daemon=True
        ).start()
    
    def _generate_hybrid_answer(self, question, chat_index):
        """Generate answer using hybrid approach"""
        try:
            # Try GPT-OSS first with very simple prompt
            gpt_response = None
            if self.gpt_oss.ready and self.current_scene_details.get('main_object'):
                self.gpt_oss_attempts += 1
                
                # Try to get a simple response from GPT-OSS
                simple_word = self.current_scene_details['main_object'][0]
                gpt_response = self.gpt_oss.get_simple_response(simple_word)
                
                if gpt_response:
                    self.gpt_oss_successes += 1
                else:
                    self.rule_based_responses += 1
            else:
                self.rule_based_responses += 1
            
            # Generate intelligent answer
            answer = self.rule_responder.answer_question(
                question, self.current_scene_details, gpt_response
            )
            
            # Determine response type
            if gpt_response and "🤖" in answer:
                response_type = 'gpt_oss'
            else:
                response_type = 'rule_based'
            
            # Update chat
            self.chat_history[chat_index]['answer'] = answer
            self.chat_history[chat_index]['response_type'] = response_type
            
            print(f"💡 Answer: {answer}")
            
        except Exception as e:
            print(f"Answer generation error: {e}")
            if chat_index < len(self.chat_history):
                self.chat_history[chat_index]['answer'] = "Error generating response"
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
        """Draw comprehensive UI"""
        h, w = frame.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Status bar with hybrid AI info
        cv2.rectangle(overlay, (8, 8), (w-8, 90), self.colors['panel'], -1)
        cv2.rectangle(overlay, (8, 8), (w-8, 90), self.colors['dark'], 2)
        
        # Hybrid AI status
        if self.gpt_oss.ready:
            gpt_rate = (self.gpt_oss_successes / max(1, self.gpt_oss_attempts)) * 100
            status = f"🤖 GPT-OSS: Ready ({gpt_rate:.0f}% success) | 🧠 Smart Rules: Active"
            color = self.colors['green']
        else:
            status = "🔄 GPT-OSS: Starting | 🧠 Smart Rules: Ready"
            color = self.colors['orange']
        
        cv2.putText(overlay, status, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Response stats
        total_responses = self.gpt_oss_successes + self.rule_based_responses
        stats = f"Objects: {self.current_scene_details.get('object_count', 0)} | FPS: {fps:.1f} | Responses: GPT-OSS:{self.gpt_oss_successes} Rules:{self.rule_based_responses}"
        cv2.putText(overlay, stats, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['gray'], 1)
        
        # Current scene info
        if self.current_scene_details.get('summary'):
            scene_text = f"Scene: {self.current_scene_details['summary']}"
            cv2.putText(overlay, scene_text, (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['blue'], 1)
        
        # Scene description panel
        scene_y = 100
        cv2.rectangle(overlay, (8, scene_y), (w-8, scene_y + 95), self.colors['panel'], -1)
        cv2.rectangle(overlay, (8, scene_y), (w-8, scene_y + 95), self.colors['dark'], 2)
        
        cv2.putText(overlay, "🎙️ Hybrid AI Scene Analysis", (15, scene_y + 22), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.colors['accent'], 2)
        
        self._draw_text_wrap(overlay, self.current_description, 15, scene_y + 45, 
                           0.45, self.colors['white'], w - 30, 16)
        
        # Chat history with response type indicators
        chat_y = 205
        chat_height = h - chat_y - 110
        cv2.rectangle(overlay, (8, chat_y), (w-8, chat_y + chat_height), self.colors['panel_light'], -1)
        cv2.rectangle(overlay, (8, chat_y), (w-8, chat_y + chat_height), self.colors['dark'], 2)
        
        cv2.putText(overlay, "💬 Hybrid Q&A (🤖=GPT-OSS, 🧠=Rules)", (15, chat_y + 22), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['accent'], 2)
        
        # Show last 2 chats with indicators
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
            if response_type == 'gpt_oss':
                answer_color = self.colors['green']
            elif response_type == 'rule_based':
                answer_color = self.colors['blue']
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
            cv2.putText(overlay, "💬 Ask hybrid AI (SPACE=Send, ESC=Cancel):", 
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
            cv2.putText(overlay, "💬 Press SPACE to ask hybrid AI (GPT-OSS + Smart Rules)", 
                       (15, input_y + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['gray'], 1)
        
        # Controls
        cv2.putText(overlay, "SPACE=Ask | R=Restart GPT-OSS | Q=Quit", (15, input_y + 75), 
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
        window_name = "Smart Vision Chat - Minimal GPT-OSS + Smart Rules"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1320, 820)
        
        print("🎬 Starting hybrid vision chat...")
        
        try:
            for result in self.yolo_model(source=camera_id, stream=True,
                                         conf=self.args.conf, imgsz=self.args.imgsz, 
                                         device=device, verbose=False):
                
                if not self.running:
                    break
                
                frame = result.plot()
                self.frame_count += 1
                
                elapsed = time.time() - self.start_time
                fps = self.frame_count / elapsed if elapsed > 0 else 0
                
                detections = self._extract_detections(result)
                
                try:
                    self.scene_queue.put_nowait(detections)
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
            self.gpt_oss.cleanup()
            
            print("\n" + "=" * 65)
            print("✅ Hybrid Vision Chat completed")
            print(f"📊 Frames: {self.frame_count}")
            print(f"💬 Conversations: {len(self.chat_history)}")
            print(f"🤖 GPT-OSS Success: {self.gpt_oss_successes}/{self.gpt_oss_attempts}")
            print(f"🧠 Rule-based Responses: {self.rule_based_responses}")
            print("=" * 65)
        
        return True

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Smart Vision Chat - Minimal GPT-OSS + Smart Rules")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model")
    parser.add_argument("--conf", type=float, default=0.6, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    
    args = parser.parse_args()
    
    app = MinimalVisionChat(args)
    success = app.run()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

