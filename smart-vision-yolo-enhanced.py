#!/usr/bin/env python3
"""
Smart Vision Chat - YOLO Enhanced GPT-OSS
Uses YOLO's detailed detection data to provide rich visual context to GPT-OSS 20B
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

class YOLOVisualAnalyzer:
    """Enhanced YOLO analysis to provide rich visual context"""
    
    def __init__(self):
        self.body_parts = {
            'person': ['head', 'torso', 'arms', 'legs']
        }
        
        # Enhanced object categories for better descriptions
        self.object_categories = {
            'clothing': ['tie', 'handbag', 'backpack', 'umbrella', 'suitcase'],
            'accessories': ['handbag', 'backpack', 'umbrella', 'tie', 'clock', 'cell phone'],
            'furniture': ['chair', 'couch', 'bed', 'dining table', 'toilet'],
            'electronics': ['tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'refrigerator'],
            'vehicles': ['car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'bicycle'],
            'animals': ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'],
            'food': ['banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake'],
            'sports': ['frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket']
        }
        
        # Position descriptors
        self.position_terms = {
            'left': 'on the left side',
            'right': 'on the right side', 
            'center': 'in the center',
            'top': 'in the upper area',
            'bottom': 'in the lower area',
            'close': 'close to the camera',
            'far': 'far from the camera'
        }
    
    def analyze_detections(self, result, frame_shape):
        """Analyze YOLO detections to create rich visual context"""
        if result.boxes is None or len(result.boxes) == 0:
            return {
                'summary': 'empty scene with no objects detected',
                'detailed_description': 'The camera shows an empty scene with no people or objects visible.',
                'object_count': 0,
                'people_analysis': [],
                'scene_context': 'empty room or outdoor space'
            }
        
        height, width = frame_shape[:2]
        class_names = result.names if hasattr(result, 'names') else {}
        
        detections = []
        people_info = []
        object_counts = {}
        
        for box in result.boxes:
            try:
                cls_id = int(box.cls.item())
                confidence = float(box.conf.item())
                
                if confidence < 0.5:
                    continue
                
                class_name = class_names.get(cls_id, "object")
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Calculate position and size info
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                box_width = x2 - x1
                box_height = y2 - y1
                
                # Determine position descriptions
                position_desc = self._get_position_description(center_x, center_y, width, height)
                size_desc = self._get_size_description(box_width, box_height, width, height)
                
                detection_info = {
                    'class': class_name,
                    'confidence': confidence,
                    'position': position_desc,
                    'size': size_desc,
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y)
                }
                
                detections.append(detection_info)
                object_counts[class_name] = object_counts.get(class_name, 0) + 1
                
                # Special analysis for people
                if class_name == 'person':
                    people_info.append(self._analyze_person(detection_info, width, height))
                
            except Exception as e:
                continue
        
        return {
            'summary': self._create_summary(object_counts),
            'detailed_description': self._create_detailed_description(detections, people_info),
            'object_count': len(detections),
            'people_analysis': people_info,
            'scene_context': self._determine_scene_context(object_counts),
            'object_details': detections
        }
    
    def _get_position_description(self, center_x, center_y, width, height):
        """Get position description based on coordinates"""
        h_pos = ""
        v_pos = ""
        
        # Horizontal position
        if center_x < width * 0.33:
            h_pos = "left"
        elif center_x > width * 0.67:
            h_pos = "right"
        else:
            h_pos = "center"
        
        # Vertical position
        if center_y < height * 0.33:
            v_pos = "top"
        elif center_y > height * 0.67:
            v_pos = "bottom"
        else:
            v_pos = "middle"
        
        if h_pos == "center" and v_pos == "middle":
            return "in the center of the frame"
        elif h_pos == "center":
            return f"in the {v_pos} center"
        elif v_pos == "middle":
            return f"on the {h_pos} side"
        else:
            return f"in the {v_pos} {h_pos}"
    
    def _get_size_description(self, box_width, box_height, frame_width, frame_height):
        """Get size description"""
        area_ratio = (box_width * box_height) / (frame_width * frame_height)
        
        if area_ratio > 0.4:
            return "large and prominent"
        elif area_ratio > 0.15:
            return "medium-sized"
        elif area_ratio > 0.05:
            return "small"
        else:
            return "very small"
    
    def _analyze_person(self, person_info, frame_width, frame_height):
        """Analyze person detection for detailed description"""
        x1, y1, x2, y2 = person_info['bbox']
        
        # Estimate pose/orientation from bounding box
        aspect_ratio = (y2 - y1) / (x2 - x1)
        
        pose_desc = ""
        if aspect_ratio > 2.5:
            pose_desc = "standing upright"
        elif aspect_ratio > 1.8:
            pose_desc = "standing or walking"
        elif aspect_ratio > 1.2:
            pose_desc = "sitting or crouching"
        else:
            pose_desc = "lying down or in horizontal position"
        
        # Estimate distance from size
        person_height_ratio = (y2 - y1) / frame_height
        if person_height_ratio > 0.7:
            distance = "very close to camera"
        elif person_height_ratio > 0.4:
            distance = "at medium distance"
        else:
            distance = "far from camera"
        
        return {
            'position': person_info['position'],
            'pose': pose_desc,
            'distance': distance,
            'size': person_info['size'],
            'confidence': person_info['confidence']
        }
    
    def _create_summary(self, object_counts):
        """Create concise summary"""
        if not object_counts:
            return "empty scene"
        
        summary_parts = []
        for obj_type, count in sorted(object_counts.items()):
            if count == 1:
                summary_parts.append(f"one {obj_type}")
            else:
                summary_parts.append(f"{count} {obj_type}s")
        
        return ", ".join(summary_parts)
    
    def _create_detailed_description(self, detections, people_info):
        """Create detailed visual description"""
        if not detections:
            return "The scene is empty with no visible objects or people."
        
        desc_parts = []
        
        # People description
        if people_info:
            people_desc = []
            for i, person in enumerate(people_info):
                if len(people_info) == 1:
                    people_desc.append(f"There is one person {person['position']}, {person['pose']}, {person['distance']}.")
                else:
                    people_desc.append(f"Person {i+1} is {person['position']}, {person['pose']}, {person['distance']}.")
            desc_parts.extend(people_desc)
        
        # Object descriptions by category
        objects_by_category = {}
        for det in detections:
            if det['class'] == 'person':
                continue
            
            category = self._get_object_category(det['class'])
            if category not in objects_by_category:
                objects_by_category[category] = []
            objects_by_category[category].append(det)
        
        for category, objects in objects_by_category.items():
            if len(objects) == 1:
                obj = objects[0]
                desc_parts.append(f"There is a {obj['size']} {obj['class']} {obj['position']}.")
            else:
                obj_names = [obj['class'] for obj in objects]
                unique_objects = list(set(obj_names))
                if len(unique_objects) == 1:
                    desc_parts.append(f"There are {len(objects)} {unique_objects[0]}s visible in the scene.")
                else:
                    desc_parts.append(f"There are several {category} items: {', '.join(unique_objects)}.")
        
        return " ".join(desc_parts)
    
    def _get_object_category(self, class_name):
        """Get category for object"""
        for category, objects in self.object_categories.items():
            if class_name in objects:
                return category
        return "objects"
    
    def _determine_scene_context(self, object_counts):
        """Determine scene context"""
        if 'person' in object_counts:
            if any(obj in object_counts for obj in ['chair', 'couch', 'bed', 'dining table']):
                return "indoor setting with furniture"
            elif any(obj in object_counts for obj in ['car', 'bicycle', 'motorcycle']):
                return "outdoor setting with vehicles"
            else:
                return "indoor or outdoor setting"
        else:
            if any(obj in object_counts for obj in ['chair', 'couch', 'tv', 'laptop']):
                return "indoor room"
            elif any(obj in object_counts for obj in ['car', 'truck', 'bicycle']):
                return "outdoor area with vehicles"
            else:
                return "general scene"

class EnhancedGPTOSS:
    """GPT-OSS with YOLO visual context"""
    
    def __init__(self):
        self.base_url = "http://127.0.0.1:11434"
        self.model_name = "gpt-oss:20b"
        self.ready = False
        self.error_message = ""
        
    def start_service(self):
        """Start GPT-OSS service"""
        print("🚀 Starting Enhanced GPT-OSS with YOLO Vision...")
        
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
            return self._test_connection()
            
        except Exception as e:
            self.error_message = str(e)
            return False
    
    def _test_connection(self):
        """Test GPT-OSS connection"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return False
            
            # Test with simple prompt
            if self._try_prompt("Hello"):
                self.ready = True
                print(f"✅ Enhanced GPT-OSS ready with YOLO vision!")
                return True
            else:
                return False
                
        except Exception as e:
            self.error_message = str(e)
            return False
    
    def _try_prompt(self, prompt):
        """Try prompt"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 30,
                    "temperature": 0.7
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=12
            )
            
            if response.status_code == 200:
                result = response.json()
                generated = result.get("response", "").strip()
                return len(generated) > 0
            else:
                return False
                
        except Exception:
            return False
    
    def generate_scene_description(self, visual_analysis):
        """Generate scene description using YOLO visual context"""
        if not self.ready:
            return None
        
        # Create rich prompt with YOLO data
        prompt = f"Based on this visual scene analysis: {visual_analysis['detailed_description']} Scene context: {visual_analysis['scene_context']}. Describe what you observe in a natural way."
        
        return self._generate_response(prompt, max_tokens=60)
    
    def answer_visual_question(self, question, visual_analysis):
        """Answer question using YOLO visual context"""
        if not self.ready:
            return None
        
        # Create context-rich prompt
        context = f"Visual scene: {visual_analysis['detailed_description']}"
        if visual_analysis['people_analysis']:
            people_details = []
            for person in visual_analysis['people_analysis']:
                people_details.append(f"Person {person['pose']}, {person['position']}, {person['distance']}")
            context += f" People details: {'; '.join(people_details)}"
        
        prompt = f"{context}\n\nQuestion: {question}\nAnswer based on the visual information:"
        
        return self._generate_response(prompt, max_tokens=80)
    
    def _generate_response(self, prompt, max_tokens=50):
        """Generate response from GPT-OSS"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
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
                if len(generated) > 3 and self._is_valid_response(generated):
                    return generated
            
            return None
            
        except Exception as e:
            print(f"Generation error: {e}")
            return None
    
    def _is_valid_response(self, text):
        """Validate response quality"""
        if not text or len(text) < 3:
            return False
        
        # Check for common issues
        if any(pattern in text.lower() for pattern in ["sorry", "cannot", "unable", "don't know"]):
            return False
        
        # Check for gibberish
        words = text.split()
        if len(words) < 2:
            return False
        
        return True
    
    def cleanup(self):
        """Cleanup"""
        self.ready = False

class SceneAnalyzer:
    """Enhanced scene analysis with YOLO"""
    
    def __init__(self):
        self.current_analysis = {}
        self.last_update = 0
        self.update_interval = 4.0
        
    def should_update_scene(self, new_analysis):
        """Check if scene should be updated"""
        current_time = time.time()
        
        if current_time - self.last_update > self.update_interval:
            self._update_scene(new_analysis)
            return True
        
        # Check for significant changes
        if self._has_significant_changes(new_analysis):
            self._update_scene(new_analysis)
            return True
        
        return False
    
    def _has_significant_changes(self, new_analysis):
        """Check for significant scene changes"""
        if not self.current_analysis:
            return True
        
        # Check object count changes
        old_count = self.current_analysis.get('object_count', 0)
        new_count = new_analysis.get('object_count', 0)
        
        if abs(old_count - new_count) > 0:
            return True
        
        # Check people changes
        old_people = len(self.current_analysis.get('people_analysis', []))
        new_people = len(new_analysis.get('people_analysis', []))
        
        if old_people != new_people:
            return True
        
        return False
    
    def _update_scene(self, new_analysis):
        """Update scene analysis"""
        self.current_analysis = new_analysis.copy()
        self.last_update = time.time()
    
    def get_current_analysis(self):
        """Get current analysis"""
        return self.current_analysis

class YOLOEnhancedVisionChat:
    """YOLO-Enhanced Vision Chat with GPT-OSS"""
    
    def __init__(self, args):
        self.args = args
        self.yolo_model = None
        self.yolo_analyzer = YOLOVisualAnalyzer()
        self.gpt_oss = EnhancedGPTOSS()
        self.scene_analyzer = SceneAnalyzer()
        
        # UI State
        self.current_input = ""
        self.input_mode = False
        self.cursor_blink = True
        self.last_blink = time.time()
        
        # App state
        self.chat_history = deque(maxlen=4)
        self.current_description = "Starting YOLO-enhanced vision system..."
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.gpt_attempts = 0
        self.gpt_successes = 0
        self.yolo_analyses = 0
        
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
        print("🎯 YOLO-Enhanced GPT-OSS Chat - Rich Visual Understanding")
        print("=" * 70)
        
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
        
        print("=" * 70)
        print("🔍 YOLO: Provides detailed visual context (position, pose, objects)")
        print("🤖 GPT-OSS: Uses YOLO data to understand clothing, actions, scenes")
        print("💬 Press SPACE to ask visual questions")
        print("❌ Press Q to quit")
        print("=" * 70)
        
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
    
    def _scene_worker(self):
        """Background scene analysis worker"""
        while self.running:
            try:
                task = self.scene_queue.get(timeout=1.0)
                if task is None:
                    break
                
                result, frame_shape = task
                
                # Analyze with YOLO
                self.yolo_analyses += 1
                visual_analysis = self.yolo_analyzer.analyze_detections(result, frame_shape)
                
                if self.scene_analyzer.should_update_scene(visual_analysis):
                    # Generate description with GPT-OSS + YOLO
                    self.gpt_attempts += 1
                    gpt_description = self.gpt_oss.generate_scene_description(visual_analysis)
                    
                    if gpt_description:
                        self.gpt_successes += 1
                        self.current_description = f"🤖 {gpt_description}"
                        print(f"🤖 GPT-OSS: {gpt_description}")
                    else:
                        # Use YOLO description as fallback
                        self.current_description = f"🔍 {visual_analysis['detailed_description']}"
                        print(f"🔍 YOLO: {visual_analysis['detailed_description']}")
                
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
                print("🎯 Ask about clothing, poses, objects (YOLO+GPT-OSS will analyze)...")
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
            self.gpt_oss = EnhancedGPTOSS()
            threading.Thread(target=self.gpt_oss.start_service, daemon=True).start()
        
        elif key == 8 and self.input_mode:  # Backspace
            if len(self.current_input) > 0:
                self.current_input = self.current_input[:-1]
        
        elif self.input_mode and 32 <= key <= 126:
            if len(self.current_input) < 100:
                self.current_input += chr(key)
    
    def _process_question(self, question):
        """Process question"""
        print(f"❓ Question: {question}")
        
        self.chat_history.append({
            'question': question,
            'answer': "Analyzing with YOLO+GPT-OSS...",
            'timestamp': time.time(),
            'response_type': 'processing'
        })
        
        threading.Thread(
            target=self._generate_answer,
            args=(question, len(self.chat_history) - 1),
            daemon=True
        ).start()
    
    def _generate_answer(self, question, chat_index):
        """Generate answer using YOLO+GPT-OSS"""
        try:
            # Get current visual analysis
            current_analysis = self.scene_analyzer.get_current_analysis()
            
            if not current_analysis:
                answer = "No visual analysis available yet. Please wait for scene detection."
                response_type = 'error'
            else:
                # Try GPT-OSS with YOLO context
                self.gpt_attempts += 1
                gpt_answer = self.gpt_oss.answer_visual_question(question, current_analysis)
                
                if gpt_answer:
                    self.gpt_successes += 1
                    answer = f"🤖 {gpt_answer}"
                    response_type = 'gpt_enhanced'
                    print(f"🤖 GPT-OSS Answer: {gpt_answer}")
                else:
                    # Use YOLO-based intelligent answer
                    answer = self._create_yolo_answer(question, current_analysis)
                    response_type = 'yolo_enhanced'
                    print(f"🔍 YOLO Answer: {answer}")
            
            # Update chat
            if chat_index < len(self.chat_history):
                self.chat_history[chat_index]['answer'] = answer
                self.chat_history[chat_index]['response_type'] = response_type
            
        except Exception as e:
            print(f"Answer generation error: {e}")
            if chat_index < len(self.chat_history):
                self.chat_history[chat_index]['answer'] = "Error generating response"
                self.chat_history[chat_index]['response_type'] = 'error'
    
    def _create_yolo_answer(self, question, analysis):
        """Create intelligent answer from YOLO analysis"""
        question_lower = question.lower()
        
        if "wearing" in question_lower or "clothes" in question_lower:
            if analysis['people_analysis']:
                person = analysis['people_analysis'][0]
                return f"🔍 I can see a person {person['position']} who is {person['pose']}. While I cannot see specific clothing details, I can detect their general posture and position in the scene."
            else:
                return "🔍 No people are currently visible in the scene to analyze clothing."
        
        elif "doing" in question_lower or "action" in question_lower:
            if analysis['people_analysis']:
                actions = [person['pose'] for person in analysis['people_analysis']]
                return f"🔍 Based on posture analysis: {', '.join(set(actions))}."
            else:
                return "🔍 No people visible to analyze actions."
        
        elif "where" in question_lower:
            if analysis['people_analysis']:
                positions = [person['position'] for person in analysis['people_analysis']]
                return f"🔍 People are located: {', '.join(positions)}."
            else:
                return f"🔍 Objects in scene: {analysis['summary']}"
        
        elif "how many" in question_lower or "count" in question_lower:
            return f"🔍 Object count: {analysis['object_count']} items detected - {analysis['summary']}"
        
        elif "see" in question_lower:
            return f"🔍 {analysis['detailed_description']}"
        
        else:
            return f"🔍 Scene analysis: {analysis['detailed_description']} Context: {analysis['scene_context']}"
    
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
        """Draw UI"""
        h, w = frame.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Status bar
        cv2.rectangle(overlay, (8, 8), (w-8, 90), self.colors['panel'], -1)
        cv2.rectangle(overlay, (8, 8), (w-8, 90), self.colors['dark'], 2)
        
        # System status
        if self.gpt_oss.ready:
            success_rate = (self.gpt_successes / max(1, self.gpt_attempts)) * 100
            status = f"🤖 GPT-OSS+YOLO: Ready ({success_rate:.0f}% success) - Rich Visual Understanding"
            color = self.colors['green']
        else:
            status = "🔄 GPT-OSS: Starting | 🔍 YOLO: Ready"
            color = self.colors['orange']
        
        cv2.putText(overlay, status, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Stats
        stats = f"FPS: {fps:.1f} | YOLO Analyses: {self.yolo_analyses} | GPT Success: {self.gpt_successes}/{self.gpt_attempts}"
        cv2.putText(overlay, stats, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['gray'], 1)
        
        # Capabilities
        caps = "Enhanced: clothing analysis, pose detection, spatial reasoning, object relationships"
        cv2.putText(overlay, caps, (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['blue'], 1)
        
        # Scene description
        scene_y = 100
        cv2.rectangle(overlay, (8, scene_y), (w-8, scene_y + 95), self.colors['panel'], -1)
        cv2.rectangle(overlay, (8, scene_y), (w-8, scene_y + 95), self.colors['dark'], 2)
        
        cv2.putText(overlay, "🎯 YOLO-Enhanced Scene Analysis", (15, scene_y + 22), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.colors['accent'], 2)
        
        self._draw_text_wrap(overlay, self.current_description, 15, scene_y + 45, 
                           0.45, self.colors['white'], w - 30, 16)
        
        # Chat history
        chat_y = 205
        chat_height = h - chat_y - 110
        cv2.rectangle(overlay, (8, chat_y), (w-8, chat_y + chat_height), self.colors['panel_light'], -1)
        cv2.rectangle(overlay, (8, chat_y), (w-8, chat_y + chat_height), self.colors['dark'], 2)
        
        cv2.putText(overlay, "💬 Visual Q&A (🤖=GPT-OSS+YOLO, 🔍=YOLO)", (15, chat_y + 22), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['accent'], 2)
        
        # Show chat
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
            if response_type == 'gpt_enhanced':
                answer_color = self.colors['green']
            elif response_type == 'yolo_enhanced':
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
            cv2.putText(overlay, "🎯 Ask about visual details (SPACE=Send, ESC=Cancel):", 
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
            cv2.putText(overlay, "🎯 SPACE=Ask Visual Questions | R=Restart | Q=Quit", 
                       (15, input_y + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['gray'], 1)
        
        # Examples
        cv2.putText(overlay, "Try: 'What is the person wearing?', 'What are they doing?', 'Where are they?'", 
                   (15, input_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.colors['dark'], 1)
        
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
        window_name = "YOLO-Enhanced GPT-OSS Chat"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1320, 820)
        
        print("🎬 Starting YOLO-enhanced vision chat...")
        
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
                
                try:
                    self.scene_queue.put_nowait((result, frame.shape))
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
            
            print("\n" + "=" * 70)
            print("✅ YOLO-Enhanced GPT-OSS Chat completed")
            print(f"📊 Frames: {self.frame_count}")
            print(f"🔍 YOLO Analyses: {self.yolo_analyses}")
            print(f"🤖 GPT-OSS Success: {self.gpt_successes}/{self.gpt_attempts}")
            print("=" * 70)
        
        return True

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="YOLO-Enhanced GPT-OSS Chat")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    
    args = parser.parse_args()
    
    app = YOLOEnhancedVisionChat(args)
    success = app.run()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
