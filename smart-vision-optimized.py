#!/usr/bin/env python3
"""
Smart Vision Chat - Optimized Version with Better Prompts
Uses Llama 3.2 with optimized prompts for object detection and scene analysis
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
    """Scene analysis with smart updates"""
    
    def __init__(self):
        self.current_scene = {}
        self.last_update = 0
        self.update_interval = 4.0
        
    def should_update_scene(self, new_detections):
        """Check if scene should be updated"""
        current_time = time.time()
        
        if current_time - self.last_update > self.update_interval:
            self.current_scene = new_detections.copy()
            self.last_update = current_time
            return True
        
        if self.current_scene != new_detections:
            self.current_scene = new_detections.copy()
            self.last_update = current_time
            return True
        
        return False
    
    def get_scene_summary(self):
        """Get clean scene summary"""
        if not self.current_scene:
            return "empty"
        
        items = []
        for obj_type, count in self.current_scene.items():
            if count == 1:
                items.append(obj_type)
            else:
                items.append(f"{count} {obj_type}s")
        
        return ", ".join(items)

class OptimizedLlama:
    """Optimized Llama 3.2 with better prompts for vision tasks"""
    
    def __init__(self):
        self.base_url = "http://127.0.0.1:11434"
        self.model_name = "llama3.2:1b"
        self.ready = False
        self.error_message = ""
        
    def start_service(self):
        """Start and verify service"""
        print("🚀 Starting Optimized Llama 3.2...")
        
        try:
            time.sleep(2)
            return self._test_model()
        except Exception as e:
            self.error_message = str(e)
            return False
    
    def _test_model(self):
        """Test model with vision-optimized prompts"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return False
            
            # Test with simple object description
            test_result = self._generate("Objects detected: person. Describe briefly:", max_tokens=15)
            if test_result and len(test_result.strip()) > 5:
                self.ready = True
                print(f"✅ Optimized AI ready! Test: '{test_result[:60]}'")
                return True
            else:
                return False
                
        except Exception as e:
            self.error_message = str(e)
            return False
    
    def _generate(self, prompt, max_tokens=80):
        """Generate with optimized parameters"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.5,  # Lower for more focused responses
                    "top_p": 0.8,
                    "stop": ["\n\n", "Note:", "I'm sorry", "I can't", "I cannot"]
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
                
                # Filter out unhelpful responses
                unhelpful_phrases = ["i'm sorry", "i can't", "i cannot", "without more context"]
                if any(phrase in generated.lower() for phrase in unhelpful_phrases):
                    return None
                
                return generated
            else:
                return None
                
        except Exception:
            return None
    
    def describe_scene(self, scene_items):
        """Generate scene description with optimized prompts"""
        if not scene_items or scene_items == "empty":
            return "Camera shows an empty space."
        
        # Use focused, specific prompts that work well
        prompts = [
            f"Camera view shows: {scene_items}. Brief description:",
            f"Objects detected: {scene_items}. What's happening:",
            f"Scene contains: {scene_items}. Describe in one sentence:",
        ]
        
        for prompt in prompts:
            result = self._generate(prompt, 50)
            if result and len(result) > 8:
                # Clean up the response
                result = result.replace("Based on the description", "").strip()
                result = result.replace("The scene shows", "").strip()
                if result:
                    return result
        
        # Fallback to simple description
        return f"Camera view shows {scene_items}."
    
    def answer_question(self, question, scene_items):
        """Answer questions with scene context"""
        if not scene_items:
            scene_context = "empty room"
        else:
            scene_context = scene_items
        
        # Optimize prompts for Q&A
        prompts = [
            f"Current view: {scene_context}. Q: {question} A:",
            f"Scene has {scene_context}. Question: {question} Answer:",
            f"Visible objects: {scene_context}. {question} Response:",
        ]
        
        for prompt in prompts:
            result = self._generate(prompt, 80)
            if result and len(result) > 8:
                # Clean response
                result = result.replace("Based on", "").strip()
                result = result.replace("According to", "").strip()
                if result:
                    return result
        
        # Smart fallback based on question type
        question_lower = question.lower()
        if "what" in question_lower and "wearing" in question_lower:
            if "person" in scene_context:
                return "I can see a person but cannot determine specific clothing details from object detection."
            else:
                return "No person visible in the current view."
        elif "who" in question_lower:
            if "person" in scene_context:
                return "I can detect a person but cannot identify specific individuals."
            else:
                return "No people detected in the current scene."
        elif "where" in question_lower:
            return f"In the camera view, I can see: {scene_context}."
        elif "how many" in question_lower:
            if scene_context != "empty room":
                return f"Current count based on detection: {scene_context}."
            else:
                return "No objects detected to count."
        else:
            return f"Regarding the scene with {scene_context}, I can provide basic object detection information."
    
    def cleanup(self):
        """Cleanup"""
        self.ready = False

class OptimizedVisionChat:
    """Optimized vision chat with better prompts"""
    
    def __init__(self, args):
        self.args = args
        self.yolo_model = None
        self.ai = OptimizedLlama()
        self.scene_analyzer = SceneAnalyzer()
        
        # UI State
        self.current_input = ""
        self.input_mode = False
        self.cursor_blink = True
        self.last_blink = time.time()
        
        # App state
        self.chat_history = deque(maxlen=4)
        self.current_description = "Starting optimized AI..."
        self.current_scene_items = ""
        
        # Performance
        self.frame_count = 0
        self.start_time = time.time()
        self.successful_responses = 0
        
        # Threading
        self.scene_queue = queue.Queue(maxsize=2)
        self.running = True
        
        # Colors
        self.colors = {
            'panel': (20, 25, 35),
            'panel_light': (35, 40, 50),
            'accent': (255, 140, 0),
            'green': (120, 255, 120),
            'orange': (0, 165, 255),
            'red': (100, 100, 255),
            'white': (255, 255, 255),
            'gray': (180, 180, 180),
            'dark': (120, 120, 120)
        }
    
    def setup(self):
        """Setup system"""
        print("🎯 Smart Vision Chat - Optimized for Object Detection")
        print("=" * 60)
        
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
        
        # Start AI
        ai_thread = threading.Thread(target=self.ai.start_service, daemon=True)
        ai_thread.start()
        
        print("=" * 60)
        print("🎥 Camera ready with optimized AI responses")
        print("💬 Press SPACE to ask questions about detected objects")
        print("🔍 AI optimized for: what/who/where/how many questions")
        print("❌ Press Q to quit")
        print("=" * 60)
        
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
                        print(f"📷 Using camera {i}")
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
        """Extract detections with better filtering"""
        detections = {}
        
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        class_names = result.names if hasattr(result, 'names') else {}
        
        for box in result.boxes:
            try:
                cls_id = int(box.cls.item())
                confidence = float(box.conf.item())
                
                if confidence < 0.6:  # Good confidence threshold
                    continue
                
                class_name = class_names.get(cls_id, "object")
                detections[class_name] = detections.get(class_name, 0) + 1
                
            except Exception:
                continue
        
        return detections
    
    def _scene_worker(self):
        """Background scene analysis"""
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
                        description = self.ai.describe_scene(scene_items)
                        if description:
                            self.current_description = description
                            self.successful_responses += 1
                            print(f"📝 Scene: {description}")
                        else:
                            self.current_description = f"Detected objects: {scene_items}"
                    else:
                        self.current_description = f"Detection: {scene_items}"
                
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
                print("💬 Ask about objects (what/who/where/how many)...")
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
            print("🔄 Restarting AI...")
            self.ai = OptimizedLlama()
            threading.Thread(target=self.ai.start_service, daemon=True).start()
        
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
            'answer': "Analyzing...",
            'timestamp': time.time()
        })
        
        threading.Thread(
            target=self._generate_answer,
            args=(question, len(self.chat_history) - 1),
            daemon=True
        ).start()
    
    def _generate_answer(self, question, chat_index):
        """Generate optimized answer"""
        try:
            if self.ai.ready:
                answer = self.ai.answer_question(question, self.current_scene_items)
                if answer:
                    self.chat_history[chat_index]['answer'] = answer
                    print(f"💡 Answer: {answer}")
                else:
                    # Provide smart fallback
                    fallback = self._smart_fallback(question)
                    self.chat_history[chat_index]['answer'] = fallback
            else:
                fallback = self._smart_fallback(question)
                self.chat_history[chat_index]['answer'] = fallback
                
        except Exception as e:
            if chat_index < len(self.chat_history):
                fallback = self._smart_fallback(question)
                self.chat_history[chat_index]['answer'] = fallback
    
    def _smart_fallback(self, question):
        """Smart fallback answers"""
        q_lower = question.lower()
        scene = self.current_scene_items or "empty"
        
        if "wearing" in q_lower or "clothes" in q_lower:
            return "Object detection can identify people but not clothing details."
        elif "who" in q_lower:
            if "person" in scene:
                return "A person is detected, but I cannot identify specific individuals."
            else:
                return "No people detected in the current view."
        elif "what" in q_lower:
            if scene != "empty":
                return f"Current detections: {scene}."
            else:
                return "No objects detected in the current scene."
        elif "where" in q_lower:
            return f"Objects are positioned within the camera's field of view: {scene}."
        elif "how many" in q_lower:
            if scene != "empty":
                return f"Object count based on detection: {scene}."
            else:
                return "No objects to count in the current view."
        else:
            return f"Based on object detection, I can see: {scene}."
    
    def _exit_input(self):
        """Exit input mode"""
        self.input_mode = False
        self.current_input = ""
    
    def _draw_text_wrap(self, img, text, x, y, font_scale, color, max_width, line_height=18):
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
        """Draw optimized UI"""
        h, w = frame.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Status bar
        cv2.rectangle(overlay, (8, 8), (w-8, 75), self.colors['panel'], -1)
        cv2.rectangle(overlay, (8, 8), (w-8, 75), self.colors['dark'], 2)
        
        # AI status
        if self.ai.ready:
            status = f"🤖 Optimized AI Ready ({self.successful_responses} responses)"
            color = self.colors['green']
        else:
            status = "🔄 AI Initializing..."
            color = self.colors['orange']
        
        cv2.putText(overlay, status, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Stats
        total_objects = sum(self.scene_analyzer.current_scene.values())
        stats = f"Objects: {total_objects} | FPS: {fps:.1f} | Frame: {self.frame_count}"
        cv2.putText(overlay, stats, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.colors['gray'], 1)
        
        # Scene description
        scene_y = 85
        cv2.rectangle(overlay, (8, scene_y), (w-8, scene_y + 95), self.colors['panel'], -1)
        cv2.rectangle(overlay, (8, scene_y), (w-8, scene_y + 95), self.colors['dark'], 2)
        
        cv2.putText(overlay, "🎙️ Optimized Scene Analysis", (15, scene_y + 22), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.colors['accent'], 2)
        
        if self.current_scene_items:
            cv2.putText(overlay, f"Objects: {self.current_scene_items}", (15, scene_y + 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['gray'], 1)
        
        self._draw_text_wrap(overlay, self.current_description, 15, scene_y + 65, 
                           0.45, self.colors['white'], w - 30, 16)
        
        # Chat history
        chat_y = 190
        chat_height = h - chat_y - 115
        cv2.rectangle(overlay, (8, chat_y), (w-8, chat_y + chat_height), self.colors['panel_light'], -1)
        cv2.rectangle(overlay, (8, chat_y), (w-8, chat_y + chat_height), self.colors['dark'], 2)
        
        cv2.putText(overlay, "💬 Smart Object Q&A", (15, chat_y + 22), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.colors['accent'], 2)
        
        # Show last 2 chats
        y_pos = chat_y + 40
        for chat in list(self.chat_history)[-2:]:
            if y_pos + 55 > chat_y + chat_height:
                break
            
            # Question
            q_height = self._draw_text_wrap(overlay, f"Q: {chat['question']}", 15, y_pos, 
                                          0.4, self.colors['white'], w - 30, 15)
            y_pos += q_height + 5
            
            # Answer
            a_height = self._draw_text_wrap(overlay, f"A: {chat['answer']}", 15, y_pos, 
                                          0.4, self.colors['green'], w - 30, 15)
            y_pos += a_height + 15
        
        # Input area
        input_y = h - 95
        input_color = self.colors['accent'] if self.input_mode else self.colors['panel']
        cv2.rectangle(overlay, (8, input_y), (w-8, h-8), input_color, -1)
        cv2.rectangle(overlay, (8, input_y), (w-8, h-8), self.colors['dark'], 2)
        
        if self.input_mode:
            cv2.putText(overlay, "💬 Ask about objects (SPACE=Send, ESC=Cancel):", 
                       (15, input_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.colors['white'], 1)
            
            # Input with cursor
            current_time = time.time()
            if current_time - self.last_blink > 0.5:
                self.cursor_blink = not self.cursor_blink
                self.last_blink = current_time
            
            display_text = self.current_input + ("|" if self.cursor_blink else "")
            cv2.putText(overlay, display_text, (15, input_y + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['white'], 2)
        else:
            cv2.putText(overlay, "💬 Press SPACE to ask about detected objects", 
                       (15, input_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['gray'], 1)
        
        # Controls
        cv2.putText(overlay, "SPACE=Ask | R=Restart | Q=Quit", (15, input_y + 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.colors['dark'], 1)
        
        # Blend
        alpha = 0.87
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
        window_name = "Smart Vision Chat - Optimized Object Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 800)
        
        print("🎬 Starting optimized vision chat...")
        
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
            self.ai.cleanup()
            
            print("\n" + "=" * 60)
            print("✅ Optimized Vision Chat completed")
            print(f"📊 Frames: {self.frame_count}")
            print(f"💬 Conversations: {len(self.chat_history)}")
            print(f"🎯 Successful AI responses: {self.successful_responses}")
            print("=" * 60)
        
        return True

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Smart Vision Chat - Optimized Object Detection")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model")
    parser.add_argument("--conf", type=float, default=0.6, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    
    args = parser.parse_args()
    
    app = OptimizedVisionChat(args)
    success = app.run()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

