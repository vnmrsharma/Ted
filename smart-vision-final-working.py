#!/usr/bin/env python3
"""
Smart Vision Chat - Final Working Version with Llama 3.2
Uses reliable Llama 3.2:1B model that actually works
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
    """Scene analysis with change detection"""
    
    def __init__(self):
        self.current_scene = {}
        self.last_update = 0
        self.update_interval = 3.0
        
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
            return "empty scene"
        
        items = []
        for obj_type, count in self.current_scene.items():
            if count == 1:
                items.append(obj_type)
            else:
                items.append(f"{count} {obj_type}s")
        
        return ", ".join(items)

class WorkingLlama:
    """Reliable Llama 3.2 model that actually works"""
    
    def __init__(self):
        self.base_url = "http://127.0.0.1:11434"
        self.model_name = "llama3.2:1b"
        self.ready = False
        self.error_message = ""
        self.process = None
        
    def start_service(self):
        """Start service and verify it works"""
        print("🚀 Starting Llama 3.2 AI Service...")
        
        try:
            # Test if server is already running
            time.sleep(3)  # Give server time to start
            
            return self._test_model()
            
        except Exception as e:
            self.error_message = str(e)
            return False
    
    def _test_model(self):
        """Test if model responds correctly"""
        try:
            # Test connection
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                self.error_message = "Server not responding"
                return False
            
            # Test generation
            test_result = self._generate("Hello, how are you?", max_tokens=10)
            if test_result and len(test_result.strip()) > 0:
                self.ready = True
                print(f"✅ Llama 3.2 ready! Test: '{test_result[:50]}'")
                return True
            else:
                self.error_message = "Model not responding"
                return False
                
        except Exception as e:
            self.error_message = str(e)
            return False
    
    def _generate(self, prompt, max_tokens=80):
        """Generate text reliably"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.7,
                    "top_p": 0.9
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
                return generated
            else:
                return None
                
        except Exception:
            return None
    
    def describe_scene(self, scene_items):
        """Generate scene description"""
        if not scene_items or scene_items == "empty scene":
            return "I see an empty space."
        
        prompt = f"Describe this scene in 1 sentence: {scene_items}"
        
        result = self._generate(prompt, 60)
        if result and len(result) > 5:
            return result
        else:
            return f"I can see {scene_items} in the camera view."
    
    def answer_question(self, question, scene_items):
        """Answer questions about the scene"""
        if not scene_items:
            scene_context = "empty room"
        else:
            scene_context = scene_items
        
        prompt = f"Scene: {scene_context}. Question: {question}. Answer briefly:"
        
        result = self._generate(prompt, 100)
        if result and len(result) > 5:
            return result
        else:
            return f"Regarding the {scene_context}, I'm not sure about that specific detail."
    
    def cleanup(self):
        """Cleanup"""
        self.ready = False

class WorkingVisionChat:
    """Vision chat with working AI"""
    
    def __init__(self, args):
        self.args = args
        self.yolo_model = None
        self.ai = WorkingLlama()
        self.scene_analyzer = SceneAnalyzer()
        
        # UI State
        self.current_input = ""
        self.input_mode = False
        self.cursor_blink = True
        self.last_blink = time.time()
        
        # App state
        self.chat_history = deque(maxlen=5)
        self.current_description = "Starting AI system..."
        self.current_scene_items = ""
        
        # Performance
        self.frame_count = 0
        self.start_time = time.time()
        
        # Threading
        self.scene_queue = queue.Queue(maxsize=2)
        self.running = True
        
        # Colors
        self.colors = {
            'panel': (25, 30, 40),
            'panel_light': (40, 45, 55),
            'accent': (255, 140, 0),
            'green': (100, 255, 100),
            'orange': (0, 165, 255),
            'red': (100, 100, 255),
            'white': (255, 255, 255),
            'gray': (180, 180, 180),
            'dark': (120, 120, 120)
        }
    
    def setup(self):
        """Setup system"""
        print("🎯 Smart Vision Chat - Working Edition with Llama 3.2")
        print("=" * 55)
        
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
        
        print("=" * 55)
        print("🎥 Camera ready with reliable AI")
        print("💬 Press SPACE to ask questions")
        print("❌ Press Q to quit")
        print("=" * 55)
        
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
        """Extract detections properly"""
        detections = {}
        
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        class_names = result.names if hasattr(result, 'names') else {}
        
        for box in result.boxes:
            try:
                cls_id = int(box.cls.item())
                confidence = float(box.conf.item())
                
                if confidence < 0.65:
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
                        self.current_description = description
                        print(f"📝 Scene: {description}")
                    else:
                        self.current_description = f"Detected: {scene_items}"
                
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
                print("💬 Ask a question...")
            else:
                if self.current_input.strip():
                    self._process_question(self.current_input.strip())
                self._exit_input()
        
        elif key in [27, ord('q'), ord('Q')]:
            if self.input_mode:
                self._exit_input()
            else:
                self.running = False
        
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
            'answer': "Thinking...",
            'timestamp': time.time()
        })
        
        # Generate answer in background
        threading.Thread(
            target=self._generate_answer,
            args=(question, len(self.chat_history) - 1),
            daemon=True
        ).start()
    
    def _generate_answer(self, question, chat_index):
        """Generate answer"""
        try:
            if self.ai.ready:
                answer = self.ai.answer_question(question, self.current_scene_items)
                self.chat_history[chat_index]['answer'] = answer
                print(f"💡 Answer: {answer}")
            else:
                self.chat_history[chat_index]['answer'] = "AI system is starting up..."
        except Exception as e:
            if chat_index < len(self.chat_history):
                self.chat_history[chat_index]['answer'] = f"Error: {str(e)[:40]}"
    
    def _exit_input(self):
        """Exit input mode"""
        self.input_mode = False
        self.current_input = ""
    
    def _draw_text_wrap(self, img, text, x, y, font_scale, color, max_width, line_height=20):
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
        """Draw UI overlay"""
        h, w = frame.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Status bar
        cv2.rectangle(overlay, (10, 10), (w-10, 80), self.colors['panel'], -1)
        cv2.rectangle(overlay, (10, 10), (w-10, 80), self.colors['dark'], 2)
        
        # AI status
        if self.ai.ready:
            status = "🤖 Llama 3.2 Ready"
            color = self.colors['green']
        else:
            status = "🔄 Llama 3.2 Starting..."
            color = self.colors['orange']
        
        cv2.putText(overlay, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Stats
        total_objects = sum(self.scene_analyzer.current_scene.values())
        stats = f"Objects: {total_objects} | FPS: {fps:.1f} | Frame: {self.frame_count}"
        cv2.putText(overlay, stats, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['gray'], 1)
        
        # Scene description
        scene_y = 90
        cv2.rectangle(overlay, (10, scene_y), (w-10, scene_y + 100), self.colors['panel'], -1)
        cv2.rectangle(overlay, (10, scene_y), (w-10, scene_y + 100), self.colors['dark'], 2)
        
        cv2.putText(overlay, "🎙️ Live Scene (Llama 3.2)", (20, scene_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['accent'], 2)
        
        if self.current_scene_items:
            cv2.putText(overlay, f"Detected: {self.current_scene_items}", (20, scene_y + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.colors['gray'], 1)
        
        self._draw_text_wrap(overlay, self.current_description, 20, scene_y + 70, 
                           0.5, self.colors['white'], w - 40, 18)
        
        # Chat history
        chat_y = 200
        chat_height = h - chat_y - 120
        cv2.rectangle(overlay, (10, chat_y), (w-10, chat_y + chat_height), self.colors['panel_light'], -1)
        cv2.rectangle(overlay, (10, chat_y), (w-10, chat_y + chat_height), self.colors['dark'], 2)
        
        cv2.putText(overlay, "💬 Smart Q&A", (20, chat_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['accent'], 2)
        
        # Show last 2 chats
        y_pos = chat_y + 45
        for chat in list(self.chat_history)[-2:]:
            if y_pos + 60 > chat_y + chat_height:
                break
            
            # Question
            q_height = self._draw_text_wrap(overlay, f"Q: {chat['question']}", 20, y_pos, 
                                          0.45, self.colors['white'], w - 40, 16)
            y_pos += q_height + 5
            
            # Answer
            a_height = self._draw_text_wrap(overlay, f"A: {chat['answer']}", 20, y_pos, 
                                          0.45, self.colors['green'], w - 40, 16)
            y_pos += a_height + 15
        
        # Input area
        input_y = h - 100
        input_color = self.colors['accent'] if self.input_mode else self.colors['panel']
        cv2.rectangle(overlay, (10, input_y), (w-10, h-10), input_color, -1)
        cv2.rectangle(overlay, (10, input_y), (w-10, h-10), self.colors['dark'], 2)
        
        if self.input_mode:
            cv2.putText(overlay, "💬 Type question (SPACE=Send, ESC=Cancel):", 
                       (20, input_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['white'], 1)
            
            # Input with cursor
            current_time = time.time()
            if current_time - self.last_blink > 0.5:
                self.cursor_blink = not self.cursor_blink
                self.last_blink = current_time
            
            display_text = self.current_input + ("|" if self.cursor_blink else "")
            cv2.putText(overlay, display_text, (20, input_y + 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['white'], 2)
        else:
            cv2.putText(overlay, "💬 Press SPACE to ask about the scene", 
                       (20, input_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['gray'], 1)
        
        # Controls
        cv2.putText(overlay, "SPACE=Ask | Q=Quit", (20, input_y + 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['dark'], 1)
        
        # Blend
        alpha = 0.85
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
        window_name = "Smart Vision Chat - Working with Llama 3.2"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1300, 800)
        
        print("🎬 Starting vision chat with working AI...")
        
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
            
            print("\n" + "=" * 55)
            print("✅ Working Vision Chat session completed")
            print(f"📊 Frames: {self.frame_count}")
            print(f"💬 Conversations: {len(self.chat_history)}")
            print("=" * 55)
        
        return True

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Smart Vision Chat - Working with Llama 3.2")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model")
    parser.add_argument("--conf", type=float, default=0.65, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    
    args = parser.parse_args()
    
    app = WorkingVisionChat(args)
    success = app.run()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

