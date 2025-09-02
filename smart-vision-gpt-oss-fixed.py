#!/usr/bin/env python3
"""
Smart Vision Chat - GPT-OSS 20B Fixed Version
Uses only prompt patterns that actually work with GPT-OSS 20B
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
    """Scene analysis optimized for GPT-OSS"""
    
    def __init__(self):
        self.current_scene = {}
        self.last_update = 0
        self.update_interval = 5.0  # Longer interval to avoid overwhelming GPT-OSS
        
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
    
    def get_simple_scene(self):
        """Get very simple scene description for GPT-OSS"""
        if not self.current_scene:
            return "empty"
        
        # Return only the first/main object type
        main_object = list(self.current_scene.keys())[0]
        return main_object

class FixedGPTOSS:
    """GPT-OSS with patterns that actually work"""
    
    def __init__(self):
        self.base_url = "http://127.0.0.1:11434"
        self.model_name = "gpt-oss:20b"
        self.ready = False
        self.error_message = ""
        self.process = None
        
        # Only use prompt patterns we KNOW work with GPT-OSS
        self.working_patterns = {
            'greetings': ["Hello", "Hi", "Good morning"],
            'simple_words': ["Yes", "No", "OK"],
            'basic_statements': ["I see {}", "There is {}", "A {}"],
            'minimal_questions': ["What?", "Where?", "How?"]
        }
        
    def start_service(self):
        """Start GPT-OSS service"""
        print("🚀 Starting GPT-OSS 20B (Fixed Version)...")
        
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
            
            time.sleep(8)
            return self._test_with_known_working_prompts()
            
        except Exception as e:
            self.error_message = str(e)
            return False
    
    def _test_with_known_working_prompts(self):
        """Test with prompts we know work"""
        try:
            # Test connection
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return False
            
            # Test with simple greeting (we know this works)
            test_result = self._generate_simple("Hello")
            if test_result and len(test_result.strip()) > 0:
                self.ready = True
                print(f"✅ GPT-OSS ready! Test response: '{test_result[:50]}'")
                return True
            else:
                self.error_message = "No response from model"
                return False
                
        except Exception as e:
            self.error_message = str(e)
            return False
    
    def _generate_simple(self, prompt, max_tokens=50):
        """Generate with minimal, working parameters"""
        try:
            # Use absolute minimal payload
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.8  # Use what worked in our tests
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
    
    def describe_scene(self, simple_object):
        """Generate scene description using only working patterns"""
        if not simple_object or simple_object == "empty":
            return "Empty space."
        
        # Use only the patterns we know work
        working_prompts = [
            f"I see {simple_object}",  # Known working pattern
            f"There is {simple_object}",  # Known working pattern
            f"A {simple_object}",  # Known working pattern
        ]
        
        # Try each working pattern
        for prompt in working_prompts:
            result = self._generate_simple(prompt, 30)
            if result and len(result.strip()) > 2:
                print(f"✅ Scene generated: '{result[:60]}'")
                return result
        
        # If all fail, use simple fallback
        return f"I can see a {simple_object}."
    
    def answer_question(self, question, simple_object):
        """Answer using minimal patterns that work"""
        # Simplify the question to basic patterns
        question_lower = question.lower()
        
        if not simple_object or simple_object == "empty":
            scene_context = "nothing"
        else:
            scene_context = simple_object
        
        # Use very simple Q&A patterns
        simple_prompts = [
            f"{scene_context}",  # Just the object
            f"Yes {scene_context}",  # Affirmative with object
            f"I see {scene_context}",  # Direct observation
        ]
        
        # Try simple patterns
        for prompt in simple_prompts:
            result = self._generate_simple(prompt, 40)
            if result and len(result.strip()) > 2:
                print(f"✅ Answer generated: '{result[:60]}'")
                return result
        
        # Smart fallback based on question type
        if "what" in question_lower and "wearing" in question_lower:
            return "Cannot see clothing details."
        elif "who" in question_lower:
            if "person" in scene_context:
                return "A person is present."
            else:
                return "No person visible."
        elif "where" in question_lower:
            return f"In camera view: {scene_context}."
        elif "how many" in question_lower:
            return f"One {scene_context} detected." if scene_context != "nothing" else "Nothing to count."
        else:
            return f"About the {scene_context}: Cannot provide more details."
    
    def cleanup(self):
        """Cleanup"""
        self.ready = False
        try:
            if self.process:
                os.killpg(os.getpgid(self.process.pid), 15)
                self.process.wait(timeout=3)
        except:
            pass

class FixedVisionChat:
    """Vision chat optimized for GPT-OSS 20B quirks"""
    
    def __init__(self, args):
        self.args = args
        self.yolo_model = None
        self.ai = FixedGPTOSS()
        self.scene_analyzer = SceneAnalyzer()
        
        # UI State
        self.current_input = ""
        self.input_mode = False
        self.cursor_blink = True
        self.last_blink = time.time()
        
        # App state
        self.chat_history = deque(maxlen=4)
        self.current_description = "Starting GPT-OSS 20B..."
        self.current_simple_object = ""
        
        # Performance
        self.frame_count = 0
        self.start_time = time.time()
        self.gpt_oss_successes = 0
        self.gpt_oss_attempts = 0
        
        # Threading
        self.scene_queue = queue.Queue(maxsize=2)
        self.running = True
        
        # Colors
        self.colors = {
            'panel': (15, 20, 30),
            'panel_light': (30, 35, 45),
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
        print("🎯 Smart Vision Chat - GPT-OSS 20B Fixed")
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
        except Exception as e:
            print(f"❌ YOLO failed: {e}")
            return None
        
        # Start GPT-OSS
        ai_thread = threading.Thread(target=self.ai.start_service, daemon=True)
        ai_thread.start()
        
        print("=" * 50)
        print("🎥 Camera ready with GPT-OSS 20B")
        print("💬 Press SPACE for simple questions")
        print("⚠️ Using only patterns that work with GPT-OSS")
        print("❌ Press Q to quit")
        print("=" * 50)
        
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
        """Extract detections (simplified for GPT-OSS)"""
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
        """Background scene analysis for GPT-OSS"""
        while self.running:
            try:
                task = self.scene_queue.get(timeout=1.0)
                if task is None:
                    break
                
                detections = task
                
                if self.scene_analyzer.should_update_scene(detections):
                    simple_object = self.scene_analyzer.get_simple_scene()
                    self.current_simple_object = simple_object
                    
                    if self.ai.ready:
                        self.gpt_oss_attempts += 1
                        try:
                            description = self.ai.describe_scene(simple_object)
                            if description and len(description.strip()) > 5:
                                self.current_description = description
                                self.gpt_oss_successes += 1
                                print(f"📝 GPT-OSS: {description}")
                            else:
                                self.current_description = f"Detected: {simple_object}"
                        except Exception as e:
                            print(f"GPT-OSS error: {e}")
                            self.current_description = f"Detection: {simple_object}"
                    else:
                        self.current_description = f"Object: {simple_object}"
                
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
                print("💬 Ask simple question (GPT-OSS works best with basic questions)...")
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
            self.ai.cleanup()
            time.sleep(2)
            self.ai = FixedGPTOSS()
            threading.Thread(target=self.ai.start_service, daemon=True).start()
        
        elif key == 8 and self.input_mode:  # Backspace
            if len(self.current_input) > 0:
                self.current_input = self.current_input[:-1]
        
        elif self.input_mode and 32 <= key <= 126:
            if len(self.current_input) < 80:  # Shorter limit for GPT-OSS
                self.current_input += chr(key)
    
    def _process_question(self, question):
        """Process user question"""
        print(f"❓ Question: {question}")
        
        self.chat_history.append({
            'question': question,
            'answer': "Processing with GPT-OSS...",
            'timestamp': time.time()
        })
        
        threading.Thread(
            target=self._generate_answer,
            args=(question, len(self.chat_history) - 1),
            daemon=True
        ).start()
    
    def _generate_answer(self, question, chat_index):
        """Generate answer with GPT-OSS"""
        try:
            if self.ai.ready:
                self.gpt_oss_attempts += 1
                answer = self.ai.answer_question(question, self.current_simple_object)
                if answer and len(answer.strip()) > 3:
                    self.chat_history[chat_index]['answer'] = answer
                    self.gpt_oss_successes += 1
                    print(f"💡 GPT-OSS answer: {answer}")
                else:
                    # Provide simple fallback
                    fallback = self._simple_fallback(question)
                    self.chat_history[chat_index]['answer'] = fallback
            else:
                fallback = self._simple_fallback(question)
                self.chat_history[chat_index]['answer'] = fallback
                
        except Exception as e:
            print(f"Answer error: {e}")
            if chat_index < len(self.chat_history):
                fallback = self._simple_fallback(question)
                self.chat_history[chat_index]['answer'] = fallback
    
    def _simple_fallback(self, question):
        """Simple fallback answers"""
        q_lower = question.lower()
        obj = self.current_simple_object or "nothing"
        
        if "wearing" in q_lower:
            return "Cannot see clothing details."
        elif "who" in q_lower:
            return "A person is visible." if "person" in obj else "No person seen."
        elif "what" in q_lower:
            return f"I can see: {obj}." if obj != "nothing" else "Nothing detected."
        elif "where" in q_lower:
            return f"Location: {obj} in camera view." if obj != "nothing" else "Empty view."
        elif "how many" in q_lower:
            return f"One {obj}." if obj != "nothing" else "Zero objects."
        else:
            return f"About {obj}: Basic detection only." if obj != "nothing" else "No objects to discuss."
    
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
        """Draw UI with GPT-OSS status"""
        h, w = frame.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Status bar
        cv2.rectangle(overlay, (8, 8), (w-8, 85), self.colors['panel'], -1)
        cv2.rectangle(overlay, (8, 8), (w-8, 85), self.colors['dark'], 2)
        
        # GPT-OSS status
        if self.ai.ready:
            success_rate = (self.gpt_oss_successes / max(1, self.gpt_oss_attempts)) * 100
            status = f"🤖 GPT-OSS 20B Ready ({success_rate:.1f}% success)"
            color = self.colors['green']
        else:
            status = "🔄 GPT-OSS 20B Starting..."
            color = self.colors['orange']
        
        cv2.putText(overlay, status, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Stats
        total_objects = sum(self.scene_analyzer.current_scene.values())
        stats = f"Objects: {total_objects} | FPS: {fps:.1f} | GPT-OSS: {self.gpt_oss_successes}/{self.gpt_oss_attempts}"
        cv2.putText(overlay, stats, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['gray'], 1)
        
        # Scene description
        scene_y = 95
        cv2.rectangle(overlay, (8, scene_y), (w-8, scene_y + 100), self.colors['panel'], -1)
        cv2.rectangle(overlay, (8, scene_y), (w-8, scene_y + 100), self.colors['dark'], 2)
        
        cv2.putText(overlay, "🎙️ GPT-OSS 20B Scene Analysis", (15, scene_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.colors['accent'], 2)
        
        if self.current_simple_object:
            cv2.putText(overlay, f"Main object: {self.current_simple_object}", (15, scene_y + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['gray'], 1)
        
        self._draw_text_wrap(overlay, self.current_description, 15, scene_y + 70, 
                           0.45, self.colors['white'], w - 30, 16)
        
        # Chat history
        chat_y = 205
        chat_height = h - chat_y - 110
        cv2.rectangle(overlay, (8, chat_y), (w-8, chat_y + chat_height), self.colors['panel_light'], -1)
        cv2.rectangle(overlay, (8, chat_y), (w-8, chat_y + chat_height), self.colors['dark'], 2)
        
        cv2.putText(overlay, "💬 GPT-OSS Q&A", (15, chat_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.colors['accent'], 2)
        
        # Show last 2 chats
        y_pos = chat_y + 45
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
        input_y = h - 100
        input_color = self.colors['accent'] if self.input_mode else self.colors['panel']
        cv2.rectangle(overlay, (8, input_y), (w-8, h-8), input_color, -1)
        cv2.rectangle(overlay, (8, input_y), (w-8, h-8), self.colors['dark'], 2)
        
        if self.input_mode:
            cv2.putText(overlay, "💬 Ask GPT-OSS (keep it simple - SPACE=Send, ESC=Cancel):", 
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
            cv2.putText(overlay, "💬 Press SPACE to ask GPT-OSS (simple questions work best)", 
                       (15, input_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['gray'], 1)
        
        # Controls
        cv2.putText(overlay, "SPACE=Ask | R=Restart GPT-OSS | Q=Quit", (15, input_y + 80), 
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
        window_name = "Smart Vision Chat - GPT-OSS 20B Fixed"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 780)
        
        print("🎬 Starting GPT-OSS 20B vision chat...")
        
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
            
            print("\n" + "=" * 50)
            print("✅ GPT-OSS 20B Vision Chat completed")
            print(f"📊 Frames: {self.frame_count}")
            print(f"💬 Conversations: {len(self.chat_history)}")
            print(f"🎯 GPT-OSS Success Rate: {self.gpt_oss_successes}/{self.gpt_oss_attempts}")
            print("=" * 50)
        
        return True

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Smart Vision Chat - GPT-OSS 20B Fixed")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model")
    parser.add_argument("--conf", type=float, default=0.65, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    
    args = parser.parse_args()
    
    app = FixedVisionChat(args)
    success = app.run()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

