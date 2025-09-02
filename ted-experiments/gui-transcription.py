#!/usr/bin/env python3
"""
GUI YOLO + GPT-OSS Transcription with In-Window Q&A
Type questions directly in the camera window and see AI responses
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

class SilentOllama:
    """Silent Ollama interface for GUI"""
    
    def __init__(self):
        self.ollama_url = "http://127.0.0.1:11434"
        self.model_name = "gpt-oss:20b"
        self.ready = False
        self.server_process = None
        
    def start_silent_server(self):
        """Start Ollama server silently"""
        try:
            # Kill existing processes
            subprocess.run(["pkill", "-f", "ollama"], capture_output=True)
            time.sleep(1)
            
            # Start silently
            with open(os.devnull, 'w') as devnull:
                self.server_process = subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=devnull,
                    stderr=devnull,
                    stdin=devnull
                )
            
            # Wait and check
            time.sleep(5)
            
            for attempt in range(10):
                try:
                    response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
                    if response.status_code == 200:
                        models = response.json().get('models', [])
                        for model in models:
                            if self.model_name in model.get('name', ''):
                                self.ready = True
                                return True
                except:
                    time.sleep(2)
            
            return False
            
        except Exception:
            return False
    
    def generate_description(self, detections):
        """Generate scene description"""
        if not self.ready:
            return self._basic_description(detections)
        
        try:
            detection_text = self._format_detections(detections)
            prompt = f"Describe this camera scene in 1-2 sentences: {detection_text}"
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 50, "temperature": 0.7}
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate", 
                json=payload, 
                timeout=8
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                return self._basic_description(detections)
                
        except Exception:
            return self._basic_description(detections)
    
    def answer_question(self, question, detections=None):
        """Answer user question"""
        if not self.ready:
            return "AI model starting... please wait."
        
        try:
            if detections:
                detection_text = self._format_detections(detections)
                prompt = f"Scene: {detection_text}\nQuestion: {question}\nAnswer:"
            else:
                prompt = f"Question: {question}\nAnswer:"
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 100, "temperature": 0.7}
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate", 
                json=payload, 
                timeout=12
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                return "Sorry, couldn't generate response."
                
        except Exception as e:
            return f"Connection error: {str(e)[:30]}"
    
    def _format_detections(self, detections):
        """Format detections"""
        if not detections:
            return "no objects detected"
        
        items = []
        for obj_name, info in list(detections.items())[:5]:
            count = info['count']
            if count == 1:
                items.append(obj_name)
            else:
                items.append(f"{count} {obj_name}s")
        
        return ", ".join(items)
    
    def _basic_description(self, detections):
        """Basic fallback description"""
        if not detections:
            return "Empty scene."
        
        obj_count = sum(info['count'] for info in detections.values())
        obj_types = list(detections.keys())[:3]
        
        if obj_count == 1:
            return f"I see a {obj_types[0]}."
        else:
            objects = ", ".join(obj_types)
            return f"Scene with {obj_count} objects: {objects}."
    
    def cleanup(self):
        """Clean shutdown"""
        try:
            if self.server_process:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
        except:
            pass
        
        try:
            subprocess.run(["pkill", "-f", "ollama"], capture_output=True)
        except:
            pass

class GUITranscriptionSystem:
    """GUI-based transcription with in-window Q&A"""
    
    def __init__(self, args):
        self.args = args
        self.yolo_model = None
        self.ollama = SilentOllama()
        self.distance_estimator = None
        
        # GUI state
        self.current_input = ""
        self.input_active = False
        self.cursor_visible = True
        self.cursor_blink_time = 0
        self.qa_history = deque(maxlen=5)  # Keep last 5 Q&A pairs
        self.pending_question = None
        self.pending_answer = None
        self.answer_ready = False
        
        # Transcription state
        self.description_queue = queue.Queue(maxsize=2)
        self.current_description = "Starting camera..."
        self.current_detections = None
        self.last_description_time = 0
        self.description_interval = 4.0
        self.running = True
        
    def setup(self):
        """Setup system"""
        print("🚀 GUI YOLO + GPT-OSS Transcription")
        print("=" * 40)
        
        # Find camera
        camera_index = self._find_camera()
        if camera_index is None:
            print("❌ No camera found!")
            return None
        
        # Load YOLO
        try:
            self.yolo_model = YOLO(self.args.model)
            print("✅ YOLO loaded")
        except Exception as e:
            print(f"❌ YOLO failed: {e}")
            return None
        
        # Start Ollama
        print("🔄 Starting AI model...")
        if self.ollama.start_silent_server():
            print("✅ AI model ready!")
        else:
            print("⚠️  AI model loading...")
        
        device = self._get_device()
        
        print("=" * 40)
        print("🎥 GUI Mode Active")
        print("💬 Click in video window to type questions")
        print("💬 Press ENTER to ask, ESC to cancel")
        print("💬 Press Q to quit")
        print("=" * 40)
        
        return camera_index, device
    
    def _find_camera(self):
        for i in range(4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                if ret:
                    return i
        return None
    
    def _get_device(self):
        if torch.cuda.is_available():
            return 0
        try:
            if torch.backends.mps.is_available():
                return "mps"
        except:
            pass
        return "cpu"
    
    def _extract_detections(self, result):
        """Extract detections"""
        detections = {}
        
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else []
        class_names = result.names if hasattr(result, 'names') else {}
        
        for i, cls_id in enumerate(classes):
            cls_name = class_names.get(int(cls_id), "object")
            conf = confidences[i] if i < len(confidences) else 0.0
            
            if conf < 0.5:
                continue
            
            if cls_name not in detections:
                detections[cls_name] = {'count': 0}
            
            detections[cls_name]['count'] += 1
        
        return detections
    
    def description_worker(self):
        """Background description generator"""
        while self.running:
            try:
                task = self.description_queue.get(timeout=1.0)
                if task is None:
                    break
                
                detections = task
                description = self.ollama.generate_description(detections)
                self.current_description = description
                
                self.description_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception:
                pass
    
    def answer_worker(self):
        """Background answer generator"""
        while self.running:
            try:
                if self.pending_question and not self.answer_ready:
                    self.pending_answer = self.ollama.answer_question(
                        self.pending_question, 
                        self.current_detections
                    )
                    self.answer_ready = True
                    
                time.sleep(0.1)
                
            except Exception:
                pass
    
    def handle_keyboard_input(self, key):
        """Handle keyboard input for GUI"""
        if key == 13:  # Enter
            if self.input_active and self.current_input.strip():
                # Submit question
                self.pending_question = self.current_input.strip()
                self.pending_answer = "Thinking..."
                self.answer_ready = False
                
                # Add to history
                self.qa_history.append({
                    'question': self.pending_question,
                    'answer': "Thinking...",
                    'timestamp': time.time()
                })
                
                # Clear input
                self.current_input = ""
                self.input_active = False
                
            else:
                # Activate input
                self.input_active = True
                self.current_input = ""
                
        elif key == 27:  # Escape
            # Cancel input
            self.input_active = False
            self.current_input = ""
            
        elif key == 8:  # Backspace
            if self.input_active and len(self.current_input) > 0:
                self.current_input = self.current_input[:-1]
                
        elif key == ord('q') or key == ord('Q'):
            self.running = False
            
        elif self.input_active and key >= 32 and key <= 126:  # Printable characters
            if len(self.current_input) < 60:  # Limit input length
                self.current_input += chr(key)
    
    def wrap_text(self, text, max_width, font_scale=0.5):
        """Wrap text to fit within width"""
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
        
        return lines
    
    def draw_gui_overlay(self, frame, result, fps, frame_count):
        """Draw GUI overlay with Q&A interface"""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Main info panel
        detection_count = len(result.boxes) if result.boxes is not None else 0
        
        cv2.rectangle(overlay, (10, 10), (400, 70), (0, 0, 0), -1)
        cv2.rectangle(overlay, (10, 10), (400, 70), (255, 255, 255), 2)
        
        cv2.putText(overlay, f"Objects: {detection_count} | FPS: {fps:.1f}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(overlay, f"Frame: {frame_count}", (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Current description panel
        desc_height = 100
        cv2.rectangle(overlay, (10, 80), (w-10, 80 + desc_height), (0, 0, 0), -1)
        cv2.rectangle(overlay, (10, 80), (w-10, 80 + desc_height), (0, 255, 0), 2)
        
        cv2.putText(overlay, "🎙️ Scene Description:", (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Wrap description text
        desc_lines = self.wrap_text(self.current_description, w-40, 0.5)
        for i, line in enumerate(desc_lines[:3]):  # Max 3 lines
            cv2.putText(overlay, line, (20, 130 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Q&A History panel
        qa_start_y = 190
        qa_height = h - qa_start_y - 120  # Leave space for input
        
        cv2.rectangle(overlay, (10, qa_start_y), (w-10, qa_start_y + qa_height), (0, 0, 0), -1)
        cv2.rectangle(overlay, (10, qa_start_y), (w-10, qa_start_y + qa_height), (255, 165, 0), 2)
        
        cv2.putText(overlay, "💬 Q&A History:", (20, qa_start_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        
        # Display recent Q&A
        y_pos = qa_start_y + 50
        for qa in list(self.qa_history)[-3:]:  # Show last 3
            if y_pos + 60 > qa_start_y + qa_height:
                break
                
            # Question
            q_lines = self.wrap_text(f"Q: {qa['question']}", w-40, 0.4)
            for i, line in enumerate(q_lines[:2]):  # Max 2 lines per question
                cv2.putText(overlay, line, (20, y_pos + i * 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 255), 1)
            y_pos += len(q_lines[:2]) * 15 + 5
            
            # Answer
            answer = qa['answer'] if qa['answer'] != "Thinking..." else self.pending_answer or "Thinking..."
            a_lines = self.wrap_text(f"A: {answer}", w-40, 0.4)
            for i, line in enumerate(a_lines[:2]):  # Max 2 lines per answer
                cv2.putText(overlay, line, (20, y_pos + i * 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 100), 1)
            y_pos += len(a_lines[:2]) * 15 + 10
        
        # Input panel
        input_y = h - 100
        cv2.rectangle(overlay, (10, input_y), (w-10, h-10), (0, 0, 0), -1)
        
        if self.input_active:
            cv2.rectangle(overlay, (10, input_y), (w-10, h-10), (0, 255, 255), 3)
            cv2.putText(overlay, "💬 Type your question:", (20, input_y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.rectangle(overlay, (10, input_y), (w-10, h-10), (128, 128, 128), 2)
            cv2.putText(overlay, "💬 Press ENTER to ask a question:", (20, input_y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
        
        # Current input text
        input_text = self.current_input
        if self.input_active:
            # Add blinking cursor
            current_time = time.time()
            if current_time - self.cursor_blink_time > 0.5:
                self.cursor_visible = not self.cursor_visible
                self.cursor_blink_time = current_time
            
            if self.cursor_visible:
                input_text += "|"
        
        cv2.putText(overlay, input_text, (20, input_y + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Instructions
        cv2.putText(overlay, "ENTER=Ask | ESC=Cancel | Q=Quit", (20, input_y + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return overlay
    
    def run(self):
        """Main execution"""
        setup_result = self.setup()
        if setup_result is None:
            return False
        
        camera_index, device = setup_result
        
        # Import distance estimator
        try:
            from main import SimpleDistanceEstimator
        except:
            pass
        
        # Start background workers
        desc_thread = threading.Thread(target=self.description_worker, daemon=True)
        desc_thread.start()
        
        answer_thread = threading.Thread(target=self.answer_worker, daemon=True)
        answer_thread.start()
        
        # Setup window
        win = "GUI YOLO + GPT-OSS Transcription"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1200, 900)
        
        t0, n = time.time(), 0
        
        try:
            for result in self.yolo_model(source=camera_index, stream=True,
                                         conf=self.args.conf, imgsz=self.args.imgsz, device=device):
                
                if not self.running:
                    break
                
                frame = result.plot()
                
                # Initialize distance estimator
                if self.distance_estimator is None and 'SimpleDistanceEstimator' in globals():
                    h, w = frame.shape[:2]
                    self.distance_estimator = SimpleDistanceEstimator(w, h)
                
                n += 1
                dt = time.time() - t0
                fps = (n / dt) if dt > 0 else 0.0
                
                # Extract detections
                detections = self._extract_detections(result)
                self.current_detections = detections
                
                # Update Q&A history with new answers
                if self.answer_ready and self.qa_history:
                    self.qa_history[-1]['answer'] = self.pending_answer
                    self.answer_ready = False
                    self.pending_question = None
                    self.pending_answer = None
                
                # Schedule description
                current_time = time.time()
                if (current_time - self.last_description_time) >= self.description_interval:
                    try:
                        self.description_queue.put_nowait(detections)
                        self.last_description_time = current_time
                    except queue.Full:
                        pass
                
                # Draw GUI
                gui_frame = self.draw_gui_overlay(frame, result, fps, n)
                cv2.imshow(win, gui_frame)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if key != 255:  # Key pressed
                    self.handle_keyboard_input(key)
                
                if not self.running:
                    break
                
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"\nError: {e}")
        finally:
            self.running = False
            self.description_queue.put(None)
            cv2.destroyAllWindows()
            self.ollama.cleanup()
            
            print("\n" + "=" * 40)
            print("✅ GUI session complete")
            print("=" * 40)
        
        return True

def main():
    parser = argparse.ArgumentParser(description="GUI YOLO + GPT-OSS Transcription")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model")
    parser.add_argument("--conf", type=float, default=0.6, help="Confidence")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    
    args = parser.parse_args()
    
    system = GUITranscriptionSystem(args)
    system.run()

if __name__ == "__main__":
    main()
