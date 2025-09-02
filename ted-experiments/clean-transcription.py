#!/usr/bin/env python3
"""
Clean YOLO + GPT-OSS Transcription with Silent Ollama
No logs, clean terminal for typing questions
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
    """Completely silent Ollama interface"""
    
    def __init__(self):
        self.ollama_url = "http://127.0.0.1:11434"
        self.model_name = "gpt-oss:20b"
        self.ready = False
        self.server_process = None
        
    def start_silent_server(self):
        """Start Ollama server completely silently"""
        try:
            # Kill any existing Ollama processes
            subprocess.run(["pkill", "-f", "ollama"], capture_output=True)
            time.sleep(1)
            
            # Start Ollama server with all output suppressed
            with open(os.devnull, 'w') as devnull:
                self.server_process = subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=devnull,
                    stderr=devnull,
                    stdin=devnull
                )
            
            # Wait for server to start
            time.sleep(5)
            
            # Check if model is available
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
                "options": {"num_predict": 80, "temperature": 0.7}
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate", 
                json=payload, 
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                return "Sorry, couldn't generate response."
                
        except Exception as e:
            return f"Connection error: {str(e)[:50]}"
    
    def _format_detections(self, detections):
        """Format detections for prompt"""
        if not detections:
            return "no objects detected"
        
        items = []
        for obj_name, info in list(detections.items())[:5]:  # Limit to 5 objects
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

class CleanTranscriptionSystem:
    """Clean transcription system with no log interference"""
    
    def __init__(self, args):
        self.args = args
        self.yolo_model = None
        self.ollama = SilentOllama()
        self.distance_estimator = None
        
        # Queues for clean threading
        self.description_queue = queue.Queue(maxsize=2)
        self.input_queue = queue.Queue()
        
        # State
        self.current_description = "Starting..."
        self.current_detections = None
        self.last_description_time = 0
        self.description_interval = 4.0  # Every 4 seconds
        self.running = True
        
    def setup(self):
        """Setup all components"""
        print("🚀 Clean YOLO + GPT-OSS Transcription")
        print("=" * 40)
        
        # Find camera
        camera_index = self._find_camera()
        if camera_index is None:
            print("❌ No camera found!")
            return None
        
        # Load YOLO
        try:
            self.yolo_model = YOLO(self.args.model)
            print(f"✅ YOLO loaded")
        except Exception as e:
            print(f"❌ YOLO failed: {e}")
            return None
        
        # Start Ollama silently
        print("🔄 Starting AI model (silent)...")
        if self.ollama.start_silent_server():
            print("✅ AI model ready!")
        else:
            print("⚠️  AI model loading, using basic descriptions")
        
        # Get device
        device = self._get_device()
        
        print("=" * 40)
        print("📝 TRANSCRIPTION ACTIVE")
        print("💬 Type questions anytime!")
        print("💬 Type 'q' to quit")
        print("=" * 40)
        
        return camera_index, device
    
    def _find_camera(self):
        """Find working camera"""
        for i in range(4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                if ret:
                    return i
        return None
    
    def _get_device(self):
        """Get best device"""
        if torch.cuda.is_available():
            return 0
        try:
            if torch.backends.mps.is_available():
                return "mps"
        except:
            pass
        return "cpu"
    
    def _extract_detections(self, result):
        """Extract detection information"""
        detections = {}
        
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else []
        class_names = result.names if hasattr(result, 'names') else {}
        
        for i, cls_id in enumerate(classes):
            cls_name = class_names.get(int(cls_id), f"object")
            conf = confidences[i] if i < len(confidences) else 0.0
            
            if conf < 0.5:  # Only confident detections
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
                
                # Clean output
                timestamp = time.strftime("%H:%M:%S")
                print(f"[{timestamp}] {description}")
                
                self.description_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception:
                pass
    
    def input_worker(self):
        """Background input handler - completely silent"""
        while self.running:
            try:
                # This blocks until user types something
                question = input()
                if question.strip():
                    self.input_queue.put(question.strip())
            except (EOFError, KeyboardInterrupt):
                self.running = False
                break
            except:
                pass
    
    def process_input(self):
        """Process user input"""
        try:
            while not self.input_queue.empty():
                question = self.input_queue.get_nowait()
                
                if question.lower() in ['q', 'quit', 'exit']:
                    self.running = False
                    return
                
                # Clean Q&A output
                print(f"\n🤔 Q: {question}")
                answer = self.ollama.answer_question(question, self.current_detections)
                print(f"🤖 A: {answer}\n")
                
        except queue.Empty:
            pass
    
    def draw_overlay(self, frame, result, fps, frame_count):
        """Minimal overlay"""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Info panel
        detection_count = len(result.boxes) if result.boxes is not None else 0
        
        cv2.rectangle(overlay, (10, 10), (350, 60), (0, 0, 0), -1)
        cv2.rectangle(overlay, (10, 10), (350, 60), (255, 255, 255), 2)
        
        cv2.putText(overlay, f"Objects: {detection_count} | FPS: {fps:.1f}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(overlay, f"Frame: {frame_count}", (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Description
        cv2.rectangle(overlay, (10, 70), (w-10, 120), (0, 0, 0), -1)
        cv2.rectangle(overlay, (10, 70), (w-10, 120), (0, 255, 0), 2)
        
        # Wrap description
        desc = self.current_description[:80]
        cv2.putText(overlay, desc, (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Controls
        cv2.putText(overlay, "Type questions in terminal | Q to quit", (20, h-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
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
            self.distance_estimator = None
        
        # Start background workers
        desc_thread = threading.Thread(target=self.description_worker, daemon=True)
        desc_thread.start()
        
        input_thread = threading.Thread(target=self.input_worker, daemon=True)
        input_thread.start()
        
        # Setup window
        win = "Clean Transcription"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1000, 700)
        
        t0, n = time.time(), 0
        
        try:
            # Main detection loop
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
                
                # Process any user input
                self.process_input()
                
                # Schedule description
                current_time = time.time()
                if (current_time - self.last_description_time) >= self.description_interval:
                    try:
                        self.description_queue.put_nowait(detections)
                        self.last_description_time = current_time
                    except queue.Full:
                        pass
                
                # Display
                enhanced_frame = self.draw_overlay(frame, result, fps, n)
                cv2.imshow(win, enhanced_frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")) or not self.running:
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
            print("✅ Clean shutdown complete")
            print("=" * 40)
        
        return True

def main():
    parser = argparse.ArgumentParser(description="Clean YOLO + GPT-OSS Transcription")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model")
    parser.add_argument("--conf", type=float, default=0.6, help="Confidence")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    
    args = parser.parse_args()
    
    system = CleanTranscriptionSystem(args)
    system.run()

if __name__ == "__main__":
    main()
