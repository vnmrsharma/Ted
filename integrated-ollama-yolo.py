#!/usr/bin/env python3
"""
Integrated YOLOv8n + Ollama GPT-OSS Real-time Scene Transcription
Clean terminal output with Q&A functionality
"""

import argparse
import time
import cv2
import torch
import sys
import numpy as np
import psutil
import os
import threading
import queue
import requests
import json
from collections import deque
from ultralytics import YOLO

class OllamaTranscriber:
    """Ollama-powered scene transcription system"""
    
    def __init__(self):
        self.ollama_url = "http://127.0.0.1:11434"
        self.model_name = "gpt-oss:20b"
        self.loaded = False
        
    def check_ollama_status(self):
        """Check if Ollama server is running and model is available"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            if response.status_code == 200:
                models = response.json().get('models', [])
                for model in models:
                    if self.model_name in model.get('name', ''):
                        self.loaded = True
                        return True
            return False
        except:
            return False
    
    def start_ollama_silent(self):
        """Start Ollama server in background without logs"""
        try:
            # Start Ollama server silently
            os.system("ollama serve > /dev/null 2>&1 &")
            time.sleep(3)  # Give it time to start
            return self.check_ollama_status()
        except:
            return False
    
    def generate_scene_description(self, detections, frame_info):
        """Generate natural language description of the scene"""
        if not self.loaded:
            if not self.check_ollama_status():
                return "Loading AI model..."
        
        try:
            # Create detection summary
            detection_text = self._format_detections(detections)
            
            # Create prompt for scene description
            prompt = f"Describe what you see in this camera view in 1-2 sentences. Keep it natural and conversational.\n\nDetected: {detection_text}"
            
            return self._call_ollama(prompt, max_tokens=60)
            
        except Exception as e:
            return self._basic_description(detections, frame_info)
    
    def answer_question(self, question, detections=None):
        """Answer user questions about the scene or general topics"""
        if not self.loaded:
            if not self.check_ollama_status():
                return "AI model not ready yet. Please wait..."
        
        try:
            # Create context-aware prompt
            if detections:
                detection_text = self._format_detections(detections)
                prompt = f"Current scene: {detection_text}\n\nQuestion: {question}\n\nAnswer helpfully:"
            else:
                prompt = f"Question: {question}\n\nAnswer:"
            
            return self._call_ollama(prompt, max_tokens=100)
            
        except Exception as e:
            return f"Sorry, error: {str(e)}"
    
    def _call_ollama(self, prompt, max_tokens=100):
        """Make API call to Ollama"""
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
                f"{self.ollama_url}/api/generate", 
                json=payload, 
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                return "AI response error"
                
        except Exception as e:
            return f"Connection error: {str(e)}"
    
    def _format_detections(self, detections):
        """Format detections for prompt"""
        if not detections:
            return "no objects"
        
        items = []
        for obj_name, info in detections.items():
            count = info['count']
            distance = info['avg_distance']
            
            if distance < 1.0:
                dist_text = f"{distance*100:.0f}cm"
            elif distance < 10.0:
                dist_text = f"{distance:.1f}m"
            else:
                dist_text = f"{distance:.0f}m"
            
            if count == 1:
                items.append(f"{obj_name} ({dist_text})")
            else:
                items.append(f"{count} {obj_name}s ({dist_text})")
        
        return ", ".join(items)
    
    def _basic_description(self, detections, frame_info):
        """Fallback description"""
        if not detections:
            return "Empty scene with no detected objects."
        
        obj_count = sum(info['count'] for info in detections.values())
        obj_types = list(detections.keys())
        
        if obj_count == 1:
            return f"Camera shows a {obj_types[0]}."
        elif len(obj_types) == 1:
            return f"Camera shows {obj_count} {obj_types[0]}s."
        else:
            main_objects = ", ".join(obj_types[:3])
            return f"Scene with {obj_count} objects: {main_objects}."

class IntegratedSystem:
    """Clean integrated YOLO + Ollama system"""
    
    def __init__(self, args):
        self.args = args
        self.yolo_model = None
        self.transcriber = OllamaTranscriber()
        self.distance_estimator = None
        self.description_queue = queue.Queue(maxsize=3)
        self.input_queue = queue.Queue()
        self.current_description = "Starting camera..."
        self.last_description_time = 0
        self.description_interval = 3.0
        self.current_detections = None
        self.running = True
        
    def find_working_camera(self):
        """Find working camera"""
        for i in range(4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                if ret:
                    return i
        return None
    
    def load_yolo_model(self):
        """Load YOLO model"""
        try:
            self.yolo_model = YOLO(self.args.model)
            return True
        except Exception as e:
            print(f"❌ YOLO load failed: {e}")
            return False
    
    def pick_device(self):
        """Select device"""
        if torch.cuda.is_available():
            return 0
        try:
            if torch.backends.mps.is_available():
                return "mps"
        except:
            pass
        return "cpu"
    
    def extract_detections(self, result):
        """Extract detection info"""
        detections = {}
        
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else []
        class_names = result.names if hasattr(result, 'names') else {}
        
        for i, cls_id in enumerate(classes):
            cls_name = class_names.get(int(cls_id), f"object_{int(cls_id)}")
            conf = confidences[i] if i < len(confidences) else 0.0
            
            if conf < 0.5:  # Only high-confidence detections
                continue
            
            # Calculate distance
            if self.distance_estimator:
                distance = self.distance_estimator.estimate_distance(result.boxes[i], cls_name)
            else:
                distance = 3.0
            
            if cls_name not in detections:
                detections[cls_name] = {'count': 0, 'distances': []}
            
            detections[cls_name]['count'] += 1
            detections[cls_name]['distances'].append(distance)
        
        # Calculate averages
        for cls_name in detections:
            if detections[cls_name]['distances']:
                detections[cls_name]['avg_distance'] = np.mean(detections[cls_name]['distances'])
            else:
                detections[cls_name]['avg_distance'] = 3.0
        
        return detections
    
    def description_worker(self):
        """Background description generator"""
        while self.running:
            try:
                task = self.description_queue.get(timeout=1.0)
                if task is None:
                    break
                
                detections, frame_info = task
                description = self.transcriber.generate_scene_description(detections, frame_info)
                
                self.current_description = description
                
                # Clean output with timestamp
                timestamp = time.strftime("%H:%M:%S")
                print(f"[{timestamp}] {description}")
                
                self.description_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception:
                pass
    
    def input_worker(self):
        """Background input handler"""
        while self.running:
            try:
                question = input()
                if question.strip():
                    self.input_queue.put(question.strip())
            except (EOFError, KeyboardInterrupt):
                break
            except:
                pass
    
    def process_questions(self):
        """Process user questions"""
        try:
            while not self.input_queue.empty():
                question = self.input_queue.get_nowait()
                
                if question.lower() in ['q', 'quit', 'exit']:
                    self.running = False
                    return
                
                print(f"\n🤔 Q: {question}")
                answer = self.transcriber.answer_question(question, self.current_detections)
                print(f"🤖 A: {answer}")
                print("💬 Next question: ", end="", flush=True)
                
        except queue.Empty:
            pass
    
    def draw_overlay(self, frame, result, fps, frame_count):
        """Draw minimal overlay"""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Basic info
        detection_count = len(result.boxes) if result.boxes is not None else 0
        cv2.rectangle(overlay, (10, 10), (400, 80), (0, 0, 0), -1)
        cv2.rectangle(overlay, (10, 10), (400, 80), (255, 255, 255), 2)
        
        cv2.putText(overlay, f"Objects: {detection_count} | FPS: {fps:.1f}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(overlay, f"Frame: {frame_count}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Current description
        desc_lines = self.current_description[:100].split()
        line1 = " ".join(desc_lines[:8])
        line2 = " ".join(desc_lines[8:]) if len(desc_lines) > 8 else ""
        
        cv2.rectangle(overlay, (10, 90), (w-10, 150), (0, 0, 0), -1)
        cv2.rectangle(overlay, (10, 90), (w-10, 150), (0, 255, 0), 2)
        cv2.putText(overlay, line1, (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if line2:
            cv2.putText(overlay, line2, (20, 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Controls
        cv2.rectangle(overlay, (w-200, 10), (w-10, 60), (0, 0, 0), -1)
        cv2.putText(overlay, "Q/ESC: Quit", (w-190, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(overlay, "Type questions!", (w-190, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        return overlay
    
    def run(self):
        """Main execution"""
        print("🚀 YOLO + GPT-OSS Scene Transcription")
        print("=" * 50)
        
        # Setup
        camera_index = self.find_working_camera()
        if camera_index is None:
            print("❌ No camera found!")
            return False
        
        if not self.load_yolo_model():
            return False
        
        # Start Ollama
        print("🔄 Starting Ollama...")
        if not self.transcriber.start_ollama_silent():
            print("⚠️  Ollama not ready, using basic descriptions")
        else:
            print("✅ Ollama ready!")
        
        device = self.pick_device()
        
        # Import distance estimator
        from main import SimpleDistanceEstimator
        
        # Start workers
        desc_thread = threading.Thread(target=self.description_worker, daemon=True)
        desc_thread.start()
        
        input_thread = threading.Thread(target=self.input_worker, daemon=True)
        input_thread.start()
        
        # Setup window
        win = "🎥 YOLO + GPT-OSS Transcription"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1200, 800)
        
        print("=" * 50)
        print("📝 LIVE TRANSCRIPTION:")
        print("=" * 50)
        print("💬 Type questions anytime and press Enter!")
        print("💬 Ask about the scene or anything else")
        print("💬 Type 'q' to quit")
        print("=" * 50)
        
        t0, n = time.time(), 0
        
        try:
            for result in self.yolo_model(source=camera_index, stream=True,
                                         conf=self.args.conf, imgsz=self.args.imgsz, device=device):
                
                if not self.running:
                    break
                
                frame = result.plot()
                
                # Initialize distance estimator
                if self.distance_estimator is None:
                    h, w = frame.shape[:2]
                    self.distance_estimator = SimpleDistanceEstimator(w, h)
                
                n += 1
                dt = time.time() - t0
                fps = (n / dt) if dt > 0 else 0.0
                
                # Extract detections
                detections = self.extract_detections(result)
                self.current_detections = detections
                
                # Process questions
                self.process_questions()
                
                # Schedule descriptions
                current_time = time.time()
                if (current_time - self.last_description_time) >= self.description_interval:
                    frame_info = {'frame': n, 'time': dt, 'fps': fps}
                    
                    try:
                        self.description_queue.put_nowait((detections, frame_info))
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
            print("\n⏹️  Stopping...")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        finally:
            self.running = False
            self.description_queue.put(None)
            cv2.destroyAllWindows()
            print("\n" + "=" * 50)
            print("✅ Session complete")
            print("=" * 50)
        
        return True

def main():
    parser = argparse.ArgumentParser(description="YOLO + GPT-OSS Scene Transcription")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model")
    parser.add_argument("--conf", type=float, default=0.6, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    
    args = parser.parse_args()
    
    system = IntegratedSystem(args)
    system.run()

if __name__ == "__main__":
    main()
