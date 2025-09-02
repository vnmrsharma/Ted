#!/usr/bin/env python3
"""
Integrated YOLOv8n + GPT-OSS-20B Real-time Scene Transcription
Combines computer vision detection with AI-powered scene description
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
from collections import deque
from ultralytics import YOLO
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc
try:
    import select
    import termios
    import tty
    TERMINAL_INPUT_AVAILABLE = True
except ImportError:
    TERMINAL_INPUT_AVAILABLE = False

class SceneTranscriber:
    """GPT-OSS-20B powered scene transcription system"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.loaded = False
        self.loading = False
        
    def load_model(self):
        """Load GPT-OSS-20B model for scene description"""
        if self.loading or self.loaded:
            return
            
        self.loading = True
        print("\n🧠 Loading GPT-OSS-20B for scene transcription...")
        
        try:
            model_path = "./gpt-oss-20b/original"
            
            # Load tokenizer
            print("📥 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b", trust_remote_code=True)
            
            # Clear memory
            gc.collect()
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
            # Load model with memory efficiency
            print("📂 Loading GPT-OSS-20B model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                local_files_only=True,
                max_memory={"cpu": "15GB"}
            )
            
            self.loaded = True
            print("✅ GPT-OSS-20B loaded successfully for scene transcription!")
            
        except Exception as e:
            print(f"❌ Failed to load GPT-OSS-20B: {e}")
            print("💡 Scene transcription will use basic descriptions")
            self.loaded = False
        finally:
            self.loading = False
    
    def generate_scene_description(self, detections, frame_info):
        """Generate natural language description of the scene"""
        if not self.loaded or not self.model:
            return self._basic_description(detections, frame_info)
        
        try:
            # Create detection summary
            detection_text = self._format_detections(detections)
            
            # Create prompt for scene description
            prompt = f"""Describe what you see in this camera view in a natural, conversational way. Keep it brief (1-2 sentences).

Detected objects: {detection_text}
Time: {frame_info['time']}
Frame: {frame_info['frame']}

Scene description:"""

            return self._generate_response(prompt, max_tokens=60)
            
        except Exception as e:
            return self._basic_description(detections, frame_info)
    
    def answer_question(self, question, detections=None):
        """Answer user questions about the scene or general topics"""
        if not self.loaded or not self.model:
            return "Sorry, the GPT-OSS model is not loaded yet. Please wait..."
        
        try:
            # Create context-aware prompt
            if detections:
                detection_text = self._format_detections(detections)
                prompt = f"""The user is asking about the current camera view. Here's what's currently visible: {detection_text}

User question: {question}

Answer the question naturally and helpfully:"""
            else:
                prompt = f"""User question: {question}

Please provide a helpful and informative answer:"""
            
            return self._generate_response(prompt, max_tokens=150)
            
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
    
    def _generate_response(self, prompt, max_tokens=100):
        """Common response generation method"""
        try:
            # Format for GPT-OSS
            formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
            
            # Tokenize
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=512)
            
            # Move to model device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract response
            if "<|assistant|>" in full_response:
                response = full_response.split("<|assistant|>")[-1].strip()
            else:
                response = full_response[len(formatted_prompt):].strip()
            
            # Clean up response
            response = response.split('\n')[0]  # Take first line only
            if max_tokens <= 60:  # For scene descriptions
                response = response.strip('."').strip() + "."
            
            return response
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _format_detections(self, detections):
        """Format detections for GPT prompt"""
        if not detections:
            return "No objects detected"
        
        detection_list = []
        for obj_name, info in detections.items():
            count = info['count']
            distance = info['avg_distance']
            
            if distance < 1.0:
                dist_text = f"{distance*100:.0f}cm away"
            elif distance < 10.0:
                dist_text = f"{distance:.1f}m away"
            else:
                dist_text = f"{distance:.0f}m away"
            
            if count == 1:
                detection_list.append(f"{obj_name} ({dist_text})")
            else:
                detection_list.append(f"{count} {obj_name}s ({dist_text})")
        
        return ", ".join(detection_list)
    
    def _basic_description(self, detections, frame_info):
        """Fallback basic description when GPT is not available"""
        if not detections:
            return "The camera shows an empty scene with no detected objects."
        
        obj_count = sum(info['count'] for info in detections.values())
        obj_types = list(detections.keys())
        
        if obj_count == 1:
            return f"I can see a {obj_types[0]} in the camera view."
        elif len(obj_types) == 1:
            return f"I can see {obj_count} {obj_types[0]}s in the camera view."
        else:
            main_objects = ", ".join(obj_types[:3])
            return f"The camera shows {obj_count} objects including {main_objects}."

class TerminalInputHandler:
    """Handle terminal input for questions while video is running"""
    
    def __init__(self, transcriber, detection_system):
        self.transcriber = transcriber
        self.detection_system = detection_system
        self.input_queue = queue.Queue()
        self.running = True
        
    def input_worker(self):
        """Background worker for handling terminal input"""
        if not TERMINAL_INPUT_AVAILABLE:
            # Fallback to simple input for compatibility
            while self.running:
                try:
                    line = input()
                    if line.strip():
                        self.input_queue.put(line.strip())
                except:
                    break
            return
            
        try:
            # Save terminal settings
            old_settings = termios.tcgetattr(sys.stdin)
            
            while self.running:
                try:
                    # Simple input method for better compatibility
                    line = input()
                    if line.strip():
                        self.input_queue.put(line.strip())
                            
                except (EOFError, KeyboardInterrupt):
                    break
                except Exception:
                    pass
                    
        except Exception:
            pass
        finally:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            except:
                pass
    
    def process_input(self):
        """Process any pending input"""
        try:
            while not self.input_queue.empty():
                question = self.input_queue.get_nowait()
                
                # Get current detections for context
                current_detections = getattr(self.detection_system, 'current_detections', None)
                
                # Generate answer
                print(f"\n🤔 Question: {question}")
                answer = self.transcriber.answer_question(question, current_detections)
                print(f"🤖 Answer: {answer}")
                print("💬 Ask another question (or press Q to quit): ", end="", flush=True)
                
        except queue.Empty:
            pass
    
    def stop(self):
        """Stop the input handler"""
        self.running = False

class IntegratedDetectionSystem:
    """Integrated YOLO + GPT-OSS system for real-time scene analysis"""
    
    def __init__(self, args):
        self.args = args
        self.yolo_model = None
        self.transcriber = SceneTranscriber()
        self.distance_estimator = None
        self.memory_monitor = None
        self.description_queue = queue.Queue(maxsize=5)
        self.current_description = "Initializing camera..."
        self.description_history = deque(maxlen=10)
        self.last_description_time = 0
        self.description_interval = 3.0  # Generate description every 3 seconds
        self.current_detections = None  # Store current detections for Q&A context
        self.input_handler = None
        
    def pick_device(self):
        """Select the best available device"""
        if torch.cuda.is_available():
            return 0
        try:
            if torch.backends.mps.is_available():
                return "mps"
        except AttributeError:
            pass
        return "cpu"
    
    def find_working_camera(self):
        """Find the first working camera"""
        for i in range(4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                if ret:
                    print(f"✓ Camera {i} is working")
                    return i
                else:
                    print(f"✗ Camera {i} opened but failed to read frame")
            else:
                print(f"✗ Failed to open camera {i}")
        return None
    
    def load_yolo_model(self):
        """Load YOLO model"""
        try:
            self.yolo_model = YOLO(self.args.model)
            print(f"✅ YOLO model loaded: {self.args.model}")
            return True
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")
            return False
    
    def extract_detections(self, result):
        """Extract and process detection information"""
        detections = {}
        
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else []
        class_names = result.names if hasattr(result, 'names') else {}
        
        # Process each detection
        for i, cls_id in enumerate(classes):
            cls_name = class_names.get(int(cls_id), f"Class {int(cls_id)}")
            conf = confidences[i] if i < len(confidences) else 0.0
            
            # Only include high-confidence detections for description
            if conf < 0.5:
                continue
            
            # Calculate distance
            if self.distance_estimator:
                distance = self.distance_estimator.estimate_distance(result.boxes[i], cls_name)
            else:
                distance = 5.0  # Default distance
            
            # Update detection info
            if cls_name not in detections:
                detections[cls_name] = {
                    'count': 0,
                    'max_conf': 0.0,
                    'distances': []
                }
            
            detections[cls_name]['count'] += 1
            detections[cls_name]['max_conf'] = max(detections[cls_name]['max_conf'], conf)
            detections[cls_name]['distances'].append(distance)
        
        # Calculate average distances
        for cls_name in detections:
            if detections[cls_name]['distances']:
                detections[cls_name]['avg_distance'] = np.mean(detections[cls_name]['distances'])
            else:
                detections[cls_name]['avg_distance'] = 5.0
        
        return detections
    
    def description_worker(self):
        """Background worker for generating scene descriptions"""
        while True:
            try:
                # Get task from queue (blocks until available)
                task = self.description_queue.get(timeout=1.0)
                if task is None:  # Shutdown signal
                    break
                
                detections, frame_info = task
                
                # Generate description
                description = self.transcriber.generate_scene_description(detections, frame_info)
                
                # Update current description
                self.current_description = description
                self.description_history.append({
                    'time': time.time(),
                    'description': description,
                    'frame': frame_info['frame']
                })
                
                # Clean terminal output for transcription
                timestamp = time.strftime("%H:%M:%S")
                print(f"[{timestamp}] {description}")
                
                self.description_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                pass  # Silent error handling for clean output
    
    def draw_enhanced_overlay(self, frame, result, fps, frame_count, elapsed_time):
        """Draw enhanced overlay with scene transcription"""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Main info panel
        detection_count = len(result.boxes) if result.boxes is not None else 0
        cv2.rectangle(overlay, (10, 10), (600, 120), (0, 0, 0), -1)
        cv2.rectangle(overlay, (10, 10), (600, 120), (255, 255, 255), 2)
        
        # Detection stats
        cv2.putText(overlay, f"Detections: {detection_count} | FPS: {fps:.1f}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(overlay, f"Frame: {frame_count} | Time: {elapsed_time:.1f}s", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Model status
        gpt_status = "✅ Active" if self.transcriber.loaded else "⏳ Loading" if self.transcriber.loading else "❌ Offline"
        cv2.putText(overlay, f"GPT-OSS: {gpt_status}", (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Memory info
        if self.memory_monitor:
            memory_info = self.memory_monitor.get_memory_info()
            if memory_info:
                ram_mb = memory_info['process']['rss_mb']
                cv2.putText(overlay, f"RAM: {ram_mb:.0f}MB", (20, 105),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Scene description panel
        desc_panel_height = min(150, h - 140)
        cv2.rectangle(overlay, (10, 130), (w - 10, 130 + desc_panel_height), (0, 0, 0), -1)
        cv2.rectangle(overlay, (10, 130), (w - 10, 130 + desc_panel_height), (0, 255, 0), 2)
        
        cv2.putText(overlay, "🎙️  AI Scene Description:", (20, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Word wrap the description
        description = self.current_description
        words = description.split()
        lines = []
        current_line = ""
        max_chars = 80  # Approximate characters per line
        
        for word in words:
            if len(current_line + " " + word) < max_chars:
                current_line += (" " + word) if current_line else word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        # Display description lines
        y_pos = 180
        for i, line in enumerate(lines[:4]):  # Show max 4 lines
            cv2.putText(overlay, line, (20, y_pos + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Instructions
        cv2.rectangle(overlay, (w - 250, 10), (w - 10, 100), (0, 0, 0), -1)
        cv2.rectangle(overlay, (w - 250, 10), (w - 10, 100), (255, 255, 255), 2)
        cv2.putText(overlay, "Controls:", (w - 240, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(overlay, "Q/ESC - Quit", (w - 240, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(overlay, "R - Reload GPT", (w - 240, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(overlay, "D - Force Description", (w - 240, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return overlay
    
    def run(self):
        """Main execution loop"""
        print("🚀 Integrated YOLO + GPT-OSS Scene Transcription System")
        print("=" * 70)
        
        # Initialize components
        camera_index = self.find_working_camera()
        if camera_index is None:
            print("❌ No working camera found!")
            return False
        
        if not self.load_yolo_model():
            return False
        
        device = self.pick_device()
        print(f"🔧 Using device: {device}")
        
        # Initialize memory monitor
        from main import MemoryMonitor, SimpleDistanceEstimator
        self.memory_monitor = MemoryMonitor()
        
        # Start GPT model loading in background
        gpt_thread = threading.Thread(target=self.transcriber.load_model, daemon=True)
        gpt_thread.start()
        
        # Start description worker
        desc_thread = threading.Thread(target=self.description_worker, daemon=True)
        desc_thread.start()
        
        # Setup input handler for Q&A
        self.input_handler = TerminalInputHandler(self.transcriber, self)
        input_thread = threading.Thread(target=self.input_handler.input_worker, daemon=True)
        input_thread.start()
        
        # Setup window
        win = "🎥 YOLO + GPT-OSS Scene Transcription"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1400, 900)
        
        print(f"🎥 Starting camera detection on device {camera_index}")
        print("🧠 GPT-OSS-20B loading for scene transcription...")
        print("=" * 70)
        print("📝 LIVE TRANSCRIPTION:")
        print("=" * 70)
        print("💬 Type questions anytime and press Enter!")
        print("💬 Ask about the scene or general topics")
        print("Q or ESC to quit")
        print("=" * 70)
        
        t0, n = time.time(), 0
        
        try:
            # Main detection loop
            for result in self.yolo_model(source=camera_index, stream=True,
                                         conf=self.args.conf, imgsz=self.args.imgsz, device=device):
                
                frame = result.plot()
                
                # Initialize distance estimator
                if self.distance_estimator is None:
                    h, w = frame.shape[:2]
                    self.distance_estimator = SimpleDistanceEstimator(w, h)
                
                n += 1
                dt = time.time() - t0
                fps = (n / dt) if dt > 0 else 0.0
                
                # Extract detections for transcription
                detections = self.extract_detections(result)
                self.current_detections = detections  # Store for Q&A context
                
                # Process any pending questions from terminal
                if self.input_handler:
                    self.input_handler.process_input()
                
                # Schedule description generation
                current_time = time.time()
                if (current_time - self.last_description_time) >= self.description_interval:
                    frame_info = {
                        'frame': n,
                        'time': dt,
                        'fps': fps
                    }
                    
                    # Add to queue if not full
                    try:
                        self.description_queue.put_nowait((detections, frame_info))
                        self.last_description_time = current_time
                    except queue.Full:
                        pass  # Skip if queue is full
                
                # Silent memory logging (no console output)
                if n % 300 == 0:  # Every 300 frames (less frequent)
                    pass  # Silent monitoring for clean output
                
                # Draw enhanced overlay
                enhanced_frame = self.draw_enhanced_overlay(frame, result, fps, n, dt)
                cv2.imshow(win, enhanced_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):  # ESC or Q
                    break
                elif key == ord("r"):  # Reload GPT (silent)
                    self.transcriber.loaded = False
                    gpt_thread = threading.Thread(target=self.transcriber.load_model, daemon=True)
                    gpt_thread.start()
                elif key == ord("d"):  # Force description
                    frame_info = {'frame': n, 'time': dt, 'fps': fps}
                    try:
                        self.description_queue.put_nowait((detections, frame_info))
                    except queue.Full:
                        pass
                
        except KeyboardInterrupt:
            print("\n⏹️  Interrupted by user")
        except Exception as e:
            print(f"\n❌ Error during detection: {e}")
        finally:
            # Cleanup
            self.description_queue.put(None)  # Shutdown signal
            if self.input_handler:
                self.input_handler.stop()
            cv2.destroyAllWindows()
            print(f"\n=" * 70)
            print(f"📊 Session completed: {n} frames in {time.time() - t0:.1f}s")
            print("✅ System shutdown complete")
            print("=" * 70)
        
        return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Integrated YOLOv8n + GPT-OSS-20B Scene Transcription System")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model path")
    parser.add_argument("--source", type=int, default=0, help="Camera index")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    
    args = parser.parse_args()
    
    # Create and run integrated system
    system = IntegratedDetectionSystem(args)
    success = system.run()
    
    if success:
        print("✅ System completed successfully")
    else:
        print("❌ System failed to start")
        sys.exit(1)

if __name__ == "__main__":
    main()
