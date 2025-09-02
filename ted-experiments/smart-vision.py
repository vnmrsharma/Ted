#!/usr/bin/env python3
"""
Smart Vision Chat - Continuous Scene Analysis with GPT-OSS
Real-time scene understanding with context-aware AI responseslive 
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
from collections import dequeh the 
from ultralytics import YOLO

class SceneAnalyzer:
    """Intelligent scene analysis and change detection"""
    
    def __init__(self):
        self.current_scene = {}
        self.scene_history = deque(maxlen=10)
        self.last_significant_change = 0
        self.change_threshold = 0.3  # 30% change triggers update
        
    def analyze_scene_change(self, new_detections):
        """Detect if scene has changed significantly"""
        if not self.current_scene:
            # First scene
            self.current_scene = new_detections.copy()
            return True, "initial_scene"
        
        # Calculate change metrics
        total_objects_old = sum(self.current_scene.values())
        total_objects_new = sum(new_detections.values())
        
        # Check for new/removed object types
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
            change_reasons.append(f"New: {', '.join(added_types)}")
        if removed_types:
            change_reasons.append(f"Removed: {', '.join(removed_types)}")
        if quantity_changes:
            for obj_type, old, new in quantity_changes:
                change_reasons.append(f"{obj_type}: {old}→{new}")
        
        # Calculate overall change percentage
        if total_objects_old > 0:
            change_percentage = abs(total_objects_new - total_objects_old) / total_objects_old
        else:
            change_percentage = 1.0 if total_objects_new > 0 else 0.0
        
        # Determine if change is significant
        is_significant = (
            len(added_types) > 0 or 
            len(removed_types) > 0 or 
            len(quantity_changes) > 0 or
            change_percentage > self.change_threshold
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
            return "Empty scene with no objects detected."
        
        context_parts = []
        for obj_type, count in self.current_scene.items():
            if count == 1:
                context_parts.append(f"1 {obj_type}")
            else:
                context_parts.append(f"{count} {obj_type}s")
        
        return f"Current scene contains: {', '.join(context_parts)}."

class SmartGPTOSS:
    """Enhanced GPT-OSS with continuous scene understanding"""
    
    def __init__(self):
        self.base_url = "http://127.0.0.1:11434"
        self.model_name = "gpt-oss:20b"
        self.ready = False
        self.loading = False
        self.error_message = ""
        self.process = None
        self.scene_memory = deque(maxlen=5)  # Remember last 5 scene descriptions
        
    def start_service(self):
        """Start Ollama service with better error handling"""
        print("🚀 Starting AI Vision System...")
        
        try:
            # Clean shutdown of existing processes
            subprocess.run(["pkill", "-f", "ollama"], capture_output=True, text=True)
            time.sleep(3)
            
            # Start fresh service
            with open(os.devnull, 'w') as devnull:
                self.process = subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=devnull,
                    stderr=devnull,
                    preexec_fn=os.setsid
                )
            
            # Wait for proper startup
            print("⏳ Initializing AI service...")
            time.sleep(8)
            
            return self._verify_and_setup_model()
            
        except FileNotFoundError:
            self.error_message = "Ollama not installed. Visit ollama.com"
            return False
        except Exception as e:
            self.error_message = f"Service error: {str(e)[:40]}"
            return False
    
    def _verify_and_setup_model(self):
        """Verify service and ensure model is ready"""
        max_retries = 15
        
        for attempt in range(max_retries):
            try:
                # Test connection
                response = requests.get(f"{self.base_url}/api/tags", timeout=3)
                if response.status_code != 200:
                    time.sleep(2)
                    continue
                
                # Check for model
                models = response.json().get("models", [])
                model_exists = any(self.model_name in model.get("name", "") for model in models)
                
                if not model_exists:
                    print(f"📥 Downloading {self.model_name}...")
                    self.loading = True
                    
                    # Pull model with timeout
                    pull_response = requests.post(
                        f"{self.base_url}/api/pull",
                        json={"name": self.model_name},
                        timeout=600  # 10 minutes
                    )
                    
                    if pull_response.status_code != 200:
                        self.error_message = "Model download failed"
                        return False
                
                # Test model with simple prompt
                test_result = self._generate_safe("Say 'Ready'", max_tokens=5)
                if "ready" in test_result.lower() or len(test_result) > 0:
                    self.ready = True
                    self.loading = False
                    print("✅ AI Vision System Ready!")
                    return True
                
                time.sleep(2)
                
            except requests.exceptions.RequestException:
                time.sleep(2)
                continue
            except Exception as e:
                print(f"Setup attempt {attempt + 1}/{max_retries} failed")
                time.sleep(2)
                continue
        
        self.error_message = "Service verification failed"
        return False
    
    def _generate_safe(self, prompt, max_tokens=100, timeout=15):
        """Safe text generation with proper error handling"""
        if not self.ready:
            return "AI system not ready"
        
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.8,
                    "top_p": 0.9,
                    "stop": ["\n\n", "User:", "Question:"]
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("response", "").strip()
                return generated_text if generated_text else "No response generated"
            else:
                return f"API Error {response.status_code}"
                
        except requests.exceptions.Timeout:
            return "Response timeout - AI is thinking too hard!"
        except requests.exceptions.RequestException as e:
            return f"Connection issue: {str(e)[:30]}"
        except Exception as e:
            return f"Generation error: {str(e)[:30]}"
    
    def analyze_scene_continuously(self, scene_context, change_description=""):
        """Generate scene description based on current context"""
        if not scene_context or "Empty scene" in scene_context:
            return "The camera view is clear with no objects detected."
        
        # Create context-aware prompt
        if change_description and change_description != "initial_scene":
            prompt = f"""Current scene: {scene_context}

Recent change: {change_description}

Describe what you observe in this camera view naturally and conversationally (1-2 sentences). Focus on the current state and any notable changes."""
        else:
            prompt = f"""Current scene: {scene_context}

Describe what you see in this camera view in a natural, friendly way (1-2 sentences)."""
        
        description = self._generate_safe(prompt, max_tokens=80, timeout=20)
        
        # Store in memory
        self.scene_memory.append({
            'timestamp': time.time(),
            'context': scene_context,
            'description': description,
            'change': change_description
        })
        
        return description
    
    def answer_contextual_question(self, question, current_scene_context):
        """Answer questions with full scene awareness"""
        # Build context from recent scene memory
        scene_history = ""
        if len(self.scene_memory) > 1:
            recent_scenes = list(self.scene_memory)[-3:]  # Last 3 scenes
            scene_parts = []
            for scene in recent_scenes:
                scene_parts.append(f"- {scene['description']}")
            scene_history = f"\n\nRecent scene context:\n" + "\n".join(scene_parts)
        
        prompt = f"""Current live camera view: {current_scene_context}{scene_history}

User question: {question}

Answer the question naturally, considering what you can currently see in the camera view. Be helpful and specific about the current scene."""
        
        return self._generate_safe(prompt, max_tokens=150, timeout=25)
    
    def cleanup(self):
        """Clean shutdown"""
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

class ContinuousVisionChat:
    """Main application with continuous scene analysis"""
    
    def __init__(self, args):
        self.args = args
        self.yolo_model = None
        self.ai = SmartGPTOSS()
        self.scene_analyzer = SceneAnalyzer()
        
        # UI State
        self.current_input = ""
        self.input_mode = False
        self.cursor_blink = True
        self.last_blink = time.time()
        
        # Chat and Scene State
        self.chat_history = deque(maxlen=8)
        self.current_description = "Starting camera..."
        self.current_scene_context = ""
        self.last_update_time = 0
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        
        # Threading
        self.scene_queue = queue.Queue(maxsize=3)
        self.running = True
        
        # Enhanced colors
        self.colors = {
            'bg_dark': (25, 25, 35),
            'panel_dark': (40, 45, 55),
            'panel_light': (55, 60, 70),
            'accent_blue': (255, 140, 0),
            'accent_green': (0, 200, 100),
            'accent_orange': (0, 165, 255),
            'accent_red': (60, 60, 255),
            'text_bright': (255, 255, 255),
            'text_dim': (180, 180, 180),
            'text_dark': (120, 120, 120)
        }
    
    def setup(self):
        """Initialize the complete system"""
        print("🎯 Smart Vision Chat v2.0")
        print("=" * 50)
        
        # Setup camera
        camera_id = self._find_working_camera()
        if camera_id is None:
            print("❌ No camera available!")
            return None
        
        # Setup YOLO
        try:
            self.yolo_model = YOLO(self.args.model)
            device = self._get_optimal_device()
            print(f"✅ YOLO model loaded on {device}")
        except Exception as e:
            print(f"❌ YOLO initialization failed: {e}")
            return None
        
        # Start AI system in background
        ai_thread = threading.Thread(target=self._initialize_ai_system, daemon=True)
        ai_thread.start()
        
        print("=" * 50)
        print("🎥 Camera active - Real-time scene analysis")
        print("💬 Press SPACE to ask questions about what you see")
        print("🔄 Press R to restart AI | Q/ESC to quit")
        print("=" * 50)
        
        return camera_id, device
    
    def _find_working_camera(self):
        """Find the first working camera"""
        for i in range(6):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    cap.release()
                    if ret and frame is not None:
                        print(f"📷 Using camera {i}")
                        return i
            except:
                continue
        return None
    
    def _get_optimal_device(self):
        """Get the best available device for YOLO"""
        if torch.cuda.is_available():
            return 0  # GPU
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"  # Apple Silicon
        return "cpu"
    
    def _initialize_ai_system(self):
        """Initialize AI system in background"""
        success = self.ai.start_service()
        if success:
            print("🤖 AI system online and ready!")
        else:
            print(f"⚠️ AI initialization issue: {self.ai.error_message}")
    
    def _extract_detections_improved(self, result):
        """Extract detections with proper array handling"""
        detections = {}
        
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        # Get class names
        class_names = result.names if hasattr(result, 'names') else {}
        
        # Process each detection with proper array indexing
        for i, box in enumerate(result.boxes):
            try:
                # Fix NumPy warnings by proper array element extraction
                cls_tensor = box.cls.cpu()
                conf_tensor = box.conf.cpu()
                
                # Extract single values properly
                cls_id = int(cls_tensor[0].item() if cls_tensor.numel() > 0 else 0)
                confidence = float(conf_tensor[0].item() if conf_tensor.numel() > 0 else 0.0)
                
                # Filter by confidence
                if confidence < 0.65:  # Higher confidence for cleaner results
                    continue
                
                # Get class name
                class_name = class_names.get(cls_id, f"object_{cls_id}")
                
                # Count detections
                detections[class_name] = detections.get(class_name, 0) + 1
                
            except Exception as e:
                print(f"Detection processing error: {e}")
                continue
        
        return detections
    
    def _scene_analysis_worker(self):
        """Background worker for continuous scene analysis"""
        while self.running:
            try:
                task = self.scene_queue.get(timeout=1.0)
                if task is None:  # Shutdown signal
                    break
                
                detections, timestamp = task
                
                # Analyze scene changes
                scene_changed, change_desc = self.scene_analyzer.analyze_scene_change(detections)
                
                if scene_changed:
                    # Get updated scene context
                    self.current_scene_context = self.scene_analyzer.get_scene_context()
                    
                    # Generate new description
                    new_description = self.ai.analyze_scene_continuously(
                        self.current_scene_context, 
                        change_desc
                    )
                    
                    self.current_description = new_description
                    self.last_update_time = timestamp
                
                self.scene_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Scene analysis error: {e}")
                continue
    
    def _handle_user_input(self, key):
        """Handle all user input"""
        if key == 32:  # Space
            if not self.input_mode:
                # Start input mode
                self.input_mode = True
                self.current_input = ""
                print("💬 Ask about the current scene...")
            else:
                # Submit question
                if self.current_input.strip():
                    self._process_user_question(self.current_input.strip())
                self._exit_input_mode()
        
        elif key in [27, ord('q'), ord('Q')]:  # ESC or Q
            if self.input_mode:
                self._exit_input_mode()
            else:
                self.running = False
        
        elif key == ord('r') or key == ord('R'):  # Restart AI
            print("🔄 Restarting AI system...")
            threading.Thread(target=self._restart_ai, daemon=True).start()
        
        elif key == 8 and self.input_mode:  # Backspace
            if len(self.current_input) > 0:
                self.current_input = self.current_input[:-1]
        
        elif self.input_mode and 32 <= key <= 126:  # Printable characters
            if len(self.current_input) < 100:  # Reasonable limit
                self.current_input += chr(key)
    
    def _process_user_question(self, question):
        """Process user question with scene context"""
        # Add to chat history
        self.chat_history.append({
            'question': question,
            'answer': "Analyzing scene...",
            'timestamp': time.time(),
            'scene_context': self.current_scene_context
        })
        
        # Generate answer in background
        threading.Thread(
            target=self._generate_contextual_answer,
            args=(question, len(self.chat_history) - 1),
            daemon=True
        ).start()
    
    def _generate_contextual_answer(self, question, chat_index):
        """Generate answer with current scene context"""
        try:
            answer = self.ai.answer_contextual_question(question, self.current_scene_context)
            
            # Update the chat entry
            if chat_index < len(self.chat_history):
                self.chat_history[chat_index]['answer'] = answer
                
        except Exception as e:
            if chat_index < len(self.chat_history):
                self.chat_history[chat_index]['answer'] = f"Error: {str(e)[:40]}"
    
    def _restart_ai(self):
        """Restart AI system"""
        self.ai.cleanup()
        time.sleep(2)
        self.ai = SmartGPTOSS()
        self._initialize_ai_system()
    
    def _exit_input_mode(self):
        """Exit input mode"""
        self.input_mode = False
        self.current_input = ""
    
    def _draw_modern_panel(self, img, x1, y1, x2, y2, color, border_color=None):
        """Draw modern panel with subtle borders"""
        # Main panel
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        
        # Border
        if border_color:
            cv2.rectangle(img, (x1, y1), (x2, y2), border_color, 2)
    
    def _draw_text_wrapped(self, img, text, x, y, font_scale, color, max_width, line_height=25):
        """Draw text with word wrapping"""
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
                    lines.append(word)  # Single word longer than max_width
        
        if current_line:
            lines.append(current_line)
        
        # Draw lines
        for i, line in enumerate(lines):
            cv2.putText(img, line, (x, y + i * line_height), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)
        
        return len(lines) * line_height
    
    def _draw_enhanced_ui(self, frame, fps):
        """Draw the complete enhanced UI"""
        h, w = frame.shape[:2]
        
        # Create overlay
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Status bar
        self._draw_modern_panel(overlay, 15, 15, w-15, 85, self.colors['panel_dark'], self.colors['text_dark'])
        
        # AI status indicator
        if self.ai.loading:
            status_text = "🔄 AI Loading"
            status_color = self.colors['accent_orange']
        elif self.ai.ready:
            status_text = "🤖 AI Online"
            status_color = self.colors['accent_green']
        else:
            status_text = f"❌ {self.ai.error_message[:25]}"
            status_color = self.colors['accent_red']
        
        cv2.putText(overlay, status_text, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Performance stats
        elapsed = time.time() - self.start_time
        stats = f"Objects: {sum(self.scene_analyzer.current_scene.values())} | FPS: {fps:.1f} | Time: {elapsed:.0f}s"
        cv2.putText(overlay, stats, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text_dim'], 1)
        
        # Scene description panel
        desc_y = 95
        desc_height = 120
        self._draw_modern_panel(overlay, 15, desc_y, w-15, desc_y + desc_height, self.colors['panel_dark'])
        
        cv2.putText(overlay, "🎙️ Live Scene Analysis", (30, desc_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['accent_blue'], 2)
        
        # Scene description with wrapping
        if self.current_description:
            self._draw_text_wrapped(overlay, self.current_description, 30, desc_y + 60, 
                                  0.55, self.colors['text_bright'], w - 60, 22)
        
        # Chat panel
        chat_y = desc_y + desc_height + 10
        chat_height = h - chat_y - 140
        self._draw_modern_panel(overlay, 15, chat_y, w-15, chat_y + chat_height, self.colors['panel_light'])
        
        cv2.putText(overlay, "💬 Contextual Q&A", (30, chat_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['accent_blue'], 2)
        
        # Chat history
        y_pos = chat_y + 60
        for chat in list(self.chat_history)[-3:]:  # Show last 3
            if y_pos + 80 > chat_y + chat_height:
                break
            
            # Question
            q_height = self._draw_text_wrapped(overlay, f"Q: {chat['question']}", 30, y_pos, 
                                             0.5, self.colors['text_bright'], w - 60, 20)
            y_pos += q_height + 5
            
            # Answer
            a_height = self._draw_text_wrapped(overlay, f"A: {chat['answer']}", 30, y_pos, 
                                             0.5, self.colors['accent_green'], w - 60, 20)
            y_pos += a_height + 15
        
        # Input panel
        input_y = h - 120
        input_color = self.colors['accent_blue'] if self.input_mode else self.colors['panel_dark']
        self._draw_modern_panel(overlay, 15, input_y, w-15, h-15, input_color)
        
        # Input text and prompt
        if self.input_mode:
            cv2.putText(overlay, "💬 Ask about the scene (SPACE=Send, ESC=Cancel):", 
                       (30, input_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text_bright'], 1)
            
            # Input with blinking cursor
            current_time = time.time()
            if current_time - self.last_blink > 0.5:
                self.cursor_blink = not self.cursor_blink
                self.last_blink = current_time
            
            display_text = self.current_input + ("|" if self.cursor_blink else "")
            cv2.putText(overlay, display_text, (30, input_y + 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text_bright'], 2)
        else:
            cv2.putText(overlay, "💬 Press SPACE to ask about what you see", 
                       (30, input_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text_dim'], 1)
        
        # Controls
        controls = "SPACE=Ask | R=Restart | Q=Quit"
        cv2.putText(overlay, controls, (30, input_y + 95), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.colors['text_dark'], 1)
        
        # Blend overlay with frame
        alpha = 0.88
        result = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
        
        return result
    
    def run(self):
        """Main execution loop"""
        setup_result = self.setup()
        if setup_result is None:
            return False
        
        camera_id, device = setup_result
        
        # Start scene analysis worker
        scene_thread = threading.Thread(target=self._scene_analysis_worker, daemon=True)
        scene_thread.start()
        
        # Setup display window
        window_name = "Smart Vision Chat v2.0 - Continuous Scene Analysis"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1400, 900)
        
        print("🎬 Starting continuous scene analysis...")
        
        try:
            # Main detection loop
            for result in self.yolo_model(source=camera_id, stream=True,
                                         conf=self.args.conf, imgsz=self.args.imgsz, 
                                         device=device, verbose=False):
                
                if not self.running:
                    break
                
                # Get frame and update counters
                frame = result.plot()
                self.frame_count += 1
                elapsed_time = time.time() - self.start_time
                fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
                
                # Extract detections
                detections = self._extract_detections_improved(result)
                
                # Queue for scene analysis
                current_time = time.time()
                try:
                    self.scene_queue.put_nowait((detections, current_time))
                except queue.Full:
                    # Skip if queue is full (system is keeping up)
                    pass
                
                # Draw enhanced UI
                enhanced_frame = self._draw_enhanced_ui(frame, fps)
                cv2.imshow(window_name, enhanced_frame)
                
                # Handle input
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
            # Cleanup
            print("\n🧹 Cleaning up...")
            self.running = False
            self.scene_queue.put(None)  # Signal worker to stop
            cv2.destroyAllWindows()
            self.ai.cleanup()
            
            print("\n" + "=" * 50)
            print("✅ Smart Vision Chat session completed")
            print(f"📊 Analyzed {self.frame_count} frames")
            print(f"💬 Processed {len(self.chat_history)} conversations")
            print("=" * 50)
        
        return True

def main():
    """Application entry point"""
    parser = argparse.ArgumentParser(
        description="Smart Vision Chat v2.0 - Continuous Scene Analysis with GPT-OSS"
    )
    parser.add_argument("--model", default="yolov8n.pt", 
                       help="YOLO model path (default: yolov8n.pt)")
    parser.add_argument("--conf", type=float, default=0.65, 
                       help="Detection confidence threshold (default: 0.65)")
    parser.add_argument("--imgsz", type=int, default=640, 
                       help="Input image size (default: 640)")
    
    args = parser.parse_args()
    
    # Create and run the application
    app = ContinuousVisionChat(args)
    success = app.run()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()