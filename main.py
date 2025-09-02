import argparse
import time
import cv2
import torch
import sys
import numpy as np
import psutil
import os
from ultralytics import YOLO

def pick_device():
    # Prefer CUDA, then Apple MPS, else CPU
    if torch.cuda.is_available():
        return 0  # "0" == first CUDA GPU
    try:
        if torch.backends.mps.is_available():  # Apple Silicon
            return "mps"
    except AttributeError:
        pass
    return "cpu"

def check_camera_access(camera_index=0):
    """Check if camera is accessible and working"""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return False, f"Failed to open camera {camera_index}"
    
    # Try to read a frame to ensure camera is working
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        return False, f"Camera {camera_index} opened but failed to read frame"
    
    return True, f"Camera {camera_index} is working"

def find_working_camera():
    """Find the first working camera"""
    for i in range(4):  # Check first 4 camera indices
        success, message = check_camera_access(i)
        if success:
            print(f"✓ {message}")
            return i
        else:
            print(f"✗ {message}")
    
    return None

class MemoryMonitor:
    """Monitor program-specific memory and storage usage for Raspberry Pi compatibility"""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_time = time.time()
        self.memory_history = []
        self.max_history = 100
        self.peak_memory = 0
        self.peak_storage = 0
        
    def get_memory_info(self):
        """Get current program-specific memory and storage information"""
        try:
            # Process memory only
            process_memory = self.process.memory_info()
            process_rss = process_memory.rss / 1024 / 1024  # MB
            process_vms = process_memory.vms / 1024 / 1024  # MB
            
            # Update peak memory
            if process_rss > self.peak_memory:
                self.peak_memory = process_rss
            
            # Program-specific storage requirements
            storage_info = self._get_program_storage()
            
            # Update peak storage
            if storage_info and storage_info['total_mb'] > self.peak_storage:
                self.peak_storage = storage_info['total_mb']
            
            # Update history
            self.memory_history.append({
                'timestamp': time.time(),
                'process_rss': process_rss,
                'process_vms': process_vms
            })
            
            # Keep only recent history
            if len(self.memory_history) > self.max_history:
                self.memory_history = self.memory_history[-self.max_history:]
            
            return {
                'process': {
                    'rss_mb': process_rss,
                    'vms_mb': process_vms,
                    'peak_rss_mb': self.peak_memory
                },
                'storage': storage_info,
                'uptime': time.time() - self.start_time
            }
        except Exception as e:
            print(f"Warning: Could not get memory info: {e}")
            return None
    
    def _get_program_storage(self):
        """Get storage requirements specific to this program"""
        try:
            current_path = os.getcwd()
            total_mb = 0
            file_details = []
            
            # Check essential files for this program
            essential_files = [
                'yolov8n.pt',  # YOLO model
                'main.py',      # Main program
                'requirements.txt'  # Dependencies
            ]
            
            for filename in essential_files:
                if os.path.exists(filename):
                    size_mb = os.path.getsize(filename) / 1024 / 1024
                    total_mb += size_mb
                    file_details.append({
                        'name': filename,
                        'size_mb': size_mb
                    })
            
            # Estimate Python environment size (approximate)
            venv_size_mb = 0
            if os.path.exists('venv'):
                venv_size_mb = self._estimate_venv_size('venv')
                total_mb += venv_size_mb
            
            # Add estimated runtime storage (logs, temp files)
            runtime_storage_mb = 50  # Conservative estimate for logs and temp files
            
            return {
                'total_mb': total_mb + runtime_storage_mb,
                'essential_files_mb': total_mb - venv_size_mb,
                'venv_mb': venv_size_mb,
                'runtime_mb': runtime_storage_mb,
                'file_details': file_details,
                'raspberry_pi_compatible': (total_mb + runtime_storage_mb) <= 5000  # 5GB limit
            }
        except Exception as e:
            print(f"Warning: Could not get storage info: {e}")
            return None
    
    def _estimate_venv_size(self, venv_path):
        """Estimate virtual environment size"""
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(venv_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
            return total_size / 1024 / 1024  # Convert to MB
        except Exception:
            return 0
    
    def get_memory_trends(self):
        """Get memory usage trends over time"""
        if len(self.memory_history) < 2:
            return None
        
        try:
            recent = self.memory_history[-10:]  # Last 10 samples
            
            rss_values = [h['process_rss'] for h in recent]
            vms_values = [h['process_vms'] for h in recent]
            
            trends = {
                'rss_trend': '↗' if rss_values[-1] > rss_values[0] else '↘' if rss_values[-1] < rss_values[0] else '→',
                'vms_trend': '↗' if vms_values[-1] > vms_values[0] else '↘' if vms_values[-1] < vms_values[0] else '→',
                'rss_change': rss_values[-1] - rss_values[0],
                'vms_change': vms_values[-1] - vms_values[0],
                'rss_stable': abs(rss_values[-1] - rss_values[0]) < 10  # Stable if change < 10MB
            }
            
            return trends
        except Exception:
            return None
    
    def log_memory_status(self, frame_count, fps):
        """Log program-specific memory status for Raspberry Pi compatibility"""
        memory_info = self.get_memory_info()
        if not memory_info:
            return
        
        trends = self.get_memory_trends()
        
        print("\n" + "="*70)
        print(f"📊 PROGRAM RESOURCE USAGE - Frame: {frame_count}, FPS: {fps:.1f}")
        print("="*70)
        
        # Program Memory
        print(f"🔹 PROGRAM MEMORY (Raspberry Pi Compatibility):")
        print(f"   Current RAM:     {memory_info['process']['rss_mb']:.1f} MB {trends['rss_trend'] if trends else ''}")
        print(f"   Peak RAM:        {memory_info['process']['peak_rss_mb']:.1f} MB")
        print(f"   Virtual Memory:  {memory_info['process']['vms_mb']:.1f} MB {trends['vms_trend'] if trends else ''}")
        
        if trends:
            print(f"   RAM Change:      {trends['rss_change']:+.1f} MB")
            print(f"   Memory Stable:   {'✓ Yes' if trends['rss_stable'] else '⚠ No (potential leak)'}")
        
        # Raspberry Pi RAM Assessment
        rss_mb = memory_info['process']['rss_mb']
        peak_mb = memory_info['process']['peak_rss_mb']
        
        if peak_mb <= 1024:  # 1GB limit
            ram_status = "✅ COMPATIBLE with 1GB RAM"
        elif peak_mb <= 1536:  # 1.5GB limit
            ram_status = "⚠️  MARGINAL - Consider 2GB RAM"
        else:
            ram_status = "❌ NOT COMPATIBLE with 1GB RAM"
        
        print(f"   RAM Assessment:  {ram_status}")
        
        # Storage Requirements
        if memory_info['storage']:
            print(f"\n🔹 STORAGE REQUIREMENTS:")
            print(f"   Essential Files: {memory_info['storage']['essential_files_mb']:.1f} MB")
            print(f"   Virtual Env:     {memory_info['storage']['venv_mb']:.1f} MB")
            print(f"   Runtime:         {memory_info['storage']['runtime_mb']:.1f} MB")
            print(f"   TOTAL NEEDED:    {memory_info['storage']['total_mb']:.1f} MB")
            
            # File breakdown
            if memory_info['storage']['file_details']:
                print(f"   File Details:")
                for file_info in memory_info['storage']['file_details']:
                    print(f"     {file_info['name']}: {file_info['size_mb']:.1f} MB")
            
            # Raspberry Pi Storage Assessment
            total_storage = memory_info['storage']['total_mb']
            if total_storage <= 5000:  # 5GB limit
                storage_status = "✅ COMPATIBLE with 5GB SD card"
            elif total_storage <= 8000:  # 8GB limit
                storage_status = "⚠️  MARGINAL - Consider 8GB+ SD card"
            else:
                storage_status = "❌ NOT COMPATIBLE with 5GB SD card"
            
            print(f"   Storage Assessment: {storage_status}")
        
        # Runtime
        uptime = memory_info['uptime']
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)
        seconds = int(uptime % 60)
        print(f"\n🔹 RUNTIME: {hours:02d}:{minutes:02d}:{seconds:02d}")
        
        # Overall Compatibility
        ram_ok = peak_mb <= 1024
        storage_ok = memory_info['storage']['raspberry_pi_compatible'] if memory_info['storage'] else False
        
        if ram_ok and storage_ok:
            overall_status = "✅ FULLY COMPATIBLE with Raspberry Pi (1GB RAM, 5GB SD)"
        elif ram_ok or storage_ok:
            overall_status = "⚠️  PARTIALLY COMPATIBLE - Check requirements above"
        else:
            overall_status = "❌ NOT COMPATIBLE with Raspberry Pi"
        
        print(f"\n🔹 OVERALL ASSESSMENT: {overall_status}")
        print("="*70)
    
    def get_memory_summary(self):
        """Get a brief memory summary for overlay display"""
        memory_info = self.get_memory_info()
        if not memory_info:
            return "Memory: N/A"
        
        rss_mb = memory_info['process']['rss_mb']
        peak_mb = memory_info['process']['peak_rss_mb']
        
        return f"RAM: {rss_mb:.0f}MB | Peak: {peak_mb:.0f}MB"

class SimpleDistanceEstimator:
    """Simple and effective distance estimation using known object dimensions"""
    
    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.focal_length = 1000  # Standard webcam focal length
        
        # Known average dimensions of common objects (in meters)
        self.known_dimensions = {
            'person': {'width': 0.5, 'height': 1.7},      # Average person
            'car': {'width': 1.8, 'height': 1.5},         # Average car
            'truck': {'width': 2.5, 'height': 2.5},       # Average truck
            'bus': {'width': 2.5, 'height': 3.0},         # Average bus
            'bicycle': {'width': 0.6, 'height': 1.5},     # Bicycle
            'motorcycle': {'width': 0.8, 'height': 1.4},  # Motorcycle
            'airplane': {'width': 35.0, 'height': 12.0},  # Commercial airplane
            'train': {'width': 3.0, 'height': 4.0},       # Train
            'boat': {'width': 5.0, 'height': 2.0},        # Small boat
            'cat': {'width': 0.3, 'height': 0.4},         # Cat
            'dog': {'width': 0.4, 'height': 0.6},         # Dog
            'horse': {'width': 1.0, 'height': 1.6},       # Horse
            'sheep': {'width': 0.6, 'height': 0.8},       # Sheep
            'cow': {'width': 1.2, 'height': 1.4},         # Cow
            'elephant': {'width': 3.0, 'height': 3.0},    # Elephant
            'bear': {'width': 1.2, 'height': 1.8},        # Bear
            'zebra': {'width': 1.2, 'height': 1.4},       # Zebra
            'giraffe': {'width': 1.0, 'height': 4.0},     # Giraffe
            'backpack': {'width': 0.3, 'height': 0.4},    # Backpack
            'umbrella': {'width': 0.1, 'height': 1.0},    # Umbrella
            'handbag': {'width': 0.3, 'height': 0.2},     # Handbag
            'suitcase': {'width': 0.5, 'height': 0.3},    # Suitcase
            'frisbee': {'width': 0.25, 'height': 0.25},   # Frisbee
            'skis': {'width': 0.1, 'height': 1.7},        # Skis
            'snowboard': {'width': 0.25, 'height': 1.5},  # Snowboard
            'sports ball': {'width': 0.22, 'height': 0.22}, # Soccer ball
            'kite': {'width': 0.5, 'height': 0.5},        # Kite
            'baseball bat': {'width': 0.07, 'height': 0.9}, # Baseball bat
            'baseball glove': {'width': 0.25, 'height': 0.25}, # Baseball glove
            'skateboard': {'width': 0.2, 'height': 0.8},  # Skateboard
            'surfboard': {'width': 0.6, 'height': 2.7},   # Surfboard
            'tennis racket': {'width': 0.3, 'height': 0.7}, # Tennis racket
            'bottle': {'width': 0.07, 'height': 0.25},    # Bottle
            'wine glass': {'width': 0.08, 'height': 0.15}, # Wine glass
            'cup': {'width': 0.08, 'height': 0.12},       # Cup
            'fork': {'width': 0.02, 'height': 0.2},       # Fork
            'knife': {'width': 0.02, 'height': 0.2},      # Knife
            'spoon': {'width': 0.02, 'height': 0.2},      # Spoon
            'bowl': {'width': 0.15, 'height': 0.08},      # Bowl
            'banana': {'width': 0.02, 'height': 0.18},    # Banana
            'apple': {'width': 0.08, 'height': 0.08},     # Apple
            'sandwich': {'width': 0.12, 'height': 0.08},  # Sandwich
            'orange': {'width': 0.08, 'height': 0.08},    # Orange
            'broccoli': {'width': 0.15, 'height': 0.15},  # Broccoli
            'carrot': {'width': 0.02, 'height': 0.2},     # Carrot
            'hot dog': {'width': 0.15, 'height': 0.05},  # Hot dog
            'pizza': {'width': 0.3, 'height': 0.3},       # Pizza
            'donut': {'width': 0.08, 'height': 0.08},     # Donut
            'cake': {'width': 0.2, 'height': 0.1},        # Cake
            'chair': {'width': 0.5, 'height': 0.9},       # Chair
            'couch': {'width': 2.0, 'height': 0.8},       # Couch
            'potted plant': {'width': 0.3, 'height': 0.5}, # Potted plant
            'bed': {'width': 1.6, 'height': 0.5},         # Bed
            'dining table': {'width': 1.2, 'height': 0.75}, # Dining table
            'toilet': {'width': 0.4, 'height': 0.7},      # Toilet
            'tv': {'width': 1.2, 'height': 0.7},          # TV
            'laptop': {'width': 0.35, 'height': 0.25},    # Laptop
            'mouse': {'width': 0.12, 'height': 0.06},     # Mouse
            'remote': {'width': 0.15, 'height': 0.05},    # Remote
            'keyboard': {'width': 0.45, 'height': 0.15},  # Keyboard
            'cell phone': {'width': 0.08, 'height': 0.15}, # Cell phone
            'microwave': {'width': 0.5, 'height': 0.3},   # Microwave
            'oven': {'width': 0.6, 'height': 0.6},        # Oven
            'toaster': {'width': 0.3, 'height': 0.25},    # Toaster
            'sink': {'width': 0.5, 'height': 0.2},        # Sink
            'refrigerator': {'width': 0.8, 'height': 1.8}, # Refrigerator
            'book': {'width': 0.15, 'height': 0.22},      # Book
            'clock': {'width': 0.2, 'height': 0.2},       # Clock
            'vase': {'width': 0.15, 'height': 0.25},      # Vase
            'scissors': {'width': 0.15, 'height': 0.05},  # Scissors
            'teddy bear': {'width': 0.3, 'height': 0.4},  # Teddy bear
            'hair drier': {'width': 0.15, 'height': 0.25}, # Hair drier
            'toothbrush': {'width': 0.02, 'height': 0.2}, # Toothbrush
        }
    
    def estimate_distance(self, box, class_name):
        """
        Simple and effective distance estimation using known object dimensions
        """
        # Get bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        box_width = x2 - x1
        box_height = y2 - y1
        
        # Get known dimensions for this class
        if class_name in self.known_dimensions:
            real_width = self.known_dimensions[class_name]['width']
            real_height = self.known_dimensions[class_name]['height']
            
            # Calculate distance using both width and height, take the average
            # Distance = (Known Size * Focal Length) / Apparent Size
            
            # Distance calculations
            dist_by_width = (real_width * self.focal_length) / box_width
            dist_by_height = (real_height * self.focal_length) / box_height
            
            # Average the two estimates
            distance = (dist_by_width + dist_by_height) / 2
            
            # Ensure reasonable bounds (0.1m to 100m)
            distance = max(0.1, min(100.0, distance))
            
            return distance
        else:
            # For unknown classes, use a generic estimation based on box size
            # Assume average object size of 0.5m
            avg_size = 0.5
            distance = (avg_size * self.focal_length) / max(box_width, box_height)
            distance = max(0.1, min(100.0, distance))
            return distance

def draw_detection_info(frame, result, fps, frame_count, elapsed_time, distance_estimator, memory_monitor):
    """Draw comprehensive detection information on the frame"""
    h, w = frame.shape[:2]
    
    # Create semi-transparent overlay for info
    overlay = frame.copy()
    
    # Draw detection count and FPS info
    detection_count = len(result.boxes) if result.boxes is not None else 0
    
    # Background rectangle for text
    cv2.rectangle(overlay, (10, 10), (500, 160), (0, 0, 0), -1)
    cv2.rectangle(overlay, (10, 10), (500, 160), (255, 255, 255), 2)
    
    # Detection statistics
    cv2.putText(overlay, f"Detections: {detection_count}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(overlay, f"FPS: {fps:.1f}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(overlay, f"Frame: {frame_count}", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(overlay, f"Time: {elapsed_time:.1f}s", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    
    # Memory information
    if memory_monitor:
        memory_summary = memory_monitor.get_memory_summary()
        cv2.putText(overlay, memory_summary, (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Draw detection details if objects are found
    if detection_count > 0 and result.boxes is not None:
        # Create detection list overlay
        y_offset = 180
        max_height = min(350, h - y_offset - 20)  # Limit height to fit in frame
        
        cv2.rectangle(overlay, (10, y_offset), (500, y_offset + max_height), (0, 0, 0), -1)
        cv2.rectangle(overlay, (10, y_offset), (500, y_offset + max_height), (0, 255, 0), 2)
        
        cv2.putText(overlay, "Detected Objects (with Simple Distance):", (20, y_offset + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Get unique classes and their counts with distances
        if hasattr(result.boxes, 'cls') and result.boxes.cls is not None:
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else []
            
            # Get class names
            class_names = result.names if hasattr(result, 'names') else {}
            
            # Count occurrences of each class and calculate distances
            class_counts = {}
            for i, cls_id in enumerate(classes):
                cls_name = class_names.get(int(cls_id), f"Class {int(cls_id)}")
                conf = confidences[i] if i < len(confidences) else 0.0
                
                if cls_name not in class_counts:
                    class_counts[cls_name] = {'count': 0, 'max_conf': 0.0, 'distances': []}
                
                class_counts[cls_name]['count'] += 1
                class_counts[cls_name]['max_conf'] = max(class_counts[cls_name]['max_conf'], conf)
                
                # Calculate distance using advanced estimator
                distance = distance_estimator.estimate_distance(
                    result.boxes[i], cls_name
                )
                class_counts[cls_name]['distances'].append(distance)
            
            # Display class counts with distance information
            y_pos = y_offset + 45
            for cls_name, info in class_counts.items():
                if y_pos < y_offset + max_height - 25:  # Don't draw outside frame
                    # Calculate average distance for this class
                    avg_distance = np.mean(info['distances']) if info['distances'] else 0
                    
                    # Format distance text with precision
                    if avg_distance < 1.0:
                        distance_text = f"{avg_distance*100:.1f}cm"
                    elif avg_distance < 10.0:
                        distance_text = f"{avg_distance:.2f}m"
                    else:
                        distance_text = f"{avg_distance:.1f}m"
                    
                    # Add confidence indicator
                    conf_indicator = "✓" if info['max_conf'] > 0.8 else "~" if info['max_conf'] > 0.6 else "?"
                    
                    text = f"{conf_indicator} {cls_name}: {info['count']} ({info['max_conf']:.2f}) - {distance_text}"
                    cv2.putText(overlay, text, (20, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y_pos += 25
    
    # Instructions overlay
    cv2.rectangle(overlay, (w - 250, 10), (w - 10, 80), (0, 0, 0), -1)
    cv2.rectangle(overlay, (w - 250, 10), (w - 10, 80), (255, 255, 255), 2)
    cv2.putText(overlay, "Press 'q' to quit", (w - 240, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(overlay, "ESC to exit", (w - 240, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(overlay, "Simple Distance Active", (w - 240, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    return overlay

def main():
    ap = argparse.ArgumentParser(description="YOLOv8n realtime webcam detection with advanced dynamic distance estimation and Raspberry Pi compatibility monitoring")
    ap.add_argument("--model", default="yolov8n.pt", help="model path or name")
    ap.add_argument("--source", type=int, default=0, help="webcam index (0,1,...)")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    ap.add_argument("--imgsz", type=int, default=640, help="inference image size")
    ap.add_argument("--memory-log-interval", type=int, default=30, help="frames between memory logging (default: 30)")
    args = ap.parse_args()

    # Check camera access first
    print("Checking camera access...")
    camera_index = find_working_camera()
    
    if camera_index is None:
        print("\n❌ No working camera found!")
        print("Please check:")
        print("1. Camera permissions in System Preferences > Security & Privacy > Camera")
        print("2. Camera is not being used by another application")
        print("3. Camera is properly connected")
        print("\nYou can also try specifying a different camera with --source flag")
        sys.exit(1)
    
    # Override source if we found a working camera
    if camera_index != args.source:
        print(f"Using camera {camera_index} instead of requested camera {args.source}")
        args.source = camera_index

    device = pick_device()
    print(f"Using device: {device}")
    
    try:
        model = YOLO(args.model)
        print(f"Model loaded: {args.model}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)

    win = "YOLOv8n • Simple Distance Detection with Memory Monitor"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)  # Set a good default size

    print(f"Starting detection on camera {args.source}...")
    print("Press 'q' or ESC to quit")
    print("Simple and effective distance estimation system active")
    print("System uses known object dimensions for reliable accuracy")
    print(f"Raspberry Pi compatibility monitoring active - logging every {args.memory_log_interval} frames")
    
    t0, n = time.time(), 0
    distance_estimator = None
    memory_monitor = MemoryMonitor()
    
    try:
        # Stream results frame-by-frame straight from the camera
        for result in model(source=args.source, stream=True,
                            conf=args.conf, imgsz=args.imgsz, device=device):
            # Get the frame with YOLO detections drawn
            frame = result.plot()  # draw boxes, labels, etc.
            
            # Initialize distance estimator on first frame
            if distance_estimator is None:
                h, w = frame.shape[:2]
                distance_estimator = SimpleDistanceEstimator(w, h)
                print(f"Distance estimator initialized for {w}x{h} resolution")
            
            n += 1
            dt = time.time() - t0
            fps = (n / dt) if dt > 0 else 0.0
            
            # Log memory status at specified intervals
            if n % args.memory_log_interval == 0:
                memory_monitor.log_memory_status(n, fps)
            
            # Enhance frame with advanced distance estimation and memory info
            enhanced_frame = draw_detection_info(frame, result, fps, n, dt, distance_estimator, memory_monitor)
            
            # Display the enhanced frame
            cv2.imshow(win, enhanced_frame)
            
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):  # ESC or q to quit
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\n❌ Error during detection: {e}")
    finally:
        # Final resource status
        print("\n" + "="*80)
        print("📊 FINAL RASPBERRY PI COMPATIBILITY ASSESSMENT")
        print("="*80)
        memory_monitor.log_memory_status(n, fps if 'fps' in locals() else 0)
        
        cv2.destroyAllWindows()
        print(f"Detection stopped. Processed {n} frames in {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()