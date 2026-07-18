import sys
import types
import torch

# Define pure PyTorch NMS to bypass the broken C++ torchvision::nms operator
def pure_pytorch_nms(boxes, scores, iou_threshold):
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])
        w = torch.clamp(xx2 - xx1, min=0.0)
        h = torch.clamp(yy2 - yy1, min=0.0)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (ovr <= iou_threshold).nonzero().flatten()
        order = order[ids + 1]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)

# Intercept and mock torchvision to prevent importing compilation errors
mock_torchvision = types.ModuleType("torchvision")
mock_torchvision.ops = types.ModuleType("ops")
mock_torchvision.ops.nms = pure_pytorch_nms
mock_torchvision.__version__ = "0.28.0"

sys.modules["torchvision"] = mock_torchvision
sys.modules["torchvision.ops"] = mock_torchvision.ops

import cv2
import time
import numpy as np

try:
    from ultralytics import YOLO
    yolo_available = True
except ImportError:
    yolo_available = False

# COCO Dataset ID mapping for common targets you requested
# Leaving this empty tracking list [] means tracking ALL 80 classes simultaneously.
# Example: [0, 39, 67] will track ONLY persons, bottles, and cell phones.
TRACK_CLASSES = [] 

def run_tracking():
    if not yolo_available:
        print("[-] Error: 'ultralytics' library (YOLOv8) is not installed.")
        print("[*] Please run: pip install ultralytics")
        sys.exit(1)
        
    print("[*] Loading pre-trained YOLOv8 Nano model (yolov8n.pt)...")
    try:
        model = YOLO("yolov8n.pt")
    except Exception as e:
        print(f"[-] Error loading YOLOv8 model: {e}")
        sys.exit(1)
        
    print("[*] Connecting to webcam feed...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[!] Warning: Could not access the webcam.")
        print("[*] Fallback: Simulating tracking environment...")
        run_simulated_tracker(model)
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[+] Webcam stream successfully established!")
    print("[*] Press 'q' in the window to quit.")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Pass our custom explicit list filtering to YOLO if defined
            if len(TRACK_CLASSES) > 0:
                results = model.track(source=frame, persist=True, verbose=False, classes=TRACK_CLASSES)
            else:
                results = model.track(source=frame, persist=True, verbose=False)
            
            annotated_frame = results[0].plot()
            
            # Calculate and display FPS
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imshow("CodeAlpha Object Detection & Tracking (YOLOv8)", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

def run_simulated_tracker(model):
    """
    Enhanced simulation loop. Because the AI model looks for realistic objects, 
    the simulation uses labeled bounding targets to show how any class will render.
    """
    print("[*] Starting Enhanced Simulated Tracker...")
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Track positions of mock objects
    t1_x, t1_y = 120, 150
    t2_x, t2_y = 450, 250
    
    t1_dx, t1_dy = 3, 2
    t2_dx, t2_dy = -2, 3
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            frame_count += 1
            frame = canvas.copy()
            
            # Draw grid lines
            for i in range(0, 640, 80):
                cv2.line(frame, (i, 0), (i, 480), (30, 30, 30), 1)
            for j in range(0, 480, 60):
                cv2.line(frame, (0, j), (640, j), (30, 30, 30), 1)
                
            # Physics updates
            t1_x = (t1_x + t1_dx) % 640
            t1_y = (t1_y + t1_dy) % 480
            t2_x = (t2_x + t2_dx) % 640
            t2_y = (t2_y + t2_dy) % 480
            
            # Target 1: Mimic a bottle/phone silhouette space
            cv2.rectangle(frame, (t1_x-30, t1_y-60), (t1_x+30, t1_y+60), (180, 100, 50), -1)
            cv2.rectangle(frame, (t1_x-10, t1_y-80), (t1_x+10, t1_y-60), (140, 80, 40), -1)
            
            # Target 2: Mimic a wider object space (like a laptop or book)
            cv2.rectangle(frame, (t2_x-70, t2_y-45), (t2_x+70, t2_y+45), (50, 80, 180), -1)

            if len(TRACK_CLASSES) > 0:
                results = model.track(source=frame, persist=True, verbose=False, classes=TRACK_CLASSES)
            else:
                results = model.track(source=frame, persist=True, verbose=False)
                
            annotated_frame = results[0].plot()
            
            # Draw manual indicators in simulator to show flexibility
            cv2.putText(annotated_frame, "SIMULATOR: YOLOv8 scanning all 80 classes", (20, 420),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(annotated_frame, "Try running with a real webcam to track phones, bottles, etc.", (20, 445),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)
            
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imshow("CodeAlpha Object Detection & Tracking (YOLOv8 SIMULATOR)", annotated_frame)
            
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_tracking()