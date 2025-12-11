
import cv2
import numpy as np
import json
import asyncio
import os
import glob
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import time
import torch

# -------------------------
# Initialize FastAPI
# -------------------------
app = FastAPI(title="YOLOv11 Vehicle Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Global Model State
# -------------------------
MODELS_DIR = "./models"
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# Global variable to hold the model
current_model = None
model_info = {"name": "None", "classes": []}

def load_yolo_model(model_name: str):
    global current_model, model_info
    
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        # Fallback for root file if not in models dir (backward compatibility)
        if os.path.exists(model_name):
            model_path = model_name
        else:
            raise FileNotFoundError(f"Model {model_name} not found.")

    print(f"Loading Model: {model_name}...")
    current_model = YOLO(model_path)
    
    # Force GPU if available
    if torch.cuda.is_available():
        current_model.to("cuda")
        print("Using GPU: CUDA")
    else:
        print("Using CPU")
        
    # Update Metadata
    model_info = {
        "name": model_name,
        "classes": list(current_model.names.values()),
        "count": len(current_model.names)
    }
    print("Model Loaded Successfully.")

# Load default model on startup (Scan dir, take first, or default)
pt_files = glob.glob(os.path.join(MODELS_DIR, "*.pt"))
if pt_files:
    default_model = os.path.basename(pt_files[0])
    load_yolo_model(default_model)
else:
    # Fallback if user hasn't put anything in /models yet
    try:
        load_yolo_model("yolo11n.pt") # Auto-download from Ultralytics if missing
    except:
        print("No models found. Please add .pt files to ./models/")


# -------------------------
# Model Management API
# -------------------------
@app.get("/models/list")
def list_models():
    """List all .pt files in the ./models directory."""
    files = glob.glob(os.path.join(MODELS_DIR, "*.pt"))
    return {"models": [os.path.basename(f) for f in files]}

@app.get("/models/current")
def get_current_model():
    """Get info about currently loaded model."""
    return model_info

@app.post("/models/select")
def select_model(model_name: str):
    """Switch the active model."""
    try:
        load_yolo_model(model_name)
        return {"status": "success", "model": model_info}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

# -------------------------
# System Stats
# -------------------------
@app.get("/system/stats")
def system_stats():
    return {
        "status": "online",
        "timestamp": time.time(),
        # Basic mock for GPU memory if nvidia-smi isn't piped
        "gpu_utilization_percent": 0 # You can implement actual nvidia-smi check here if needed
    }

@app.get("/")
def home():
    return {"status": "running", "current_model": model_info["name"]}

# -------------------------
# Websocket Video Detection
# -------------------------
@app.websocket("/ws/detect")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")

    try:
        while True:
            # 1. Receive Raw Bytes
            try:
                data = await websocket.receive_bytes()
            except RuntimeError:
                continue

            # 2. Decode Image
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None or current_model is None:
                continue

            # 3. Run YOLO + Track
            # We use the global 'current_model'
            results = current_model.track(
                frame,
                persist=True,
                tracker="botsort.yaml",
                verbose=False,
                iou=0.5,
                conf=0.45
            )

            detections_output = []
            
            # Auto-generate a dict for ALL classes in THIS model
            class_counts = {name: 0 for name in current_model.names.values()}

            for r in results:
                boxes = r.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                xyxy = boxes.xyxy.cpu().numpy()
                conf = boxes.conf.cpu().numpy()
                cls = boxes.cls.cpu().numpy().astype(int)

                if boxes.id is not None:
                    track_ids = boxes.id.cpu().numpy().astype(int)
                else:
                    track_ids = [-1] * len(boxes)

                for box, c, class_id, tid in zip(xyxy, conf, cls, track_ids):
                    if class_id in current_model.names:
                        class_name = current_model.names[class_id]
                        class_counts[class_name] += 1

                        x1, y1, x2, y2 = box
                        detections_output.append({
                            "id": f"{class_name}_{tid}",
                            "class": class_name,
                            "confidence": float(c),
                            "x": float(x1),
                            "y": float(y1),
                            "w": float(x2 - x1),
                            "h": float(y2 - y1)
                        })

            # Send full JSON
            await websocket.send_json({
                "detections": detections_output,
                "stats": class_counts
            })

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Server Error: {e}")
        try:
            await websocket.close()
        except:
            pass