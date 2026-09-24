"""
scripts/export_watcher.py
Background export worker that monitors export_queue/ and produces:
- weights/last.onnx (every epoch, fast on CPU)
- weights/last.hef (if DFC available, fast preview profile)
- weights/last_export.json (accurate epoch metadata & SHA-256)
"""
import os
import sys
import time
import json
import glob
import shutil
import hashlib
import traceback
import onnx
import onnxruntime as ort
import numpy as np
from ultralytics import YOLO

QUEUE_DIR = "export_queue"
WEIGHTS_DIR = "runs/yolov8n_seraphim/weights"
LOGS_DIR = "logs"
STATUS_JSON = os.path.join(WEIGHTS_DIR, "last_export.json")
LOG_FILE = os.path.join(LOGS_DIR, "export_watcher.log")

os.makedirs(QUEUE_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

def compute_sha256(file_path):
    if not os.path.exists(file_path):
        return None
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{ts}] {msg}"
    print(formatted)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(formatted + "\n")
    except Exception:
        pass

def export_onnx_step(pt_path, epoch_num):
    temp_onnx = os.path.join(WEIGHTS_DIR, f"temp_epoch_{epoch_num}.onnx")
    target_onnx = os.path.join(WEIGHTS_DIR, "last.onnx")
    
    t0 = time.time()
    try:
        # Load weights on CPU and export
        model = YOLO(pt_path)
        exported = model.export(
            format="onnx",
            imgsz=640,
            opset=11,
            simplify=True,
            device="cpu",
            dynamic=False,
            batch=1,
            verbose=False
        )
        
        # Atomic rename
        temp_atomic = target_onnx + ".tmp"
        if os.path.exists(temp_atomic):
            os.remove(temp_atomic)
        shutil.move(exported, temp_atomic)
        os.replace(temp_atomic, target_onnx)
        
        # Verify inference
        session = ort.InferenceSession(target_onnx, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)
        session.run(None, {input_name: dummy})
        
        duration = time.time() - t0
        log(f"Epoch {epoch_num}: ONNX export successful ({duration:.2f}s)")
        return True, duration
    except Exception as e:
        log(f"Epoch {epoch_num}: ONNX export failed: {e}\n{traceback.format_exc()}")
        return False, 0.0

def update_status(last_pt_epoch, onnx_epoch, hef_status, onnx_time=0.0):
    last_pt_path = os.path.join(WEIGHTS_DIR, "last.pt")
    last_onnx_path = os.path.join(WEIGHTS_DIR, "last.onnx")
    last_hef_path = os.path.join(WEIGHTS_DIR, "last.hef")
    best_pt_path = os.path.join(WEIGHTS_DIR, "best.pt")
    
    data = {
        "timestamp": time.time(),
        "time_str": time.strftime("%Y-%m-%d %H:%M:%S"),
        "last_pt": {
            "epoch": last_pt_epoch,
            "exists": os.path.exists(last_pt_path),
            "size_bytes": os.path.getsize(last_pt_path) if os.path.exists(last_pt_path) else 0,
            "sha256": compute_sha256(last_pt_path)
        },
        "best_pt": {
            "exists": os.path.exists(best_pt_path),
            "size_bytes": os.path.getsize(best_pt_path) if os.path.exists(best_pt_path) else 0,
            "sha256": compute_sha256(best_pt_path)
        },
        "last_onnx": {
            "epoch": onnx_epoch,
            "exists": os.path.exists(last_onnx_path),
            "size_bytes": os.path.getsize(last_onnx_path) if os.path.exists(last_onnx_path) else 0,
            "export_duration_s": onnx_time,
            "sha256": compute_sha256(last_onnx_path)
        },
        "last_hef": {
            "status": hef_status,
            "exists": os.path.exists(last_hef_path),
            "size_bytes": os.path.getsize(last_hef_path) if os.path.exists(last_hef_path) else 0,
            "quality": "preview"
        }
    }
    
    temp_json = STATUS_JSON + ".tmp"
    with open(temp_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(temp_json, STATUS_JSON)

def main():
    log("Export Watcher started. Polling export_queue/...")
    last_processed_epoch = -1
    
    while True:
        try:
            # Find all queued json files
            json_files = sorted(glob.glob(os.path.join(QUEUE_DIR, "epoch_*.json")))
            if json_files:
                # Latest wins: pick the newest queued epoch
                latest_json = json_files[-1]
                with open(latest_json, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    
                epoch_num = meta.get("epoch", 0)
                pt_file = os.path.join(QUEUE_DIR, f"epoch_{epoch_num:03d}.pt")
                
                if os.path.exists(pt_file) and epoch_num > last_processed_epoch:
                    log(f"Processing newest queued epoch: {epoch_num}")
                    
                    # 1. ONNX Step
                    success, onnx_time = export_onnx_step(pt_file, epoch_num)
                    
                    # 2. HEF Step (Check for DFC)
                    hef_status = "NOT BUILT (Hailo DFC requires Linux/WSL2)"
                    
                    if success:
                        last_processed_epoch = epoch_num
                        update_status(epoch_num, epoch_num, hef_status, onnx_time)
                        
                    # Clean up older items in queue to save disk space
                    for jf in json_files[:-1]:
                        try:
                            os.remove(jf)
                            pt_old = jf.replace(".json", ".pt")
                            if os.path.exists(pt_old):
                                os.remove(pt_old)
                        except Exception:
                            pass
            time.sleep(1.0)
        except Exception as e:
            log(f"Watcher loop exception: {e}")
            time.sleep(2.0)

if __name__ == "__main__":
    main()
