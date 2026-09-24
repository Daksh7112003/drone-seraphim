"""
scripts/status.py
Status inspection tool for YOLOv8 Seraphim training & exports.
"""
import os
import time
import json

WEIGHTS_DIR = "runs/yolov8n_seraphim/weights"
STATUS_JSON = os.path.join(WEIGHTS_DIR, "last_export.json")

def format_bytes(size):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"

def get_file_info(path):
    if not os.path.exists(path):
        return "Not found", 0, 0
    stat = os.stat(path)
    age = time.time() - stat.st_mtime
    return format_bytes(stat.st_size), age, stat.st_size

def main():
    print("\n" + "="*60)
    print("SERAPHIM YOLOV8N TRAINING & EXPORT STATUS")
    print("="*60)
    
    last_pt = os.path.join(WEIGHTS_DIR, "last.pt")
    best_pt = os.path.join(WEIGHTS_DIR, "best.pt")
    last_onnx = os.path.join(WEIGHTS_DIR, "last.onnx")
    last_hef = os.path.join(WEIGHTS_DIR, "last.hef")
    best_onnx = os.path.join(WEIGHTS_DIR, "best.onnx")
    best_hef = os.path.join(WEIGHTS_DIR, "best.hef")
    
    meta = {}
    if os.path.exists(STATUS_JSON):
        try:
            with open(STATUS_JSON, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            pass
            
    pt_epoch = meta.get("last_pt", {}).get("epoch", "N/A")
    onnx_epoch = meta.get("last_onnx", {}).get("epoch", "N/A")
    hef_status = meta.get("last_hef", {}).get("status", "N/A")
    
    files = [
        ("last.pt", last_pt, f"Epoch {pt_epoch}"),
        ("best.pt", best_pt, "Best Validation Fitness"),
        ("last.onnx", last_onnx, f"Epoch {onnx_epoch} (Hailo-compatible)"),
        ("last.hef", last_hef, hef_status),
        ("best.onnx", best_onnx, "Final Post-Training Build"),
        ("best.hef", best_hef, "Final Hailo-10H Build")
    ]
    
    for name, path, extra in files:
        size_str, age_s, _ = get_file_info(path)
        if age_s > 0:
            age_str = f"{age_s:.1f}s ago"
        else:
            age_str = "N/A"
        print(f"{name:12s} | Size: {size_str:10s} | Age: {age_str:10s} | Info: {extra}")
        
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
