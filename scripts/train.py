"""
scripts/train.py
Main training script for YOLOv8n on Seraphim Drone Dataset (Single Class: drone).
Configured for maximum throughput on NVIDIA RTX 5070 Ti (Blackwell sm_120, 16 GB VRAM).
"""
import os
import sys
import time
import json
import shutil
import argparse
import torch
from ultralytics import YOLO

# Enable Tensor Cores and cuDNN benchmark for maximum GPU throughput
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

WORK_DIR = r"c:\Users\dhruvbomber\drone-seraphim"
WEIGHTS_DIR = os.path.join(WORK_DIR, "runs", "yolov8n_seraphim", "weights")
QUEUE_DIR = os.path.join(WORK_DIR, "export_queue")
LOGS_DIR = os.path.join(WORK_DIR, "logs")
PROGRESS_JSON = os.path.join(LOGS_DIR, "training_progress.json")

os.makedirs(QUEUE_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

def on_fit_epoch_end(trainer):
    """
    Non-blocking callback to queue weights for asynchronous Hailo/ONNX export.
    """
    try:
        epoch = getattr(trainer, 'epoch', 0) + 1
        total_epochs = getattr(trainer, 'epochs', 100)
        last_pt = os.path.join(trainer.save_dir, 'weights', 'last.pt')
        best_pt = os.path.join(trainer.save_dir, 'weights', 'best.pt')
        
        if os.path.exists(last_pt):
            # Sync to central weights dir as well
            os.makedirs(WEIGHTS_DIR, exist_ok=True)
            shutil.copyfile(last_pt, os.path.join(WEIGHTS_DIR, "last.pt"))
            if os.path.exists(best_pt):
                shutil.copyfile(best_pt, os.path.join(WEIGHTS_DIR, "best.pt"))
            
            dst_pt = os.path.join(QUEUE_DIR, f"epoch_{epoch:03d}.pt")
            dst_json = os.path.join(QUEUE_DIR, f"epoch_{epoch:03d}.json")
            
            # Atomic copy to export queue
            tmp_pt = dst_pt + ".tmp"
            shutil.copyfile(last_pt, tmp_pt)
            os.replace(tmp_pt, dst_pt)
            
            # Extract validation metrics
            metrics = {}
            if hasattr(trainer, 'metrics') and isinstance(trainer.metrics, dict):
                metrics = {k: float(v) for k, v in trainer.metrics.items() if isinstance(v, (int, float))}
            elif hasattr(trainer, 'validator') and hasattr(trainer.validator, 'metrics'):
                val_m = trainer.validator.metrics
                if hasattr(val_m, 'results_dict'):
                    metrics = {k: float(v) for k, v in val_m.results_dict.items() if isinstance(v, (int, float))}
            
            meta = {
                "epoch": epoch,
                "total_epochs": total_epochs,
                "timestamp": time.time(),
                "time_str": time.strftime("%Y-%m-%d %H:%M:%S"),
                "metrics": metrics
            }
            
            tmp_json = dst_json + ".tmp"
            with open(tmp_json, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            os.replace(tmp_json, dst_json)
            
            # Update progress tracking file
            tmp_prog = PROGRESS_JSON + ".tmp"
            with open(tmp_prog, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            os.replace(tmp_prog, PROGRESS_JSON)
    except Exception:
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    
    weights_path = "yolov8n.pt"
    if args.resume:
        candidates = [
            "runs/detect/runs/yolov8n_seraphim/weights/last.pt",
            "runs/yolov8n_seraphim/weights/last.pt",
            os.path.join(WORK_DIR, "runs", "detect", "runs", "yolov8n_seraphim", "weights", "last.pt"),
            os.path.join(WORK_DIR, "runs", "yolov8n_seraphim", "weights", "last.pt")
        ]
        for c in candidates:
            if os.path.exists(c):
                weights_path = c
                break
        if weights_path == "yolov8n.pt":
            print("Warning: last.pt not found for resume. Starting from yolov8n.pt")
            args.resume = False

    print(f"============================================================")
    print(f"STARTING HIGH-THROUGHPUT YOLOV8N TRAINING (RESUME={args.resume})")
    print(f"  Model / Checkpoint: {weights_path}")
    print(f"  Device:             NVIDIA RTX 5070 Ti (sm_120)")
    print(f"  Batch Size:         {args.batch}")
    print(f"  Workers:            {args.workers}")
    print(f"  Total Epochs:       {args.epochs}")
    print(f"  Dataset:            configs/data.yaml (74,732 train / 4,400 val)")
    print(f"  Precision:          AMP fp16 / TensorFloat-32")
    print(f"============================================================")
    
    model = YOLO(weights_path)
    
    # Register export watcher callback
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    
    if args.resume:
        results = model.train(resume=True)
    else:
        results = model.train(
            data="configs/data.yaml",
            imgsz=640,
            epochs=args.epochs,
            patience=20,
            batch=args.batch,
            workers=args.workers,
            cache=False,
            device=0,
            amp=True,
            close_mosaic=10,
            deterministic=False,
            seed=0,
            project="runs",
            name="yolov8n_seraphim",
            exist_ok=True,
            plots=False,
            verbose=True
        )
    print("Full training run completed successfully!")

if __name__ == "__main__":
    main()
