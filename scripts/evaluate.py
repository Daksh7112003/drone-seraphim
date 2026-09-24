"""
scripts/evaluate.py
Evaluates YOLOv8n model on test sets (positives-only and mixed).
Computes mAP50, mAP50-95, precision, recall, and size-bucketed recall (tiny, small, medium, large).
"""
import os
import sys
import json
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

def evaluate_size_buckets(model_path, test_txt="data/test_pos.txt"):
    print(f"\nEvaluating recall across size buckets on {test_txt}...")
    model = YOLO(model_path)
    
    lines = [l.strip() for l in open(test_txt).readlines() if l.strip()]
    
    bucket_counts = {"tiny": 0, "small": 0, "medium": 0, "large": 0}
    bucket_hits = {"tiny": 0, "small": 0, "medium": 0, "large": 0}
    
    for img_path in tqdm(lines[:1000], desc="Evaluating size buckets (sample 1000)"):
        lbl_path = img_path.replace("/images/", "/labels/").replace("\\images\\", "\\labels\\").rsplit(".", 1)[0] + ".txt"
        if not os.path.exists(lbl_path):
            continue
            
        with open(lbl_path, "r", encoding="utf-8") as f:
            gt_boxes = []
            for l in f.readlines():
                p = l.strip().split()
                if len(p) == 5:
                    bw = float(p[3]) * 640.0
                    bh = float(p[4]) * 640.0
                    max_dim = max(bw, bh)
                    gt_boxes.append((float(p[1]), float(p[2]), float(p[3]), float(p[4]), max_dim))
                    
        if not gt_boxes:
            continue
            
        # Run prediction
        preds = model(img_path, conf=0.25, verbose=False, device=0)[0]
        pred_boxes = preds.boxes.xywhn.cpu().numpy() if len(preds.boxes) > 0 else []
        
        for g_cx, g_cy, g_w, g_h, max_dim in gt_boxes:
            if max_dim < 16.0:
                b_name = "tiny"
            elif max_dim < 32.0:
                b_name = "small"
            elif max_dim < 96.0:
                b_name = "medium"
            else:
                b_name = "large"
                
            bucket_counts[b_name] += 1
            
            # Check if any prediction overlaps IoU > 0.5
            hit = False
            for p_cx, p_cy, p_w, p_h in pred_boxes:
                # Approximate IoU
                xA = max(g_cx - g_w/2, p_cx - p_w/2)
                yA = max(g_cy - g_h/2, p_cy - p_h/2)
                xB = min(g_cx + g_w/2, p_cx + p_w/2)
                yB = min(g_cy + g_h/2, p_cy + p_h/2)
                inter = max(0, xB - xA) * max(0, yB - yA)
                union = (g_w * g_h) + (p_w * p_h) - inter
                iou = inter / (union + 1e-6)
                if iou >= 0.5:
                    hit = True
                    break
            if hit:
                bucket_hits[b_name] += 1
                
    results = {}
    print("\nSize Bucket Recall Results:")
    for b in ["tiny", "small", "medium", "large"]:
        tot = bucket_counts[b]
        hits = bucket_hits[b]
        rec = (hits / tot * 100.0) if tot > 0 else 0.0
        results[b] = {"total_gt": tot, "detected": hits, "recall_%": round(rec, 2)}
        print(f"  - {b:8s}: {hits:4d} / {tot:4d} ({rec:5.2f}%)")
    return results

def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "runs/yolov8n_seraphim/weights/best.pt"
    
    print("="*60)
    print("EVALUATION ON TEST SETS")
    print(f"Model: {model_path}")
    print("="*60)
    
    model = YOLO(model_path)
    
    # 1. Positives Only
    print("\n---> Running Validation on Positives-Only Test Set...")
    res_pos = model.val(data="configs/data_test_pos.yaml", device=0, imgsz=640, verbose=False)
    
    # 2. Mixed Test Set (if available)
    res_mixed = None
    if os.path.exists("configs/data_test_mixed.yaml"):
        print("\n---> Running Validation on Mixed Test Set (with Negatives)...")
        res_mixed = model.val(data="configs/data_test_mixed.yaml", device=0, imgsz=640, verbose=False)
        
    # 3. Size buckets
    bucket_res = evaluate_size_buckets(model_path, "data/test_pos.txt")
    
    summary = {
        "model": model_path,
        "positives_test": {
            "mAP50": round(float(res_pos.box.map50), 4),
            "mAP50-95": round(float(res_pos.box.map), 4),
            "precision": round(float(res_pos.box.mp), 4),
            "recall": round(float(res_pos.box.mr), 4)
        },
        "size_buckets": bucket_res
    }
    
    if res_mixed is not None:
        summary["mixed_test"] = {
            "mAP50": round(float(res_mixed.box.map50), 4),
            "mAP50-95": round(float(res_mixed.box.map), 4),
            "precision": round(float(res_mixed.box.mp), 4),
            "recall": round(float(res_mixed.box.mr), 4)
        }
        
    out_file = "logs/test_evaluation_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print(f"\nEvaluation summary written to {out_file}")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
