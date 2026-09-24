"""
scripts/false_alarms.py
Evaluates false alarm rates on dedicated negatives test set (negatives_test.txt).
Computes False Positives per 1,000 images across confidence thresholds (0.10, 0.25, 0.50, 0.75).
Saves top 50 highest-confidence detections on background images for manual inspection of potential hidden drones.
"""
import os
import sys
import json
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

OUT_CROPS_DIR = "logs/false_alarm_crops"
os.makedirs(OUT_CROPS_DIR, exist_ok=True)

def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "runs/yolov8n_seraphim/weights/best.pt"
    neg_test_file = "data/negatives_test.txt"
    
    if not os.path.exists(neg_test_file):
        print(f"Negatives test file {neg_test_file} not found. False alarms cannot be measured.")
        return
        
    neg_paths = [l.strip() for l in open(neg_test_file).readlines() if l.strip()]
    total_neg = len(neg_paths)
    print(f"Measuring false alarms on {total_neg} negative images using {model_path}...")
    
    model = YOLO(model_path)
    
    thresholds = [0.10, 0.25, 0.50, 0.75]
    fp_counts = {t: 0 for t in thresholds}
    all_detections = []
    
    for img_path in tqdm(neg_paths, desc="Scanning negatives"):
        preds = model(img_path, conf=0.10, verbose=False, device=0)[0]
        if len(preds.boxes) > 0:
            for box in preds.boxes:
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                for t in thresholds:
                    if conf >= t:
                        fp_counts[t] += 1
                all_detections.append({
                    "img_path": img_path,
                    "conf": conf,
                    "box": [float(x) for x in xyxy]
                })
                
    # Sort detections by confidence descending
    all_detections.sort(key=lambda d: d["conf"], reverse=True)
    
    # Save top 50 crops for inspection
    print(f"\nSaving top {min(50, len(all_detections))} highest-confidence detections to {OUT_CROPS_DIR}...")
    for i, d in enumerate(all_detections[:50]):
        img = cv2.imread(d["img_path"])
        if img is not None:
            x1, y1, x2, y2 = [int(v) for v in d["box"]]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"Conf: {d['conf']:.2f}", (x1, max(y1-5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            crop_fname = os.path.join(OUT_CROPS_DIR, f"top_fp_{i:02d}_conf_{d['conf']:.2f}.jpg")
            cv2.imwrite(crop_fname, img)
            
    print("\n" + "="*60)
    print("FALSE ALARM MEASUREMENTS (PER 1,000 NEGATIVE IMAGES)")
    print("="*60)
    rates = {}
    for t in thresholds:
        count = fp_counts[t]
        rate_per_k = (count / total_neg * 1000.0) if total_neg > 0 else 0.0
        rates[str(t)] = {
            "total_false_positives": count,
            "false_positives_per_1000": round(rate_per_k, 2)
        }
        print(f"  Confidence >= {t:.2f} : {count:4d} FPs total -> {rate_per_k:6.2f} FPs / 1,000 images")
    print("="*60)
    
    with open("logs/false_alarm_summary.json", "w", encoding="utf-8") as f:
        json.dump(rates, f, indent=2)

if __name__ == "__main__":
    main()
