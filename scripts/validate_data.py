"""
scripts/validate_data.py
Comprehensive validation script for Seraphim Drone Detection Dataset.
"""
import os
import sys
import glob
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

DATASET_PATH = r"C:\Users\dhruvbomber\drone-seraphim\Data"
IMG_DIR = os.path.join(DATASET_PATH, "images")
LBL_DIR = os.path.join(DATASET_PATH, "labels_backup_class0")

def validate_single_pair(filename_stem):
    img_path = os.path.join(IMG_DIR, filename_stem + ".jpg")
    lbl_path = os.path.join(LBL_DIR, filename_stem + ".txt")
    
    corrupt_reason = None
    boxes = []
    
    # Check image readability and dimensions
    try:
        with Image.open(img_path) as img:
            img.verify()
        # Verify with cv2 or PIL open
        with Image.open(img_path) as img:
            w, h = img.size
            if w <= 0 or h <= 0:
                corrupt_reason = f"Invalid dimensions: {w}x{h}"
    except Exception as e:
        corrupt_reason = f"Image corrupt: {str(e)}"
        
    if corrupt_reason:
        return {"stem": filename_stem, "corrupt": corrupt_reason, "boxes": []}
        
    # Check label file
    if not os.path.exists(lbl_path):
        return {"stem": filename_stem, "corrupt": "Missing label file", "boxes": []}
        
    try:
        with open(lbl_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            
        if len(lines) == 0:
            return {"stem": filename_stem, "empty_label": True, "boxes": []}
            
        for line_num, line in enumerate(lines):
            parts = line.split()
            if len(parts) != 5:
                return {"stem": filename_stem, "corrupt": f"Label line {line_num} has {len(parts)} values (expected 5)", "boxes": []}
            cls_id = int(parts[0])
            cx = float(parts[1])
            cy = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])
            
            if cls_id != 0:
                return {"stem": filename_stem, "corrupt": f"Non-zero class ID {cls_id}", "boxes": []}
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 <= bw <= 1.0 and 0.0 <= bh <= 1.0):
                return {"stem": filename_stem, "corrupt": f"Box coords out of bounds: {parts}", "boxes": []}
            
            # Pixel dimensions at 640x640
            px_w = bw * 640.0
            px_h = bh * 640.0
            boxes.append((px_w, px_h, bw, bh))
            
    except Exception as e:
        return {"stem": filename_stem, "corrupt": f"Label read error: {str(e)}", "boxes": []}
        
    return {"stem": filename_stem, "corrupt": None, "empty_label": False, "boxes": boxes}

def main():
    print(f"Validating dataset at: {DATASET_PATH}")
    img_files = os.listdir(IMG_DIR)
    stems = [os.path.splitext(f)[0] for f in img_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total = len(stems)
    print(f"Found {total} image files to validate.")
    
    corrupt_files = []
    empty_labels = []
    all_boxes = []
    boxes_per_image = []
    
    tiny_count = 0    # < 16x16 px (max dimension < 16)
    small_count = 0   # 16-32 px
    medium_count = 0  # 32-96 px
    large_count = 0   # >= 96 px
    
    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(tqdm(executor.map(validate_single_pair, stems), total=total, desc="Validating"))
        
    for res in results:
        if res.get("corrupt"):
            corrupt_files.append((res["stem"], res["corrupt"]))
        elif res.get("empty_label"):
            empty_labels.append(res["stem"])
        else:
            b_list = res["boxes"]
            boxes_per_image.append(len(b_list))
            for px_w, px_h, bw, bh in b_list:
                all_boxes.append((px_w, px_h))
                max_dim = max(px_w, px_h)
                if max_dim < 16.0:
                    tiny_count += 1
                elif max_dim < 32.0:
                    small_count += 1
                elif max_dim < 96.0:
                    medium_count += 1
                else:
                    large_count += 1
                    
    os.makedirs("data", exist_ok=True)
    with open("data/corrupt_files.txt", "w", encoding="utf-8") as f:
        for stem, reason in corrupt_files:
            f.write(f"{stem}: {reason}\n")
            
    print("\n" + "="*50)
    print("VALIDATION SUMMARY")
    print("="*50)
    print(f"Total images checked: {total}")
    print(f"Corrupt files found: {len(corrupt_files)}")
    print(f"Empty label files found: {len(empty_labels)} (Must be 0 as confirmed: positive-only)")
    print(f"Total bounding boxes: {len(all_boxes)}")
    print(f"Average boxes per image: {np.mean(boxes_per_image):.2f} (Min: {np.min(boxes_per_image)}, Max: {np.max(boxes_per_image)})")
    print("\nBox Size Histogram at 640x640 resolution (max dimension):")
    print(f"  - Tiny   (< 16 px)    : {tiny_count:6d} ({tiny_count/len(all_boxes)*100:5.2f}%)")
    print(f"  - Small  (16 - 32 px) : {small_count:6d} ({small_count/len(all_boxes)*100:5.2f}%)")
    print(f"  - Medium (32 - 96 px) : {medium_count:6d} ({medium_count/len(all_boxes)*100:5.2f}%)")
    print(f"  - Large  (>= 96 px)   : {large_count:6d} ({large_count/len(all_boxes)*100:5.2f}%)")
    print("="*50)

if __name__ == "__main__":
    main()
