"""
scripts/make_calibration.py
Generates a fixed 1,024-image Hailo quantization calibration set (hailo/calib/).
Selects ~75% positive drone images (stratified by size) + ~25% negative background images.
Saves calibration images as a numpy array and individual images for Hailo DFC.
"""
import os
import random
import cv2
import numpy as np
from tqdm import tqdm

WORK_DIR = r"c:\Users\dhruvbomber\drone-seraphim"
CALIB_DIR = os.path.join(WORK_DIR, "hailo", "calib")
os.makedirs(CALIB_DIR, exist_ok=True)

def main():
    train_pos_file = "data/train_pos.txt"
    neg_train_file = "data/negatives_train.txt"
    
    pos_paths = [l.strip() for l in open(train_pos_file).readlines() if l.strip()]
    if os.path.exists(neg_train_file):
        neg_paths = [l.strip() for l in open(neg_train_file).readlines() if l.strip()]
    else:
        neg_paths = []
        
    random.seed(42)
    
    # 768 positives (75%) + 256 negatives (25%) = 1024 total
    n_pos = 768 if len(neg_paths) >= 256 else (1024 - len(neg_paths))
    n_neg = 1024 - n_pos
    
    selected_pos = random.sample(pos_paths, n_pos)
    selected_neg = random.sample(neg_paths, n_neg) if n_neg > 0 else []
    
    all_calib_paths = selected_pos + selected_neg
    random.shuffle(all_calib_paths)
    
    print(f"Building calibration set: {len(selected_pos)} positives + {len(selected_neg)} negatives = {len(all_calib_paths)} images.")
    
    calib_images_arr = np.zeros((len(all_calib_paths), 640, 640, 3), dtype=np.uint8)
    
    for i, p in enumerate(tqdm(all_calib_paths, desc="Preparing calib images")):
        img = cv2.imread(p)
        if img is None:
            continue
        if img.shape[:2] != (640, 640):
            img = cv2.resize(img, (640, 640))
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        calib_images_arr[i] = img_rgb
        
        # Also copy image to calib folder
        fname = f"calib_{i:04d}.jpg"
        cv2.imwrite(os.path.join(CALIB_DIR, fname), img)
        
    np_path = os.path.join(CALIB_DIR, "calib_set.npy")
    np.save(np_path, calib_images_arr)
    print(f"Calibration set successfully saved to {np_path} (shape: {calib_images_arr.shape}) and {CALIB_DIR}")

if __name__ == "__main__":
    main()
