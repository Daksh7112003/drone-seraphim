"""
scripts/make_negatives.py
Downloads and prepares ~8,513 negative background and distractor images from COCO 2017.
Letterboxes all images to 640x640 with (114, 114, 114) gray padding.
Creates empty label files for single-class background suppression.
"""
import os
import sys
import zipfile
import urllib.request
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import random

COCO_VAL_URL = "http://images.cocodataset.org/zips/val2017.zip"  # 5,000 images (~777 MB)
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

WORK_DIR = r"c:\Users\dhruvbomber\drone-seraphim"
EXTRACT_DIR = os.path.join(WORK_DIR, "data", "extracted")
NEG_IMG_DIR = os.path.join(WORK_DIR, "data", "negatives", "images")
NEG_LBL_DIR = os.path.join(WORK_DIR, "data", "negatives", "labels")

os.makedirs(EXTRACT_DIR, exist_ok=True)
os.makedirs(NEG_IMG_DIR, exist_ok=True)
os.makedirs(NEG_LBL_DIR, exist_ok=True)

def download_with_progress(url, dest_path):
    if os.path.exists(dest_path):
        print(f"File already exists: {dest_path}")
        return
    print(f"Downloading {url} to {dest_path}...")
    def hook(t):
        last_b = [0]
        def update_to(b=1, bsize=1, tsize=None):
            if tsize is not None:
                t.total = tsize
            t.update((b - last_b[0]) * bsize)
            last_b[0] = b
        return update_to

    with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=os.path.basename(dest_path)) as t:
        urllib.request.urlretrieve(url, filename=dest_path, reporthook=hook(t))

def letterbox_image(img, target_size=(640, 640), fill_color=(114, 114, 114)):
    h, w = img.shape[:2]
    tw, th = target_size
    scale = min(tw / w, th / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((th, tw, 3), fill_color, dtype=np.uint8)
    
    dx = (tw - nw) // 2
    dy = (th - nh) // 2
    canvas[dy:dy+nh, dx:dx+nw] = resized
    return canvas

def main():
    val_zip = os.path.join(EXTRACT_DIR, "val2017.zip")
    download_with_progress(COCO_VAL_URL, val_zip)
    
    # Extract val2017
    extracted_val_folder = os.path.join(EXTRACT_DIR, "val2017")
    if not os.path.exists(extracted_val_folder):
        print("Extracting val2017.zip...")
        with zipfile.ZipFile(val_zip, 'r') as zf:
            zf.extractall(EXTRACT_DIR)
        print("Extraction complete.")
        
    all_coco_imgs = [os.path.join(extracted_val_folder, f) for f in os.listdir(extracted_val_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(all_coco_imgs)} images in COCO val2017.")
    
    # Process images: letterbox to 640x640 with (114, 114, 114) and save to data/negatives/images
    # Create empty labels in data/negatives/labels
    processed_count = 0
    neg_img_paths = []
    
    print("Processing negative images (letterbox 640x640, gray 114 fill)...")
    for src_path in tqdm(all_coco_imgs, desc="Letterboxing"):
        fname = os.path.basename(src_path)
        stem = os.path.splitext(fname)[0]
        dst_img = os.path.join(NEG_IMG_DIR, f"coco_neg_{stem}.jpg")
        dst_lbl = os.path.join(NEG_LBL_DIR, f"coco_neg_{stem}.txt")
        
        if not os.path.exists(dst_img):
            img = cv2.imread(src_path)
            if img is not None:
                boxed = letterbox_image(img, (640, 640), fill_color=(114, 114, 114))
                cv2.imwrite(dst_img, boxed, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        
        # Create empty label file (0 bytes)
        if not os.path.exists(dst_lbl):
            with open(dst_lbl, "w", encoding="utf-8") as f:
                pass
                
        norm_path = os.path.abspath(dst_img).replace('\\', '/')
        neg_img_paths.append(norm_path)
        processed_count += 1
        
    print(f"Successfully prepared {processed_count} negative images and empty label files.")
    
    # Split negatives: 3,500 train negatives, 400 val negatives, 1,100 test negatives
    random.seed(42)
    random.shuffle(neg_img_paths)
    
    n_neg_val = 400
    n_neg_test = 1000
    n_neg_train = len(neg_img_paths) - n_neg_val - n_neg_test
    
    neg_val = neg_img_paths[:n_neg_val]
    neg_test = neg_img_paths[n_neg_val:n_neg_val + n_neg_test]
    neg_train = neg_img_paths[n_neg_val + n_neg_test:]
    
    print(f"Negatives split: Train = {len(neg_train)}, Val = {len(neg_val)}, Negatives Test = {len(neg_test)}")
    
    with open("data/negatives_train.txt", "w", encoding="utf-8") as f:
        for p in neg_train:
            f.write(p + "\n")
            
    with open("data/negatives_val.txt", "w", encoding="utf-8") as f:
        for p in neg_val:
            f.write(p + "\n")
            
    with open("data/negatives_test.txt", "w", encoding="utf-8") as f:
        for p in neg_test:
            f.write(p + "\n")
            
    # Combine positive + negative list files
    train_pos = [l.strip() for l in open("data/train_pos.txt").readlines() if l.strip()]
    val_pos = [l.strip() for l in open("data/val_pos.txt").readlines() if l.strip()]
    test_pos = [l.strip() for l in open("data/test_pos.txt").readlines() if l.strip()]
    
    train_combined = train_pos + neg_train
    val_combined = val_pos + neg_val
    test_mixed = test_pos + neg_test[:835]  # ~10% negatives in mixed test
    
    random.seed(42)
    random.shuffle(train_combined)
    random.shuffle(val_combined)
    random.shuffle(test_mixed)
    
    with open("data/train.txt", "w", encoding="utf-8") as f:
        for p in train_combined:
            f.write(p + "\n")
            
    with open("data/val.txt", "w", encoding="utf-8") as f:
        for p in val_combined:
            f.write(p + "\n")
            
    with open("data/test_mixed.txt", "w", encoding="utf-8") as f:
        for p in test_mixed:
            f.write(p + "\n")
            
    print(f"Final combined splits written:")
    print(f"  - train.txt: {len(train_combined)} images ({len(train_pos)} pos + {len(neg_train)} neg)")
    print(f"  - val.txt: {len(val_combined)} images ({len(val_pos)} pos + {len(neg_val)} neg)")
    print(f"  - test_pos.txt: {len(test_pos)} images (100% pos)")
    print(f"  - test_mixed.txt: {len(test_mixed)} images ({len(test_pos)} pos + 835 neg)")
    print(f"  - negatives_test.txt: {len(neg_test)} images (100% pure neg)")

if __name__ == "__main__":
    main()
