"""
scripts/make_splits.py
Generates train_pos.txt, val_pos.txt, test_pos.txt and checks for imagehash leakage.
"""
import os
import random
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import imagehash

DATASET_PATH = r"C:\Users\dhruvbomber\drone-seraphim\Data"
IMG_DIR = os.path.join(DATASET_PATH, "images")
CORRUPT_FILE = r"data/corrupt_files.txt"

def compute_hash(img_path):
    try:
        with Image.open(img_path) as img:
            # Perceptual hash (dhash or phash)
            h = imagehash.phash(img)
            return img_path, str(h)
    except Exception:
        return img_path, None

def main():
    # Read corrupt files to exclude
    corrupt_stems = set()
    if os.path.exists(CORRUPT_FILE):
        with open(CORRUPT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    corrupt_stems.add(line.split(":")[0].strip())
                    
    print(f"Excluding {len(corrupt_stems)} corrupt stems: {corrupt_stems}")
    
    all_files = sorted(os.listdir(IMG_DIR))
    valid_paths = []
    for f in all_files:
        stem = os.path.splitext(f)[0]
        if stem not in corrupt_stems and f.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Store normalized forward-slash absolute path
            full_path = os.path.abspath(os.path.join(IMG_DIR, f)).replace('\\', '/')
            valid_paths.append(full_path)
            
    total = len(valid_paths)
    print(f"Total valid positive images: {total}")
    
    # Deterministic shuffle
    random.seed(42)
    random.shuffle(valid_paths)
    
    # Split counts: ~8,348 test (10%), 4,000 val, remaining (~71,132) train
    n_test = 8348
    n_val = 4000
    n_train = total - n_test - n_val
    
    test_paths = sorted(valid_paths[:n_test])
    val_paths = sorted(valid_paths[n_test:n_test + n_val])
    train_paths = sorted(valid_paths[n_test + n_val:])
    
    print(f"Splits: Train = {len(train_paths)}, Val = {len(val_paths)}, Test = {len(test_paths)}")
    
    # Write split list files
    os.makedirs("data", exist_ok=True)
    with open("data/train_pos.txt", "w", encoding="utf-8") as f:
        for p in train_paths:
            f.write(p + "\n")
            
    with open("data/val_pos.txt", "w", encoding="utf-8") as f:
        for p in val_paths:
            f.write(p + "\n")
            
    with open("data/test_pos.txt", "w", encoding="utf-8") as f:
        for p in test_paths:
            f.write(p + "\n")
            
    print("Positives split files written.")
    
    # Leakage check: sample 1000 from val and 1000 from test against train
    print("\nRunning perceptual hash leakage analysis...")
    print("Computing hashes for sample of train (5000), val (1000), and test (1000)...")
    
    train_sample = random.sample(train_paths, min(5000, len(train_paths)))
    val_sample = random.sample(val_paths, min(1000, len(val_paths)))
    test_sample = random.sample(test_paths, min(1000, len(test_paths)))
    
    with ThreadPoolExecutor(max_workers=16) as ex:
        train_hashes = dict(list(tqdm(ex.map(compute_hash, train_sample), total=len(train_sample), desc="Train hashes")))
        val_hashes = dict(list(tqdm(ex.map(compute_hash, val_sample), total=len(val_sample), desc="Val hashes")))
        test_hashes = dict(list(tqdm(ex.map(compute_hash, test_sample), total=len(test_sample), desc="Test hashes")))
        
    train_hash_set = {h for h in train_hashes.values() if h is not None}
    
    val_exact_leaks = sum(1 for h in val_hashes.values() if h in train_hash_set)
    test_exact_leaks = sum(1 for h in test_hashes.values() if h in train_hash_set)
    
    print(f"Val exact pHash matches in Train sample: {val_exact_leaks} / {len(val_sample)} ({val_exact_leaks/len(val_sample)*100:.2f}%)")
    print(f"Test exact pHash matches in Train sample: {test_exact_leaks} / {len(test_sample)} ({test_exact_leaks/len(test_sample)*100:.2f}%)")

if __name__ == "__main__":
    main()
