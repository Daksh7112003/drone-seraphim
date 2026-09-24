# YOLOv8n Training & Hailo-10H Pipeline on Seraphim Drone Dataset

## Phase 0: System & Dataset Inspection Report

### 1. Hardware & System Specifications
- **Operating System:** Microsoft Windows 11 Home (Version 10.0.22000, 64-bit)
- **CPU:** Intel(R) Core(TM) i9-14900KS
  - **Physical Cores:** 24 (8 Performance cores + 16 Efficient cores)
  - **Logical Processors:** 32
- **System Memory (RAM):** 64.0 GB
- **Storage:** Drive C: ~245.5 GB free (NVMe/SSD)
- **GPU:** NVIDIA GeForce RTX 5070 Ti (Blackwell architecture, sm_120)
  - **VRAM:** 16,303 MiB (16 GB)
  - **Driver Version:** 591.86
  - **CUDA Version:** 13.1 (Driver) / PyTorch 2.12.0.dev+cu128
  - **Power Limits:**
    - Default Power Limit: **300.00 W**
    - Enforced/Current Power Limit: **300.00 W**
    - Minimum Power Limit: **250.00 W**
    - Maximum Power Limit: **350.00 W**
  - **Hardware Verification:** PyTorch CUDA initialization, GPU matmul, Conv2d forward pass, and YOLOv8n inference successfully verified on device `cuda:0` (`(12, 0)` compute capability).

### 2. Dataset Inspection (`DATASET_PATH = C:\Users\dhruvbomber\drone-seraphim\Data`)
- **Total Images:** 83,483 `.jpg` files located in `Data/images/` (Total size: ~8.81 GB).
- **Total Labels:** 83,483 `.txt` files located in `Data/labels_backup_class0/` (Total size: ~3.60 MB).
- **Integrity Check:**
  - Missing labels: **0**
  - Unmatched images/labels: **0** (Exact 1:1 match across all 83,483 items)
  - Positive-only check: Verified sample labels contain class `0` (`drone`).
- **Original Dataset Immutability:** The original `Data` directory remains completely untouched and unmodified.

### 3. Hailo Toolchain & Compiler Status
- **Hailo DFC & HailoMZ:** Not installed on host Windows (DFC requires Linux x86-64, e.g. Ubuntu 22.04/24.04).
- **Docker / WSL:** Neither Docker nor an active WSL2 Linux distro is currently initialized in PATH.
- **Export Strategy:**
  - Per-epoch ONNX export (`last.onnx`, `best.onnx`) with Hailo-compatible opset and end-nodes runs locally on CPU via `scripts/export_watcher.py`.
  - A complete `hailo_package/` with calibration set, network YAML, `.alls` scripts (`fast` and `full`), NMS configs, and compilation script (`hailo_worker.sh` / `HAILO_STEPS.md`) will be generated.
  - If WSL2/Ubuntu or external Linux with DFC is connected, HEFs (`last.hef`, `best.hef`) compile automatically.

---

## Phase-by-Phase Plan

### Phase 1: Environment & Tooling Verification
1. Freeze working dependencies into `requirements.txt` (`torch`, `torchvision`, `ultralytics`, `onnx`, `onnxslim`, `onnxruntime`, `imagehash`, `opencv-python`, `pillow`, `psutil`).
2. Verify repeatable script execution and set seeds.

### Phase 2: Data Validation, Splits & Negatives
1. **Positives Validation (`validate_data.py`):**
   - Check bounding box formats (0-1 range, 5 values per row, class 0).
   - Compute box size distribution (tiny <16x16, small 16-32, medium 32-96, large >96 px).
   - Check for corrupt images (logged to `data/corrupt_files.txt`).
2. **Padding Inspection:** Analyze 50 random images to determine exact letterbox padding color and aspect ratio treatment.
3. **Split Generation (`make_splits.py`):**
   - Train Positives: ~71,134 images (`data/train_pos.txt`)
   - Val Positives: ~4,000 images (`data/val_pos.txt`)
   - Test Positives: ~8,349 images (`data/test_pos.txt`)
   - Perceptual hash leakage test between train and val/test sets.
4. **Negatives Strategy (`make_negatives.py`):**
   - Target: ~10% negatives in train/val sets + dedicated `negatives_test` set (>1,000 images).
   - Mix: Generic background (sky, clouds, landscape, urban) + Hard negatives (birds, airplanes, helicopters, kites).
   - Prepare empty label files in `data/negatives/labels/`.
   - Record sources, licences, and splits in `data/NEGATIVES.md`.
5. **Config Creation:**
   - Generate `configs/data.yaml`, `configs/data_test_pos.yaml`, `configs/data_test_mixed.yaml`.
6. **Stop & Review Checkpoint:** Present negative counts, sample previews, and split statistics for approval.

### Phase 3: GPU Throughput Tuning (RTX 5070 Ti)
1. Launch background GPU and CPU resource logging (`scripts/benchmark_gpu.py`).
2. Execute 1-2 epoch trials on a fixed 10,000-image subset varying:
   - Batch sizes: 64, 128, 192, 256, and dynamic fraction `0.90`.
   - Worker threads: 8, 16, 24, 32.
   - Cache mode: `disk` (using high-speed NVMe) vs `none` vs `ram` (evaluated against 64 GB RAM).
3. Record images/sec, GPU utilization (median, p95), VRAM allocation, power draw, and SM clocks.
4. Stop & Review Checkpoint: Present GPU tuning table and optimal hyperparameters.

### Phase 3B: Per-Epoch Hailo-Compatible Export Pipeline
1. Inspect YOLOv8n ONNX graph to identify exact 6 detection head output convolution node names.
2. Build fixed 1,024-image calibration set (`hailo/calib/`).
3. Generate `fast.alls` and `full.alls` scripts, `yolov8n_seraphim.yaml`, and `nms_config.json`.
4. Implement `scripts/export_watcher.py` (atomic replacements, `last_export.json`, non-blocking).
5. Benchmark watcher overhead on a 2-epoch trial and present timing metrics.

### Phase 4: Full Model Training
1. Train YOLOv8n (`epochs=100`, `patience=20`, `imgsz=640`, `amp=True`, optimal batch/workers/cache).
2. Ultralytics training callback copies weights to `export_queue/`.
3. Monitor GPU power, temperature, and epoch progression via `scripts/status.py`.

### Phase 5: Evaluation & False-Alarm Analysis
1. Evaluate on `data_test_pos.yaml` (mAP50, mAP50-95, recall by size bucket).
2. Evaluate on `data_test_mixed.yaml` (realistic operational scenario).
3. Compute False Positives per 1,000 negative images across confidence thresholds and categories.
4. Export top 50 highest-confidence detections on negatives for manual inspection of potential unlabeled drones.
5. Save 50 false positives and 50 false negatives on test set for error analysis.

### Phase 6: Final Export & Packaging
1. Export `best.onnx` and perform parity verification against FP32 PyTorch model on 100 test images.
2. Create standard ONNX export and documentation (`exports/ONNX_NOTES.md`).
3. Compile `best.hef` (full-quality profile) if DFC is available, or package complete compile bundle in `hailo_package/` with `HAILO_STEPS.md`.

---

## Directory Layout
```
PROJECT_DIR/ (c:\Users\dhruvbomber\drone-seraphim\)
├── PLAN.md
├── README.md
├── NOTICE
├── requirements.txt
├── configs/
│   ├── data.yaml
│   ├── data_test_pos.yaml
│   ├── data_test_mixed.yaml
│   └── train.yaml
├── data/
│   ├── train.txt
│   ├── val.txt
│   ├── test_pos.txt
│   ├── test_mixed.txt
│   ├── negatives_test.txt
│   ├── train_pos.txt
│   ├── val_pos.txt
│   ├── corrupt_files.txt
│   ├── NEGATIVES.md
│   └── negatives/
│       ├── images/
│       └── labels/
├── hailo/
│   ├── calib/
│   ├── yolov8n_seraphim.yaml
│   ├── fast.alls
│   ├── full.alls
│   └── nms_config.json
├── scripts/
│   ├── validate_data.py
│   ├── make_splits.py
│   ├── make_negatives.py
│   ├── benchmark_gpu.py
│   ├── train.py
│   ├── evaluate.py
│   ├── false_alarms.py
│   ├── export_watcher.py
│   ├── status.py
│   ├── export_onnx.py
│   ├── parity_check.py
│   └── make_calibration.py
├── export_queue/
├── runs/
│   └── yolov8n_seraphim/
│       └── weights/
│           ├── last.pt
│           ├── best.pt
│           ├── last.onnx
│           ├── last.hef
│           └── last_export.json
├── exports/
│   ├── yolov8n_seraphim_standard.onnx
│   └── ONNX_NOTES.md
├── hailo_package/
└── logs/
    ├── gpu_summary.md
    └── export_watcher.log
```

---

## Technical Notes & Risks
1. **Windows Dataloader Workers:** Windows spawns separate processes for dataloader workers, creating higher IPC overhead compared to Linux forks. In Phase 3, we will test whether NVMe disk caching (`cache="disk"`) or pinned RAM caching helps maintain >95% GPU utilization.
2. **Seraphim Label Directory Name:** The original dataset folder is `Data/labels_backup_class0`. Ultralytics default path resolver replaces `/images/` with `/labels/`. To strictly adhere to Hard Rule 2 (never modify/move original dataset files), our split list files and dataset loader will map image paths to labels via a symlink/junction or custom loader mapping in `PROJECT_DIR/data/`.
3. **Hailo DFC Compatibility:** DFC targets Linux x86-64. The pipeline will execute all ONNX steps natively and generate all required Hailo artifacts and configs ready for 1-click execution in WSL2/Docker/Linux.
