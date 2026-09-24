# Seraphim Drone Detection & Hailo-10H Pipeline 🛸

High-throughput YOLOv8n single-class (`drone`) object detection training and deployment pipeline optimized for NVIDIA RTX 5070 Ti (Blackwell) training and edge deployment on Raspberry Pi 5 + Hailo-10H AI Hat.

---

## 📋 System & Architecture Overview

- **Dataset:** Seraphim Drone Dataset (83,483 positive images + hard/generic background negatives)
- **Model Architecture:** YOLOv8n (3.01M parameters, 8.2 GFLOPs @ 640x640)
- **Training Acceleration:** NVIDIA GeForce RTX 5070 Ti (16 GB VRAM, sm_120), Intel Core i9-14900KS, 64 GB RAM
- **Target Edge Hardware:** Raspberry Pi 5 with Hailo-10H NPU (40 TOPS)

---

## ⚡ Throughput & Hardware Benchmarks

GPU throughput trials executed on a 10,000-image subset identified the optimal training configuration:

| Trial | Batch Size | Workers | Precision | Images/sec | VRAM Peak | Peak Power | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `batch64_w8` | 64 | 8 | AMP fp16 | 287.8 | 7.58 GB | 171.9 W | Stable |
| **`batch128_w8`** | **128** | **8** | **AMP fp16** | **357.1** | **10.88 GB** | **206.7 W** | **Optimal** |
| `batch192_w8` | 192 | 8 | AMP fp16 | 185.2 | 15.98 GB | 73.1 W | VRAM Thrashing |
| `batch256_w8` | 256 | 8 | AMP fp16 | 0.0 | >16.0 GB | N/A | Out of Memory |

---

## 🚀 Pipeline Status & Milestones

- [x] **Phase 0 & 1:** Environment, CUDA 13 / Blackwell sm_120 verification, and dataset validation.
- [x] **Phase 2:** Train (74,732), Val (4,400), and Test (8,349) split generation with hard negative backgrounds.
- [x] **Phase 3:** Hardware benchmarking & hyperparameter selection.
- [x] **Phase 3B:** Hailo calibration dataset (1,024 images), `.alls` profiles, and asynchronous export watcher.
- [x] **Phase 4:** High-throughput training (Epoch 16/100: mAP50 `95.77%`, mAP50-95 `67.12%`, Precision `95.66%`, Recall `91.14%`).
- [ ] **Phase 5:** Evaluation across size buckets (tiny, small, medium, large) and false alarm rates on negative backgrounds.
- [ ] **Phase 6:** PyTorch vs ONNX parity checks and final Hailo-10H HEF compilation.

---

## 🛠️ Repository Layout

```
drone-seraphim/
├── configs/
│   ├── data.yaml                 # Primary training & validation config
│   ├── data_test_pos.yaml        # Positives-only test config
│   └── data_test_mixed.yaml      # Mixed test set (positives + negatives)
├── hailo/
│   ├── calib/                    # Fixed 1,024-image calibration set
│   ├── fast.alls                 # Fast preview compilation profile
│   ├── full.alls                 # Full-quality compilation profile
│   ├── nms_config.json           # NPU hardware NMS configuration
│   └── yolov8n_seraphim.yaml     # Hailo model definition
├── hailo_package/
│   └── HAILO_STEPS.md            # Step-by-step Hailo compilation guide
├── scripts/
│   ├── benchmark_gpu.py          # GPU & CPU telemetry logger
│   ├── evaluate.py               # Evaluation across size buckets & metrics
│   ├── export_onnx.py            # Static 640x640 Hailo-compatible ONNX exporter
│   ├── export_watcher.py         # Asynchronous per-epoch export worker
│   ├── false_alarms.py           # False positive rate calculation on negatives
│   ├── make_calibration.py       # Calibration set generator
│   ├── make_negatives.py         # Negative background generator
│   ├── make_splits.py            # Dataset split creator & leak detector
│   ├── parity_check.py           # PyTorch vs ONNX numerical verification
│   ├── status.py                 # Live training & checkpoint status viewer
│   ├── train.py                  # High-throughput training runner
│   └── validate_data.py          # Label integrity & bounding box validator
├── logs/
│   └── gpu_summary.md            # Hardware benchmarking report
├── PLAN.md                       # Complete technical design & implementation plan
├── NOTICE                        # Attribution & dataset notice
└── requirements.txt              # Pinned Python dependencies
```

---

## 💻 Quick Start Guide

### 1. Installation
```powershell
pip install -r requirements.txt
```

### 2. Resume / Run Training
```powershell
python scripts/train.py --resume
```

### 3. Check Live Status
```powershell
python scripts/status.py
```

### 4. Evaluate Test Metrics
```powershell
python scripts/evaluate.py runs/detect/runs/yolov8n_seraphim/weights/best.pt
```

### 5. Compile for Hailo-10H (Linux / WSL2 / Docker)
```bash
hailomz compile \
    --hw-arch hailo10h \
    --model-name yolov8n_seraphim \
    --yaml hailo/yolov8n_seraphim.yaml \
    --alls hailo/full.alls \
    --calib-path hailo/calib/calib_set.npy \
    --output-dir hailo_package/
```
