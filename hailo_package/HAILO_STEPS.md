# Hailo-10H HEF Compilation Instructions

## Prerequisites
Hailo Dataflow Compiler (DFC) runs on **Linux x86-64** (Ubuntu 22.04 LTS recommended, Python 3.8 - 3.10).

### Recommended Options:
1. **WSL2 on Windows:**
   ```bash
   # In Windows PowerShell, install Ubuntu:
   wsl --install -d Ubuntu-22.04
   ```
2. **Docker with Hailo DFC Image:**
   ```bash
   docker run -it --rm -v $(pwd):/workspace -w /workspace hailotarget/dfc:latest bash
   ```

---

## Step-by-Step Compilation Commands

### 1. Compile Hailo-10H HEF with Fast Profile (Preview):
```bash
hailomz compile \
    --hw-arch hailo10h \
    --model-name yolov8n_seraphim \
    --yaml hailo/yolov8n_seraphim.yaml \
    --alls hailo/fast.alls \
    --calib-path hailo/calib/calib_set.npy \
    --output-dir runs/yolov8n_seraphim/weights/
```

### 2. Compile Hailo-10H HEF with Full Quality Profile (Final Release):
```bash
hailomz compile \
    --hw-arch hailo10h \
    --model-name yolov8n_seraphim \
    --yaml hailo/yolov8n_seraphim.yaml \
    --alls hailo/full.alls \
    --calib-path hailo/calib/calib_set.npy \
    --output-dir runs/yolov8n_seraphim/weights/
```

---

## Deployment on Raspberry Pi + Hailo-10H
- Target Hardware: **Raspberry Pi 5 with Hailo-10H AI Hat**
- Install HailoRT:
  ```bash
  sudo apt install hailo-all
  ```
- Run Python inference on device:
  ```python
  import hailo_platform as hp
  # Load HEF and configure 640x640 input stream
  ```
