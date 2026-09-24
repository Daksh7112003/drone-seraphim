# GPU Throughput Tuning & Benchmark Summary

## System & Hardware
- **GPU:** NVIDIA GeForce RTX 5070 Ti (16 GB VRAM, Blackwell architecture, sm_120)
- **Power Limit:** 300.00 W default
- **Driver:** 591.86 | **PyTorch:** 2.12.0.dev+cu128
- **OS:** Windows 11 Home (64-bit)

---

## Benchmark Trials & Telemetry

| Trial | Batch Size | Workers | Cache Mode | Images/sec | VRAM Peak (MB) | VRAM % | Peak Power (W) | Median GPU Util (%) | p95 GPU Util (%) | Status / Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `batch64_w8` | 64 | 8 | None | 287.8 | 7,578 | 46.5% | 171.9 W | 92.0% | 98.0% | Stable |
| `batch128_w8` | **128** | **8** | **None** | **357.1** | **10,882** | **66.7%** | **206.7 W** | **99.0%** | **100.0%** | **Optimal (Best throughput & headroom)** |
| `batch192_w8` | 192 | 8 | None | 185.2 | 15,978 | 98.0% | 73.1 W | 100.0% | 100.0% | WDDM RAM thrashing at >95% VRAM |
| `batch256_w8` | 256 | 8 | None | 0.0 | >16,303 | >100% | N/A | 0.0% | 0.0% | Out of Memory (OOM) |
| `batch128_w16` | 128 | 16 | None | N/A | N/A | N/A | N/A | N/A | N/A | Win32 Error 1455 (pagefile handle limit) |

---

## Power & Hardware Utilization Analysis
1. **GPU Power Draw:** Peak power reached **206.7 W** at `batch=128`. Average power during training cycles is ~70-130 W.
2. **Why YOLOv8n draws ~70-207 W (below the 300 W board limit):**
   - YOLOv8n is a nano architecture with only ~3.0 million parameters and 8.2 GFLOPs.
   - At 640x640 resolution, tensor operations complete in sub-millisecond bursts.
   - Even with continuous 100% GPU utilization, a nano model does not switch enough silicon gates simultaneously to saturate the full 300 W thermal envelope.
3. **Chosen Full Training Hyperparameters:**
   - `batch: 128`
   - `workers: 8`
   - `amp: True`
   - `imgsz: 640`
   - `cache: False` (relying on high-speed NVMe image loading)
