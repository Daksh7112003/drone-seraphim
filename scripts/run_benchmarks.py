"""
scripts/run_benchmarks.py
Runs isolated subprocess benchmark trials across batch sizes (64, 128, 192) and worker configurations.
Monitors and logs GPU power, temperature, SM clock, utilization, and throughput.
"""
import os
import sys
import time
import subprocess
import csv
import numpy as np

SUBSET_DATA = "configs/data_bench.yaml"
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

TRIALS = [
    {"name": "batch64_w8", "batch": 64, "workers": 8, "cache": False},
    {"name": "batch128_w8", "batch": 128, "workers": 8, "cache": False},
    {"name": "batch192_w8", "batch": 192, "workers": 8, "cache": False},
    {"name": "batch128_w4", "batch": 128, "workers": 4, "cache": False},
    {"name": "batch128_w12", "batch": 128, "workers": 12, "cache": False},
]

def run_isolated_trial(t):
    name = t["name"]
    batch = t["batch"]
    workers = t["workers"]
    cache = t["cache"]
    log_csv = os.path.join(LOGS_DIR, f"gpu_bench_{name}.csv")
    
    print(f"\n========================================================")
    print(f"BENCHMARK TRIAL: {name} (Batch={batch}, Workers={workers}, Cache={cache})")
    print(f"========================================================")
    
    # Start GPU monitor subprocess
    mon_cmd = [
        "nvidia-smi",
        "--query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw,memory.used,clocks.sm,temperature.gpu",
        "--format=csv,nounits",
        "-l", "1"
    ]
    with open(log_csv, "w", encoding="utf-8") as out_f:
        mon_proc = subprocess.Popen(mon_cmd, stdout=out_f)
        
    t0 = time.time()
    code = f"""
import time
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(
    data='{SUBSET_DATA}',
    imgsz=640,
    epochs=1,
    batch={batch},
    workers={workers},
    cache={repr(cache)},
    device=0,
    amp=True,
    deterministic=False,
    project='runs/benchmarks',
    name='{name}',
    exist_ok=True,
    verbose=False
)
"""
    train_proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    t1 = time.time()
    
    mon_proc.terminate()
    try:
        mon_proc.wait(timeout=2.0)
    except Exception:
        mon_proc.kill()
        
    duration = t1 - t0
    success = (train_proc.returncode == 0)
    
    if not success:
        print(f"Trial {name} Failed: {train_proc.stderr[:300]}")
        return None
        
    # Parse monitor CSV
    gpu_utils, pwr_draws, mem_useds = [], [], []
    if os.path.exists(log_csv):
        with open(log_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 7 and "utilization" not in row[1]:
                    try:
                        gpu_utils.append(float(row[1]))
                        pwr_draws.append(float(row[3]))
                        mem_useds.append(float(row[4]))
                    except Exception:
                        pass
                        
    img_per_sec = 8000.0 / max(duration, 0.001)
    res = {
        "Trial": name,
        "Batch": batch,
        "Workers": workers,
        "Cache": str(cache),
        "Duration_s": round(duration, 2),
        "Images_per_sec": round(img_per_sec, 1),
        "GPU_Util_Median_%": round(float(np.median(gpu_utils)), 1) if gpu_utils else 0.0,
        "GPU_Util_p95_%": round(float(np.percentile(gpu_utils, 95)), 1) if gpu_utils else 0.0,
        "VRAM_Peak_MB": round(float(np.max(mem_useds)), 0) if mem_useds else 0.0,
        "Power_Avg_W": round(float(np.mean(pwr_draws)), 1) if pwr_draws else 0.0,
        "Power_Peak_W": round(float(np.max(pwr_draws)), 1) if pwr_draws else 0.0
    }
    print(f"Result: {res['Images_per_sec']} img/s | GPU Median: {res['GPU_Util_Median_%']}% | Peak Power: {res['Power_Peak_W']}W | Peak VRAM: {res['VRAM_Peak_MB']}MB")
    return res

def main():
    results = []
    for t in TRIALS:
        r = run_isolated_trial(t)
        if r:
            results.append(r)
            
    summary_path = os.path.join(LOGS_DIR, "gpu_tuning_summary.csv")
    if results:
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
            
    md_path = os.path.join(LOGS_DIR, "gpu_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# GPU Throughput Tuning & Benchmark Summary\n\n")
        f.write("| Trial | Batch | Workers | Cache | Img/s | Duration (s) | GPU Util Median (%) | GPU Util p95 (%) | VRAM Peak (MB) | Power Avg (W) | Power Peak (W) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for r in results:
            f.write(f"| {r['Trial']} | {r['Batch']} | {r['Workers']} | {r['Cache']} | {r['Images_per_sec']} | {r['Duration_s']} | {r['GPU_Util_Median_%']} | {r['GPU_Util_p95_%']} | {r['VRAM_Peak_MB']} | {r['Power_Avg_W']} | {r['Power_Peak_W']} |\n")
            
    print("\n" + "="*85)
    print("GPU TUNING BENCHMARK TABLE")
    print("="*85)
    print(open(md_path).read())
    print("="*85)

if __name__ == "__main__":
    main()
