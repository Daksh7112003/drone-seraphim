"""
scripts/benchmark_gpu.py
Benchmarking script for RTX 5070 Ti throughput tuning on YOLOv8n.
Evaluates batch sizes, worker thread counts, and caching modes.
"""
import os
import sys
import time
import subprocess
import threading
import csv
import numpy as np
import psutil
from ultralytics import YOLO

SUBSET_DATA = "configs/data_bench.yaml"
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

class ResourceMonitor:
    def __init__(self, log_csv_path, interval=0.5):
        self.log_csv_path = log_csv_path
        self.interval = interval
        self.running = False
        self.records = []
        self._thread = None

    def _monitor(self):
        while self.running:
            try:
                cmd = [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,utilization.memory,power.draw,memory.used,clocks.sm,temperature.gpu",
                    "--format=csv,noheader,nounits"
                ]
                res = subprocess.run(cmd, capture_output=True, text=True, check=True)
                parts = [p.strip() for p in res.stdout.strip().split(",")]
                if len(parts) >= 6:
                    gpu_util = float(parts[0])
                    mem_util = float(parts[1])
                    pwr_draw = float(parts[2])
                    mem_used = float(parts[3])
                    sm_clock = float(parts[4])
                    gpu_temp = float(parts[5])
                    cpu_pct = psutil.cpu_percent()
                    
                    self.records.append({
                        "timestamp": time.time(),
                        "gpu_util": gpu_util,
                        "mem_util": mem_util,
                        "pwr_draw": pwr_draw,
                        "mem_used_mb": mem_used,
                        "sm_clock": sm_clock,
                        "gpu_temp": gpu_temp,
                        "cpu_pct": cpu_pct
                    })
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        self.running = True
        self.records = []
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        if self.records:
            keys = self.records[0].keys()
            with open(self.log_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.records)
                
            gpu_utils = [r["gpu_util"] for r in self.records]
            pwr_draws = [r["pwr_draw"] for r in self.records]
            mem_useds = [r["mem_used_mb"] for r in self.records]
            sm_clocks = [r["sm_clock"] for r in self.records]
            gpu_temps = [r["gpu_temp"] for r in self.records]
            cpu_pcts = [r["cpu_pct"] for r in self.records]
            
            return {
                "gpu_util_median": float(np.median(gpu_utils)),
                "gpu_util_p95": float(np.percentile(gpu_utils, 95)),
                "pwr_avg": float(np.mean(pwr_draws)),
                "pwr_peak": float(np.max(pwr_draws)),
                "vram_peak_mb": float(np.max(mem_useds)),
                "sm_clock_avg": float(np.mean(sm_clocks)),
                "gpu_temp_max": float(np.max(gpu_temps)),
                "cpu_util_avg": float(np.mean(cpu_pcts))
            }
        return {
            "gpu_util_median": 0.0, "gpu_util_p95": 0.0, "pwr_avg": 0.0,
            "pwr_peak": 0.0, "vram_peak_mb": 0.0, "sm_clock_avg": 0.0,
            "gpu_temp_max": 0.0, "cpu_util_avg": 0.0
        }

def run_trial(trial_name, batch, workers, cache):
    log_csv = os.path.join(LOGS_DIR, f"gpu_bench_{trial_name}.csv")
    monitor = ResourceMonitor(log_csv, interval=0.5)
    
    print(f"\n---> Starting Trial: {trial_name} | Batch={batch}, Workers={workers}, Cache={cache}")
    model = YOLO("yolov8n.pt")
    
    monitor.start()
    t0 = time.time()
    try:
        results = model.train(
            data=SUBSET_DATA,
            imgsz=640,
            epochs=1,
            batch=batch,
            workers=workers,
            cache=cache,
            device=0,
            amp=True,
            deterministic=False,
            project="runs/benchmarks",
            name=trial_name,
            exist_ok=True,
            verbose=False
        )
    finally:
        t1 = time.time()
        stats = monitor.stop()
        
    duration = t1 - t0
    # 8000 images in subset
    imgs_per_sec = 8000.0 / max(duration, 0.001)
    
    res_dict = {
        "Trial": trial_name,
        "Batch": batch,
        "Workers": workers,
        "Cache": str(cache),
        "Duration_s": round(duration, 2),
        "Images_per_sec": round(imgs_per_sec, 1),
        "GPU_Util_Median_%": stats["gpu_util_median"],
        "GPU_Util_p95_%": stats["gpu_util_p95"],
        "VRAM_Peak_MB": stats["vram_peak_mb"],
        "Power_Avg_W": round(stats["pwr_avg"], 1),
        "Power_Peak_W": round(stats["pwr_peak"], 1),
        "CPU_Avg_%": round(stats["cpu_util_avg"], 1)
    }
    return res_dict

def main():
    trials = [
        {"name": "batch64_w16_nocache", "batch": 64, "workers": 16, "cache": False},
        {"name": "batch128_w16_nocache", "batch": 128, "workers": 16, "cache": False},
        {"name": "batch192_w16_nocache", "batch": 192, "workers": 16, "cache": False},
        {"name": "batch256_w16_nocache", "batch": 256, "workers": 16, "cache": False},
        {"name": "batch192_w8_nocache", "batch": 192, "workers": 8, "cache": False},
        {"name": "batch192_w24_nocache", "batch": 192, "workers": 24, "cache": False},
        {"name": "batch192_w16_diskcache", "batch": 192, "workers": 16, "cache": "disk"},
    ]
    
    results = []
    for t in trials:
        try:
            res = run_trial(t["name"], t["batch"], t["workers"], t["cache"])
            results.append(res)
            print(f"Result: {res['Images_per_sec']} img/s | GPU Util Median: {res['GPU_Util_Median_%']}% | Peak Power: {res['Power_Peak_W']}W")
        except Exception as e:
            print(f"Trial {t['name']} failed: {e}")
            
    summary_path = os.path.join(LOGS_DIR, "gpu_tuning_summary.csv")
    if results:
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
            
    print("\n" + "="*80)
    print("GPU TUNING BENCHMARK RESULTS")
    print("="*80)
    header = f"{'Trial':24s} | {'Batch':5s} | {'Workers':7s} | {'Cache':9s} | {'Img/s':7s} | {'GPU%':5s} | {'Pwr_Peak':8s} | {'VRAM(MB)':8s}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['Trial']:24s} | {r['Batch']:5d} | {r['Workers']:7d} | {r['Cache']:9s} | {r['Images_per_sec']:7.1f} | {r['GPU_Util_Median_%']:5.1f} | {r['Power_Peak_W']:8.1f} | {r['VRAM_Peak_MB']:8.0f}")
    print("="*80)
    print(f"Results saved to {summary_path}")

if __name__ == "__main__":
    main()
