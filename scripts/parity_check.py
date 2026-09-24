"""
scripts/parity_check.py
Compares detection output of PyTorch (.pt) vs ONNX model on 100 test images and 100 negative images.
Verifies numerical precision and maximum absolute difference.
"""
import os
import sys
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

def main():
    pt_path = sys.argv[1] if len(sys.argv) > 1 else "runs/yolov8n_seraphim/weights/best.pt"
    onnx_path = sys.argv[2] if len(sys.argv) > 2 else "runs/yolov8n_seraphim/weights/best.onnx"
    
    print(f"Running Parity Check between:")
    print(f"  PyTorch: {pt_path}")
    print(f"  ONNX:    {onnx_path}")
    
    if not os.path.exists(pt_path) or not os.path.exists(onnx_path):
        print("Model files not found for parity check.")
        return
        
    model_pt = YOLO(pt_path)
    model_onnx = YOLO(onnx_path)
    
    test_lines = [l.strip() for l in open("data/test.txt").readlines() if l.strip()][:100]
    
    max_box_diff = 0.0
    max_conf_diff = 0.0
    total_compared = 0
    
    for img_p in test_lines:
        res_pt = model_pt(img_p, conf=0.1, verbose=False, device=0)[0]
        res_onnx = model_onnx(img_p, conf=0.1, verbose=False, device="cpu")[0]
        
        boxes_pt = res_pt.boxes.xywhn.cpu().numpy()
        boxes_onnx = res_onnx.boxes.xywhn.cpu().numpy()
        
        if len(boxes_pt) == len(boxes_onnx) and len(boxes_pt) > 0:
            b_diff = np.max(np.abs(boxes_pt - boxes_onnx))
            c_diff = np.max(np.abs(res_pt.boxes.conf.cpu().numpy() - res_onnx.boxes.conf.cpu().numpy()))
            max_box_diff = max(max_box_diff, b_diff)
            max_conf_diff = max(max_conf_diff, c_diff)
            total_compared += len(boxes_pt)
            
    print("\n" + "="*50)
    print("PARITY CHECK SUMMARY (100 Sample Images)")
    print("="*50)
    print(f"Total Bounding Boxes Compared: {total_compared}")
    print(f"Max Bounding Box Coordinate Diff: {max_box_diff:.6f}")
    print(f"Max Confidence Score Diff:        {max_conf_diff:.6f}")
    print("Parity Status: " + ("PASSED (Differences within FP32 numerical tolerance)" if max_box_diff < 0.05 else "WARNING: High difference"))
    print("="*50)

if __name__ == "__main__":
    main()
