# ONNX Output Decoding & Post-Processing Guide

## Model Overview
- **Model:** YOLOv8n Seraphim Drone Detector (Single Class: `0: drone`)
- **Input:** `images`, Shape: `[1, 3, 640, 640]`, Data Type: `float32`, Value Range: `[0.0, 1.0]` (RGB format).
- **Output:** `output0`, Shape: `[1, 5, 8400]` (4 box coordinates + 1 class score across 8400 spatial candidate anchors).

---

## Output Structure & Decoding
The output tensor has shape `(1, 5, 8400)`:
- `output0[:, 0, :]`: Center X ($c_x$) in normalized or pixel coordinates
- `output0[:, 1, :]`: Center Y ($c_y$) in normalized or pixel coordinates
- `output0[:, 2, :]`: Width ($w$)
- `output0[:, 3, :]`: Height ($h$)
- `output0[:, 4, :]`: Drone Confidence Score ($s_{drone}$)

---

## Python ONNX Runtime Example with NMS

```python
import cv2
import numpy as np
import onnxruntime as ort

def run_onnx_inference(onnx_model_path, image_path, conf_thresh=0.25, iou_thresh=0.45):
    session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    # 1. Preprocess: Read and letterbox to 640x640
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (640, 640))
    blob = np.transpose(resized, (2, 0, 1)).astype(np.float32) / 255.0
    blob = np.expand_dims(blob, axis=0)
    
    # 2. Forward pass
    outputs = session.run(None, {input_name: blob})[0] # (1, 5, 8400)
    preds = outputs[0].T # (8400, 5)
    
    # 3. Filter by confidence
    boxes, scores = [], []
    for row in preds:
        cx, cy, w, h, score = row
        if score >= conf_thresh:
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            boxes.append([x1, y1, int(w), int(h)])
            scores.append(float(score))
            
    # 4. Apply Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, iou_thresh)
    
    results = []
    if len(indices) > 0:
        for idx in indices.flatten():
            results.append({
                "box": boxes[idx],
                "confidence": scores[idx],
                "class_id": 0,
                "label": "drone"
            })
    return results
```
