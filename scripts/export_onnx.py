"""
scripts/export_onnx.py
Exports a YOLOv8 PyTorch model (.pt) to Hailo-compatible ONNX (opset 11, static shape 640x640).
Inspects the ONNX graph and identifies the 6 output convolution nodes of the detection head.
"""
import os
import sys
import onnx
import onnxruntime as ort
import numpy as np
from ultralytics import YOLO

def export_hailo_compatible_onnx(pt_path, onnx_path, opset=11):
    print(f"Exporting {pt_path} to Hailo-compatible ONNX ({onnx_path}, opset={opset})...")
    model = YOLO(pt_path)
    
    # Export via Ultralytics
    exported_file = model.export(
        format="onnx",
        imgsz=640,
        opset=opset,
        simplify=True,
        device="cpu",
        dynamic=False,
        batch=1
    )
    
    if os.path.exists(exported_file) and exported_file != onnx_path:
        # Atomic rename/move to destination
        temp_dest = onnx_path + ".tmp"
        if os.path.exists(temp_dest):
            os.remove(temp_dest)
        os.replace(exported_file, temp_dest)
        os.replace(temp_dest, onnx_path)
        
    print(f"Exported to {onnx_path}")
    
    # Verify and inspect graph nodes
    model_proto = onnx.load(onnx_path)
    onnx.checker.check_model(model_proto)
    print("ONNX checker passed successfully!")
    
    # Find head output Conv nodes
    conv_nodes = [node.name for node in model_proto.graph.node if node.op_type == "Conv"]
    print(f"Total Conv nodes in graph: {len(conv_nodes)}")
    
    # Detect head convolution nodes (cv2 box and cv3 class branches)
    head_convs = [n for n in conv_nodes if "cv2" in n or "cv3" in n or "model.22" in n]
    print(f"Candidate Detection Head nodes: {head_convs[-6:]}")
    
    # Run 1 sample inference in ONNX Runtime (CPU)
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    dummy_input = np.zeros((1, 3, 640, 640), dtype=np.float32)
    outputs = session.run(None, {input_name: dummy_input})
    
    print(f"Input: {input_name}, shape: {input_shape}")
    for i, out in enumerate(outputs):
        print(f"Output {i} shape: {out.shape}")
        
    return head_convs[-6:]

if __name__ == "__main__":
    pt_file = sys.argv[1] if len(sys.argv) > 1 else "yolov8n.pt"
    onnx_file = sys.argv[2] if len(sys.argv) > 2 else "runs/yolov8n_seraphim/weights/last.onnx"
    os.makedirs(os.path.dirname(onnx_file), exist_ok=True)
    export_hailo_compatible_onnx(pt_file, onnx_file)
