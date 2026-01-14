from ultralytics import YOLO

# Carica il modello YOLOv8
model = YOLO("yolo26n.pt")  # puoi usare yolov8s.pt o il tuo modello custom

# Esporta in ONNX FP16
model.export(
    format="onnx",   # formato ONNX
    opset=17,        # versione ONNX
    dynamic=False,   # dimensione fissa (es. 640x640)
    half=True        # usa FP16
)

import onnxruntime as ort

sess = ort.InferenceSession("yolo26n.onnx", providers=["DmlExecutionProvider"])
print(sess.get_inputs()[0].type)
