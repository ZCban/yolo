import time
import cv2
import numpy as np
import onnxruntime
from ScreenCapture import ScreenCaptureMSS, ScreenCaptureBETTERCAM

# ------------------- Classe YOLOv10 -------------------
class YOLOv10:

    def __init__(self, path: str, conf_thres: float = 0.2):
        self.conf_threshold = conf_thres

        # Inizializza il modello ONNX con DirectML
        providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(path, providers=providers)

        # Recupera input/output details
        self.get_input_details()
        self.get_output_details()

        # Carica classi dal metadata (se presenti)
        self.classes = self.load_class_names()

        # Preallocazione tensore input per velocità
        self.input_tensor = np.zeros((1, 3, self.input_height, self.input_width), dtype=np.float32)

    def load_class_names(self):
        try:
            metadata = self.session.get_modelmeta().custom_metadata_map['names']
            class_names = [item.split(": ")[1].strip(" {}'") for item in metadata.split("', ")]
        except:
            class_names = [str(i) for i in range(80)]  # fallback COCO
        return class_names

    def __call__(self, image: np.ndarray):
        return self.detect_objects(image)

    def detect_objects(self, image: np.ndarray):
        input_tensor = self.prepare_input(image)
        outputs = self.inference(input_tensor)
        return self.process_output(outputs[0])

    def prepare_input(self, image: np.ndarray):
        # Assumi immagine già 640x640
        self.img_height, self.img_width = image.shape[:2]

        # BGRA -> BGR se necessario
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        # Normalizza e trasponi
        img = image.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)

        # Scrivi nel tensore preallocato
        self.input_tensor[0] = img
        return self.input_tensor

    def inference(self, input_tensor):
        return self.session.run(self.output_names, {self.input_names[0]: input_tensor})

    def process_output(self, output):
        output = output.squeeze()
        if output.ndim == 1:  # nessun oggetto
            return np.array([]), np.array([]), np.array([])

        boxes = output[:, :-2]
        confidences = output[:, -2]
        class_ids = output[:, -1].astype(int)

        # Filtro confidenza
        mask = confidences > self.conf_threshold
        boxes = boxes[mask, :]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        # Non serve rescale perché input già 640x640
        return class_ids, boxes, confidences

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [inp.name for inp in model_inputs]
        input_shape = model_inputs[0].shape
        self.input_height = input_shape[2] if isinstance(input_shape[2], int) else 640
        self.input_width = input_shape[3] if isinstance(input_shape[3], int) else 640

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [out.name for out in model_outputs]

    # ------------------- Disegna box con nome classe -------------------
    def draw_detections(self, img, boxes, class_ids, confidences):
        for box, class_id, conf in zip(boxes, class_ids, confidences):
            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{self.classes[class_id]}: {conf:.2f}"
            cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)



def get_onnx_input_size(onnx_path):
    session = onnxruntime.InferenceSession(
        onnx_path,
        providers=['CPUExecutionProvider']
    )

    inp = session.get_inputs()[0]
    input_shape = inp.shape
    input_type = inp.type

    # input_shape = [1, 3, H, W]
    h = input_shape[2]
    w = input_shape[3]

    # Dimensioni
    if isinstance(h, int) and isinstance(w, int):
        width, height = w, h
    else:
        width, height = 640, 640  # fallback sicuro YOLO

    # Precisione
    if input_type == "tensor(float16)":
        dtype = np.float16
        precision = "FP16"
    else:
        dtype = np.float32
        precision = "FP32"

    return width, height, dtype, precision


model_path = 'yolo26n.onnx'
w, h, input_dtype, precision = get_onnx_input_size(model_path)

print(f"ONNX input: {w}x{h} - {precision}")

# ------------------- Config -------------------
screenshot = w
screeneginegpu = True
countfps = True
visual = True

center = screenshot / 2

# ------------------- Screen Capture -------------------
if screeneginegpu:
    screen_capture = ScreenCaptureBETTERCAM(screenshot)
else:
    screen_capture = ScreenCaptureMSS(screenshot)

# ------------------- Modello -------------------
model = YOLOv10(model_path, conf_thres=0.49)

fps = 0
frame_count = 0
start_time = time.time()
current_target = None

# ------------------- Loop principale -------------------
while True:
    img = screen_capture.capture()
    #img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    if img is not None:
        class_ids, boxes, confidences = model(img)
        targets = []

        for box, cls, conf in zip(boxes, class_ids, confidences):
            x1, y1, x2, y2 = box
            target_x = (x1 + x2) / 2 - center
            target_y = (y1 + y2) / 2 - center
            targets.append((target_x, target_y))

            if visual:
                color = (0, 255, 0)
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                class_name = model.classes[cls] if cls < len(model.classes) else str(cls)
                cv2.putText(img, f"{class_name}:{conf:.2f}", (int(x1), int(y1)-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Target più vicino al centro
        if targets:
            targets_array = np.array(targets)
            dist_from_center = np.linalg.norm(targets_array, axis=1)
            min_dist_idx = np.argmin(dist_from_center)
            current_target = targets_array[min_dist_idx]

        if visual:
            cv2.imshow("Detected Objects", img)
            cv2.waitKey(1)

        # Calcolo FPS
        if countfps:
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 1:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
                print(f"FPS: {fps:.2f}")
