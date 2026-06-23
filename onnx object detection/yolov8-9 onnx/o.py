import onnxruntime as ort
import cv2
import numpy as np
import kmNet
import win32api
from win32api import GetSystemMetrics
import time
import threading
import queue
import os
import torch
import torchvision.ops as ops
import math
import keyboard
import winsound
import gpu_capture



#gpu_capture.initialize_capture(512, 512)
#inizializza 512x512 rgba, zomm
gpu_capture.initialize_capture(512, 512,False,False)

print("Engine GPU-Image avviato. Loop attivo...")


screenshot=512
screeneginegpu = False
countfps = True
visual =  False                               
modelname = 'r62026.onnx'
center = screenshot / 2
trigger = False
data =False
weapon_mode_close = False



# Inizializza dispositivo
ret = kmNet.init("192.168.2.188", "1408", "9FC05414")
if ret == -8997:
    print('Dati inseriti sbagliati: controllare IP, port e MAC')
else:
    print("Dispositivo inizializzato correttamente")
#kmNet.monitor(5001) 


last_time = time.perf_counter()
fps_counter_time = last_time
frame_count = 0
start_time = time.time()


class Predict:
    def __init__(self, onnx_model, confidence_thres=0.4, iou_thres=0.5):
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            onnx_model,
            sess_options=options,
            providers=['DmlExecutionProvider', 'CPUExecutionProvider']
        )

        self.get_input_details()
        self.get_output_details()
        self.classes = self.get_class_names()
        self.classes_array = np.array(self.classes)
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes),3)).astype(np.uint8)

        self.binding = self.session.io_binding()
        for name in self.output_names:
            self.binding.bind_output(name, device_type='cpu')

    # ------------------- Detect -------------------
    def detect_objects1(self, image):
        input_tensor = self.preprocess(image)
        outputs = self.inference(input_tensor)
        return self.postprocess(image, outputs)

    def detect_objects(self, image):
        #outputs = self.inference(image)
        #return self.postprocess(image, outputs)
        input_tensor = image  # già pronto da C++ (NO preprocess)
        if input_tensor is None:
            return None
        self.binding.clear_binding_inputs()

        self.binding.bind_input(
            name=self.input_names[0],
            device_type='cpu',
            device_id=0,
            element_type=np.float32,
            shape=[1, 3, self.input_height, self.input_width],
            buffer_ptr=input_tensor.ctypes.data )

        self.session.run_with_iobinding(self.binding)

        ort_outputs = self.binding.get_outputs()
        outputs = ort_outputs[0].numpy()
        return self.postprocess(image, outputs)

    # ------------------- Preprocess -------------------
    def preprocess(self, image):
        blob = np.ascontiguousarray(image.transpose(2, 0, 1)[np.newaxis, ...], dtype=np.float32) / 255.0
        return blob

    def inference(self, input_tensor):
        return self.session.run(self.output_names, {self.input_names[0]: input_tensor})

    # ------------------- Postprocess -------------------
    def postprocess(self, input_image, outputs):
        outputs = np.squeeze(outputs[0]).T
        if outputs.size == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        scores = outputs[:, 4:].max(axis=1)
        mask = scores > self.confidence_thres
        outputs = outputs[mask]
        scores = scores[mask]

        if not len(scores):
            return np.array([]), np.array([]), np.array([]), np.array([])

        class_ids = outputs[:, 4:].argmax(axis=1)
        boxes = self.extract_boxes(outputs)
        indices = self.fast_nms(boxes, scores, self.iou_thres)

        if not len(indices):
            return np.array([]), np.array([]), np.array([]), np.array([])

        boxes = boxes[indices]
        scores = scores[indices]
        class_ids = class_ids[indices]
        class_names = self.classes_array[class_ids]

        return boxes, scores, class_ids, class_names

    # ------------------- NMS -------------------
    def fast_nms(self, boxes, scores, iou_threshold):
        if len(boxes) == 0:
            return np.empty(0, dtype=np.int32)

        sorted_idx = np.argsort(scores)[::-1]
        boxes = boxes[sorted_idx]

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        inter_x1 = np.maximum(x1[:, None], x1[None, :])
        inter_y1 = np.maximum(y1[:, None], y1[None, :])
        inter_x2 = np.minimum(x2[:, None], x2[None, :])
        inter_y2 = np.minimum(y2[:, None], y2[None, :])

        inter = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
        union = areas[:, None] + areas[None, :] - inter
        iou = inter / np.maximum(union, 1e-9)

        triu_iou = np.triu(iou, k=1)
        pick = np.where((triu_iou >= iou_threshold).sum(axis=0) == 0)[0]
        return sorted_idx[pick]

    def non_max_suppression(self, boxes, scores, iou_threshold):
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            union = areas[i] + areas[order[1:]] - inter
            iou = inter / union
            order = order[1:][iou <= iou_threshold]

        return np.array(keep, dtype=np.int32)

    # ------------------- Utils -------------------
    def xywh2xyxy(self, x):
        y = x.copy()
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y

    def extract_boxes(self, predictions):
        boxes = predictions[:, :4]
        boxes = self.xywh2xyxy(boxes)
        return boxes

    # ------------------- ONNX model info -------------------
    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    def get_class_names(self):
        metadata = self.session.get_modelmeta().custom_metadata_map.get('names', '')
        class_names = [item.split(": ")[1].strip(" {}'") for item in metadata.split("', ")] if metadata else []
        return class_names

    # ------------------- Draw detections -------------------
    def draw_detections(self, img, box, score, class_id):
        x1, y1, x2, y2 = map(int, box)
        color = self.color_palette[class_id].tolist()
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{self.classes[class_id]}: {score:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_x, label_y = x1, y1 - 10 if y1 - 10 > label_height else y1 + 10
        cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED)
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

# ------------------- Classe wrapper ONNX -------------------
class ONNX:
    def __init__(self, onnx_model, confidence_thres=0.3, iou_thres=0.5):
        self.predict_model = Predict(onnx_model, confidence_thres, iou_thres)

    def __call__(self, image):
        return self.predict_model.detect_objects(image)






def save_image_worker(image_queue):
    last_save_time = 0
    interval = 0.5  # 32 millisecondi
    
    while True:
        image = image_queue.get()
        
        if image is None:
            image_queue.task_done()
            break
            
        current_time = time.time()
        
        # Salviamo solo se sono passati almeno 32ms dall'ultimo salvataggio
        if (current_time - last_save_time) >= interval:
            # Trasformazione veloce
            img = np.transpose(image[0], (1, 2, 0))
            img = (img * 255).astype(np.uint8)
            
            filename = f"data/detected_{int(current_time * 1000)}.jpg"
            # cv2.imwrite è bloccante, ma in un thread separato va bene
            cv2.imwrite(filename, img)
            
            last_save_time = current_time
        
        image_queue.task_done()

if data:
    # 1. Assicurati che la cartella esista
    if not os.path.exists('data'):
        os.makedirs('data')
    # 2. Inizializza la coda PRIMA di avviare il thread
    image_queue = queue.Queue()
    # 3. Avvia il thread passandogli la coda come argomento
    # Nota: args deve essere una tupla, quindi usiamo (image_queue,)
    save_thread = threading.Thread(target=save_image_worker, args=(image_queue,), daemon=True)
    save_thread.start()

# Initialize the ONNX model
model = ONNX(modelname, 0.37, 0.50)

def human_smooth2(dx, dy):
    ax = dx if dx >= 0 else -dx
    ay = dy if dy >= 0 else -dy

    dist = ax if ax > ay else ay

    t = dist * 0.01
    if t > 1.0:
        t = 1.0

    t = t * t
    inv_speed = 1.0 / (1.1 + t * 1.5)

    rx = dx * inv_speed
    ry = dy * inv_speed

    return (int(rx + (0.5 if rx >= 0 else -0.5)),int(ry + (0.5 if ry >= 0 else -0.5)))




def toggle_weapon(e=None):
    global weapon_mode_close
    weapon_mode_close = not weapon_mode_close

    if weapon_mode_close:
        # Suono per arma attiva
        winsound.Beep(1000, 100)  # 1000 Hz, 100 ms
    else:
        # Suono per arma disattiva
        winsound.Beep(400, 150)   # 600 Hz, 100 ms

    print(f"[INFO] Weapon mode close: {weapon_mode_close}")


keyboard.on_press_key('p', toggle_weapon)

while True:
    img = gpu_capture.get_frame_image(False)
    if img is None:
        continue

    result = model(img) 
    
    if result is None:
        continue

    boxes, scores, class_ids, class_names = result

    # ================= TARGET LOGIC =================
    if boxes.size > 0:
        centers_x = (boxes[:, 0] + boxes[:, 2]) * 0.5 - center
        centers_y = boxes[:, 1] + (boxes[:, 3] - boxes[:, 1]) * 0.35 - center
        dists = centers_x**2 + centers_y**2
        min_index = np.argmin(dists)

        target_box = boxes[min_index]

        x = centers_x[min_index]
        y = centers_y[min_index]
        dx, dy = human_smooth2(x, y)
        delta_x1  = x if x >= 0 else -x
        delta_y1 = y if y >= 0 else -y

        if weapon_mode_close:
            half_w = (target_box[2] - target_box[0]) * 0.95
            half_h = (target_box[3] - target_box[1]) * 0.95
        else:
            half_w = (target_box[2] - target_box[0]) * 0.95
            half_h = (target_box[3] - target_box[1]) * 0.95

        
        #if triggerbot:
        #if delta_x1 <= half_w and delta_y1 <= half_h:
        #    button_state = 1
        #else:
        #    button_state = 0
        in_target = (delta_x1 <= half_w and delta_y1 <= half_h)
        button_state = 1 if in_target else 0
        #button_state = 1  click_now else 0
        if win32api.GetKeyState(0x05) < 0:
            if weapon_mode_close:
                if in_target:
                    kmNet.mouse(1, dx, dy, 0)
                    time.sleep(0.01)
                    kmNet.mouse(0, 0, 0, 0)
                    trigger = True
                else:
                    kmNet.mouse(0, dx, dy, 0)
            else:
                kmNet.mouse(button_state, dx, dy, 0)
                trigger = True
        else:
            if trigger:
                kmNet.mouse(0, 0, 0, 0)
                trigger = False

    else:
        if trigger:
            kmNet.mouse(0, 0, 0, 0)
            trigger=False
        if data: #and frame_id % SAVE_EVERY == 0:
            image_queue.put((img))

    # ================= VISUAL (SEMPRE) =================
    if visual:
        img = np.transpose(img[0], (1, 2, 0))
        img_uint8 = (img * 255).astype(np.uint8).copy()
        #img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
        for box, score, class_id, cls in zip(boxes, scores, class_ids, class_names):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_uint8, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{cls}: {score:.2f}"  # usa cls per il nome leggibile
            cv2.putText(
                img_uint8,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
                )

        cv2.imshow("Detected Objects", img_uint8)
        cv2.waitKey(1)


    #Update FPS if needed
    if countfps:
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
            print(fps)
