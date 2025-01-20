import onnxruntime as ort
import cv2
import numpy as np
from mss import mss
import ctypes
import os
import torch
import torchvision.ops as ops
import time
from PyQt5 import QtWidgets, QtGui, QtCore
import sys
import queue
from win32api import GetSystemMetrics
import win32api

# Configurazioni iniziali
screenshot = 512
fov = 380  # FOV x2
fov_x = (screenshot - fov) // 2  # Calcola le coordinate del rettangolo FOV
fov_y = (screenshot - fov) // 2
countfps = True
visual = True
data = False
modelname = 'best.onnx'
aimpoint = 2.7
center = screenshot / 2
centerx, centery = center, center
TARGET_FPS = 90
FRAME_TIME = 1 / TARGET_FPS



# Inizializzazione kmNet
#kmNet.init('192.168.2.188', '1408', '9FC05414')

# Esempio di risoluzione e FOV del gioco
screen_width = GetSystemMetrics(0)
screen_height = GetSystemMetrics(1)
#for overlay
x = (screen_width - screenshot) // 2
y = (screen_height - screenshot) // 2
fov_horizontal=90
sensitivity_game=0.6 #from 0.1 to 1
  

# Screenshot
screen_capture = mss()
screen_region = screen_capture.monitors[1]
screen_region['left'] = int((screen_region['width'] / 2) - (screenshot / 2))
screen_region['top'] = int((screen_region['height'] / 2) - (screenshot / 2))
screen_region['width'] = screenshot
screen_region['height'] = screenshot



# Assicurarsi che la directory 'data' esista
if not os.path.exists('data'):
    os.makedirs('data')

# Variables for FPS calculation
fps = 0
frame_count = 0
start_time = time.time()
last_saved_time = time.time()

class Predict:
    def __init__(self, onnx_model, confidence_thres, iou_thres):
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        # Crea opzioni per la sessione ONNX
        options = ort.SessionOptions()
        # Aumenta il livello di ottimizzazione del modello
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        self.session = ort.InferenceSession(onnx_model, sess_options=options,providers=['DmlExecutionProvider', 'CPUExecutionProvider'])

        self.get_input_details()
        self.get_output_details()
        self.classes = self.get_class_names()
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3)).astype(np.uint8)

    def detect_objects(self, image):
        input_tensor = self.preprocess(image)
        outputs = self.inference(input_tensor)
        postprocess = self.postprocess(image, outputs)
        return postprocess

    
    def preprocess(self, image):
        blob = np.ascontiguousarray(image.transpose(2, 0, 1)[np.newaxis, ...], dtype=np.float32) / 255.0

        return blob


    def inference(self, input_tensor):
        #outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        #return outputs
        return self.session.run(self.output_names, {self.input_names[0]: input_tensor})


    def postprocess(self, input_image, outputs):
        outputs = np.squeeze(outputs[0]).T
        scores = np.max(outputs[:, 4:], axis=1)
        outputs = outputs[scores > self.confidence_thres, :]
        scores = scores[scores > self.confidence_thres]
        class_ids = np.argmax(outputs[:, 4:], axis=1)
        boxes = self.extract_boxes(outputs)
        indices = self.non_max_suppression(boxes, scores, self.iou_thres)
        #indices = self.nms_cod(boxes, scores, input_image, self.iou_thres)
        arry_box = []
        for i in indices:
            box, score, class_id = boxes[i, :4], scores[i], class_ids[i]
            arry_box.append((box, score, class_id, self.classes[class_id]))
        return arry_box, input_image

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

    def extract_boxes_rescale_boxes(self, predictions):
        boxes = predictions[:, :4]
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_height, self.img_width, self.img_height, self.img_width])
        boxes = self.xywh2xyxy(boxes)
        return boxes

    def non_max_suppression1(self,boxes, scores, iou_threshold):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while len(order) > 0:
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
            order = order[np.where(iou <= iou_threshold)[0] + 1]

        return keep



    def non_max_suppression(self, boxes, scores, iou_threshold):
        boxes = torch.tensor(boxes, dtype=torch.float32)#.cuda()  # Ensure boxes are on GPU
        scores = torch.tensor(scores, dtype=torch.float32)#.cuda()  # Ensure scores are on GPU
        keep = ops.nms(boxes, scores, iou_threshold)#.cpu().numpy()

        return keep



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
        metadata = self.session.get_modelmeta().custom_metadata_map['names']
        class_names = [item.split(": ")[1].strip(" {}'") for item in metadata.split("', ")]
        return class_names

    def save_image(self, image, timestamp):
        filename = f"data/detected_{timestamp}.jpg"
        cv2.imwrite(filename, image)


class ONNX:
    def __init__(self, onnx_model, confidence_thres, iou_thres):
        self.predict_model = Predict(onnx_model, confidence_thres, iou_thres)

    def __call__(self, image):
        self.results, self.images = self.predict_model.detect_objects(image)
        return self.results


class DetectionThread(QtCore.QThread):
    update_bboxes_signal = QtCore.pyqtSignal(list)

    def __init__(self, model,  screen_region, width, height, center, frame_time, parent=None):
        super().__init__(parent)
        self.model = model
        self.screen_region = screen_region
        self.width = width
        self.height = height
        self.center = center
        self.frame_time = frame_time
        self.frame_count = 0  # Inizializza frame_count come variabile di istanza
        self.start_time = time.time()
        self.running = True
        

    def run(self):
        screen_capture = mss()
        while self.running:
            
            frame_start = time.time()
            img = np.array(screen_capture.grab(self.screen_region))[:, :, :3]
            results = self.model(img)


            #targets =[(((box[0] + box[2])/2 -center), (box[1]  - center +6 ))for box, _, _, _ in results]
            targets = [(((box[0] + box[2]) / 2 - center), (box[1] - center + 6))for box, _, _, _ in results
                       if -fov_x <= (box[0] + box[2]) / 2 <= fov_x  # Controllo X
                       and -fov_y <= (box[1] + box[3]) / 2 <= fov_y ]  # Controllo Y
            new_bboxes = [(int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2), int(box[2] - box[0]), int(box[3] - box[1]))for box, _, _, _ in results]


            targets_array = np.array(targets)

            if len(targets_array) > 0:
                dist_from_center = np.linalg.norm(targets_array[:, :1], axis=1)
                min_dist_idx = np.argmin(dist_from_center)
                current_target = targets_array[min_dist_idx]

                # Ottieni le coordinate del bersaglio rispetto al centro dello schermo
                delta_x = current_target[0]
                delta_y = current_target[1]

                # Applica i valori calcolati per lo spostamento
                step_x = int(delta_x/2)
                step_y = int(delta_y/2)


                #if win32api.GetKeyState(0x05)<0 :
                #        kmNet.move(step_x, step_y)
                # Determina lo stato desiderato in base alle condizioni
                #if -1.3 <= step_x <= 1.3 and -1 <= step_y <= 1:
                #    kmNet.left(1)
                #    kmNet.left(0)
            
            self.update_bboxes_signal.emit(new_bboxes)
            frame_time = time.time() - frame_start
            sleep_time = self.frame_time - frame_time
            if sleep_time > 0:
                time.sleep(sleep_time)

            if countfps:
                self.frame_count += 1
                elapsed_time = time.time() - self.start_time
                if elapsed_time > 1:
                    fps = self.frame_count / elapsed_time
                    self.frame_count = 0
                    self.start_time = time.time()
                    print(f"FPS: {fps:.2f}")

    def stop(self):
        self.running = False
        self.wait()


class OverlayWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowTransparentForInput)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setGeometry(x, y, screenshot, screenshot)
        self.bboxes = []
        self.show()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        pen = QtGui.QPen(QtGui.QColor(255, 0, 0), 2)
        painter.setPen(pen)
        font = QtGui.QFont("Arial", 8)  # Imposta il font per il testo
        painter.setFont(font)

        # Rettangolo immagine (screenshot)
        painter.drawRect(0, 0, screenshot, screenshot)
        painter.drawText(5, 10, "Image Size")  # Testo sopra il rettangolo immagine

        # Rettangolo FOV aim
        painter.drawRect(fov_x, fov_y, fov, fov)
        painter.drawText(fov_x + 5, fov_y - 5, "FOV Aim")  # Testo sopra il rettangolo FOV
        for bbox in self.bboxes:
            painter.drawRect(*bbox)

    def update_bboxes(self, bboxes):
        self.bboxes = [(int(cx - w / 2), int(cy - h /2 ), int(w), int(h)) for cx, cy, w, h in bboxes]
        self.update()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    # Inizializzazione del modello
    model = ONNX(modelname, 0.5, 0.6)

    if visual:
        # Creazione dell'overlay
        overlay_window = OverlayWindow()
        # Avvio del thread di rilevamento
        detection_thread = DetectionThread(model, screen_region,screenshot, screenshot, center, FRAME_TIME)
        detection_thread.update_bboxes_signal.connect(overlay_window.update_bboxes)
        detection_thread.start()
    else:
        detection_thread = DetectionThread(model,  screen_region, screenshot, screenshot, center, FRAME_TIME)
        detection_thread.start()

    # Avvio dell'applicazione Qt
    sys.exit(app.exec_())

    # Arresto del thread di rilevamento
    detection_thread.stop()
