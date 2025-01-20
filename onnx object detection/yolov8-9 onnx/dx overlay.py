import onnxruntime as ort
import cv2
import numpy as np
import kmNet
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
import threading

# Configurazioni iniziali
screenshot = 512
fov = 380  # FOV x2
fovaim = fov/2  # FOV aim
fov_x = (screenshot - fov) // 2  # Calcola le coordinate del rettangolo FOV
fov_y = (screenshot - fov) // 2  # Calcola le coordinate del rettangolo FOV
countfps = True
visual = True
data = False
modelname = 'best.onnx'
aimpoint = 12
center = screenshot / 2
centerx, centery = center, center
TARGET_FPS = 90
FRAME_TIME = 1 / TARGET_FPS
# Crea una coda per le immagini da salvare
image_queue = queue.Queue(maxsize=10) 


# Inizializzazione kmNet
kmNet.init('192.168.2.188', '1408', '9FC05414')

# Load DLL dxgi + setup
screen_width = GetSystemMetrics(0)
screen_height = GetSystemMetrics(1)
x = (screen_width - screenshot) // 2
y = (screen_height - screenshot) // 2
buffer_size = screenshot * screenshot * 4
buffer = (ctypes.c_ubyte * buffer_size)()
dxgx_dll = ctypes.CDLL(r'C:\Users\Admin\Desktop\ghg\old backup\Pyt\dxgi_shot-main\x64\Release\dxgx.dll')
dxgx_dll.create.restype = ctypes.c_void_p
dxgx_dll.init.argtypes = [ctypes.c_void_p]
dxgx_dll.init.restype = ctypes.c_bool
dxgx_dll.shot.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_ubyte)]
dxgx_dll.shot.restype = ctypes.POINTER(ctypes.c_ubyte)
dxgx_dll.destroy.argtypes = [ctypes.c_void_p]
duplicator = dxgx_dll.create()
if not dxgx_dll.init(duplicator):
    raise RuntimeError("Failed to initialize DXGIDuplicator")

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



class ONNX:
    def __init__(self, onnx_model, confidence_thres, iou_thres):
        self.predict_model = Predict(onnx_model, confidence_thres, iou_thres)

    def __call__(self, image):
        self.results, self.images = self.predict_model.detect_objects(image)
        return self.results

def save_image_worker():
    while True:
        
        # Prendi un'immagine dalla coda
        image = image_queue.get()
        
        # Se l'elemento è None, significa che dobbiamo uscire dal thread
        if image is None:
            break

        # Ottieni il timestamp in millisecondi
        timestamp = int(time.time() * 1)

        # Salva l'immagine
        filename = f"data/detected_{timestamp}.jpg"
        cv2.imwrite(filename, image)

        # Segnala alla coda che il compito è stato completato
        image_queue.task_done()

class DetectionThread(QtCore.QThread):
    update_bboxes_signal = QtCore.pyqtSignal(list)

    def __init__(self, model, duplicator, buffer, width, height, center, frame_time,image_queue, parent=None):
        super().__init__(parent)
        self.model = model
        self.image_queue = image_queue
        self.duplicator = duplicator
        self.buffer = buffer
        self.width = width
        self.height = height
        self.center = center
        self.frame_time = frame_time
        self.frame_count = 0  # Inizializza frame_count come variabile di istanza
        self.start_time = time.time()
        self.running = True

    def run(self):
        while self.running:
            frame_start = time.time()
            image_data_ptr = dxgx_dll.shot(self.duplicator, x, y, self.width, self.height, self.buffer)
            image_data = np.ctypeslib.as_array(self.buffer)
            img = image_data.reshape((self.height, self.width, 4))[:, :, :3]
            results = self.model(img)

            #targets = []
            #targets =[(((box[0] + box[2])/2 -center), (box[1]  - center +6 ))for box, _, _, _ in results]
            targets = [(((box[0] + box[2]) / 2 - center), (box[1] - center + aimpoint))for box, _, _, _ in results
                       if -fovaim <= (box[0] + box[2]) / 2- center <= fovaim  # Controllo X
                       and -fovaim <= (box[1] + box[3]) / 2- center + aimpoint <= fovaim ]  # Controllo Y

            new_bboxes = [(int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2), int(box[2] - box[0]), int(box[3] - box[1]))for box, _, _, _ in results]

            #for box, score, class_id, cls in results:
                #target_x = (box[0] + box[2])/2 -center
                #target_y = (box[1]  - center +9 ) #box[3]) 
                #target_height = box[3] - box[1]
                #target_y = target_y - (target_height / aimpoint)
                #if  -fovaim <= target_x <= fovaim and -fovaim <= target_y <= fovaim:
                #targets.append((target_x, target_y, ))#target_height

            targets_array = np.array(targets)

            if len(targets_array) > 0:
                dist_from_center = np.linalg.norm(targets_array[:, :1], axis=1)
                min_dist_idx = np.argmin(dist_from_center)
                current_target = targets_array[min_dist_idx]

                # Ottieni le coordinate del bersaglio rispetto al centro dello schermo
                delta_x = current_target[0]
                delta_y = current_target[1]

                # Applica i valori calcolati per lo spostamento
                step_x = int(delta_x/1.7)
                step_y = int(delta_y/1.7)


                if win32api.GetKeyState(0x05)<0 :
                    kmNet.move(step_x, step_y)
                    # Determina lo stato desiderato in base alle condizioni
                    if -1.4 <= step_x <= 1.4 and -1 <= step_y <= 1:
                        kmNet.left(1)
                        kmNet.left(0)
            
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

            # Aggiungi l'immagine alla coda per il salvataggio se `data` è True
            if data:
                image_queue.put((img))

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
    model = ONNX(modelname, 0.5, 0.7)

    if visual:
        
        # Creazione dell'overlay
        overlay_window = OverlayWindow()
        # Avvio del thread di rilevamento
        detection_thread = DetectionThread(model, duplicator, buffer, screenshot, screenshot, center, FRAME_TIME,image_queue)
        detection_thread.update_bboxes_signal.connect(overlay_window.update_bboxes)
        detection_thread.start()

    else:
        detection_thread = DetectionThread(model, duplicator, buffer, screenshot, screenshot, center, FRAME_TIME,image_queue)
        detection_thread.start()

    # Avvia il thread di salvataggio se `data` è True
    if data:
        save_thread = threading.Thread(target=save_image_worker)
        save_thread.daemon = True
        save_thread.start()

    # Avvio dell'applicazione Qt
    sys.exit(app.exec_())

    # Arresto del thread di rilevamento
    detection_thread.stop()
