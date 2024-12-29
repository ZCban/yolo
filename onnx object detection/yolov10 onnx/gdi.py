import onnxruntime as ort
import cv2
import numpy as np
import bettercam
import win32api
from win32api import GetSystemMetrics
import win32con
import win32ui
import os
import win32gui
import time
import kmNet
import threading

# Example usage
screenshot = 512
center_square_size = 300
countfps =True
visual = False
modelname='best.onnx'
rust= 3.45
valorant=2.75
aimpoint=3.35
take_image = False
current_target = None
center = screenshot/2
half_square = center_square_size / 2

screen_width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
screen_height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
screen_region = (
    int((screen_width / 2) - (screenshot / 2)),
    int((screen_height / 2) - (screenshot / 2)),
    int((screen_width / 2) + (screenshot / 2)),
    int((screen_height / 2) + (screenshot / 2))
)

left, top, right, bottom = screen_region
width = right - left
height = bottom - top

# Configurazione GDI
hdesktop = win32gui.GetDesktopWindow()
hwindc = win32gui.GetWindowDC(hdesktop)
srcdc = win32ui.CreateDCFromHandle(hwindc)
memdc = srcdc.CreateCompatibleDC()
bmp = win32ui.CreateBitmap()
bmp.CreateCompatibleBitmap(srcdc, width, height)
memdc.SelectObject(bmp)

def capture_screen_gdi():
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)
    bmp_info = bmp.GetInfo()
    bmp_str = bmp.GetBitmapBits(True)
    
    img = np.frombuffer(bmp_str, dtype='uint8')
    img.shape = (bmp_info['bmHeight'], bmp_info['bmWidth'], 4)
    img = img[:, :, :3]
    return img

# Ensure 'collectdata' directory exists
if not os.path.exists('data'):
    os.makedirs('data')

##setup kmnet
kmNet.init('192.168.2.188', '1408', '9FC05414')

# Load the ONNX model and extract input dimensions
session = ort.InferenceSession(modelname, providers=['DmlExecutionProvider', 'CPUExecutionProvider'])
input_shape = session.get_inputs()[0].shape
input_height = input_shape[2]
input_width = input_shape[3]

# Calculate gain, pad_w, and pad_h dynamically
gain = min(input_width / screenshot, input_height / screenshot)
pad_w = (input_width - screenshot * gain) / 2
pad_h = (input_height - screenshot * gain) / 2

# Funzione per gestire il movimento e il clic del mouse in un thread separato
def handle_movement(step_x, step_y):
    if win32api.GetKeyState(0x05) < 0:
        if -3 <= step_x <= 3 and -1.4 <= step_y <= 1.4:
            kmNet.left(1)
            kmNet.left(0)
        else:
            kmNet.move(step_x, step_y)

class Predict:
    def __init__(self, session, confidence_thres):#, iou_thres):
        self.confidence_thres = confidence_thres
        #self.iou_thres = iou_thres
        self.session = session
        self.get_input_details()
        self.get_output_details()
        self.classes = self.get_class_names()
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3)).astype(np.uint8)

        # Variabili per il calcolo degli FPS
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()

    def detect_objects(self, image):
        input_tensor = self.preprocess(image)
        outputs = self.inference(input_tensor)
        postprocess = self.postprocess(image, outputs)
        return postprocess

    def detect_objects_resize(self, image):
        input_tensor = self.preprocess_resize(image)
        outputs = self.inference(input_tensor)
        postprocess = self.postprocess_resize(image, outputs)
        return postprocess

    def preprocess_resize(self, image):
        self.img_height, self.img_width = image.shape[:2]
        resized_image = cv2.resize(image, (self.input_width, self.input_height))
        blob = np.ascontiguousarray(resized_image.transpose(2, 0, 1)[np.newaxis, ...], dtype=np.float32) / 255.0
        return blob

    def preprocess(self, image):
        self.img_height, self.img_width = image.shape[:2]
        blob = np.ascontiguousarray(image.transpose(2, 0, 1)[np.newaxis, ...], dtype=np.float32) / 255.0
        return blob

    def inference(self, input_tensor):
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        return outputs

    def postprocess_resize(self, input_image, outputs):
        outputs = outputs[0][0]
        mask = outputs[:, 4] >= self.confidence_thres
        outputs = outputs[mask]

        boxes = outputs[:, :4]
        scores = outputs[:, 4]
        class_ids = outputs[:, 5].astype(int)

        # Riscalare le box alle dimensioni originali dell'immagine (350x350)
        boxes = self.extract_boxes_rescale_boxes(boxes)
        
        # Calcolare i target
        targets = []
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = map(int, box)
            x_center = (x1 + x2) // 2 - center
            y_center = (y1 + y2) // 2 - center
            width = x2 - x1
            height = y2 - y1
            y_center = y_center-(height/3)
            targets.append((x_center, y_center, width, height))

        targets_array = np.array(targets)

        return boxes, scores, class_ids, input_image, targets_array

    def postprocess(self, input_image, outputs):
        outputs = outputs[0][0]
        mask = outputs[:, 4] >= self.confidence_thres
        outputs = outputs[mask]

        boxes = outputs[:, :4]
        scores = outputs[:, 4]
        class_ids = outputs[:, 5].astype(int)

        # Convertire le box in formato intero
        boxes[:, [0, 2]] = boxes[:, [0, 2]]
        boxes[:, [1, 3]] = boxes[:, [1, 3]]
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, screenshot)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, screenshot)
        
        # Calcolare i target
        targets = []
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = map(int, box)
            x_center = (x1 + x2) // 2 - center
            y_center = (y1 + y2) // 2 - center
            width = x2 - x1
            height = y2 - y1
            y_center = y_center-(height/3)
            targets.append((x_center, y_center, width, height))

        targets_array = np.array(targets)

        return boxes, scores, class_ids, input_image, targets_array

    def extract_boxes_rescale_boxes(self, boxes):
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / gain
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / gain
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, screenshot)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, screenshot)
        return boxes

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    def draw_detections(self, img, box, score, class_id):
        x1, y1, x2, y2 = map(int, box)  # Assicurarsi che le coordinate siano intere
        color = self.color_palette[class_id].tolist()
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)  # Aumentare lo spessore a 2 per una migliore visibilità
        label = f"{self.classes[class_id]}: {score:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_x, label_y = x1, y1 - 10 if y1 - 10 > label_height else y1 + 10
        cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED)
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def get_class_names(self):
        metadata = self.session.get_modelmeta().custom_metadata_map['names']
        class_names = [item.split(": ")[1].strip(" {}'") for item in metadata.split("', ")]
        return class_names

    def update_fps(self):
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 1:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()
            print(self.fps)

class ONNX:
    def __init__(self, session, confidence_thres):
        self.predict_model = Predict(session, confidence_thres)

    def __call__(self, image):
        self.boxes, self.scores, self.class_ids, self.images, self.targets = self.predict_model.detect_objects(image)
        return self.boxes, self.scores, self.class_ids, self.targets


# Inizializzare il modello ONNX
model = ONNX(session,0.50)


while True:
    img = capture_screen_gdi()
    boxes, scores, class_ids, targets_array = model(img)

    if len(targets_array) > 0:
        # Calculate the distance from the center for each target
        dist_from_center = np.sqrt((targets_array[:, 0]) ** 2)
        min_dist_idx = np.argmin(dist_from_center)
        current_target = targets_array[min_dist_idx]
        

        # Calculate the deltas for x and y
        delta_x = current_target[0]
        delta_y = current_target[1]
        # Calculate the steps for movement
        step_y = round(delta_y/2)
        step_x = round(delta_x/2)

        # Creare un nuovo thread per eseguire il movimento senza bloccare il rilevamento
        movement_thread = threading.Thread(target=handle_movement, args=(step_x, step_y))
        movement_thread.start()


    if visual:
        for box, score, class_id in zip(boxes, scores, class_ids):
            #image_data=np.array(image_data)
            model.predict_model.draw_detections(img, box, score, class_id)
        cv2.imshow("Detected Objects", img)
        cv2.waitKey(1)

    if countfps:
        model.predict_model.update_fps()

srcdc.DeleteDC()
memdc.DeleteDC()
win32gui.ReleaseDC(hdesktop, hwindc)
win32gui.DeleteObject(bmp.GetHandle())
