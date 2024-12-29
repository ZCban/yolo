import onnxruntime as ort
import cv2
import numpy as np
import kmNet
import ctypes
import win32api
import win32gui
import win32ui
import win32con
from win32api import GetSystemMetrics
import time
import threading
import queue
import os
import torch
import torchvision.ops as ops
import math

# Example usage
screenshot = 512
countfps = True
visual =  False                                   
data = True
modelname = 'best.onnx'
aimpoint=3
current_target = None
center = screenshot / 2
trigger=False
# Costanti
TARGET_FPS = 100
FRAME_TIME = 1 / TARGET_FPS  # Tempo target per ogni frame (in secondi)

# Esempio di risoluzione e FOV del gioco
screen_width = GetSystemMetrics(0)
screen_height = GetSystemMetrics(1)
fov_horizontal=90
sensitivity_game=0.5 #from 0.1 to 1

# Load the DLL
dxgx_dll = ctypes.CDLL(r'C:\Users\Admin\Desktop\ghg\old backup\Pyt\dxgi_shot-main\x64\Release\dxgx.dll')

# Define the argument and return types of the DLL functions
dxgx_dll.create.restype = ctypes.c_void_p

dxgx_dll.init.argtypes = [ctypes.c_void_p]
dxgx_dll.init.restype = ctypes.c_bool

dxgx_dll.shot.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_ubyte)]
dxgx_dll.shot.restype = ctypes.POINTER(ctypes.c_ubyte)

dxgx_dll.destroy.argtypes = [ctypes.c_void_p]

# Create an instance of DXGIDuplicator
duplicator = dxgx_dll.create()

# Initialize the duplicator
if not dxgx_dll.init(duplicator):
    raise RuntimeError("Failed to initialize DXGIDuplicator")

# Get the screen dimensions using ctypes
user32 = ctypes.windll.user32
screen_width = user32.GetSystemMetrics(0)
screen_height = user32.GetSystemMetrics(1)

# Calculate the center region (350x350) of the screen
width, height =screenshot,screenshot
x = (screen_width - width) // 2
y = (screen_height - height) // 2

buffer_size = width * height * 4  # Assuming 4 bytes per pixel (RGBA)
buffer = (ctypes.c_ubyte * buffer_size)()
center = np.array(screenshot / 2)

# Ensure 'data' directory exists
if not os.path.exists('data'):
    os.makedirs('data')

# Setup kmNet
kmNet.init('192.168.2.188', '1408', '9FC05414')

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



    def detect_objects_resize(self, image):
        self.img_height, self.img_width = 1080,1080
        resized_image = cv2.resize(image, (512, 512))
        input_tensor = self.preprocess(resized_image)
        outputs = self.inference(input_tensor)
        postprocess = self.postprocess_resize(resized_image, outputs)
        return postprocess

    def detect_objects(self, image):
        input_tensor = self.preprocess(image)
        outputs = self.inference(input_tensor)
        postprocess = self.postprocess(image, outputs)
        return postprocess


    def postprocess_resize(self, input_image, outputs):
        outputs = np.squeeze(outputs[0]).T
        scores = np.max(outputs[:, 4:], axis=1)
        outputs = outputs[scores > self.confidence_thres, :]
        scores = scores[scores > self.confidence_thres]
        class_ids = np.argmax(outputs[:, 4:], axis=1)
        boxes = self.extract_boxes_rescale_boxes(outputs)
        indices = self.non_max_suppression(boxes, scores, self.iou_thres)
        arry_box = []
        for i in indices:
            box, score, class_id = boxes[i, :4], scores[i], class_ids[i]
            arry_box.append((box, score, class_id, self.classes[class_id]))
        return arry_box, input_image
    

    
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

    # Funzione per filtro colore e NMS
    def nms_valo(self, boxes, scores, img, iou_threshold=0.5, violet_threshold=0.05):
        adjusted_scores = []
        valid_boxes = []
        valid_scores = []


        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = map(int, box)
            box_img = img[y1:y2, x1:x2]

            if box_img.size == 0:
                adjusted_scores.append(0)
                continue

            hsv_box_img = cv2.cvtColor(box_img, cv2.COLOR_BGR2HSV)
            lower_violet = np.array([140, 70, 70])
            upper_violet = np.array([160, 255, 255])
            mask = cv2.inRange(hsv_box_img, lower_violet, upper_violet)

            violet_pixels = cv2.countNonZero(mask)
            total_pixels = box_img.shape[0] * box_img.shape[1]
            violet_ratio = violet_pixels / total_pixels if total_pixels > 0 else 0

            # Verifica se la box contiene sufficiente colore viola
            if violet_ratio > violet_threshold:
                valid_boxes.append(box)
                valid_scores.append(score * (1 + violet_ratio))  # Punteggio aggiustato in base al viola

        # Se nessun box ha soddisfatto i criteri, ritorna lista vuota
        if not valid_boxes:
            return []

        # Converti box e punteggi validi in tensori per GPU e applica NMS
        boxes_tensor = torch.tensor(valid_boxes, dtype=torch.float32).cuda()
        scores_tensor = torch.tensor(valid_scores, dtype=torch.float32).cuda()
        keep = ops.nms(boxes_tensor, scores_tensor, iou_threshold).cpu().numpy()
        return keep

    def nms_cod(self, boxes, scores, img, iou_threshold=0.5):
        valid_boxes = []
        valid_scores = []

        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = map(int, box)

            # Calcola l'offset come metà dell'altezza del bounding box
            box_height = y2 - y1
            offset = box_height // 2  # Arrotondato per eccesso

            # Definizione dell'area sopra il box
            x1_top = max(0, x1)
            y1_top = max(0, y1 - offset)  # Area sopra il bounding box
            x2_top = min(img.shape[1], x2)
            y2_top = y1  # Fine dell'area sopra il box

            # Ritaglio dell'area sopra il bounding box
            top_area_img = img[y1_top:y2_top, x1_top:x2_top]

            if top_area_img.size == 0:
                continue

            # Definizione delle maschere per il colore 
            lower_green = np.array([92, 230, 92])  # Modifica in base alla tonalità desiderata
            upper_green = np.array([130, 255, 130])  # Modifica in base alla tonalità desiderata

            lower_pink = np.array([170, 13, 200])
            upper_pink = np.array([195, 100, 255])

            # Crea maschera per rilevare il colore 
            mask = cv2.inRange(top_area_img, lower_green, upper_green)
            mask1 = cv2.inRange(top_area_img, lower_pink, upper_pink)

            if cv2.countNonZero(mask) == 0:  # Se non c'è verde, il box è valido
                valid_boxes.append(box)
                valid_scores.append(score)
            elif cv2.countNonZero(mask1) > 0:  # Se c'è viola, aumenta la confidenza
                score = max(score, 0.75)  # Aumenta la confidenza a 0.75 se necessario
                valid_boxes.append(box)
                valid_scores.append(score)

        # Se nessun box ha soddisfatto i criteri, ritorna lista vuota
        if not valid_boxes:
            return []

        # Converti box e punteggi validi in tensori per GPU e applica NMS
        boxes_tensor = torch.tensor(valid_boxes, dtype=torch.float32).cuda()
        scores_tensor = torch.tensor(valid_scores, dtype=torch.float32).cuda()
        keep = ops.nms(boxes_tensor, scores_tensor, iou_threshold).cpu().numpy()
        return keep


    def non_max_suppression(self, boxes, scores, iou_threshold):
        boxes = torch.tensor(boxes, dtype=torch.float32).cuda()  # Ensure boxes are on GPU
        scores = torch.tensor(scores, dtype=torch.float32).cuda()  # Ensure scores are on GPU
        keep = ops.nms(boxes, scores, iou_threshold).cpu().numpy()  # Return to CPU for further processing
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


    def draw_detections(self, img, box, score, class_id):
        x1, y1, x2, y2 = map(int, box)
        color = self.color_palette[class_id].tolist()
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{self.classes[class_id]}: {score:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_x, label_y = x1, y1 - 10 if y1 - 10 > label_height else y1 + 10
        cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED)
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def save_image(self, image, timestamp):
        filename = f"data/detected_{timestamp}.jpg"
        cv2.imwrite(filename, image)


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

# Funzione per calcolare Cx e Cy basati su risoluzione e FOV
def calculate_Cx_Cy(resolution_width, resolution_height, fov_horizontal):
    # Calcola il FOV verticale in base al rapporto d'aspetto
    aspect_ratio = resolution_width / resolution_height
    fov_vertical = fov_horizontal / aspect_ratio

    # Calcola i pixel per grado in base alla risoluzione e al FOV
    pixel_per_grado_x = (resolution_width / fov_horizontal) * sensitivity_game
    pixel_per_grado_y = (resolution_height / fov_vertical) * sensitivity_game

    # Risultati finali con sensibilità applicata
    Cx = pixel_per_grado_x 
    Cy = pixel_per_grado_y 
    
    return Cx, Cy

# Initialize the ONNX model
model = ONNX(modelname, 0.49, 0.5)

# Crea una coda per le immagini da salvare
image_queue = queue.Queue()

# Calcola i corretti valori di Cx e Cy in base alla risoluzione e al FOV
Cx, Cy = calculate_Cx_Cy(screen_width, screen_height, fov_horizontal)

# Variables for FPS calculation
fps = 0
frame_count = 0
start_time = time.time()
last_saved_time = time.time()

if data:
    save_thread = threading.Thread(target=save_image_worker)
    save_thread.daemon = True  # Rende il thread daemon, quindi si chiude quando il programma termina
    save_thread.start()


while True:
    frame_start = time.time()
    image_data_ptr = dxgx_dll.shot(duplicator, x, y, width, height, buffer)
    image_data = np.ctypeslib.as_array(buffer)
    img = image_data.reshape((height, width, 4))[:, :, :3]
    results = model(img)
    targets = []

    for box, score, class_id, cls in results:
        target_x = (box[0] + box[2]) / 2 - center
        target_y = (box[1] + box[3]) /2 - center
        target_height = box[3] - box[1]
        target_y = target_y - (target_height / aimpoint)
        #if not -256 <= target_x <= -30 and -100 <= target_y <= 256:
        targets.append((target_x, target_y, target_height))     

    targets_array = np.array(targets)

    if len(targets_array) > 0:
        # Determina il bersaglio attuale
        if len(targets_array) == 1:
            current_target = targets_array[0]
        else:
            # Se ci sono più bersagli, calcoliamo la distanza dal centro per trovare quello più vicino.
            #dist_from_center = [abs(target[0]) for target in targets_array]
            #min_dist_idx = dist_from_center.index(min(dist_from_center))
            dist_from_center = np.linalg.norm(targets_array[:, :2], axis=1)+ 0.1 * targets_array[:, 0]
            min_dist_idx = np.argmin(dist_from_center)
            current_target = targets_array[min_dist_idx]

        # Ottieni le coordinate del bersaglio rispetto al centro dello schermo
        delta_x = current_target[0]
        delta_y = current_target[1]

        # Calcola la velocità (speed)
        mx = math.atan2(delta_x, Cx) * Cx
        my = math.atan2(delta_y, math.sqrt(delta_x ** 2 + Cx ** 2)) * Cy

        # Applica i valori calcolati per lo spostamento
        step_x = int(mx)
        step_y = int(my)

        # Determina lo stato desiderato in base alle condizioni
        if -4 <= step_x <= 4 and -2 <= step_y <= 2:
            kmNet.left(1)
            kmNet.left(0)
        # Muovi il cursore se il tasto è attivo
        if win32api.GetKeyState(0x05)<0 :#
            kmNet.move(step_x, step_y)
           

    # Aggiungi l'immagine alla coda per il salvataggio se `data` è True
    if data:
        image_queue.put((img))

    
    # Keep visualization on the main thread
    if visual:
        img=np.array(img)
        # Calcola il centro dell'immagine
        center_x = img.shape[1] // 2  # Larghezza divisa per 2
        center_y = img.shape[0] // 2  # Altezza divisa per 2
        # Trasforma le coordinate logiche in coordinate assolute
        exclude_rect = [
            center_x - 256,  # x_min
            center_y + 256, # y_min
            center_x -30,  # x_max
            center_y - 80   # y_max
            ]
        # Disegna il rettangolo da escludere
        cv2.rectangle(img,(exclude_rect[0], exclude_rect[1]),  # Punto in alto a sinistra
                      (exclude_rect[2], exclude_rect[3]),  # Punto in basso a destra
                      (0, 0, 255),  # Colore rosso (BGR)
                      2  # Spessore della linea
        )
        for box, score, class_id, cls in results:
            model.predict_model.draw_detections(img, box, score, class_id)
        cv2.imshow("Detected Objects", img)
        cv2.waitKey(1)

    # Calcola il tempo rimanente per raggiungere il target FPS
    frame_time = time.time() - frame_start
    sleep_time = FRAME_TIME - frame_time

    # Attendi se necessario per stabilizzare gli FPS
    if sleep_time > 0:
        time.sleep(sleep_time)

    # Update FPS if needed
    if countfps:
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
            print(fps)

            
# Terminazione del thread di salvataggio in modo pulito (opzionale se il programma deve terminare)
image_queue.put((None, None))  # Inserisce un segnale di arresto nella coda
save_thread.join()  # Attende che il thread termini
