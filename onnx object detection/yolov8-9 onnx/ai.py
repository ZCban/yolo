import onnxruntime as ort
import cv2
import numpy as np
import kmNet
from ScreenCapture import ScreenCaptureMSS,ScreenCaptureBETTERCAM
import win32api
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
screeneginegpu = True
countfps = True
visual =  False                                 
data = False
antirecoil = True
recoily = 3
modelname = 'bestv8.onnx'
aimpoint=3.5
current_target = None
center = screenshot / 2

if screeneginegpu:
    screen_capture = ScreenCaptureBETTERCAM(screenshot)
else:
    screen_capture = ScreenCaptureMSS(screenshot)

# Setup kmNet
kmNet.init('192.168.2.188', '1408', '9FC05414')
kmNet.monitor(5001)
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



    def detect_objects_resize(self, image):
        self.img_height, self.img_width = 800,800
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

    def non_max_suppression(self,boxes, scores, iou_threshold):
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]  # Ordina i punteggi in ordine decrescent
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
            # Seleziona solo i box con IoU inferiore alla soglia
            order = order[1:][iou <= iou_threshold]

        return np.array(keep, dtype=np.int32)

    def non_max_suppression1(self, boxes, scores, iou_threshold):
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
        timestamp = int(time.time() * 3)

        # Salva l'immagine
        filename = f"data/detected_{timestamp}.jpg"
        cv2.imwrite(filename, image)

        # Segnala alla coda che il compito è stato completato
        image_queue.task_done()

if data:
    # Ensure 'data' directory exists
    if not os.path.exists('data'):
        os.makedirs('data')
    # Crea una coda per le immagini da salvare
    image_queue = queue.Queue()
    save_thread = threading.Thread(target=save_image_worker)
    save_thread.daemon = True  # Rende il thread daemon, quindi si chiude quando il programma termina
    save_thread.start()

# Initialize the ONNX model
model = ONNX(modelname, 0.49, 0.5)





while True:
    img = screen_capture.capture()

    if img is not None:
        results = model(img)
        targets = []

        for box, score, class_id, cls in results:
            target_x = (box[0] + box[2]) / 2 - center
            target_y = box[1] + (box[3] - box[1]) / 3.5 - center#(box[1] + 20)  - center
            targets.append((target_x, target_y))
        targets_array = np.array(targets)

        if len(targets_array) > 0:
            # Se ci sono più bersagli, calcoliamo la distanza dal centro per trovare quello più vicino aggiungi valore fisso per rendere distanze univoche.
            dist_from_center = np.linalg.norm(targets_array, axis=1)
            min_dist_idx = np.argmin(dist_from_center)
            current_target = targets_array[min_dist_idx]
            # Ottieni le coordinate del bersaglio rispetto al centro dello schermo
            delta_x = int(current_target[0]/2.4)
            delta_y = int(current_target[1]/2.4)
            # Muovi il cursore se il tasto è attivo
            if kmNet.isdown_side1()==1:
                kmNet.move(delta_x, delta_y)
        #else:
        #    if data:
        #        image_queue.put((img))

        # Keep visualization on the main thread
        if visual:
            for box, score, class_id, cls in results:
                model.predict_model.draw_detections(img, box, score, class_id)
            cv2.imshow("Detected Objects", img)
            cv2.waitKey(1)

        # Update FPS if needed
        if countfps:
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 1:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
                print(fps)



    if antirecoil and kmNet.isdown_left()==1 and kmNet.isdown_right()==1 :
        kmNet.move(int(0), int(recoily))
