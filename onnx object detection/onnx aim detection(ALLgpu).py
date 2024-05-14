import onnxruntime as ort
import cv2
import numpy as np
import bettercam
import kmNet
import win32api
from win32api import GetSystemMetrics
import time

# Get screen resolution dynamically
screen_width = GetSystemMetrics(0)
screen_height = GetSystemMetrics(1)

# Initialize your webcam capture
screenshot = 448
left, top = (screen_width - screenshot) // 2, (screen_height - screenshot) // 2
right, bottom = left + screenshot, top + screenshot
region = (left, top, right, bottom)
cam = bettercam.create(output_idx=0, output_color="BGR")
cam.start(region=region, video_mode=True, target_fps=85)
center = screenshot / 2
countfps = False
movespeed = 2
visual = True

# Initialize the ONNX model
model = ONNX('448.onnx', 0.6, 0.5)

kmNet.init('192.168.2.188', '1408', '9FC05414')

# Variables for FPS calculation
fps = 0
frame_count = 0
start_time = time.time()

class Predict:
    def __init__(self, onnx_model, confidence_thres, iou_thres):
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.session = ort.InferenceSession(onnx_model, providers=['DmlExecutionProvider', 'CPUExecutionProvider'])
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
        self.img_height, self.img_width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=(self.input_width, self.input_height), swapRB=True, crop=False)
        return blob

    def inference(self, input_tensor):
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        return outputs

    def postprocess(self, input_image, outputs):
        outputs = np.squeeze(outputs[0]).T
        scores = np.max(outputs[:, 4:], axis=1)
        outputs = outputs[scores > self.confidence_thres, :]
        scores = scores[scores > self.confidence_thres]
        class_ids = np.argmax(outputs[:, 4:], axis=1)
        boxes = self.extract_boxes(outputs)
        indices = self.multiclass_nms(boxes, scores, class_ids, self.iou_thres)
        arry_box = []
        for i in indices:
            box, score, class_id = boxes[i, :4], scores[i], class_ids[i]
            arry_box.append((box, score, class_id, self.classes[class_id]))
        return arry_box, input_image

    def xywh2xyxy(self, x):
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y

    def extract_boxes(self, predictions):
        boxes = predictions[:, :4]
        boxes = self.rescale_boxes(boxes)
        boxes = self.xywh2xyxy(boxes)
        return boxes

    def rescale_boxes(self, boxes):
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    def intersection_over_union(self, boxA, boxB):
        xA = np.maximum(boxA[0], boxB[0])
        yA = np.maximum(boxA[1], boxB[1])
        xB = np.minimum(boxA[2], boxB[2])
        yB = np.minimum(boxA[3], boxB[3])
        inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = inter_area / float(boxA_area + boxB_area - inter_area)
        return iou

    def non_max_suppression(self, boxes, scores, iou_threshold):
        indices = np.argsort(scores)[::-1]
        selected_indices = []
        while len(indices) > 0:
            current_index = indices[0]
            selected_indices.append(current_index)
            current_box = boxes[current_index]
            other_boxes = boxes[indices[1:]]
            iou_values = np.array([self.intersection_over_union(current_box, other_box) for other_box in other_boxes])
            indices = indices[np.where(iou_values <= iou_threshold)[0] + 1]
        return np.array(selected_indices)

    def multiclass_nms(self, boxes, scores, class_ids, iou_threshold):
        unique_class_ids = np.unique(class_ids)
        selected_indices = []
        for class_id in unique_class_ids:
            class_mask = (class_ids == class_id)
            class_boxes = boxes[class_mask]
            class_scores = scores[class_mask]
            indices = self.non_max_suppression(class_boxes, class_scores, iou_threshold)
            selected_indices.extend(np.where(class_mask)[0][indices])
        return np.array(selected_indices)

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
        x1, y1, x2, y2 = box.astype(int)
        color = self.color_palette[class_id]
        cv2.rectangle(img, (x1, y1), (x2, y2), tuple(color.tolist()), 2)
        label = f"{self.classes[class_id]}: {score:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_x, label_y = x1, y1 - 10 if y1 - 10 > label_height else y1 + 10
        cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), tuple(color.tolist()), cv2.FILLED)
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

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



while True:
    img = cam.get_latest_frame()
    results = model(img)
    targets = []

    for box, score, class_id, cls in results:
        if visual:
            model.predict_model.draw_detections(img, box, score, class_id)
        target_x = int((box[0] + box[2]) / 2)
        target_y = int((box[1] + box[3]) / 2)
        # Calculate the height of the rectangle
        target_height = int(box[3] - box[1])
        targets.append((target_x, target_y, target_height))

    if len(targets) > 0:
        targets_array = np.array(targets)
        # Calculate Euclidean distances between targets and center
        distances = np.linalg.norm(targets_array[:, :2] - center, axis=1)
        # Find the index of the nearest target
        nearest_index = np.argmin(distances)
        nearest_distance = distances[nearest_index]
        nearest_target = targets[nearest_index]
        delta_x = int(nearest_target[0] - center)
        delta_y = int(nearest_target[1] - center)
        delta_y -= int(nearest_target[2] / 3)
        if win32api.GetKeyState(0x05) < 0:
            kmNet.move(int(delta_x / movespeed), int(delta_y / movespeed))
            if -2.5 <= delta_x / movespeed <= 2.5 and -1 <= delta_y / movespeed <= 1:
                kmNet.left(1)
                kmNet.left(0)
            if not (-2.5 <= delta_x / movespeed <= 2.5 and -1 <= delta_y / movespeed <= 1):
                kmNet.move(int(delta_x / movespeed), int(delta_y / movespeed))

    if countfps:
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1:  # Update FPS every second
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
            if not visual:
                print(fps)

    if visual:
        if countfps:
            # Display FPS on the image
            cv2.putText(img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Detected Objects", img)
        cv2.waitKey(1)

cv2.destroyAllWindows()
cam.stop()

