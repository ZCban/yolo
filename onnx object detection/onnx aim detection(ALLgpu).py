import onnxruntime as ort
import cv2
import numpy as np
import bettercam
import kmNet
import win32api
from win32api import GetSystemMetrics
import time

# setting int
screenshot = 448
countfps = True
movespeed = 2
visual = False
fovx=2.2
fovy=1.2
activationkey=0x05
modelname='448.onnx'
fpslimit=80

# Get screen resolution dynamically
screen_width = GetSystemMetrics(0)
screen_height = GetSystemMetrics(1)

##setup bettercam##
left, top = (screen_width - screenshot) // 2, (screen_height - screenshot) // 2
right, bottom = left + screenshot, top + screenshot
region = (left, top, right, bottom)
cam = bettercam.create(output_idx=0, output_color="BGR")
cam.start(region=region, video_mode=True, target_fps=fpslimit)
center = screenshot / 2

##setup kmnet
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
        blob = np.ascontiguousarray(image.transpose(2, 0, 1)[np.newaxis, ...], dtype=np.float32) / 255.0
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
        indices = self.non_max_suppression(boxes, scores, self.iou_thres)
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
        boxes = self.xywh2xyxy(boxes)
        return boxes

    def non_max_suppression(self, boxes, scores, iou_threshold):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
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
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[np.where(iou <= iou_threshold)[0] + 1]

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

    def draw_detections(self, img, box, score, class_id):
        x1, y1, x2, y2 = box.astype(int)
        color = self.color_palette[class_id]
        cv2.rectangle(img, (x1, y1), (x2, y2), tuple(color.tolist()), 1)
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


# Initialize the ONNX model
model = ONNX(modelname, 0.52, 0.55)

while True:
    img = cam.get_latest_frame()
    results = model(img)
    targets = []

    for box, score, class_id, cls in results:
        target_x = int((box[0] + box[2]) / 2)
        target_y = int((box[1] + box[3]) / 2)
        target_height = int(box[3] - box[1])
        targets.append((target_x, target_y, target_height))

    targets_array = np.array(targets)
    if len(targets_array) > 0:       
        # Calculate Euclidean distances between targets and center
        distances = np.linalg.norm(targets_array[:, :2] - center, axis=1)
        # Find the index of the nearest target
        nearest_index = np.argmin(distances)
        nearest_distance = distances[nearest_index]
        nearest_target = targets[nearest_index]
        delta_x = int(nearest_target[0] - center)
        delta_y = int(nearest_target[1] - center)
        delta_y -= int(nearest_target[2] / 2.8)
        if win32api.GetKeyState(activationkey) < 0:
            kmNet.move(int(delta_x / movespeed), int(delta_y / movespeed))
            if -fovx <= delta_x / movespeed <= fovx and -fovy <= delta_y / movespeed <= fovy:
                kmNet.left(1)
                kmNet.left(0)

    if visual:
        for box, score, class_id, cls in results:
            model.predict_model.draw_detections(img, box, score, class_id)
        cv2.imshow("Detected Objects", img)
        cv2.waitKey(1)

    if countfps:
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1:  # Update FPS every second
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
            if not visual:
                print(fps)

cv2.destroyAllWindows()
cam.stop()

