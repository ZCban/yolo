import math
import time
import cv2
import numpy as np
import onnxruntime
import os
import bettercam
import requests
from io import BytesIO
import win32api
import kmNet
from win32api import GetSystemMetrics



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

center_x_screenshot = screenshot // 2
center_y_screenshot = screenshot // 2



kmNet.init('192.168.2.188', '1408', '9FC05414')

class YOLOSeg:

    def __init__(self, path, conf_thres=0.5, iou_thres=0.5, num_masks=32):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.num_masks = num_masks
        self.class_names = ['car', 'track', 'out track']

        # Create a list of colors for each class where each color is a tuple of 3 integer values
        rng = np.random.default_rng(3)
        self.colors = rng.uniform(0, 255, size=(len(self.class_names), 3))

        # Initialize model
        self.initialize_model(path)

    def __call__(self, image):
        return self.segment_objects(image)

    def initialize_model(self, path):
        self.session = onnxruntime.InferenceSession(path, providers=['DmlExecutionProvider', 'CPUExecutionProvider'])
        # Get model info
        self.get_input_details()
        self.get_output_details()

    def segment_objects(self, image):
        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)

        self.boxes, self.scores, self.class_ids, mask_pred = self.process_box_output(outputs[0])
        self.mask_maps = self.process_mask_output(mask_pred, outputs[1])

        return self.boxes, self.scores, self.class_ids, self.mask_maps

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def inference(self, input_tensor):
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        return outputs

    def process_box_output(self, box_output):
        predictions = np.squeeze(box_output).T
        num_classes = box_output.shape[1] - self.num_masks - 4

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:4+num_classes], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], [], np.array([])

        box_predictions = predictions[..., :num_classes+4]
        mask_predictions = predictions[..., num_classes+4:]

        # Get the class with the highest confidence
        class_ids = np.argmax(box_predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(box_predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = self.nms(boxes, scores, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices], mask_predictions[indices]

    def process_mask_output(self, mask_predictions, mask_output):
        if mask_predictions.shape[0] == 0:
            return []

        mask_output = np.squeeze(mask_output)

        # Calculate the mask maps for each box
        num_mask, mask_height, mask_width = mask_output.shape  # CHW
        masks = self.sigmoid(mask_predictions @ mask_output.reshape((num_mask, -1)))
        masks = masks.reshape((-1, mask_height, mask_width))

        # Downscale the boxes to match the mask size
        scale_boxes = self.rescale_boxes(self.boxes,
                                         (self.img_height, self.img_width),
                                         (mask_height, mask_width))

        # For every box/mask pair, get the mask map
        mask_maps = np.zeros((len(scale_boxes), self.img_height, self.img_width))
        blur_size = (int(self.img_width / mask_width), int(self.img_height / mask_height))
        for i in range(len(scale_boxes)):
            scale_x1 = int(math.floor(scale_boxes[i][0]))
            scale_y1 = int(math.floor(scale_boxes[i][1]))
            scale_x2 = int(math.ceil(scale_boxes[i][2]))
            scale_y2 = int(math.ceil(scale_boxes[i][3]))

            x1 = int(math.floor(self.boxes[i][0]))
            y1 = int(math.floor(self.boxes[i][1]))
            x2 = int(math.ceil(self.boxes[i][2]))
            y2 = int(math.ceil(self.boxes[i][3]))

            scale_crop_mask = masks[i][scale_y1:scale_y2, scale_x1:scale_x2]
            crop_mask = cv2.resize(scale_crop_mask,
                                   (x2 - x1, y2 - y1),
                                   interpolation=cv2.INTER_CUBIC)

            crop_mask = cv2.blur(crop_mask, blur_size)

            crop_mask = (crop_mask > 0.5).astype(np.uint8)
            mask_maps[i, y1:y2, x1:x2] = crop_mask

        return mask_maps

    def extract_boxes(self, box_predictions):
        # Extract boxes from predictions
        boxes = box_predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes,
                                   (self.input_height, self.input_width),
                                   (self.img_height, self.img_width))

        # Convert boxes to xyxy format
        boxes = self.xywh2xyxy(boxes)

        # Check the boxes are within the image
        boxes[:, 0] = np.clip(boxes[:, 0], 0, self.img_width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, self.img_height)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, self.img_width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, self.img_height)

        return boxes

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):
        return self._draw_detections(image, self.boxes, self.scores,
                                     self.class_ids, mask_alpha)

    def draw_masks(self, image, draw_scores=True, mask_alpha=0.5):
        return self._draw_detections(image, self.boxes, self.scores,
                                     self.class_ids, mask_alpha, mask_maps=self.mask_maps)

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    @staticmethod
    def rescale_boxes(boxes, input_shape, image_shape):
        input_shape = np.array([input_shape[1], input_shape[0], input_shape[1], input_shape[0]])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([image_shape[1], image_shape[0], image_shape[1], image_shape[0]])
        return boxes

    @staticmethod
    def xywh2xyxy(x):
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def nms(self, boxes, scores, iou_threshold):
        sorted_indices = np.argsort(scores)[::-1]
        keep_boxes = []
        while sorted_indices.size > 0:
            box_id = sorted_indices[0]
            keep_boxes.append(box_id)
            ious = self.compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])
            keep_indices = np.where(ious < iou_threshold)[0]
            sorted_indices = sorted_indices[keep_indices + 1]
        return keep_boxes

    @staticmethod
    def compute_iou(box, boxes):
        xmin = np.maximum(box[0], boxes[:, 0])
        ymin = np.maximum(box[1], boxes[:, 1])
        xmax = np.minimum(box[2], boxes[:, 2])
        ymax = np.minimum(box[3], boxes[:, 3])
        intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - intersection_area
        iou = intersection_area / union_area
        return iou

    def _draw_detections(self, image, boxes, scores, class_ids, mask_alpha=0.3, mask_maps=None):
        img_height, img_width = image.shape[:2]
        size = min([img_height, img_width]) * 0.0006
        text_thickness = int(min([img_height, img_width]) * 0.001)
        mask_img = self._draw_masks(image, boxes, class_ids, mask_alpha, mask_maps)

        for box, score, class_id in zip(boxes, scores, class_ids):
            if class_id == target_class_id:
                color = self.colors[class_id]
                x1, y1, x2, y2 = box.astype(int)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                error_x = center_x - center_x_screenshot
                cv2.circle(mask_img, (center_x, center_y), 5, (0, 0, 255), -1)
                cv2.rectangle(mask_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            else :
                color = self.colors[class_id]
                x1, y1, x2, y2 = box.astype(int)

            

            label = self.class_names[class_id]
            caption = f'{label} {int(score * 100)}%'
            (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                          fontScale=size, thickness=text_thickness)
            th = int(th * 1.2)
            cv2.rectangle(mask_img, (x1, y1), (x1 + tw, y1 - th), color, -1)
            cv2.putText(mask_img, caption, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

        return mask_img

    def _draw_masks(self, image, boxes, class_ids, mask_alpha=0.3, mask_maps=None):
        mask_img = image
        for i, (box, class_id) in enumerate(zip(boxes, class_ids)):
            color = self.colors[class_id]
            x1, y1, x2, y2 = box.astype(int)
            if mask_maps is None:
                cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)
            else:
                crop_mask = mask_maps[i][y1:y2, x1:x2, np.newaxis]
                crop_mask_img = mask_img[y1:y2, x1:x2]
                crop_mask_img = crop_mask_img * (1 - crop_mask) + crop_mask * color
                mask_img[y1:y2, x1:x2] = crop_mask_img
        return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)


model_path = "best.onnx"
yoloseg = YOLOSeg(model_path, conf_thres=0.5, iou_thres=0.5)
# Initialize the camera
cam = bettercam.create(output_idx=0, output_color="BGR")
cam.start(region=region, video_mode=True, target_fps=60)
target_class_id = 1

while True:
    img = cam.get_latest_frame()
    boxes, scores, class_ids, mask_maps = yoloseg(img)  # Perform object detection and segmentation


    combined_img = yoloseg.draw_masks(img)  # Draw detections and masks

    cv2.imshow("Output", combined_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cv2.destroyAllWindows()
