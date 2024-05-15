import onnxruntime as ort
import cv2
import os
import shutil
import pytube
from tqdm import tqdm
import torch
from PIL import Image
import ssl
import numpy as np

modelname='640.onnx'
video1_name = 'mw3'
num_videos = 1
# Dimensione desiderata per il ridimensionamento dei frame
resize_dimension = 640 #dimesion for  cropped image
saveimageevrynumframe=15 #save evry 10 frames
ssl._create_default_https_context = ssl._create_unverified_context

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


def findvideo():
    script_dir = os.path.dirname(__file__)
    download_path = os.path.join(script_dir, "download")
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    log_file = os.path.join(script_dir, "log.txt")
    if os.path.isfile(log_file):
        with open(log_file, "r") as f:
            downloaded_files = f.read().splitlines()
    else:
        downloaded_files = []

    video_list = pytube.Search(video1_name).results
    while not video_list:
        print("Nessun video trovato con il nome specificato")
        return
    video_list = [v for v in video_list if v.title not in downloaded_files]
    video_list = video_list[:num_videos]

    for video1 in tqdm(video_list, desc="Download progress", unit="video"):
        yt = pytube.YouTube(video1.watch_url)
        #stream = yt.streams.filter(progressive=True, file_extension="mp4").order_by('resolution').desc().first()
        stream = yt.streams.get_highest_resolution()
        stream.download(download_path)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(video1.title + "\n")

def video_to_frame(target_size, frames_to_skip=100):
    videos_folder = "download"
    frames_folder = "frames_extracted"
    os.makedirs(frames_folder, exist_ok=True)

    processed_videos = 0
    video_files = [filename for filename in os.listdir(videos_folder) if filename.endswith((".mp4", ".avi"))]
    total_videos = len(video_files)

    for filename in video_files:
        video_path = os.path.join(videos_folder, filename)
        video = cv2.VideoCapture(video_path)

        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = min(frame_width, frame_height, target_size)
        x = int(frame_width / 2 - frame_size / 2)
        y = int(frame_height / 2 - frame_size / 2)

        frames_extracted = 0
        frame_count = 0
        while True:
            success, frame = video.read()
            if not success:
                break

            frame_count += 1
            if frame_count % frames_to_skip == 0:
                cropped_frame = frame[y:y+frame_size, x:x+frame_size]
                resized_frame = cv2.resize(cropped_frame, (target_size, target_size))
                frame_name = os.path.join(frames_folder, f"{os.path.splitext(filename)[0]}_{frames_extracted}.jpg")
                cv2.imwrite(frame_name, resized_frame)
                frames_extracted += 1

            progress = int(frames_extracted / total_videos * 100)
            print(frames_extracted)

        video.release()
        processed_videos += 1

    shutil.rmtree(videos_folder)
    cv2.destroyAllWindows()


def mantieni0_35():
    model = ONNX(modelname, 0.35, 0.5)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(script_dir, "frames_extracted")

    for filename in tqdm(os.listdir(input_folder)):
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path)#.convert("RGB")
        image=np.array(image)
        results = model(image)
        targets = []
        # Loop attraverso i risultati della predizione
        for box, score, class_id,cls in results:
            target_x = int((box[0] + box[2]) / 2)
            target_y = int((box[1] + box[3]) / 2)
            target_height = int(box[3] - box[1])
            targets.append((target_x, target_y, target_height))

        if not  targets:
            os.remove(image_path)


def elimino0_50():
    model = ONNX(modelname, 0.5, 0.5)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(script_dir, "frames_extracted")

    for filename in tqdm(os.listdir(input_folder)):
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path)#.convert("RGB")
        image=np.array(image)
        results = model(image)
        targets = []
        # Loop attraverso i risultati della predizione
        for box, score, class_id,cls in results:
            target_x = int((box[0] + box[2]) / 2)
            target_y = int((box[1] + box[3]) / 2)
            target_height = int(box[3] - box[1])
            targets.append((target_x, target_y, target_height))

        if  targets:
            os.remove(image_path)


print('cercando video')
findvideo()
print('catturo frame')
video_to_frame(target_size=resize_dimension, frames_to_skip=saveimageevrynumframe)
print('0.35')
mantieni0_35()
print('0-50')
elimino0_50()







