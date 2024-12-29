import onnxruntime as ort
import cv2
import os
import shutil
import pytube
from tqdm import tqdm
from PIL import Image,ImageFile
import ssl
import numpy as np
import time

modelname='best.onnx'
video1_name = 'mw3'
num_videos = 1
# Dimensione desiderata per il ridimensionamento dei frame
resize_dimension = 512 #dimesion for  cropped image
saveimageevrynumframe=4 #save evry 10 frames
ssl._create_default_https_context = ssl._create_unverified_context
ImageFile.LOAD_TRUNCATED_IMAGES = True
# Constants and Configuration
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.9
MAX_IMAGES = 10000

# Paths and SSL Context
ssl._create_default_https_context = ssl._create_unverified_context
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMES_PATH = os.path.join(SCRIPT_DIR, "frames_extracted")
MANUAL_LABEL_PATH = os.path.join(SCRIPT_DIR, "manual_label")
LABELS_PATH = os.path.join(SCRIPT_DIR, "labels")
IMAGES_PATH = os.path.join(SCRIPT_DIR, "images")

# Create necessary directories
os.makedirs(FRAMES_PATH, exist_ok=True)
os.makedirs(MANUAL_LABEL_PATH, exist_ok=True)
os.makedirs(LABELS_PATH, exist_ok=True)
os.makedirs(IMAGES_PATH, exist_ok=True)


class Predict:
    def __init__(self, onnx_model, confidence_thres, iou_thres):
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.session = ort.InferenceSession(onnx_model, providers=['DmlExecutionProvider', 'CPUExecutionProvider'])
        self.get_input_details()
        self.get_output_details()
        self.classes = self.get_class_names()
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3)).astype(np.uint8)

        # Variables for FPS calculation
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()

    def detect_objects_resize(self, image):
        input_tensor = self.preprocess_resize(image)
        outputs = self.inference(input_tensor)
        postprocess = self.postprocess_resize(image, outputs)
        return postprocess

    def detect_objects(self, image):
        input_tensor = self.preprocess(image)
        outputs = self.inference(input_tensor)
        postprocess = self.postprocess(image, outputs)
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
        #y = x.copy()
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

    def non_max_suppression(self,boxes, scores, iou_threshold):
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
        x1, y1, x2, y2 = map(int, box)  # Assicurarsi che le coordinate siano intere
        color = self.color_palette[class_id].tolist()
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)  # Aumentare lo spessore a 2 per una migliore visibilità
        label = f"{self.classes[class_id]}: {score:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_x, label_y = x1, y1 - 10 if y1 - 10 > label_height else y1 + 10
        cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED)
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


class ONNX:
    def __init__(self, onnx_model, confidence_thres, iou_thres):
        self.predict_model = Predict(onnx_model, confidence_thres, iou_thres)

    def __call__(self, image):
        self.results, self.images = self.predict_model.detect_objects_resize(image)
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
    os.makedirs(videos_folder, exist_ok=True)
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
    # Initialize the model
    model = ONNX(modelname, 0.35, 0.5)
    
    # Define the input folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(script_dir, "frames_extracted")
    
    # Check if input folder exists and is not empty
    if not os.path.exists(input_folder) or not os.listdir(input_folder):
        print(f"No files found in {input_folder}")
        return

    # Process each file in the input folder
    for filename in tqdm(os.listdir(input_folder)):
        image_path = os.path.join(input_folder, filename)
        
        try:
            # Open and preprocess the image
            image = Image.open(image_path)
            image_np = np.array(image)
            
            # Run the model on the image
            results = model(image_np)
            targets = []
            
            # Loop through the prediction results
            for box, score, class_id, cls in results:
                target_x = int((box[0] + box[2]) / 2)
                target_y = int((box[1] + box[3]) / 2)
                target_height = int(box[3] - box[1])
                targets.append((target_x, target_y, target_height))

            # Remove the image if no targets are found
            if not targets:
                os.remove(image_path)
                
        except:
            # Remove the image if it is corrupted or any other error occurs
            if os.path.exists(image_path):
                try:
                    os.remove(image_path)
                    print(f"Removed corrupted or problematic image: {filename} - {e}")
                except PermissionError as pe:
                    print(f"Could not remove image {filename} due to permission error: {pe}")

def elimino0_50():
    model = ONNX(modelname, 0.5, 0.5)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    #input_folder = os.path.join(script_dir, "frames_extracted")
    input_folder = r'C:\Users\Admin\Desktop\ghg\old backup\Pyt\yolo-main\onnx object detection\yolov8-9 onnx\data'

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

def annotate_images():
    model = ONNX(modelname, 0.5, 0.5)
    for filename in tqdm(os.listdir(MANUAL_LABEL_PATH)):
        image_path = os.path.join(MANUAL_LABEL_PATH, filename)
        image = Image.open(image_path)
        image = np.array(image)
        results = model(image)
        if results:
            annotation_lines = []
            for box, score, class_id, cls in results:
                if score >= 0.50:
                    model.predict_model.draw_detections(image, box, score, class_id)
                    x_center = (box[0] + box[2]) / 2 
                    y_center = (box[1] + box[3]) / 2 
                    width = (box[2] - box[0])
                    height = (box[3] - box[1])
                    annotation_lines.append(f"{class_id} {x_center} {y_center} {width} {height}")

            # Save annotated image
            #annotated_image = Image.fromarray(image)
            #annotated_image.save(image_path)

            # Save annotations to a TXT file
            annotation_path = os.path.splitext(image_path)[0] + ".txt"
            with open(annotation_path, "w") as f:
                f.write("\n".join(annotation_lines))

def move_0_50():
    model = ONNX(modelname, 0.5, 0.5)
    for filename in tqdm(os.listdir(FRAMES_PATH)):
        image_path = os.path.join(FRAMES_PATH, filename)
        image = Image.open(image_path)
        image = np.array(image)
        results = model(image)
        if results:
            dest_image_path = os.path.join(MANUAL_LABEL_PATH, filename)
            shutil.move(image_path, dest_image_path)
            annotation_path = os.path.splitext(image_path)[0] + ".txt"
            if os.path.exists(annotation_path):
                dest_annotation_path = os.path.join(MANUAL_LABEL_PATH, os.path.basename(annotation_path))
                shutil.move(annotation_path, dest_annotation_path)

def remove_corrupted_images():
    for filename in tqdm(os.listdir(FRAMES_PATH), desc="Checking for corrupted images"):
        image_path = os.path.join(FRAMES_PATH, filename)
        try:
            with Image.open(image_path) as img:
                img.verify()  # Basic check to verify if the image file is not corrupted
                
            # Reopen the image to detect truncation issues
            try:
                with Image.open(image_path) as img:
                    img.load()  # Attempt to load the image fully to catch truncated files
            except (IOError, SyntaxError, OSError) as e:
                print(f"Deleting truncated image: {image_path} due to {e}")
                os.remove(image_path)
                
        except (IOError, SyntaxError, OSError) as e:
            print(f"Deleting corrupted image: {image_path} due to {e}")
            os.remove(image_path)

#video_to_frame(target_size=512, frames_to_skip=30)
#remove_corrupted_images()
#mantieni0_35()
#move_0_50()
#annotate_images()

elimino0_50()





