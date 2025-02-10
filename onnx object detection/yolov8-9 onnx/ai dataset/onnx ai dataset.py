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
video1_name = 'valorant mvp'
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


import os
import yt_dlp
from tqdm import tqdm

def download_video():
    script_dir = os.path.dirname(__file__)
    download_path = os.path.join(script_dir, "download")

    if not os.path.exists(download_path):
        os.makedirs(download_path)

    log_file = os.path.join(script_dir, "log.txt")

    if os.path.isfile(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            downloaded_files = f.read().splitlines()
    else:
        downloaded_files = []

    ydl_opts = {
        "quiet": False,
        "outtmpl": f"{download_path}/%(title)s.%(ext)s",
        "format": "bestvideo+bestaudio/best",
        "noplaylist": True,
        "skip_download": True,  # Fetch info first, then download selected videos
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            search_results = ydl.extract_info(f"ytsearch{num_videos * 100}:{video1_name}", download=False)

            if not search_results or "entries" not in search_results:
                print("❌ Nessun video trovato.")
                return

            # Apply filtering
            filtered_videos = []
            for video in search_results["entries"]:
                try:
                    video_duration = video.get("duration", 0)
                    if video["title"] in downloaded_files:
                        continue  # Skip already downloaded videos
                    if video_duration < 60 or video_duration > 400:  
                        continue  # Skip videos shorter than 60s or longer than 15min
                    filtered_videos.append(video)
                except KeyError:
                    continue  # Skip videos without proper metadata

            filtered_videos = filtered_videos[:num_videos]  # Get only required number

            if not filtered_videos:
                print("❌ Nessun video valido trovato dopo il filtraggio.")
                return

            ydl_opts["skip_download"] = False  # Now enable downloads

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                for video in tqdm(filtered_videos, desc="Download progress", unit="video"):
                    try:
                        ydl.download([video["webpage_url"]])
                        with open(log_file, "a", encoding="utf-8") as f:
                            f.write(video["title"] + "\n")
                        print(f"✅ Video scaricato: {video['title']} ({video['duration']} sec)")
                    except Exception as e:
                        print(f"❌ Errore nel download di {video['title']}: {e}")
        except Exception as e:
            print(f"❌ Errore durante la ricerca dei video: {e}")















import os
import cv2
import shutil
from tqdm import tqdm  # ✅ Import tqdm for progress bar

def video_to_frame(target_size, frames_to_skip=100):
    videos_folder = "download"
    frames_folder = "frames_extracted"
    os.makedirs(videos_folder, exist_ok=True)
    os.makedirs(frames_folder, exist_ok=True)

    video_files = [filename for filename in os.listdir(videos_folder) if filename.endswith((".mp4", ".avi"))]

    if not video_files:
        print("❌ No video files found in 'download' folder.")
        return

    for filename in video_files:
        video_path = os.path.join(videos_folder, filename)
        video = cv2.VideoCapture(video_path)

        if not video.isOpened():
            print(f"❌ Error opening video file: {filename}")
            continue

        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_size = min(frame_width, frame_height, target_size)
        x = int(frame_width / 2 - frame_size / 2)
        y = int(frame_height / 2 - frame_size / 2)

        frames_extracted = 0
        frame_count = 0

        print(f"🔄 Processing video: {filename} ({total_frames} total frames)")

        with tqdm(total=total_frames, desc=f"Extracting frames from {filename}", unit="frame") as pbar:
            while True:
                success, frame = video.read()
                if not success:
                    break  # Stop reading if the video ends

                if frame_count % frames_to_skip == 0:
                    try:
                        cropped_frame = frame[y:y+frame_size, x:x+frame_size]
                        resized_frame = cv2.resize(cropped_frame, (target_size, target_size))
                        frame_name = os.path.join(frames_folder, f"{os.path.splitext(filename)[0]}_{frames_extracted}.jpg")
                        cv2.imwrite(frame_name, resized_frame)
                        frames_extracted += 1
                    except Exception as e:
                        print(f"⚠️ Error processing frame {frame_count} of {filename}: {e}")

                frame_count += 1
                pbar.update(1)  # ✅ Update progress bar for each frame processed

        video.release()
        
        if frames_extracted > 0:
            os.remove(video_path)  # ✅ Only delete video if at least one frame was extracted
            print(f"✅ Processed & deleted: {filename} (Frames Extracted: {frames_extracted})")
        else:
            print(f"❌ No frames extracted from {filename}, skipping deletion.")

    cv2.destroyAllWindows()


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
    model = ONNX(modelname, 0.5, 0.5)  # Modello con threshold di 0.5
    for filename in tqdm(os.listdir(MANUAL_LABEL_PATH), desc="Annotating images"):
        image_path = os.path.join(MANUAL_LABEL_PATH, filename)

        try:
            image = Image.open(image_path).convert("RGB")  # Conversione in RGB per evitare errori
            image = np.array(image)
        except Exception as e:
            print(f"❌ Errore nell'aprire l'immagine {filename}: {e}")
            continue  # Salta le immagini corrotte

        results = model(image)
        annotation_lines = []

        for box, score, class_id, cls in results:
            if score >= 0.50:
                model.predict_model.draw_detections(image, box, score, class_id)

                # Calcolo delle coordinate normalizzate
                img_width, img_height = image.shape[1], image.shape[0]
                x_center = ((box[0] + box[2]) / 2) / img_width
                y_center = ((box[1] + box[3]) / 2) / img_height
                width = (box[2] - box[0]) / img_width
                height = (box[3] - box[1]) / img_height

                # Assicuriamoci che i valori siano float con 6 decimali per il formato YOLO
                annotation_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        if annotation_lines:
            annotation_path = os.path.join(LABELS_PATH, os.path.splitext(filename)[0] + ".txt")
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



import shutil
import numpy as np

def compare_annotations():
    """
    Confronta le annotazioni manuali con quelle generate dal modello.
    Se il modello ha rilevato meno oggetti, più oggetti o ha sbagliato classe/coordinate,
    sposta l'immagine nella cartella 'controlla/da_correggere/' per revisione manuale.
    """
    control_images_folder = os.path.join(SCRIPT_DIR, "controlla", "images")  # Immagini annotate automaticamente
    control_labels_folder = os.path.join(SCRIPT_DIR, "controlla", "labels")  # Label manuali
    correction_folder = os.path.join(SCRIPT_DIR, "controlla", "da_correggere")  # Immagini con errori

    os.makedirs(correction_folder, exist_ok=True)

    for filename in tqdm(os.listdir(control_labels_folder), desc="Comparing annotations"):
        if not filename.endswith(".txt"):
            continue  # Salta i file non di annotazione

        manual_annotation_path = os.path.join(control_labels_folder, filename)
        auto_annotation_path = os.path.join(LABELS_PATH, filename)

        # Se manca l'annotazione automatica, l'immagine è errata e va spostata
        if not os.path.exists(auto_annotation_path):
            print(f"⚠️ Annotazione automatica mancante per {filename}, spostando in 'da_correggere'")
            image_path = os.path.join(control_images_folder, filename.replace(".txt", ".jpg"))
            if os.path.exists(image_path):
                shutil.move(image_path, os.path.join(correction_folder, os.path.basename(image_path)))
            continue

        # Legge le annotazioni
        with open(manual_annotation_path, "r") as f:
            manual_labels = [line.strip() for line in f.readlines()]

        with open(auto_annotation_path, "r") as f:
            auto_labels = [line.strip() for line in f.readlines()]

        # Se il numero di oggetti non coincide, l'immagine è errata
        if len(manual_labels) != len(auto_labels):
            print(f"🚨 Differenza nel numero di oggetti per {filename}, spostando in 'da_correggere'")
            image_path = os.path.join(control_images_folder, filename.replace(".txt", ".jpg"))
            if os.path.exists(image_path):
                shutil.move(image_path, os.path.join(correction_folder, os.path.basename(image_path)))
            continue

        # Confronto dettagliato: verifica classi e coordinate
        manual_data = [list(map(float, label.split())) for label in manual_labels]
        auto_data = [list(map(float, label.split())) for label in auto_labels]

        for manual_obj, auto_obj in zip(manual_data, auto_data):
            manual_class, *manual_coords = manual_obj
            auto_class, *auto_coords = auto_obj

            # Se la classe è diversa, è un errore
            if manual_class != auto_class:
                print(f"❌ Classe diversa per {filename}, spostando in 'da_correggere'")
                image_path = os.path.join(control_images_folder, filename.replace(".txt", ".jpg"))
                if os.path.exists(image_path):
                    shutil.move(image_path, os.path.join(correction_folder, os.path.basename(image_path)))
                break

            # Se le coordinate differiscono troppo, è un errore
            manual_coords = np.array(manual_coords)
            auto_coords = np.array(auto_coords)
            if np.linalg.norm(manual_coords - auto_coords) > 0.1:  # Soglia di errore
                print(f"⚠️ Coordinate diverse per {filename}, spostando in 'da_correggere'")
                image_path = os.path.join(control_images_folder, filename.replace(".txt", ".jpg"))
                if os.path.exists(image_path):
                    shutil.move(image_path, os.path.join(correction_folder, os.path.basename(image_path)))
                break

    print("✅ Confronto delle annotazioni completato!")

import cv2
import numpy as np
import os
import shutil

def resize_images_and_check_labels():
    """
    Ridimensiona le immagini da 640x640 a 512x512 e verifica le annotazioni YOLO.
    Se ci sono bounding box oltre il limite 512x512, elimina l'immagine e il file di annotazione.
    """
    input_images_folder = os.path.join(SCRIPT_DIR, "controlla", "images")  # Cartella delle immagini
    input_labels_folder = os.path.join(SCRIPT_DIR, "controlla", "labels")  # Cartella delle label
    output_images_folder = os.path.join(SCRIPT_DIR, "controlla", "resized_images")  # Cartella per immagini corrette
    output_labels_folder = os.path.join(SCRIPT_DIR, "controlla", "resized_labels")  # Cartella per label corrette

    os.makedirs(output_images_folder, exist_ok=True)
    os.makedirs(output_labels_folder, exist_ok=True)

    target_size = 512  # Nuova dimensione dell'immagine

    for filename in os.listdir(input_images_folder):
        if not filename.endswith(".jpg") and not filename.endswith(".png"):
            continue  # Salta i file che non sono immagini

        image_path = os.path.join(input_images_folder, filename)
        label_path = os.path.join(input_labels_folder, filename.replace(".jpg", ".txt").replace(".png", ".txt"))

        # Carica e ridimensiona l'immagine
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Errore nel caricamento di {filename}, saltato.")
            continue

        original_size = image.shape[:2]  # (altezza, larghezza)
        image_resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)

        if not os.path.exists(label_path):
            print(f"⚠️ Nessuna annotazione per {filename}, solo ridimensionato e salvato.")
            cv2.imwrite(os.path.join(output_images_folder, filename), image_resized)
            continue

        # Legge e aggiorna le annotazioni YOLO
        with open(label_path, "r") as f:
            labels = [line.strip() for line in f.readlines()]

        updated_labels = []
        for label in labels:
            parts = label.split()
            class_id = parts[0]
            x_center, y_center, width, height = map(float, parts[1:])

            # Converti le coordinate da YOLO (0-1) a pixel
            x_center *= original_size[1]  # larghezza originale
            y_center *= original_size[0]  # altezza originale
            width *= original_size[1]
            height *= original_size[0]

            # Scala le coordinate in base alla nuova dimensione (512x512)
            x_center = (x_center / original_size[1]) * target_size
            y_center = (y_center / original_size[0]) * target_size
            width = (width / original_size[1]) * target_size
            height = (height / original_size[0]) * target_size

            # Normalizza di nuovo in YOLO format
            x_center /= target_size
            y_center /= target_size
            width /= target_size
            height /= target_size

            # Se il box è fuori dai limiti, scarta l'immagine
            if x_center - width / 2 < 0 or x_center + width / 2 > 1 or y_center - height / 2 < 0 or y_center + height / 2 > 1:
                print(f"🚨 Bounding box fuori dai limiti in {filename}, eliminato!")
                os.remove(image_path)  # Elimina l'immagine originale
                os.remove(label_path)  # Elimina il file di annotazione
                break  # Passa alla prossima immagine

            # Se il box è valido, lo aggiungiamo alla lista delle annotazioni aggiornate
            updated_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # Se ci sono bounding box validi, salva l'immagine e l'annotazione aggiornata
        if updated_labels:
            cv2.imwrite(os.path.join(output_images_folder, filename), image_resized)
            with open(os.path.join(output_labels_folder, filename.replace(".jpg", ".txt").replace(".png", ".txt")), "w") as f:
                f.write("\n".join(updated_labels))

    print("✅ Processo di ridimensionamento e controllo completato!")





#download_video()
#video_to_frame(target_size=512, frames_to_skip=30)
#remove_corrupted_images()
#move_0_50()
annotate_images()
#elimino0_50()
#resize_images_and_check_labels()
compare_annotations()
