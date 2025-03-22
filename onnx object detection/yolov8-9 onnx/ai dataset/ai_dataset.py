import onnxruntime as ort
import cv2
import os
import shutil
from tqdm import tqdm
from PIL import Image,ImageFile
import ssl
import numpy as np
import time
import yt_dlp


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






def download_video_from_link(video_url, download_folder="download", log_file="log.txt", cookies_file="cookies.txt"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    download_path = os.path.join(script_dir, download_folder)
    log_path = os.path.join(script_dir, log_file)
    cookies_path = os.path.join(script_dir, cookies_file)

    if not os.path.exists(download_path):
        os.makedirs(download_path)

    # Leggere il log dei video già scaricati
    if os.path.isfile(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            downloaded_files = f.read().splitlines()
    else:
        downloaded_files = []

    ydl_opts = {
        "quiet": False,
        "outtmpl": f"{download_path}/%(title)s.%(ext)s",
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
        "noplaylist": True,
        "merge_output_format": "mp4",  # Assicura che il file finale sia in MP4
        "postprocessors": [{
            "key": "FFmpegVideoConvertor",
            "preferedformat": "mp4",
        }],
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Referer": "https://www.youtube.com/",
            "Accept-Language": "en-US,en;q=0.9",
        },
        "nocheckcertificate": True,  # Evita problemi con certificati HTTPS
        "no_warnings": True,  # Riduce gli avvisi non critici
    }
    
    # Usa i cookies se disponibili
    if os.path.isfile(cookies_path):
        ydl_opts["cookiefile"] = cookies_path

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            print(f"🔄 Recupero informazioni per: {video_url}")
            info_dict = ydl.extract_info(video_url, download=False)
            video_title = info_dict.get("title", "Unknown Video")
            video_file = os.path.join(download_path, f"{video_title}.mp4")

            if video_title in downloaded_files:
                print(f"⏩ Il video '{video_title}' è già stato scaricato. Skipping...")
                return

            print(f"🔄 Download in corso per: {video_url}")
            ydl.download([video_url])

            # Controllo se il file esiste davvero
            if os.path.exists(video_file):
                # Salvare il titolo nel log
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(video_title + "\n")
                print(f"✅ Download completato: {video_title}")
            else:
                print("❌ Errore: Il file non è stato salvato correttamente.")
        except Exception as e:
            print(f"❌ Errore durante il download: {e}")

# Esempio di utilizzo
# download_video_from_link("https://www.youtube.com/watch?v=VIDEO_ID")





def download_video(video1_name,num_videos):
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
            search_results = ydl.extract_info(f"ytsearch{num_videos * 8}:{video1_name}", download=False)

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
                    if video_duration < 60 or video_duration > 600:  
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





def video_to_frame(target_size, frames_to_skip=100, videos_folder='download', frames_folder='frames_extracted'):


    video_files = [filename for filename in os.listdir(videos_folder) if filename.endswith((".mp4", ".avi",".mkv",".webm"))]

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


def elimino0_50(FRAMES_PATH):
    for filename in tqdm(os.listdir(FRAMES_PATH)):
        image_path = os.path.join(FRAMES_PATH, filename)
        image = Image.open(image_path)#.convert("RGB")
        image = np.asarray(image)
        results = model(image)       
        targets = []
        # Loop attraverso i risultati della predizione
        for box, score, class_id,cls in results:
            target_x = int((box[0] + box[2]) / 2)
            target_y = int((box[1] + box[3]) / 2)
            targets.append((target_x, target_y))

        if  targets:
            os.remove(image_path)

def annotate_images(FRAMES_PATH):
    for filename in tqdm(os.listdir(FRAMES_PATH), desc="Annotating images"):
        image_path = os.path.join(FRAMES_PATH, filename)

        try:
            image = Image.open(image_path)#.convert("RGB")  # Conversione in RGB per evitare errori
            image = np.asarray(image)
        except Exception as e:
            print(f"❌ Errore nell'aprire l'immagine {filename}: {e}")
            continue  # Salta le immagini corrotte

        results = model(image)
        annotation_lines = []

        for box, score, class_id, cls in results:
            # Calcolo delle coordinate normalizzate
            img_width, img_height = image.shape[1], image.shape[0]
            x_center = ((box[0] + box[2]) / 2) / img_width
            y_center = ((box[1] + box[3]) / 2) / img_height
            width = (box[2] - box[0]) / img_width
            height = (box[3] - box[1]) / img_height

            # Assicuriamoci che i valori siano float con 6 decimali per il formato YOLO
            annotation_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        if annotation_lines:
            annotation_path = os.path.join(FRAMES_PATH, os.path.splitext(filename)[0] + ".txt")
            try:
                with open(annotation_path, "w") as f:
                    f.write("\n".join(annotation_lines))
            except Exception as e:
                print(f"❌ Error writing annotation for {filename}: {e}")






def remove_corrupted_images(FRAMES_PATH):
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



import os
from tqdm import tqdm

def clean_folders(FRAMES_PATH,images_path,labels_path):
    """
    Funzione unica che esegue:
    1. Eliminazione di file di testo vuoti
    2. Rimozione di immagini senza etichetta
    3. Rimozione di etichette senza immagine
    """
    
    def delete_empty_files(folder_path):
        """Elimina i file di testo vuoti nella cartella specificata."""
        if not os.path.exists(folder_path):
            print(f"La cartella '{folder_path}' non esiste.")
            return
        
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path) and file_name.endswith(".txt"):
                if os.path.getsize(file_path) == 0:
                    os.remove(file_path)
                    print(f"File vuoto '{file_name}' eliminato.")
    
    def check_images_without_labels(folder_images):
        """Rimuove immagini che non hanno un'etichetta corrispondente."""
        if not os.path.exists(folder_images):
            print("Errore: La cartella delle immagini non esiste.")
            return
        
        label_basenames = set(os.path.splitext(f)[0] for f in os.listdir(folder_images) if f.endswith(".txt"))
        
        for file in tqdm(os.listdir(folder_images), desc='Checking for missing labels', unit='file'):
            filename, ext = os.path.splitext(file)
            if ext.lower() == ".jpg" and filename not in label_basenames:
                file_path = os.path.join(folder_images, file)
                os.remove(file_path)
                print(f"Immagine senza etichetta '{file}' eliminata.")
    
    def check_labels_without_images(folder_images):
        """Rimuove etichette che non hanno un'immagine corrispondente."""
        if not os.path.exists(folder_images):
            print("Errore: La cartella delle immagini non esiste.")
            return
        
        image_basenames = set(os.path.splitext(f)[0] for f in os.listdir(folder_images) if f.endswith(".jpg"))
        
        for file in tqdm(os.listdir(folder_images), desc='Checking for missing images', unit='file'):
            filename, ext = os.path.splitext(file)
            if ext.lower() == ".txt" and filename not in image_basenames:
                file_path = os.path.join(folder_images, file)
                os.remove(file_path)
                print(f"Etichetta senza immagine '{file}' eliminata.")

    def move_files_to_train(folder_path):

        
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                if file.endswith(".jpg"):
                    os.rename(file_path, os.path.join(images_path, file))
                elif file.endswith(".txt"):
                    os.rename(file_path, os.path.join(labels_path, file))
        
        print("Tutti i file sono stati spostati nelle rispettive cartelle.")
    
    # Esegui le quattro funzioni
    delete_empty_files(FRAMES_PATH)
    check_images_without_labels(FRAMES_PATH)
    check_labels_without_images(FRAMES_PATH)
    move_files_to_train(FRAMES_PATH)






modelname='bestv8.onnx'
min_conf=0.5
max_iou=0.5
model = ONNX(modelname, min_conf, max_iou)

# Paths and SSL Context
#ssl._create_default_https_context = ssl._create_unverified_context


#download_video(video1_name = 'Rainbow Six Siege best play',num_videos = 1)
#download_video_from_link("https://www.youtube.com/watch?v=izEefQhiNgU")
#video_to_frame(target_size=512, frames_to_skip=2)
#remove_corrupted_images(FRAMES_PATH)
#elimino0_50(FRAMES_PATH)
#annotate_images(FRAMES_PATH)
#clean_folders(FRAMES_PATH,images_path,labels_path)

