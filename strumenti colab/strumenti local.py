#0: person
#1: bicycle
#2: car
#3: motorcycle
#4: airplane
#5: bus
#6: train
#7: truck

import os
import glob
from pytube import YouTube
import cv2
import numpy as np
from ultralytics import YOLO
from youtube_search import YoutubeSearch

###################setting###################
folder_path = r"C:\Users\FBposta\Desktop\Nuova cartella\1"
url = "https://www.youtube.com/watch?v=-QZ7ypPgDfg&t=363s"
save_path = folder_path 
frame_rate = 60  # Numero di frame da saltare prima di salvare il successivo
target_size = (640, 640)  # Dimensione target per il ridimensionamento dei frame
model = YOLO("yolov8m-seg.pt")
output_dir = r"C:\Users\FBposta\Desktop\Nuova cartella\images"  # Output directory for annotations
output_dir1 = r"C:\Users\FBposta\Desktop\Nuova cartella\labels"  # Output directory for annotations
#####################################################################

def delete_folder_contents(folder_path):
    """Controlla se la cartella esiste e elimina tutti i file al suo interno."""
    if not os.path.exists(folder_path):
        print(f"La cartella {folder_path} non esiste.")
        # Crea la cartella se non esiste
        os.makedirs(folder_path, exist_ok=True)
    else:
        # Ottieni tutti i file nella cartella specificata
        file_list = glob.glob(folder_path + "/*")
        # Verifica se ci sono file nella cartella
        if file_list:
            # Elimina ogni file
            for file_path in file_list:
                os.remove(file_path)
            print("Contenuti della cartella eliminati con successo.")
        else:
            print("La cartella è già vuota.")


def download_youtube_video(url, save_path):
    try:
        yt = YouTube(url)
        video = yt.streams.get_highest_resolution()
        video.download(save_path)
        video_path = os.path.join(save_path, video.default_filename)
        print(f"Video scaricato con successo: {video_path}")
        return video_path  # Ensure this return statement is included
    except Exception as e:
        print(f"Si è verificato un errore durante il download del video: {e}")
        return None  # Return None or an appropriate value if the download fails



def extract_and_save_frames(video_path, save_path, target_size, frame_rate):
    cap = cv2.VideoCapture(video_path)
    count = 0

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = min(frame_width, frame_height, *target_size)
    x = int(frame_width / 2 - frame_size / 2)
    y = int(frame_height / 2 - frame_size / 2)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        if count % frame_rate == 0:
            cropped_frame = frame[y:y+frame_size, x:x+frame_size]
            resized_frame = cv2.resize(cropped_frame, target_size)
            frame_name = os.path.join(save_path, f"frame_{count}.jpg")
            cv2.imwrite(frame_name, resized_frame)
            print(f"Frame salvato: {frame_name}")

        count += 1
    cap.release()




def process_and_annotate_frames_segment(save_path, model_path, output_dir, conf_threshold=0.5, target_classes=[2,3,5,7]):
    model = YOLO(model_path)
    os.makedirs(output_dir, exist_ok=True)
    yolo_classes = list(model.names.values())
    classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]

    for filename in os.listdir(save_path):
        if filename.endswith(".jpg"):
            img_path = os.path.join(save_path, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue
            results = model.predict(img, conf=conf_threshold, classes=target_classes)
            for result in results:
                if result.masks is not None:
                    for mask, box in zip(result.masks.xy, result.boxes):
                        points = np.int32([mask])
                        class_id = int(box.cls[0])
                        normalized_points = [(p[0] / img.shape[1], p[1] / img.shape[0]) for p in points[0]]
                        annotation_str = f"{class_id} " + " ".join(f"{x:.8f}" for coord in normalized_points for x in coord)
                        with open(os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.txt"), "a") as f:
                            f.write(annotation_str + "\n")

# Use the returned video_path from the download function
delete_folder_contents(folder_path)
delete_folder_contents(output_dir)
delete_folder_contents(output_dir1)
downloaded_video_path = download_youtube_video(url, save_path)
extract_and_save_frames(downloaded_video_path, output_dir, (640, 640), 60)  # Use downloaded_video_path here
process_and_annotate_frames_segment(output_dir, 'yolov8m-seg.pt', output_dir1)

