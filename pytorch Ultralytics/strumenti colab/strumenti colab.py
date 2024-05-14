# install module
!pip install pytube ultralytics youtube_search

import os
import glob
from pytube import YouTube
import cv2
import numpy as np
from ultralytics import YOLO
from youtube_search import YoutubeSearch
import shutil
from datetime import datetime
import zipfile
import random
import string

###################setting###################
folder_path = "/content/1"
url = "https://www.youtube.com/watch?v=55VeeYLOM_k&t=280s"
save_path = "/content/1"
frame_rate = 60  # Numero di frame da saltare prima di salvare il successivo
target_size = (640, 640)  # Dimensione target per il ridimensionamento dei frame
output_dir = "/content/images"  # Output directory for frames
output_dir1 = "/content/labels"  # Output directory for annotations
base_path ="/content"
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
            # Genera un nome file casuale di 8 caratteri alfanumerici
            random_name = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
            frame_name = os.path.join(save_path, f"{random_name}.jpg")
            cv2.imwrite(frame_name, resized_frame)
            print(f"Frame salvato: {frame_name}")

        count += 1
    cap.release()



def process_and_annotate_frames_segment(save_path, model_path, output_dir, conf_threshold=0.5, target_classes=[2]):
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

def clean_up_video(folder_path):
    if not os.path.exists(folder_path):
        print(f"La cartella {folder_path} non esiste.")
        return 0  # No files to count if the folder doesn't exist

    files = os.listdir(folder_path)
    count_images = 0  # Counter for remaining image files

    # Process files in the directory
    for file in files:
        file_path = os.path.join(folder_path, file)
        if file.endswith(".mp4"):
            os.remove(file_path)
            print(f"Il file {file} è stato eliminato.")

    print("Eliminazione completata.")
    return count_images

def delete_unlabeled_images(image_dir, label_dir):
    # Collect all label filenames without extension
    label_files = {os.path.splitext(label)[0] for label in os.listdir(label_dir) if label.endswith('.txt')}

    # Iterate over all image files in the image directory
    for image in os.listdir(image_dir):
        if image.endswith((".png", ".jpg", ".jpeg")):
            image_base = os.path.splitext(image)[0]
            # Check if the base name of the image file has a corresponding label file
            if image_base not in label_files:
                image_path = os.path.join(image_dir, image)
                os.remove(image_path)
                print(f"Il file {image} è stato eliminato perché non ha un file label corrispondente.")

def prepare_and_zip(output_dir, output_dir1, base_path):
    # Create the 'train' directory if it does not exist
    train_path = os.path.join(base_path, 'train')
    if not os.path.exists(train_path):
        os.makedirs(train_path)
        print(f"Created directory: {train_path}")

    # Move the specified directories into the 'train' directory
    for directory in [output_dir, output_dir1]:
        destination_path = os.path.join(train_path, os.path.basename(directory))
        if os.path.exists(directory):
            shutil.move(directory, destination_path)
            print(f"Moved: {directory} to {destination_path}")
        else:
            print(f"The directory {directory} does not exist.")

    # Create a zip file of the 'train' directory with the current time as part of the name
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    zip_filename = os.path.join(base_path, f"manual_edit_{current_time}.zip")
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(train_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, os.path.join(train_path, '..')))
    print(f"Created zip file: {zip_filename}")




# Use the returned video_path from the download function
delete_folder_contents(folder_path)
delete_folder_contents(output_dir)
delete_folder_contents(output_dir1)
downloaded_video_path = download_youtube_video(url, save_path)
extract_and_save_frames(downloaded_video_path, output_dir, (640, 640), 60)
process_and_annotate_frames_segment(output_dir, 'yolov8m-seg.pt', output_dir1)
clean_up_video(save_path)
delete_unlabeled_images(output_dir, output_dir1)
prepare_and_zip(output_dir, output_dir1, base_path)
