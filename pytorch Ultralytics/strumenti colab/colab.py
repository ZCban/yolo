
#################################
#install module
!pip install pytube ultralytics youtube_search
#clone reposity
!git clone https://github.com/ZCban/yolov9.git

#################################

import cv2
import os
from pytube import YouTube
from youtube_search import YoutubeSearch
from ultralytics import YOLO
import numpy as np
import glob
from google.colab import drive
import random
import shutil


#mount drive path
drive.mount('/content/drive')
#setting
model = YOLO('/content/yolov9/640.pt')
folder_path = "/content/1"
url = "https://www.youtube.com/watch?v=6a6VLR0OdQ4"
titolo ='valorant pvp'
save_path = "/content/1"
drive_folder_path = "/content/drive/My Drive/colab"
frame_rate = 15  # Numero di frame da saltare prima di salvare il successivo
target_size = (640, 640)  # Dimensione target per il ridimensionamento dei frame


###############################
#cheack and delete old result
# Verifica se la cartella esiste
if not os.path.exists(folder_path):
    print(f"La cartella {folder_path} non esiste.")
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

########################################
#dowload url video
#risultati = YoutubeSearch(titolo, max_results=1).to_dict()
# Ottieni l'URL del primo video
#url = "https://www.youtube.com" + risultati[0]['url_suffix']

yt = YouTube(url)
video = yt.streams.get_highest_resolution()
video.download(save_path)
video_path = os.path.join(save_path, video.default_filename)


#########################
#from video to frame
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
        # Ritaglia il frame al centro
        cropped_frame = frame[y:y+frame_size, x:x+frame_size]
        # Ridimensiona il frame alla dimensione desiderata
        resized_frame = cv2.resize(cropped_frame, target_size)
        frame_name = os.path.join(save_path, f"frame_{count}.jpg")
        cv2.imwrite(frame_name, resized_frame)
        print(f"Frame salvato: {frame_name}")

    count += 1

cap.release()
#########################
#remove no detection + know target

# Process each saved frame
for filename in os.listdir(save_path):
    if filename.endswith(".jpg"):
        img = cv2.imread(os.path.join(save_path, filename))  # Read the image
        results = model.predict(img, classes=[0,], conf=0.35)  # Perform object detection

        # Initialize a list to store target coordinates and confidences
        targets = []

        # Loop through the prediction results
        for result in results:
            for box in result.boxes:
                # Calculate the center of the bounding box
                target_x = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
                target_y = int((box.xyxy[0][1] + box.xyxy[0][3]) / 2)
                targets.append((target_x, target_y, box.conf))

        # Check if there are no targets detected or all targets have confidence > 0.50
        if not targets or all(conf > 0.50 for (_, _, conf) in targets):
            os.remove(os.path.join(save_path, filename))  # Delete the image

###################################################
#count img remaining + delete video dowloaded

# Verifica se la cartella esiste
if not os.path.exists(folder_path):
    print(f"La cartella {folder_path} non esiste.")
else:
    # Ottieni la lista dei file nella cartella
    files = os.listdir(folder_path)

    # Elimina i file .mp4
    for file in files:
        if file.endswith(".mp4"):
            file_path = os.path.join(folder_path, file)
            os.remove(file_path)
            print(f"Il file {file} è stato eliminato.")

    print("Eliminazione completata.")


#rename evry img
if not os.path.exists(folder_path):
    print(f"La cartella {folder_path} non esiste.")
else:
    files = os.listdir(folder_path)
    image_files = [file for file in files if file.endswith(".jpg")]

    for i, file in enumerate(image_files):
        random_suffix = str(random.randint(0, 99999999)).zfill(8)  # Genera un numero casuale di 8 cifre
        new_name = f"imgcollab{random_suffix}.jpg"
        old_path = os.path.join(folder_path, file)
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)
        print(f"File rinominato: {file} -> {new_name}")

    print("Rinominazione completata.")

# Conta le immagini rimanenti nella cartella
remaining_images = len([name for name in os.listdir(save_path) if name.endswith(".jpg")])
print(f"Numero di immagini rimanenti: {remaining_images}")

##########################################################
#move to drive

# Check if the source folder exists
if not os.path.exists(drive_folder_path):
    print(f"The folder {folder_path} does not exist.")
    # Create a folder in the root directory
    !mkdir -p "/content/drive/My Drive/colab"

# Check if the source folder exists
if not os.path.exists(folder_path):
    print(f"The folder {folder_path} does not exist.")
else:
    # Move each image to the destination folder
    for file_name in os.listdir(folder_path):
        # Construct paths
        source_path = os.path.join(folder_path, file_name)
        destination_path = os.path.join(drive_folder_path, file_name)
        
        # Move the file
        shutil.move(source_path, destination_path)
        print(f"Moved: {file_name} to {destination_path}")

    print("Move operation completed.")


