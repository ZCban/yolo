import cv2
import os
import shutil
import pytube
from tqdm import tqdm
import torch
from PIL import Image
from ultralytics import YOLO
import ssl

video1_name = 'valorant'
num_videos = 1
# Dimensione desiderata per il ridimensionamento dei frame
resize_dimension = 640 #dimesion for  cropped image
saveimageevrynumframe=10 #save evry 10 frames
ssl._create_default_https_context = ssl._create_unverified_context

model = YOLO('yolov8n.pt')


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


def mantieni0_25():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(script_dir, "frames_extracted")

    for filename in tqdm(os.listdir(input_folder)):
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path)#.convert("RGB")
        results = model.predict(image, classes=[0,], conf=0.25)
        targets = []
        # Loop attraverso i risultati della predizione
        for result in results:
            for box in result.boxes:
                # Calcola il centro del rettangolo
                target_x = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
                target_y = int((box.xyxy[0][1] + box.xyxy[0][3]) / 2)
                targets.append((target_x, target_y))

        if not targets:
            os.remove(image_path)

def elimino0_50():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(script_dir, "frames_extracted")

    for filename in tqdm(os.listdir(input_folder)):
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path)#.convert("RGB")
        results = model.predict(image, classes=[0,], conf=0.48)
        targets = []
        # Loop attraverso i risultati della predizione
        for result in results:
            for box in result.boxes:
                # Calcola il centro del rettangolo
                target_x = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
                target_y = int((box.xyxy[0][1] + box.xyxy[0][3]) / 2)
                targets.append((target_x, target_y))

        if  targets:
            os.remove(image_path)

def mantieni0_25SEG():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(script_dir, "frames_extracted")

    for filename in tqdm(os.listdir(input_folder)):
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path)#.convert("RGB")
        results = model.predict(image, classes=[0,], conf=0.25)
        targets = []
        # Loop attraverso i risultati della predizione
        for result in results:
            if result.masks is not None:
                os.remove(image_path)
                

def elimino0_50SEG():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(script_dir, "frames_extracted")

    for filename in tqdm(os.listdir(input_folder)):
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path)#.convert("RGB")
        results = model.predict(image, classes=[1,], conf=0.51)
        targets = []
        # Loop attraverso i risultati della predizione
        for result in results:
            if result.masks is not None:
                os.remove(image_path)
                
while True:
    print('cercando video')
    findvideo()
    print('catturo frame')
    video_to_frame(target_size=resize_dimension, frames_to_skip=saveimageevrynumframe)
    print('0.35')
    mantieni0_35()
    print('0-50')
    elimino0_50()




