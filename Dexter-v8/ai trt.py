import time
import numpy as np
import torch
from utils.utils_trt import TRTModule,Utility
import kmNet
#from mss import mss
from rich import print
import sys
import cv2
#import win32api
import os
#from win32api import GetSystemMetrics
#import ctypes
from ScreenCapture import ScreenCaptureMSS,ScreenCaptureBETTERCAM
import threading
import queue




# Configurazioni iniziali
screenshot = 512
screeneginegpu = True
antirecoil = True
countfps = True
visual = False
data = False
model_file = 'best32.trt'
min_conf = 0.48
center = screenshot / 2
#centerx, centery = center, center


# Crea una coda per le immagini da salvare
#image_queue = queue.Queue()

utility = Utility()
utility.check_cuda_device()

kmNet.init('192.168.2.188','1408','9FC05414')
kmNet.monitor(5001)
# Ottiene il percorso della directory in cui si trova lo script corrente
current_dir = os.path.dirname(os.path.abspath(__file__))
models_path = os.path.join(current_dir, 'models')

model_path = os.path.join(models_path, model_file)


if screeneginegpu:
    screen_capture = ScreenCaptureBETTERCAM(screenshot)
else:
    screen_capture = ScreenCaptureMSS(screenshot)

# Load model
model = TRTModule(model_path, device=0)

# Inizializza una variabile di conteggio fps
# Variabili per il calcolo degli FPS
fps = 0
frame_count = 0
start_time = time.time()


while True:
    img = screen_capture.capture()

    if img is not None:
        tensor = torch.as_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)[np.newaxis, ...], dtype=np.float32) / 255.0, device='cuda')
        data = model(tensor)
        num_dets, bboxes, scores, labels = (i[0] for i in data)
        selected = (scores >= min_conf)

        bboxes_selected = bboxes[selected].cpu().numpy()
        #scores_selected = scores[selected].cpu().numpy()  # Move to CPU and convert to NumPy
        #labels_selected = labels[selected].cpu().numpy()  # Move to CPU and convert to NumPy

        targets = []
        for x1, y1, x2, y2 in bboxes_selected:
            adjusted_x = (x1 + x2) / 2 - center
            #adjusted_y = (y1  + y2)/2 - center #for center
            adjusted_y = y1 + (y2 - y1) / 3 - center #for bust
            #if -fovaim <= adjusted_x <= fovaim and -fovaim <= adjusted_y <= fovaim:
            targets.append((adjusted_x, adjusted_y))

        targets_array = np.array(targets)
        if len(targets_array) > 0:
            dist_from_center = np.linalg.norm(targets_array, axis=1)
            min_dist_idx = np.argmin(dist_from_center)
            current_target = targets_array[min_dist_idx]
            step_x = int(current_target[0]/2.9)
            step_y = int(current_target[1]/2.9)

            #if win32api.GetKeyState(0x05)<0 :
            if kmNet.isdown_side1()==1:
                kmNet.move(step_x, step_y)



        if visual:
            # Draw bounding boxes on the image
            for bbox in bboxes_selected:
                x1, y1, x2, y2 = bbox
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                cx1,cy1=(x1 + x2) / 2 , y1 + (y2 - y1) / 3
                w, h = x2 - x1, y2 - y1
                top_left = int(cx - w / 2), int(cy - h / 2)
                bottom_right = int(cx + w / 2), int(cy + h / 2)
                cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
                # Draw a yellow circle at the center of the bounding box
                cv2.circle(img, (int(cx), int(cy1)), 5, (0, 255, 255), -1)  # Yellow (BGR: 0,255,255)

            # Display the image with detected objects and the target point
            cv2.imshow("Object Detection", img)
            cv2.waitKey(1)

        if countfps:
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 1:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
                print(f"FPS: {fps:.2f}")

        if antirecoil and  kmNet.isdown_left()==1 and kmNet.isdown_right()==1 :
            kmNet.move(int(0), int(4))










