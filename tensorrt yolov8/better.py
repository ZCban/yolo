import time
import numpy as np
import torch
from utils.utils_trt import TRTModule,Utility
import kmNet
import bettercam
from rich import print
import sys
import cv2
import win32api
import os

def calculate_movespeed(fps_limit, reference_fps=90, reference_movespeed=2):
    return (reference_movespeed / reference_fps) * fps_limit

screenshot = 448
center = screenshot/2
model_file = '448.engine'
min_conf = 0.50
fps_limit=60
movespeed=calculate_movespeed(fps_limit, reference_fps=90, reference_movespeed=2)
visual = False
fpscount=True

current_target = None


utility = Utility()
utility.check_cuda_device()

kmNet.init('192.168.2.188','1408','9FC05414')

# Ottiene il percorso della directory in cui si trova lo script corrente
current_dir = os.path.dirname(os.path.abspath(__file__))
models_path = os.path.join(current_dir, 'models')

model_path = os.path.join(models_path, model_file)
print(model_path)

left, top = (1920 - screenshot) // 2, (1080 - screenshot) // 2
right, bottom = left + screenshot, top + screenshot
region = (left, top, right, bottom)
cam=bettercam.create(output_idx=0, output_color="BGR")
cam.start(region=region,video_mode=True,target_fps=fps_limit)

# Load model
model = TRTModule(model_path, device=0)

# Inizializza una variabile di conteggio fps
fps_counter = 0
start_time = time.time()



while True:   
    img = cam.get_latest_frame()  # Cattura lo screenshot


    tensor = np.ascontiguousarray(img.transpose(2, 0, 1)[np.newaxis, ...], dtype=np.float32) / 255.0
    tensor = torch.as_tensor(tensor, device='cuda')
    data = model(tensor)
    bboxes = utility.det_postprocess6(data, confidence_threshold=min_conf)

    targets_data = bboxes[:, :4]
    
    if len(targets_data) > 0  : 
        #dist_from_center = np.sqrt(np.sum((targets_data[:, :1] - center)**2, axis=1))
        dist_from_center = targets_data[:, 0] - center#) + np.abs(targets_data[:, 1] - center)
        min_dist_idx = np.argmin(dist_from_center)
        current_target = targets_data[min_dist_idx]
        delta_x = current_target[0] - center
        delta_y = current_target[1] - center
        delta_y -= current_target[3]/2.95
        
        if win32api.GetKeyState(0x05)<0:
            kmNet.move(int(delta_x/movespeed),int(delta_y/movespeed))
            if -2.2 <= delta_x/movespeed <=2.2 and -1 <= delta_y/movespeed <= 1:
                kmNet.left(1)
                kmNet.left(0)

    if visual:
        utility.draw_visuals(img, bboxes)

    if fpscount:
        utility.count_fps()

    

