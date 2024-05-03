import bettercam
import numpy as np
import cv2
from ultralytics import YOLO
import kmNet
import win32api
import random
import string
import os
import time

screenshot = 640

center = screenshot // 2
left, top = (1920 - screenshot) // 2, (1080 - screenshot) // 2
right, bottom = left + screenshot, top + screenshot
region = (left, top, right, bottom)
cam=bettercam.create(output_idx=0, output_color="BGR")
cam.start(region=region,video_mode=True,target_fps=60)#region=region

# Initialize YOLOv9c model with pretrained weights
#model = YOLO('yolov9c.pt')
# Initialize YOLOv8n model with pretrained weights
#model = YOLO('yolov8n.pt')
model = YOLO('best.pt')

kmNet.init('192.168.2.188','1408','9FC05414')



def resize_image(img, new_size):
    return cv2.resize(img, new_size)

def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)
    return results

def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
    results = predict(chosen_model, img, classes, conf=conf)
    for result in results:
        for box in result.boxes:
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
    return img, results


def find_nearest_target(chosen_model, img, classes=0, conf=0.4, move_threshold=5):
    results = predict(chosen_model, img, classes, conf=conf)
    
    # Inizializza una lista per memorizzare le coordinate dei target
    targets = []
    
    # Loop attraverso i risultati della predizione
    for result in results:
        for box in result.boxes:
            # Calcola il centro del rettangolo
            target_x = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
            target_y = int((box.xyxy[0][1] + box.xyxy[0][3])/2 )
            # Calcola l'altezza del rettangolo
            target_height = int(box.xyxy[0][3] - box.xyxy[0][1])
            
            targets.append((target_x, target_y, target_height))
    
    if not targets:
        return None  # Se non ci sono bersagli, restituisci None
    
    targets_array = np.array(targets)
    
    # Calcola le distanze euclidee tra i bersagli e il centro
    distances = np.linalg.norm(targets_array[:, :2] - center, axis=1)
    
    # Trova l'indice del bersaglio più vicino
    nearest_index = np.argmin(distances)
    nearest_distance = distances[nearest_index]

    nearest_target = targets[nearest_index]
    delta_x =int( nearest_target[0] - center)
    delta_y = int(nearest_target[1] - center)
    delta_y -= int(nearest_target[2]  / 4.1)

    if win32api.GetKeyState(0x05)<0:
        kmNet.move_auto(delta_x, delta_y, 2)
        if -5 <= delta_x <=5 and -5 <= delta_y <= 5:
            kmNet.left(1)
            kmNet.move(delta_x, delta_y)
            kmNet.left(0)

def find_rightmost_target(chosen_model, img, classes=0, conf=0.4, move_threshold=5):
    results = predict(chosen_model, img, classes, conf=conf)
    
    # Inizializza una lista per memorizzare le coordinate dei target
    targets = []
    
    # Loop attraverso i risultati della predizione
    for result in results:
        for box in result.boxes:
            # Calcola il centro del rettangolo
            target_x = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
            target_y = int((box.xyxy[0][1] + box.xyxy[0][3]) / 2)
            # Calcola l'altezza del rettangolo
            target_height = int(box.xyxy[0][3] - box.xyxy[0][1])          
            targets.append((target_x, target_y, target_height))
    
    if not targets:
        return None  # Se non ci sono bersagli, restituisci None
    
    targets_array = np.array(targets)
    
    # Trova il bersaglio più a destra
    rightmost_index = np.argmax(targets_array[:, 0])  # Argmax sulle coordinate x
    rightmost_target = targets[rightmost_index]
    
    delta_x = int(rightmost_target[0] - center)
    delta_y = int(rightmost_target[1] - center)
    delta_y -= int(rightmost_target[2] / 4.1)
    
    if win32api.GetKeyState(0x05) < 0:
        kmNet.move_auto(delta_x, delta_y, 2)
        if -5 <= delta_x <= 5 and -5 <= delta_y <= 5:
            kmNet.left(1)
            kmNet.move(delta_x, delta_y)
            kmNet.left(0)


while True:
    # Capture image
    img = cam.get_latest_frame()

    #aim
    #find_nearest_target(model, img, classes=[0,], conf=0.48)
    #find_rightmost_target(model, img, classes=[0,], conf=0.48)

    # Perform detection
    results = predict(model, img, classes=[], conf=0.5)
    # Draw segmentation masks without bounding boxes
    result_img = draw_segmentation(img, results)
    cv2.imshow('YOLO Segmentation', result_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


#cv2.destroyAllWindows()
