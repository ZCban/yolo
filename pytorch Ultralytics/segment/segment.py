from ultralytics import YOLO
import cv2
import numpy as np
import bettercam
import torch

model = YOLO("best.pt")  # Load the model

screenshot =1080
left, top = (1920 - screenshot) // 2, (1080 - screenshot) // 2
right, bottom = left + screenshot, top + screenshot
region = (left, top, right, bottom)
cam=bettercam.create(output_idx=0, output_color="BGR")
cam.start(region=region,video_mode=True,target_fps=60)#region=region

# Define colors for each class in BGR format
colors = {0: (0, 255, 255),  # Yellow f1 car
          1: (0, 0, 255),# Red runway
          2: (255, 0, 0)}# Blue runway

conf = 0.5
desired_size = (352, 352)  # Desired size for resizing

# Use CUDA streams for asynchronous processing
stream = torch.cuda.Stream()
while True:
    img = cam.get_latest_frame()
    with torch.cuda.stream(stream):
        results = model.predict(img, conf=conf)
        # Draw polygons based on detection results
        for result in results:
            if result.masks is not None:  # Check for masks, which contain polygon points
                for mask, box in zip(result.masks.xy, result.boxes):
                    points = np.int32([mask])
                    class_id = int(box.cls[0])
                    cv2.fillPoly(img, [points], colors[class_id])  # Fill polygon with the class-specific color
        cv2.imshow("Image", img)
        cv2.waitKey(1)

cv2.destroyAllWindows()
