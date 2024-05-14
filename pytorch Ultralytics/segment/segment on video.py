from ultralytics import YOLO
import cv2
import numpy as np
import random

# Load YOLO model
model = YOLO("best.pt")
conf = 0.5

# Open a video file
cap = cv2.VideoCapture(r'C:\Users\ui\Desktop\ghg\old backup\Pyt\yolov9-main\segment\1.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
# Define colors for each class in BGR format
colors = {
    0: (0, 255, 255),  # Yellow f1 car
    1: (0, 0, 255),    # Red runway
    2: (255, 0, 0)     # Blue runway
}
# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Predict using YOLO
    results = model.predict(frame, conf=conf)  # Adjust class indices as needed

    # Draw masks and bounding boxes
    for result in results:
        if result.masks is not None:
            for mask, box in zip(result.masks.xy, result.boxes):
                points = np.int32([mask])
                class_id = int(box.cls[0])
                cv2.fillPoly(frame, points, colors[class_id])

    # Write the frame with detections
    out.write(frame)

# Release everything when job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
