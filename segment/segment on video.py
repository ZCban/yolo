from ultralytics import YOLO
import cv2
import numpy as np
import random

# Load YOLO model
model = YOLO("yolov8x-seg.pt")
conf = 0.4

# Open a video file
cap = cv2.VideoCapture(r'C:\Users\ui\Desktop\yolov9-main\1.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Predict using YOLO
    results = model.predict(frame, conf=conf, classes=[0,])  # Adjust class indices as needed
    colors = [random.choices(range(256), k=3) for _ in range(len(model.names))]

    # Draw masks and bounding boxes
    for result in results:
        if result.masks is not None:
            for mask, box in zip(result.masks.xy, result.boxes):
                points = np.int32([mask])
                color_number = int(box.cls[0])
                cv2.fillPoly(frame, points, colors[color_number])

    # Write the frame with detections
    out.write(frame)

# Release everything when job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
