from ultralytics import YOLO
import cv2
import numpy as np
import bettercam

import win32api
import time



# Load the YOLO model
model = YOLO("best.pt")
visual = True

# Set the dimensions for the screenshot
screenshot = 350
left, top = (1280 - screenshot) // 2, (1024 - screenshot) // 2
right, bottom = left + screenshot, top + screenshot
region = (left, top, right, bottom)

# Create and start the camera
cam = bettercam.create(output_idx=0, output_color="BGR")
cam.start(region=region, video_mode=True, target_fps=60)

# Set the confidence threshold for detection
conf = 0.3

# Center of the screenshot
center_x_screenshot = screenshot // 2

# Define colors for each class in BGR format
colors = {
    0: (0, 255, 255),  # Yellow f1 car
    1: (0, 0, 255),    # Red runway
    2: (255, 0, 0)     # Blue runway
}

while True:
    # Capture the latest frame from the camera
    img = cam.get_latest_frame()

    # Run the YOLO model on the frame
    results = model.predict(img, conf=conf, classes=[1],max_det=1)

    # Process the results
    for result in results:
        for box in result.boxes:
            # Extract the coordinates of the bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Calculate the error in the x-axis
            error_x = center_x - center_x_screenshot



            if visual:
                # Draw the bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Draw the centroid
                cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)

    if visual:
        # Show the image
        cv2.imshow("show", img)
        cv2.waitKey(1)

cv2.destroyAllWindows()
