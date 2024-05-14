from ultralytics import YOLO
import cv2
import numpy as np
import bettercam
import kmNet
import win32api
import time

screenshot = 540
left, top = (1920 - screenshot) // 2, (1080 - screenshot) // 2
right, bottom = left + screenshot, top + screenshot
region = (left, top, right, bottom)
cam = bettercam.create(output_idx=0, output_color="BGR")
cam.start(region=region, video_mode=True, target_fps=60)
center_x = 540 // 2  # Centro dell'immagine sull'asse X

# Initialize YOLO model with pretrained weights
model = YOLO('yolov9c.pt')
kmNet.init('192.168.2.188','1408','9FC05414')
# Initialize variables for keyboard control (Use actual key codes from your keyboard library)
KEY_RIGHT = 0x07  # Change to the actual key code for right
KEY_LEFT = 0x04   # Change to the actual key code for left
KEY_STRAIGHT = 0x1A  # Change to the actual key code for straight


while True:
    # Capture image
    img = cam.get_latest_frame()
    results = model.predict(img, conf=0.5, classes=[2])

    # Initialize a list to store the coordinates of targets
    targets = []

    # Loop through the prediction results
    for result in results:
        for box in result.boxes:
            target_x = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
            target_y = int((box.xyxy[0][1] + box.xyxy[0][3]) / 2)
            target_height = int(box.xyxy[0][3] - box.xyxy[0][1])           
            targets.append((target_x, target_y, target_height))

            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])), (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), 2)
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

    # Calculate distance and direction from fixed_point to nearest target
    if len(targets) > 0:
        targets_array = np.array(targets)
        distances = np.linalg.norm(targets_array[:, :2] - center, axis=1)
        nearest_index = np.argmin(distances)
        nearest_distance = distances[nearest_index]
        nearest_target = targets[nearest_index]
        nearest_target_center = np.array(nearest_target[:2])

        # Calculate horizontal error between fixed point and nearest target center
        error_x = nearest_target_center[0] - fixed_point[0]

        # Check for right mouse button press to adjust car direction
        if win32api.GetKeyState(0x05) < 0:
            if error_x > 8:
                kmNet.keyup(KEY_LEFT)
                kmNet.keydown(KEY_RIGHT)
            elif error_x < -8:
                kmNet.keyup(KEY_RIGHT)
                kmNet.keydown(KEY_LEFT)
            else:
                kmNet.keyup(KEY_RIGHT)
                kmNet.keyup(KEY_LEFT)
                kmNet.keydown(KEY_STRAIGHT)
        else:
            kmNet.keyup(KEY_RIGHT)
            kmNet.keyup(KEY_LEFT)
            kmNet.keyup(KEY_STRAIGHT)

    # Display the image
    cv2.imshow('Result Image', img)
    cv2.waitKey(1)

cv2.destroyAllWindows()
