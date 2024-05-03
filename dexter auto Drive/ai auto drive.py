from ultralytics import YOLO
import cv2
import numpy as np
import bettercam
import kmNet
import win32api
import time


kmNet.init('192.168.2.188','1408','9FC05414')
# Load the YOLO model
model = YOLO("best.pt")

# Camera initialization
cam = bettercam.create(output_idx=0, output_color="BGR")
cam.start(video_mode=True, target_fps=60)
conf = 0.45
desired_size = (352, 352)  # Desired size for resizing
center_x_screenshot = desired_size[0] // 2
center_y_screenshot = desired_size[1] // 2 + 50

# Define colors for each class in BGR format
colors = {
    0: (0, 255, 255),  # Yellow f1 car
    1: (0, 0, 255),    # Red runway
    2: (255, 0, 0)     # Blue runway
}


while True:
    img = cam.get_latest_frame()
    img = cv2.resize(img, desired_size)

    results = model.predict(img, conf=conf, classes=[1])

    for result in results:
        if result.masks is not None:
            for mask in result.masks.xy:
                points = np.int32([mask])
                cv2.fillPoly(img, [points], (0, 255, 0))

                # Calculate the centroid of the polygon
                M = cv2.moments(points)
                if M["m00"] != 0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
                else:
                    center_x, center_y = 0, 0

                # Draw the centroid
                cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)

                # Calculate the error in the x-axis
                error_x = center_x - center_x_screenshot
                if win32api.GetKeyState(0x14):
                    # Adjust the car direction based on error_x
                    if error_x > 1:  # Move to the right
                        kmNet.keyup(0x1A)
                        kmNet.keydown(0x07)
                        time.sleep(0.05)
                        kmNet.keyup(0x07)
                    elif error_x < -1:  # Move to the left
                        kmNet.keyup(0x1A)
                        kmNet.keydown(0x04)
                        time.sleep(0.05)
                        kmNet.keyup(0x04)

                    else:  # Center aligned, go straight
                        kmNet.keydown(0x1A)



    # Draw a red vertical line at the center of the screenshot
    #cv2.line(img, (center_x_screenshot, 0), (center_x_screenshot, img.shape[0]), (0, 0, 255), thickness=2)
    # Draw a red horizontal line at the center of the screenshot/2 + 50
    #cv2.line(img, (0, center_y_screenshot), (img.shape[1], center_y_screenshot), (0, 0, 255), thickness=2)

    # Display the image in a window
    cv2.imshow("show", img)
    cv2.setWindowProperty("show", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.waitKey(1)



cv2.destroyAllWindows()




