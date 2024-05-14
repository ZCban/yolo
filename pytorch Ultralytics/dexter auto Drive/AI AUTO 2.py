from ultralytics import YOLO
import cv2
import numpy as np
import bettercam
import kmNet
import win32api
import time

ddzoneleftright=30


ddzoceaccelletor=100
ddzonestop=100

kmNet.init('192.168.2.188','1408','9FC05414')
# Load the YOLO model
model = YOLO("best.pt")
visual=True



screenshotx = 1080
screenshoty = 1080

left, top = (1920 - screenshotx) // 2, (1080 - screenshoty)
right, bottom = left + screenshotx, top + screenshoty
region = (left, top, right, bottom)
cam=bettercam.create(output_idx=0, output_color="BGR")
cam.start(region=region,video_mode=True,target_fps=60)#region=region

conf = 0.4

center_x_screenshot=screenshotx //2
center_y_screenshot=screenshoty //2



# Define colors for each class in BGR format
colors = {
    0: (0, 255, 255),  # Yellow f1 car
    1: (0, 0, 255),    # Red runway
    2: (255, 0, 0)     # Blue runway
}

while True:
    img = cam.get_latest_frame()
    results = model.predict(img, conf=conf, classes=[1])

    turning = False  # Flag to track if turning

    for result in results:
        if result.masks is not None:
            for mask in result.masks.xy:
                points = np.int32([mask])
                
                # Calculate the centroid of the polygon
                M = cv2.moments(points)
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])+20

                # Calculate the error in the x-axis
                error_x = center_x - center_x_screenshot
                error_y = center_y - center_y_screenshot

                if win32api.GetKeyState(0x05) < 0:
                    # Adjust the car direction based on error_x
                    if error_x > ddzoneleftright:  # Move to the right
                        kmNet.keyup(0x1A)
                        kmNet.keyup(0x04)
                        kmNet.keydown(0x07)
                        turning = True  # Set turning flag

                    elif error_x < -ddzoneleftright:# Move to the left
                        kmNet.keyup(0x07)
                        kmNet.keyup(0x1A)
                        kmNet.keydown(0x04)
                        turning = True  # Set turning fla

                    #if :
                    if  error_x < -ddzoceaccelletor and error_y < ddzoceaccelletor:
                        kmNet.keyup(0x16)
                        kmNet.keyup(0x07)
                        kmNet.keyup(0x04)
                        kmNet.keydown(0x1A)                           

                    # Center aligned, go straight or backward, only if not turning
                    if not turning and error_y < -ddzonestop:  # Go backward
                        kmNet.keyup(0x1A)
                        kmNet.keyup(0x07)
                        kmNet.keyup(0x04)
                        kmNet.keydown(0x16)
                        time.sleep(0.05)
                        kmNet.keyup(0x16)
                        
                else:
                    kmNet.keyup(0x16)
                    kmNet.keyup(0x07)
                    kmNet.keyup(0x04)
                    kmNet.keyup(0x1A)
                if visual:
                    # Draw the target detection model
                    cv2.fillPoly(img, [points], (0, 255, 0))
                    # Draw the centroid
                    cv2.circle(img, (center_x, center_y), 5, (255, 0, 255), -1)
                    # Draw a red line from the center of the screenshot to the centroid
                    #cv2.line(img, ((center_x_screenshot-150), 1920), (center_x, center_y), (0, 0, 255), 5)
                    #cv2.line(img, ((center_x_screenshot+150), 1920), (center_x, center_y), (0, 0, 255), 5)

    if visual:
        # Define the window name
        window_name = "Top Window"
        # Create the window using the namedWindow function before setting properties
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # Try to set the window to be topmost using the cv2.setWindowProperty
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        # Show the image in the window
        cv2.imshow(window_name, img)
        # Wait for a key press for 1 millisecond
        cv2.waitKey(1)

cv2.destroyAllWindows()
