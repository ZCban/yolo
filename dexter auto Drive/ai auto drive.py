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
visual=True

screenshot = 540
left, top = (1920 - screenshot) // 2, (1080 - screenshot) // 2
right, bottom = left + screenshot, top + screenshot
region = (left, top, right, bottom)
cam=bettercam.create(output_idx=0, output_color="BGR")
cam.start(region=region,video_mode=True,target_fps=60)#region=region

conf = 0.3

center_x_screenshot=screenshot//2

# Define colors for each class in BGR format
colors = {
    0: (0, 255, 255),  # Yellow f1 car
    1: (0, 0, 255),    # Red runway
    2: (255, 0, 0)     # Blue runway
}


while True:
    img = cam.get_latest_frame()

    results = model.predict(img, conf=conf, classes=[1])

    for result in results:
        if result.masks is not None:
            for mask in result.masks.xy:
                points = np.int32([mask])

                
                # Calculate the centroid of the polygon
                M = cv2.moments(points)
                #if M["m00"] != 0:
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"]) 
                #else:
                    #center_x, center_y = 0, 0                               

                # Calculate the error in the x-axis
                error_x = center_x - center_x_screenshot

                if win32api.GetKeyState(0x05)<0:
                    # Adjust the car direction based on error_x
                    if error_x > 14:  # Move to the right
                        kmNet.keydown(0x16)
                        kmNet.keyup(0x1A)
                        kmNet.keyup(0x04)
                        kmNet.keyup(0x16)
                        kmNet.keydown(0x07)
                        

                    elif error_x < -14:  # Move to the left
                        kmNet.keydown(0x16)
                        kmNet.keyup(0x1A)
                        kmNet.keyup(0x07)                        
                        kmNet.keyup(0x16)
                        kmNet.keydown(0x04)
                        


                    else:  # Center aligned, go straight
                        kmNet.keyup(0x16)
                        kmNet.keyup(0x07)
                        kmNet.keyup(0x04)
                        kmNet.keydown(0x1A)
                else:
                    kmNet.keyup(0x16)
                    kmNet.keyup(0x07)
                    kmNet.keyup(0x04)
                    kmNet.keyup(0x1A)

                    

                if visual:
                    #draw target detection model
                    cv2.fillPoly(img, [points], (0, 255, 0))
                    # Draw the centroid
                    cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)



    if visual:
        #show image 
        cv2.imshow("show", img)
        cv2.waitKey(1)



cv2.destroyAllWindows()










