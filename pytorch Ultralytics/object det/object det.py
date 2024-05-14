import bettercam
import numpy as np
import cv2
from ultralytics import YOLO

screenshot = 440
left, top = (1920 - screenshot) // 2, (1080 - screenshot) // 2
right, bottom = left + screenshot, top + screenshot
region = (left, top, right, bottom)
cam=bettercam.create(output_idx=0, output_color="BGR")
cam.start(region=region,video_mode=True,target_fps=60)#region=region
center=screenshot/2

# Initialize YOLOv9c model with pretrained weights
model = YOLO('yolov9c.pt')


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


while True:
    # Capture image
    img = cam.get_latest_frame()

    # Dysplay Image
    result_img,_= predict_and_detect(model, img, classes=[0,], conf=0.5)
    cv2.imshow('YOLO Detection', result_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
