from ultralytics import YOLOv10
import multiprocessing


    
# Load a model
model = YOLOv10('best.pt')

if __name__ == '__main__':
    # Train the model
    #results = model.train(data=r'C:\Users\ui\Desktop\dataset\aim-dataset\coco.yaml', epochs=50, batch=-1,imgsz=512,val=False,amp=True,save_period=-1)
    # Export the model to ONNX format
    model.export(format='onnx')

