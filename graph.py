import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt  # Import for graph plotting
from pathlib import Path


yolov5_path = 'A:/firedetect/yolov5'
if yolov5_path not in sys.path:
    sys.path.append(yolov5_path)


from models.common import DetectMultiBackend
from utils.general import (check_img_size, non_max_suppression)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device

model_path = 'A:/firedetect/yolov5/runs/train/exp3/weights/best.pt'
device = select_device('')
model = DetectMultiBackend(model_path, device=device)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size(640, s=stride)  # check image size


cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print("Error: Could not open video capture device.")
    exit()


detection_counts = []
false_positive_counts = []


def run_inference(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, axis=0)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if model.fp16 else img.float()  
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    pred = model(img)
    pred = non_max_suppression(pred, 0.25, 0.45, agnostic=False)
    return pred

while True:
    ret, frame = cap.read()
    if not ret:
        break

  
    pred = run_inference(frame)

 
    detection_count = 0
    false_positive_count = 0

   
    for i, det in enumerate(pred): 
        annotator = Annotator(frame, line_width=2, example=str(names))
        if len(det):
            for *xyxy, conf, cls in reversed(det):
                # Convert xyxy to xywh
                x1, y1, x2, y2 = map(int, xyxy)
                w, h = x2 - x1, y2 - y1
                
              
                box_area = w * h
                if box_area < 1000:  # Adjust this threshold as needed
                    continue  # Skip small detections

             
                detection_count += 1

                
                if names[int(cls)] != 'fire':
                    false_positive_count += 1

             
                label = f'{names[int(cls)]} {conf:.2f}'
                annotator.box_label([x1, y1, x2, y2], label, color=colors(cls))
        frame = annotator.result()


    detection_counts.append(detection_count)
    false_positive_counts.append(false_positive_count)

   
    cv2.imshow('Fire Detection', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

plt.figure(figsize=(12, 6))
plt.plot(range(len(detection_counts)), detection_counts, label='Detections', color='blue')
plt.plot(range(len(false_positive_counts)), false_positive_counts, label='False Positives', color='red', linestyle='--')
plt.xlabel('Frame Number')
plt.ylabel('Count')
plt.title('Fire Detection and False Positives Over Time')
plt.legend()


plt.savefig('fire_detection_graph.png')


plt.show()
