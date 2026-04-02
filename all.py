import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import seaborn as sns
import winsound  # For Windows beep sound (use os module for macOS/Linux)


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
# Initialize video capture
cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print("Error: Could not open video capture device.")
    exit()


y_true = []
y_pred = []
detection_counts = []
false_positive_counts = []


def run_inference(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))  
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

    
    fire_detected = False
    detection_count = 0
    false_positive_count = 0

  
    for i, det in enumerate(pred):
        annotator = Annotator(frame, line_width=2, example=str(names))
        if len(det):
            for *xyxy, conf, cls in reversed(det):
              
                x1, y1, x2, y2 = map(int, xyxy)
                w, h = x2 - x1, y2 - y1
                
               
                box_area = w * h
                if box_area < 1000:  
                    continue  # Skip small detections

         
                detection_count += 1

                
                if names[int(cls)] == 'fire':  
                    fire_detected = True
                else:
                    false_positive_count += 1

           
                label = f'{names[int(cls)]} {conf:.2f}'
                annotator.box_label([x1, y1, x2, y2], label, color=colors(cls))
        frame = annotator.result()

    
    if fire_detected:
        winsound.Beep(1000, 500)  

 
    detection_counts.append(detection_count)
    false_positive_counts.append(false_positive_count)

   
    true_label = 1 if fire_detected else 0
    y_true.append(true_label)

 
    y_pred.append(1 if fire_detected else 0)


    cv2.imshow('Fire Detection', frame)

  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()


conf_matrix = confusion_matrix(y_true, y_pred)
TN, FP, FN, TP = conf_matrix.ravel()
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)


print("Confusion Matrix:\n", conf_matrix)
print(f"True Negatives (TN): {TN}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")
print(f"True Positives (TP): {TP}")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")


with open('metrics.txt', 'w') as f:
    f.write("Confusion Matrix:\n")
    f.write(f"{conf_matrix}\n")
    f.write(f"True Negatives (TN): {TN}\n")
    f.write(f"False Positives (FP): {FP}\n")
    f.write(f"False Negatives (FN): {FN}\n")
    f.write(f"True Positives (TP): {TP}\n")
    f.write(f"Accuracy: {accuracy:.2f}\n")
    f.write(f"Precision: {precision:.2f}\n")
    f.write(f"Recall: {recall:.2f}\n")


sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(range(len(detection_counts)), detection_counts, label='Detections', color='blue')
plt.plot(range(len(false_positive_counts)), false_positive_counts, label='False Positives', color='red', linestyle='--')
plt.xlabel('Frame Number')
plt.ylabel('Count')
plt.title('Fire Detection and False Positives Over Time')
plt.legend()


plt.savefig('fire_detection_graph.png')


plt.show()
