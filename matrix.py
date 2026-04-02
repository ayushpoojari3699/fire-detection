import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt


yolov5_path = 'A:/firedetect/yolov5'
if yolov5_path not in sys.path:
    sys.path.append(yolov5_path)

from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
from utils.torch_utils import select_device


model_path = 'A:/firedetect/yolov5/runs/train/exp3/weights/best.pt'
device = select_device('')  # Automatically select CPU or GPU
model = DetectMultiBackend(model_path, device=device, dnn=False)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size(640, s=stride)  # Ensure image size is compatible with model

cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print("Error: Could not open video capture device.")
    exit()


y_true = []
y_pred = []


def run_inference(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
   
    img = cv2.resize(img, (imgsz, imgsz))  
    
   
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if model.fp16 else img.float()  # uint8 to fp16/32
    img /= 255.0  # Normalize to 0-1

    pred = model(img, augment=False)
    

    pred = non_max_suppression(pred, 0.25, 0.45, agnostic=False)
    return pred

while True:
    ret, frame = cap.read()
    if not ret:
        break


    pred = run_inference(frame)


    fire_detected = False
    annotator = Annotator(frame, line_width=2, example=str(names))
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes to original image size
            det[:, :4] = scale_boxes(imgsz, det[:, :4], frame.shape).round()

            for *xyxy, conf, cls in reversed(det):
                if names[int(cls)] == 'fire':  # Assuming 'fire' is the class name for fire
                    fire_detected = True
                # Draw bounding box and label
                label = f'{names[int(cls)]} {conf:.2f}'
                annotator.box_label([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])], label, color=colors(cls))
        frame = annotator.result()

 
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
