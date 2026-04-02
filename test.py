import sys
import torch
import cv2
import numpy as np
from pathlib import Path


yolov5_path = 'A:/firedetect/yolov5'
if yolov5_path not in sys.path:
    sys.path.append(yolov5_path)

from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression
from utils.plots import Annotator, colors
from utils.torch_utils import select_device



import torch.serialization
import models.yolo

torch.serialization.add_safe_globals([models.yolo.Model])


_original_torch_load = torch.load
def unsafe_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = unsafe_torch_load


model_path = 'A:/firedetect/model/yolov5s_best.pt'
device = select_device('')

print("\nLoading YOLOv5 model safely...")
try:
    model = DetectMultiBackend(model_path, device=device)
    print(" Model loaded successfully!")
except Exception as e:
    print(" Error loading model:", e)
    exit()


stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size(640, s=stride)


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print(" Cannot access webcam.")
    exit()

print(" Fire Detection Running — press 'q' to quit")

def run_inference(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if model.fp16 else img.float()
    img /= 255.0
    pred = model(img)
    pred = non_max_suppression(pred, 0.25, 0.45, agnostic=False)
    return pred


while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame read error.")
        break

    pred = run_inference(frame)
    annotator = Annotator(frame, line_width=2, example=str(names))

    for det in pred:
        if len(det):
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(int(cls)))

    cv2.imshow(" Fire Detection", annotator.result())
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(" Detection stopped.")
