import sys
import torch
import cv2
import numpy as np
from pathlib import Path
import threading
import os
import getpass
import smtplib
from email.message import EmailMessage


EMAIL_ADDRESS = "ayushpoojari100@gmail.com"  
TO_EMAIL = "ayushpoojari100@gmail.com"       


EMAIL_PASSWORD = os.getenv("qgjdzxxbofcajlry")
if not EMAIL_PASSWORD:
    EMAIL_PASSWORD = getpass.getpass("Enter Gmail App Password (hidden): ").strip()

def send_email_alert(image_path=None):
    msg = EmailMessage()
    msg["Subject"] = " FIRE ALERT"
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = TO_EMAIL
    msg.set_content("Fire detected by your YOLOv5 Fire Detection System.")

    if image_path and Path(image_path).exists():
        with open(image_path, "rb") as f:
            img_data = f.read()
        msg.add_attachment(img_data, maintype="image", subtype="jpeg", filename="fire.jpg")

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
        print(" Email sent.")
    except Exception as e:
        print(" Email failed:", e)

yolov5_path = 'A:/firedetect/yolov5'
if yolov5_path not in sys.path:
    sys.path.append(yolov5_path)

from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import Annotator, colors
from utils.torch_utils import select_device


model_path = 'A:/firedetect/model/yolov5s_best.pt'
device = select_device('0' if torch.cuda.is_available() else '')


print(f"\nLoading YOLOv5 model on device: {device}...")
try:
    model = DetectMultiBackend(model_path, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(640, s=stride)
    print(f" Model loaded: {model_path}")
except Exception as e:
    print(" Error loading model:", e)
    exit()


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
        print(" Frame read error.")
        break

    pred = run_inference(frame)
    annotator = Annotator(frame, line_width=2, example=str(names))

    fire_detected = False

    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(torch.tensor(imgsz), det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(int(cls)))
                if names[int(cls)].lower() in ["fire", "flame"]:
                    fire_detected = True

    cv2.imshow(" Fire Detection", annotator.result())

    # Send alert if fire detected
    if fire_detected:
        snapshot = "fire_snapshot.jpg"
        cv2.imwrite(snapshot, frame)
        threading.Thread(target=send_email_alert, args=(snapshot,), daemon=True).start()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(" Detection stopped.")
