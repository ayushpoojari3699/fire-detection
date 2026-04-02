import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr
from threading import Thread


yolov5_path = 'A:/firedetect/yolov5'
if yolov5_path not in sys.path:
    sys.path.append(yolov5_path)

from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression
from utils.plots import Annotator, colors
from utils.torch_utils import select_device


model_path = 'A:/firedetect/yolov5/runs/train/exp3/weights/best.pt'
device = select_device('')
model = DetectMultiBackend(model_path, device=device)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size(640, s=stride)  # Adjust image size


cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print("Error: Could not open video capture device.")
    exit()


sender_email = "ayushpoojari100@gmail.com"
receiver_email = "ayushpoojari100@gmail.com"
password = "lcid grxf mwiy xbdc"  

def send_email(subject, body):
    msg = MIMEMultipart()
    msg['From'] = formataddr(('Fire Detection System', sender_email))
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
            print("Email sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")

def send_email_async(subject, body):
    email_thread = Thread(target=send_email, args=(subject, body))
    email_thread.start()

def run_inference(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))  
    img = np.transpose(img, (2, 0, 1))  
    img = np.expand_dims(img, axis=0)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if model.fp16 else img.float() 
    img /= 255.0  # Normalize to 0.0 - 1.0
    
    
    with torch.no_grad():
        pred = model(img)
        pred = non_max_suppression(pred, 0.4, 0.5, agnostic=False)  
    
    return pred

def color_based_detection(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
  
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    
    mask_red1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
    mask_red = mask_red1 | mask_red2
    

    kernel = np.ones((5, 5), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)

    fire_pixels = np.sum(mask_red > 0)
    total_pixels = frame.shape[0] * frame.shape[1]
    fire_percentage = (fire_pixels / total_pixels) * 100

    return mask_red, fire_percentage

def process_frame(frame):
    pred = run_inference(frame)
    fire_detected = False
    annotator = Annotator(frame, line_width=2, example=str(names))
    
    for det in pred:  # detections per image
        if len(det):
            for *xyxy, conf, cls in reversed(det):
                x1, y1, x2, y2 = map(int, xyxy)
                x, y = (x1 + x2) / 2, (y1 + y2) / 2
                w, h = x2 - x1, y2 - y1
                box_area = w * h
                if box_area < 1000:
                    continue
                
                if names[int(cls)] == 'fire':  
                    fire_detected = True
                    # Draw bounding box and label
                    label = f'{names[int(cls)]} {conf:.2f}'
                    annotator.box_label([x1, y1, x2, y2], label, color=colors(cls))
    
    frame = annotator.result()
    
    mask_red, fire_percentage = color_based_detection(frame)
    if fire_detected or fire_percentage > 5: 
        fire_detected = True
    
    return frame, fire_detected, mask_red, fire_percentage

email_sent = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        frame, fire_detected, mask_red, fire_percentage = process_frame(frame)

        if fire_detected:
            cv2.imshow('Color-based Mask', mask_red)
            if not email_sent:
                send_email_async(
                    subject="Fire Detected!",
                    body=f"Fire has been detected. Color-based fire percentage: {fire_percentage:.2f}%."
                )
                email_sent = True  # Set flag to true after sending email

    except Exception as e:
        print(f"Error during processing: {e}")

    cv2.imshow('Fire Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
