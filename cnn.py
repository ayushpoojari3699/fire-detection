import cv2
import numpy as np
import pickle

# Load the trained SVM model from the given path
svm_model_path = r"A:\firedetect\svm_fire_detection_model.pkl"  # Use 'r' for raw string
with open(svm_model_path, "rb") as f:
    svm_model = pickle.load(f)

# Initialize webcam
cap = cv2.VideoCapture(0)

def extract_features(frame):
    frame_resized = cv2.resize(frame, (64, 64)).flatten()  # Resize & flatten
    return frame_resized.reshape(1, -1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Extract features and predict
    features = extract_features(frame)
    fire_detected = svm_model.predict(features)[0] == 1  # Fire = 1, No Fire = 0

    if fire_detected:
        cv2.putText(frame, "🔥 FIRE DETECTED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the result
    cv2.imshow("Fire Detection (SVM)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
