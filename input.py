import cv2
import requests
import time
import random

API_URL = "http://13.213.18.54:8000/upload"

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

def send_frame(frame):
    try:
        # Encode frame as JPEG (in memory)
        _, img_encoded = cv2.imencode('.jpg', frame)

        files = {
            "file": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")
        }

        response = requests.post(API_URL, files=files, timeout=10)

        print("Status:", response.status_code)
        print("Response:", response.json())

    except Exception as e:
        print("Error:", e)


while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    send_frame(frame)

    delay = 0.2
    print(f"Waiting {delay:.2f} seconds...\n")
    time.sleep(delay)

cap.release()