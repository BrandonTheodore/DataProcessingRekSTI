from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import torch
import cv2
import io

from count import predict_count

app = FastAPI()

LATEST_IMAGE_PATH = "latest.jpg"
latest_count = None

def preprocess(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = np.array(image)

    img = cv2.resize(img, (512, 512))

    img = img / 255.0
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

    img = np.transpose(img, (2, 0, 1))
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

    return img

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    global latest_count

    contents = await file.read()

    with open(LATEST_IMAGE_PATH, "wb") as f:
        f.write(contents)

    img = preprocess(contents)

    latest_count = predict_count(img)

    return {
        "status": "processed",
        "count": latest_count
    }


@app.get("/latest")
def get_latest():
    if latest_count is None:
        return {"message": "No image processed yet"}

    return {
        "count": latest_count,
        "image_path": LATEST_IMAGE_PATH
    }