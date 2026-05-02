from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from torchvision import transforms
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

    image.thumbnail((512, 512), Image.Resampling.LANCZOS)

    padded_image = Image.new('RGB', (512, 512), (0, 0, 0))

    paste_x = (512 - image.width) // 2
    paste_y = (512 - image.height) // 2
    padded_image.paste(image, (paste_x, paste_y))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    img_tensor = transform(padded_image).unsqueeze(0)

    return img_tensor

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

@app.get("/latest-image")
def get_latest_image():
    if latest_count is None:
        return {"message": "No image processed yet"}

    return FileResponse(
        LATEST_IMAGE_PATH,
        media_type="image/jpeg"
    )