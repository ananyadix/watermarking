from fastapi import FastAPI, File, UploadFile, Form
import numpy as np
import cv2

# Embed
from embed.embed import embed_pipeline
from utils.image_io import save_image

# Extract
from extract.extract import extract_and_restore_numpy

app = FastAPI()

# -----------------------------
# Embed API
# -----------------------------
@app.post("/embed")
async def embed_api(file: UploadFile = File(...), text: str = Form(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    result = embed_pipeline(image, text)

    output_path = "output.png"
    save_image(result, output_path)

    return {
        "message": "Embedding successful",
        "output": output_path
    }

# -----------------------------
# Extract / Verify API
# -----------------------------
@app.post("/extract")
async def extract_api(original_file: UploadFile = File(...), watermarked_file: UploadFile = File(...)):
    orig_bytes = await original_file.read()
    orig_nparr = np.frombuffer(orig_bytes, np.uint8)
    orig_image = cv2.imdecode(orig_nparr, cv2.IMREAD_GRAYSCALE)

    water_bytes = await watermarked_file.read()
    water_nparr = np.frombuffer(water_bytes, np.uint8)
    water_image = cv2.imdecode(water_nparr, cv2.IMREAD_GRAYSCALE)

    geo, integrity, restored, psnr, mse, ssim_score = extract_and_restore_numpy(orig_image, water_image)

    restored_path = None
    if restored is not None:
        restored_path = "restored.png"
        save_image(restored, restored_path)

    return {
        "geometric_check": geo,
        "integrity_check": integrity,
        "psnr": round(psnr, 2),
        "mse": round(mse, 2),
        "ssim": round(ssim_score, 4),
        "restored_image": restored_path
    }