from fastapi import FastAPI, UploadFile, File
from PIL import Image
import onnxruntime as ort
import numpy as np
import io

app = FastAPI()

# Load onnx
sess = ort.InferenceSession('model/model.onnx', providers=['CPUExecutionProvider'])

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)[np.newaxis, :]  # (1, 3, 224, 224)
    return arr

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    input_arr = preprocess(img_bytes)
    outputs = sess.run(None, {'input': input_arr})[0]
    class_id = int(np.argmax(outputs))
    confidence = float(np.max(np.exp(outputs) / np.sum(np.exp(outputs))))
    return {"class_id": class_id, "confidence": round(confidence, 4)}
