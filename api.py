from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib, numpy as np
from tensorflow.keras.models import load_model

app = FastAPI(title="Railway Crack Detection AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class Reading(BaseModel):
    frequency: float
    amplitude: float

cnn = load_model("models/cnn_fft_model.keras")

@app.get("/")
def home():
    return {"status": "API running"}


@app.post("/predict_fft")
def predict_fft(r: Reading):
    FS = 20000
    t = np.arange(512) / FS
    wave = np.sin(2*np.pi*r.frequency*t) * (r.amplitude/700)
    fft_mag = np.abs(np.fft.rfft(wave))
    fft_mag /= np.max(fft_mag)
    X = fft_mag[np.newaxis, :, np.newaxis]
    probs = cnn.predict(X)[0]
    label = int(np.argmax(probs))
    return {
        "label": ["defectless","minor","major"][label],
        "probabilities": {
            "defectless": float(probs[0]),
            "minor": float(probs[1]),
            "major": float(probs[2])
        }
    }
