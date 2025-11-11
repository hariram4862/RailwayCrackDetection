import serial
import numpy as np
from scipy.fft import rfft
from tensorflow.keras.models import load_model
import time

PORT = "COM8"  # your Arduino port
BAUD = 115200
model = load_model("models/cnn_fft_model.keras")

def classify_waveform(values):
    fft_vals = np.abs(rfft(values))
    fft_norm = fft_vals / np.max(fft_vals)
    fft_input = fft_norm[:256].reshape(1, 256, 1)
    pred = model.predict(fft_input)
    label = np.argmax(pred)
    return label, pred[0]

ser = serial.Serial(PORT, BAUD, timeout=2)
print("🔌 Listening for Arduino...")

while True:
    line = ser.readline().decode().strip()
    print(line)
    if not line or "ARDUINO READY" in line:
        continue
    try:
        values = np.array([int(x) for x in line.split(",")])
        if len(values) == 512:
            label, prob = classify_waveform(values)
            status = "🟢 Defectless" if label == 0 else "🔴 Defective"
            print(f"{status} | probs={prob}")
        else:
            print("⚠️ incomplete data, skipping")
    except KeyboardInterrupt:
        print("\nStopped.")
        break
    except Exception as e:
        print("Error:", e)
