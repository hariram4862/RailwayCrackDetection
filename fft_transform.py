# import numpy as np
# from pathlib import Path

# data = np.load("fft_data/waves.npy")
# labels = np.load("fft_data/labels.npy")

# fft_mags = np.abs(np.fft.rfft(data, axis=1))
# fft_mags = fft_mags / np.max(fft_mags, axis=1, keepdims=True)  # normalize

# np.save("fft_data/fft_mags.npy", fft_mags)
# np.save("fft_data/labels.npy", labels)
# print("FFT data:", fft_mags.shape)



import numpy as np
import os
from scipy.fft import rfft

RAW_DIR = "data/raw"
SAVE_DIR = "data/fft"
os.makedirs(SAVE_DIR, exist_ok=True)

X, y = [], []

for label, cls in enumerate(["defectless", "defective"]):
    folder = os.path.join(RAW_DIR, cls)
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            data = np.loadtxt(os.path.join(folder, file), delimiter=",")
            # FFT transform
            fft_vals = np.abs(rfft(data))
            fft_norm = fft_vals / np.max(fft_vals)
            X.append(fft_norm[:256])  # keep half spectrum
            y.append(label)

X = np.array(X)
y = np.array(y)

np.save(os.path.join(SAVE_DIR, "fft_data.npy"), X)
np.save(os.path.join(SAVE_DIR, "fft_labels.npy"), y)

print(f"Saved {X.shape[0]} FFT samples of shape {X.shape[1]}")
