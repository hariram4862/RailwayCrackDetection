# gen_fft_data.py
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
Path("fft_data").mkdir(exist_ok=True)

SAMPLES = 512       # samples per signal
FS = 20000          # sampling rate (Hz)
t = np.arange(SAMPLES) / FS

def make_wave(freq, amp, noise_level=0.01):
    signal = amp * np.sin(2 * np.pi * freq * t)
    signal += noise_level * np.random.randn(SAMPLES)
    return signal

data, labels = [], []

for _ in range(600):  # defectless
    data.append(make_wave(5000 + np.random.uniform(-50, 50), 1.0, 0.02))
    labels.append(0)

for _ in range(400):  # minor crack
    data.append(make_wave(4800 + np.random.uniform(-80, 80), 0.6, 0.05))
    labels.append(1)

for _ in range(300):  # major crack
    data.append(make_wave(4600 + np.random.uniform(-100, 100), 0.3, 0.08))
    labels.append(2)

data = np.array(data)
labels = np.array(labels)
np.save("fft_data/waves.npy", data)
np.save("fft_data/labels.npy", labels)
print("Generated:", data.shape, "labels:", labels.shape)
