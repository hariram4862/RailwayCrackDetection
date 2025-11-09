import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft

def plot_fft(file, label):
    data = np.loadtxt(file, delimiter=",")
    fft = np.abs(rfft(data))
    fft = fft / np.max(fft)
    plt.plot(fft[:256])
    plt.title(f"FFT Magnitude – {label}")
    plt.xlabel("Frequency Bin")
    plt.ylabel("Normalized Amplitude")

plot_fft("data/raw/defectless/0_001.csv", "Defectless")
plot_fft("data/raw/defective/1_001.csv", "Defective")
plt.legend(["Defectless", "Defective"])
plt.show()
