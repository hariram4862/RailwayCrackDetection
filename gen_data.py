# gen_data.py
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
Path("data").mkdir(exist_ok=True)

def gen_samples(n, freq_mu, freq_sigma, amp_mu, amp_sigma, label):
    freqs = np.random.normal(freq_mu, freq_sigma, n)
    amps  = np.random.normal(amp_mu, amp_sigma, n)
    return pd.DataFrame({"frequency": freqs, "amplitude": amps, "label": label})

# defectless: high freq ~5000Hz, high amp
df1 = gen_samples(1000, 5000, 60, 700, 80, 0)
# minor crack: modest freq drop and lower amp
df2 = gen_samples(600, 4850, 80, 400, 100, 1)
# major crack: large freq drop and low amp
df3 = gen_samples(400, 4600, 90, 150, 80, 2)

df = pd.concat([df1, df2, df3]).reset_index(drop=True)
# Clip unrealistic ranges
df.frequency = df.frequency.clip(2000, 8000)
df.amplitude = df.amplitude.clip(0, 1023)
df.to_csv("data/synthetic.csv", index=False)
print("Saved data/synthetic.csv", df.shape)
