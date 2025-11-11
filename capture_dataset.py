import serial, os, time
import numpy as np
from pathlib import Path

PORT = "COM8"  # change if needed
BAUD = 115200
LABEL = input("Enter label (0=defectless, 1=defective): ").strip()
SAVE_DIR = Path(f"data/raw/{'defectless' if LABEL == '0' else 'defective'}")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

ser = serial.Serial(PORT, BAUD, timeout=2)
print(f"📡 Listening for Arduino on {PORT}...\n")

# --- Find next file index safely ---
existing_files = list(SAVE_DIR.glob(f"{LABEL}_*.csv"))
if existing_files:
    last_index = max([int(f.stem.split("_")[1]) for f in existing_files])
else:
    last_index = 0

i = last_index
print(f"📁 Found {len(existing_files)} existing files. Starting from index {i+1}...\n")

while True:
    try:
        line = ser.readline().decode(errors="ignore").strip()
        if not line or "ARDUINO READY" in line:
            continue

        values = [int(x) for x in line.split(",") if x.strip().isdigit()]
        if len(values) == 512:
            i += 1
            filename = SAVE_DIR / f"{LABEL}_{i:03d}.csv"
            np.savetxt(filename, values, delimiter=",", fmt="%d")
            print(f"✅ Saved {filename} ({len(values)} samples)")
        else:
            print("⚠️ Incomplete packet, skipping...")

    except KeyboardInterrupt:
        print("\n🛑 Data collection stopped by user.")
        break
    except Exception as e:
        print("❌ Error:", e)
