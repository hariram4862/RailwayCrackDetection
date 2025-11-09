import serial, os, time
import numpy as np

PORT = "COM5"  # change if needed
BAUD = 115200
LABEL = input("Enter label (0=defectless, 1=defective): ")
SAVE_DIR = f"data/raw/{'defectless' if LABEL == '0' else 'defective'}"
os.makedirs(SAVE_DIR, exist_ok=True)

ser = serial.Serial(PORT, BAUD, timeout=2)
print("Listening for Arduino...")

i = 0
while True:
    try:
        line = ser.readline().decode().strip()
        if not line or "ARDUINO READY" in line:
            continue
        values = [int(x) for x in line.split(",") if x.strip().isdigit()]
        if len(values) == 512:
            i += 1
            filename = os.path.join(SAVE_DIR, f"{LABEL}_{i:03d}.csv")
            np.savetxt(filename, values, delimiter=",", fmt="%d")
            print(f"Saved {filename} ({len(values)} samples)")
        else:
            print("⚠️ incomplete packet, skipping...")
    except KeyboardInterrupt:
        print("\nData collection stopped.")
        break
    except Exception as e:
        print("Error:", e)
