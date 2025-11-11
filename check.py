from pathlib import Path

# Set your base directory
base_dir = Path(r"D:\RailwayCrackDetection\backend\data\raw")
defective_dir = base_dir / "defective"
defectless_dir = base_dir / "defectless"

def rename_prefix(folder, old_prefix, new_prefix):
    renamed_files = []
    for file in folder.iterdir():
        # Skip non-files
        if not file.is_file():
            continue

        name = file.stem  # without extension
        suffix = file.suffix if file.suffix else ".csv"  # default to .csv if missing

        if name.startswith(f"{old_prefix}_"):
            new_name = name.replace(f"{old_prefix}_", f"{new_prefix}_", 1) + suffix
            new_path = file.with_name(new_name)
            file.rename(new_path)
            renamed_files.append((file.name, new_name))
    return renamed_files


# Perform the swaps
renamed_defective = rename_prefix(defective_dir, "0", "1")   # defective: 0_ → 1_
renamed_defectless = rename_prefix(defectless_dir, "1", "0") # defectless: 1_ → 0_

# Print results
print("\n✅ Renamed files in 'defective' (0 → 1):")
for old, new in renamed_defective:
    print(f"  {old} → {new}")

print("\n✅ Renamed files in 'defectless' (1 → 0):")
for old, new in renamed_defectless:
    print(f"  {old} → {new}")

print("\n🎉 All filenames corrected successfully!")



# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.fft import rfft

# def plot_fft(file, label):
#     data = np.loadtxt(file, delimiter=",")
#     fft = np.abs(rfft(data))
#     fft = fft / np.max(fft)
#     plt.plot(fft[:256])
#     plt.title(f"FFT Magnitude – {label}")
#     plt.xlabel("Frequency Bin")
#     plt.ylabel("Normalized Amplitude")

# plot_fft("data/raw/defectless/0_001.csv", "Defectless")
# plot_fft("data/raw/defective/1_001.csv", "Defective")
# plt.legend(["Defectless", "Defective"])
# plt.show()

