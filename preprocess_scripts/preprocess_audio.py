import os
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# =========================
# CONFIG
# =========================
RAW_ROOT = r"D:\FakeDetection\raw_datasets\audio\asvspoof2019"
OUT_ROOT = r"D:\FakeDetection\processed_datasets\audio"

RAW_TO_OUT_SPLIT = {
    "train": "train",
    "dev": "eval",
    "test": "test"
}

AUDIO_EXTENSIONS = [".flac", ".wav", ".mp3"]
FIG_SIZE = (3, 3)
DPI = 75
MAX_FILES_PER_CLASS = None   # set to 100 for testing

# =========================
# HELPERS
# =========================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def find_audio_files(folder):
    files = []
    for root, _, filenames in os.walk(folder):
        for f in filenames:
            if Path(f).suffix.lower() in AUDIO_EXTENSIONS:
                files.append(os.path.join(root, f))
    return files

def save_mel_spectrogram(audio_path, out_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)

        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        plt.figure(figsize=FIG_SIZE)
        librosa.display.specshow(mel_spec_db, sr=sr)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0, dpi=DPI)
        plt.close("all")

        return True
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return False

def process_split(raw_split, out_split):
    for class_name in ["real", "fake"]:
        in_dir = os.path.join(RAW_ROOT, raw_split, class_name)
        out_dir = os.path.join(OUT_ROOT, out_split, class_name)

        ensure_dir(out_dir)

        if not os.path.exists(in_dir):
            print(f"Missing input folder: {in_dir}")
            continue

        audio_files = find_audio_files(in_dir)

        if MAX_FILES_PER_CLASS is not None:
            audio_files = audio_files[:MAX_FILES_PER_CLASS]

        total = len(audio_files)
        count = 0

        print(f"\nProcessing {raw_split}/{class_name} → {out_split}/{class_name}")
        print(f"Total files: {total}\n")

        for audio_path in audio_files:
            file_name = Path(audio_path).stem + ".png"
            out_path = os.path.join(out_dir, file_name)

            if os.path.exists(out_path):
                count += 1
                percent = (count / total) * 100
                print(f"Skipping {file_name} ({count}/{total}) {percent:.1f}%")
                continue

            ok = save_mel_spectrogram(audio_path, out_path)
            if ok:
                count += 1
                percent = (count / total) * 100
                print(f"[{out_split}][{class_name}] {count}/{total} ({percent:.1f}%) -> {file_name}")

# =========================
# MAIN
# =========================
def main():
    for raw_split, out_split in RAW_TO_OUT_SPLIT.items():
        process_split(raw_split, out_split)

    print("\nAudio preprocessing done.")

if __name__ == "__main__":
    main()