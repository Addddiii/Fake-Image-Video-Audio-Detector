# ============================================================
# Audio Preprocessing Script
# ============================================================

from pathlib import Path
import numpy as np
import librosa
from tqdm import tqdm

RAW_BASE = Path(r"D:\audio\raw")
PROCESSED_BASE = Path(r"D:\audio\processed")

SPLITS = ["train", "dev", "eval"]
LABELS = ["real", "fake"]

TARGET_SR = 16000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256
DURATION = 4.0
FIXED_SAMPLES = int(TARGET_SR * DURATION)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}


def load_audio(file_path):
    audio, _ = librosa.load(file_path, sr=TARGET_SR, mono=True)

    # pad or trim to fixed length
    if len(audio) < FIXED_SAMPLES:
        audio = np.pad(audio, (0, FIXED_SAMPLES - len(audio)), mode="constant")
    else:
        audio = audio[:FIXED_SAMPLES]

    return audio


def create_log_mel(audio):
    # convert audio waveform to mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=TARGET_SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )

    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)


def get_audio_files(folder):
    return [
        file for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() in AUDIO_EXTENSIONS
    ]


def process_audio_file(file_path, output_dir):
    audio = load_audio(file_path)
    log_mel = create_log_mel(audio)

    output_path = output_dir / f"{file_path.stem}.npy"
    np.save(output_path, log_mel)


def main():
    processed_count = 0
    failed_count = 0

    for split in SPLITS:
        for label in LABELS:
            input_dir = RAW_BASE / split / label
            output_dir = PROCESSED_BASE / split / label
            output_dir.mkdir(parents=True, exist_ok=True)

            if not input_dir.exists():
                print(f"Skipping missing folder: {input_dir}")
                continue

            files = get_audio_files(input_dir)

            print(f"\n=== {split.upper()} / {label.upper()} ===")
            print(f"Audio files: {len(files)}")

            for file_path in tqdm(files, desc=f"{split}-{label}", unit="file", ncols=100):
                try:
                    process_audio_file(file_path, output_dir)
                    processed_count += 1

                except Exception as e:
                    failed_count += 1
                    print(f"Failed: {file_path} | {e}")

    print("\n===== DONE =====")
    print(f"Processed files: {processed_count}")
    print(f"Failed files: {failed_count}")
    print(f"Saved to: {PROCESSED_BASE}")


if __name__ == "__main__":
    main()