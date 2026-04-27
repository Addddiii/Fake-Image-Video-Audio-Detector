from pathlib import Path
import numpy as np
import librosa
from tqdm import tqdm

RAW_BASE = Path(r"D:\audio\raw")
PROCESSED_BASE = Path(r"D:\audio\processed")

SPLITS = ["train", "dev", "eval"]   # change if you really use test instead of dev
LABELS = ["real", "fake"]

TARGET_SR = 16000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256
DURATION = 4.0   # seconds
FIXED_SAMPLES = int(TARGET_SR * DURATION)

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}


def load_and_fix_length(file_path: Path, sr: int, fixed_samples: int):
    audio, _ = librosa.load(file_path, sr=sr, mono=True)

    if len(audio) < fixed_samples:
        pad = fixed_samples - len(audio)
        audio = np.pad(audio, (0, pad), mode="constant")
    else:
        audio = audio[:fixed_samples]

    return audio


def make_log_mel(audio: np.ndarray, sr: int):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)


def main():
    total_done = 0

    for split in SPLITS:
        for label in LABELS:
            in_dir = RAW_BASE / split / label
            out_dir = PROCESSED_BASE / split / label
            out_dir.mkdir(parents=True, exist_ok=True)

            if not in_dir.exists():
                print(f"Skipping missing folder: {in_dir}")
                continue

            files = [f for f in in_dir.iterdir() if f.is_file() and f.suffix.lower() in AUDIO_EXTS]
            print(f"\nProcessing {split}/{label} -> {len(files)} files")

            for file_path in tqdm(files):
                try:
                    audio = load_and_fix_length(file_path, TARGET_SR, FIXED_SAMPLES)
                    log_mel = make_log_mel(audio, TARGET_SR)

                    out_path = out_dir / f"{file_path.stem}.npy"
                    np.save(out_path, log_mel)

                    total_done += 1
                except Exception as e:
                    print(f"Failed: {file_path} | {e}")

    print(f"\nDone. Processed {total_done} files.")


if __name__ == "__main__":
    main()