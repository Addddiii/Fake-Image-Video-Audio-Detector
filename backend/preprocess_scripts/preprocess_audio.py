"""
Preprocess audio datasets for audio model training.

This script loads raw audio files, converts them into fixed-length log-mel
spectrograms, extracts handcrafted audio features, and saves the processed
data as compressed .npz files.
"""

from pathlib import Path
import json
import shutil
import warnings

import librosa
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore", message="PySoundFile failed.*")
warnings.filterwarnings("ignore", message=".*Trying audioread instead.*")

RAW_DIR = Path(r"D:\audio_raw")
OUTPUT_DIR = Path(r"D:\audio_processed")

SAMPLE_RATE = 16000
DURATION = 4.0
FIXED_SAMPLES = int(SAMPLE_RATE * DURATION)

N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256

CLEAN_OUTPUT = False

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


def load_audio(audio_path):
    """
    Load an audio file as mono and resize it to a fixed 4-second length.
    Longer clips are centre-cropped and shorter clips are padded.
    """
    audio, _ = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True,
    )

    if len(audio) > FIXED_SAMPLES:
        start = (len(audio) - FIXED_SAMPLES) // 2
        audio = audio[start:start + FIXED_SAMPLES]

    elif len(audio) < FIXED_SAMPLES:
        pad_total = FIXED_SAMPLES - len(audio)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left

        audio = np.pad(
            audio,
            (pad_left, pad_right),
            mode="constant",
        )

    return audio.astype(np.float32)


def create_log_mel(audio):
    """
    Convert audio waveform into a normalised log-mel spectrogram.
    """
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )

    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)

    return log_mel.astype(np.float32)


def extract_audio_features(audio):
    """
    Extract handcrafted features that describe loudness, silence, rhythm,
    pitch variation, and spectral behaviour.
    """
    rms = librosa.feature.rms(
        y=audio,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH,
    )[0]

    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))

    silence_threshold = rms_mean * 0.35
    silence_ratio = float(np.mean(rms < silence_threshold))

    onset_env = librosa.onset.onset_strength(
        y=audio,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
    )

    onset_count = len(
        librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=SAMPLE_RATE,
            hop_length=HOP_LENGTH,
        )
    )

    speech_rate_proxy = float(onset_count / DURATION)

    pitches, magnitudes = librosa.piptrack(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )

    pitch_values = []

    for frame_index in range(pitches.shape[1]):
        magnitude_column = magnitudes[:, frame_index]
        pitch_column = pitches[:, frame_index]

        if magnitude_column.max() > 0:
            pitch = pitch_column[magnitude_column.argmax()]

            if pitch > 0:
                pitch_values.append(pitch)

    if len(pitch_values) > 1:
        pitch_values = np.array(pitch_values)

        pitch_mean = float(np.mean(pitch_values))
        pitch_std = float(np.std(pitch_values))
        pitch_jitter = float(np.mean(np.abs(np.diff(pitch_values))))
    else:
        pitch_mean = 0.0
        pitch_std = 0.0
        pitch_jitter = 0.0

    flatness = librosa.feature.spectral_flatness(
        y=audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )[0]

    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )[0]

    zcr = librosa.feature.zero_crossing_rate(
        audio,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH,
    )[0]

    features = {
        "rms_mean": rms_mean,
        "rms_std": rms_std,
        "silence_ratio": silence_ratio,
        "speech_rate_proxy": speech_rate_proxy,
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "pitch_jitter": pitch_jitter,
        "spectral_flatness_mean": float(np.mean(flatness)),
        "spectral_flatness_std": float(np.std(flatness)),
        "centroid_mean": float(np.mean(centroid)),
        "centroid_std": float(np.std(centroid)),
        "zcr_mean": float(np.mean(zcr)),
        "zcr_std": float(np.std(zcr)),
    }

    feature_vector = np.array(list(features.values()), dtype=np.float32)

    return features, feature_vector


def collect_audio(split, label_name):
    """
    Collect all supported audio files for a dataset split and class label.
    """
    folder = RAW_DIR / split / label_name

    if not folder.exists():
        print(f"Missing folder: {folder}")
        return []

    audio_files = [
        file_path
        for file_path in folder.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in AUDIO_EXTENSIONS
    ]

    return sorted(audio_files)


def process_split_label(split, label_name):
    """
    Process one dataset split and label, then save spectrograms and features.
    """
    audio_files = collect_audio(split, label_name)
    output_folder = OUTPUT_DIR / split / label_name
    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"\n{split}/{label_name}: found {len(audio_files)} audio files")

    saved_count = 0
    skipped_count = 0

    for audio_path in tqdm(
        audio_files,
        desc=f"Processing {split}/{label_name}",
        unit="audio",
    ):
        try:
            audio = load_audio(audio_path)
            log_mel = create_log_mel(audio)
            feature_dict, feature_vector = extract_audio_features(audio)

            output_name = audio_path.stem

            np.savez_compressed(
                output_folder / f"{output_name}.npz",
                mel=log_mel,
                features=feature_vector,
                feature_names=np.array(list(feature_dict.keys())),
                label=label_name,
            )

            with open(
                output_folder / f"{output_name}.json",
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(feature_dict, file, indent=4)

            saved_count += 1

        except Exception as error:
            skipped_count += 1
            print(f"\nSkipped: {audio_path}")
            print(f"Reason: {error}")

    print(f"{split}/{label_name} saved: {saved_count}")
    print(f"{split}/{label_name} skipped: {skipped_count}")


def count_processed_files(folder):
    """
    Count processed .npz files inside a folder.
    """
    if not folder.exists():
        return 0

    return len(
        [
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() == ".npz"
        ]
    )


def print_final_counts():
    """
    Print final processed file counts for each split and class.
    """
    print("\nFinal processed folder counts:")

    for split in ["train", "eval", "test"]:
        for label_name in ["real", "fake"]:
            folder = OUTPUT_DIR / split / label_name
            count = count_processed_files(folder)

            print(f"{split}/{label_name}: {count}")


def main():
    """
    Run the full audio preprocessing pipeline.
    """
    if CLEAN_OUTPUT and OUTPUT_DIR.exists():
        print(f"Removing old processed folder: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in ["train", "eval", "test"]:
        for label_name in ["real", "fake"]:
            process_split_label(split, label_name)

    print_final_counts()
    print(f"\nProcessed audio saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()