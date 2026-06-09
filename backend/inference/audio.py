"""
Audio deepfake detection inference.

This module loads an audio file, converts it into a log-mel spectrogram,
extracts additional handcrafted audio features, and runs the trained
EfficientNet-B0 audio detection model.
"""

from pathlib import Path
from typing import Dict, Optional
import warnings

import librosa
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from architectures.audio_model import create_audio_detection_model

warnings.filterwarnings("ignore", message="PySoundFile failed.*")
warnings.filterwarnings("ignore", message=".*Trying audioread instead.*")

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "audio_model.pth"

SAMPLE_RATE = 16000
DURATION = 4.0
FIXED_SAMPLES = int(SAMPLE_RATE * DURATION)

N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256

IMAGE_SIZE = 224
EXTRA_FEATURE_DIM = 13

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True

_model_instance = None


def to_percent(value):
    """Convert a probability value into a percentage."""
    return round(float(value) * 100.0, 2)


def to_score(value):
    """Round a probability score for API output."""
    return round(float(value), 4)


def get_confidence_level(confidence):
    """Convert confidence score into a readable confidence level."""
    if confidence >= 0.90:
        return "High"
    if confidence >= 0.70:
        return "Medium"
    return "Low"


def make_summary(label, confidence):
    """Create a user-friendly prediction summary."""
    confidence_percent = to_percent(confidence)

    if confidence < 0.70:
        return (
            f"The audio shows possible signs of being {label.upper()}, "
            f"but confidence is low ({confidence_percent}%). "
            "Manual review is recommended."
        )

    if confidence < 0.90:
        return (
            f"The audio is predicted as {label.upper()} with "
            f"{confidence_percent}% confidence. "
            "This is a medium-confidence result, so review is recommended."
        )

    return (
        f"This audio is predicted to be {label.upper()} "
        f"with {confidence_percent}% confidence."
    )


def format_feature_score(value):
    """Round extracted audio feature values for the response."""
    return round(float(value), 4)


def load_audio(audio_path: str):
    """
    Load audio as mono and resize it to a fixed 4-second duration.
    Longer clips are centre-cropped and shorter clips are padded.
    """
    y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    if len(y) > FIXED_SAMPLES:
        start = (len(y) - FIXED_SAMPLES) // 2
        y = y[start:start + FIXED_SAMPLES]

    elif len(y) < FIXED_SAMPLES:
        pad_total = FIXED_SAMPLES - len(y)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        y = np.pad(y, (pad_left, pad_right), mode="constant")

    return y.astype(np.float32)


def create_log_mel(y):
    """
    Convert the waveform into a normalised log-mel spectrogram.
    The spectrogram is later converted into an image tensor.
    """
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )

    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)

    return log_mel.astype(np.float32)


def extract_audio_features(y):
    """
    Extract handcrafted audio features used alongside the CNN model.
    These features describe loudness, silence, rhythm, pitch, and spectrum.
    """
    rms = librosa.feature.rms(
        y=y,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH,
    )[0]

    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))

    silence_threshold = rms_mean * 0.35
    silence_ratio = float(np.mean(rms < silence_threshold))

    onset_env = librosa.onset.onset_strength(
        y=y,
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
        y=y,
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
        y=y,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )[0]

    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )[0]

    zcr = librosa.feature.zero_crossing_rate(
        y,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH,
    )[0]

    feature_dict = {
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

    feature_vector = np.array(list(feature_dict.values()), dtype=np.float32)

    return feature_dict, feature_vector


def mel_to_image_tensor(log_mel, transform, device):
    """
    Convert a log-mel spectrogram into a normalised image tensor.
    """
    mel = log_mel.astype(np.float32)

    mel_min = mel.min()
    mel_max = mel.max()

    mel = (mel - mel_min) / (mel_max - mel_min + 1e-6)
    mel = (mel * 255.0).clip(0, 255).astype(np.uint8)

    image = Image.fromarray(mel).convert("RGB")

    return transform(image).unsqueeze(0).to(device)


class AudioDeepfakeDetector:
    """
    Handles model loading, audio preprocessing, feature extraction, and prediction.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        device: Optional[str] = None,
    ):
        self.model_path = str(model_path)
        self.device = torch.device(device) if device else DEVICE
        self.class_names = ["real", "fake"]

        self.model = None
        self.transform = None
        self.image_size = IMAGE_SIZE
        self.use_extra_features = True

        self._load_model()
        self._setup_transforms()

    def _load_model(self):
        """
        Load the trained audio model checkpoint from disk.
        """
        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=True,
        )

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            self.image_size = checkpoint.get("image_size", IMAGE_SIZE)
            dropout = checkpoint.get("dropout", 0.4)
            extra_feature_dim = checkpoint.get(
                "extra_feature_dim",
                EXTRA_FEATURE_DIM,
            )
            self.use_extra_features = checkpoint.get("use_extra_features", True)
        else:
            state_dict = checkpoint
            dropout = 0.4
            extra_feature_dim = EXTRA_FEATURE_DIM
            self.use_extra_features = True

        self.model = create_audio_detection_model(
            num_classes=2,
            pretrained=False,
            dropout=dropout,
            extra_feature_dim=extra_feature_dim,
            use_extra_features=self.use_extra_features,
        )

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def _setup_transforms(self):
        """
        Prepare image transforms for the spectrogram input.
        """
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )

    def preprocess_audio(self, audio_path: str):
        """
        Convert raw audio into the model inputs:
        spectrogram tensor, handcrafted feature tensor, and feature dictionary.
        """
        y = load_audio(audio_path)
        log_mel = create_log_mel(y)
        feature_dict, feature_vector = extract_audio_features(y)

        audio_tensor = mel_to_image_tensor(
            log_mel,
            self.transform,
            self.device,
        )

        extra_features = torch.tensor(
            feature_vector,
            dtype=torch.float32,
        ).unsqueeze(0).to(self.device)

        return audio_tensor, extra_features, feature_dict

    def predict(self, audio_path: str) -> Dict:
        """
        Run full audio inference and return prediction details for the API.
        """
        audio_tensor, extra_features, feature_dict = self.preprocess_audio(audio_path)

        with torch.no_grad():
            with torch.amp.autocast(
                "cuda",
                enabled=(self.device.type == "cuda"),
            ):
                logits = self.model(audio_tensor, extra_features)
                probabilities = torch.softmax(logits, dim=1)[0]

        real_probability = float(probabilities[0].detach().cpu())
        fake_probability = float(probabilities[1].detach().cpu())

        predicted_index = int(torch.argmax(probabilities).item())
        label = self.class_names[predicted_index]
        confidence = float(probabilities[predicted_index].detach().cpu())

        return {
            "prediction": label,
            "label": label,
            "verdict": label.upper(),
            "confidence": to_score(confidence),
            "confidence_percent": to_percent(confidence),
            "confidence_level": get_confidence_level(confidence),
            "real_probability": to_score(real_probability),
            "fake_probability": to_score(fake_probability),
            "real_percent": to_percent(real_probability),
            "fake_percent": to_percent(fake_probability),
            "probabilities": {
                "real": to_percent(real_probability),
                "fake": to_percent(fake_probability),
            },
            "summary": make_summary(label, confidence),
            "technical_details": {
                "sample_rate": SAMPLE_RATE,
                "duration_seconds": DURATION,
                "image_size": self.image_size,
                "device": str(self.device),
                "extra_features_used": self.use_extra_features,
            },
            "audio_features": {
                "rms_mean": format_feature_score(feature_dict["rms_mean"]),
                "rms_std": format_feature_score(feature_dict["rms_std"]),
                "silence_ratio": format_feature_score(feature_dict["silence_ratio"]),
                "speech_rate_proxy": format_feature_score(
                    feature_dict["speech_rate_proxy"]
                ),
                "pitch_mean": format_feature_score(feature_dict["pitch_mean"]),
                "pitch_std": format_feature_score(feature_dict["pitch_std"]),
                "pitch_jitter": format_feature_score(feature_dict["pitch_jitter"]),
                "spectral_flatness_mean": format_feature_score(
                    feature_dict["spectral_flatness_mean"]
                ),
                "spectral_flatness_std": format_feature_score(
                    feature_dict["spectral_flatness_std"]
                ),
                "centroid_mean": format_feature_score(feature_dict["centroid_mean"]),
                "centroid_std": format_feature_score(feature_dict["centroid_std"]),
                "zcr_mean": format_feature_score(feature_dict["zcr_mean"]),
                "zcr_std": format_feature_score(feature_dict["zcr_std"]),
            },
        }


def initialize_model(model_path: str):
    """
    Initialise the global audio model instance.
    """
    global _model_instance

    if not Path(model_path).exists():
        return False

    try:
        _model_instance = AudioDeepfakeDetector(model_path)
        return True
    except Exception:
        _model_instance = None
        return False


def get_model() -> Optional[AudioDeepfakeDetector]:
    """
    Return the currently loaded global audio model instance.
    """
    return _model_instance


def load_audio_detector(
    model_path: str = DEFAULT_MODEL_PATH,
    device: Optional[str] = None,
) -> AudioDeepfakeDetector:
    """
    Create and return an audio detector instance.
    """
    return AudioDeepfakeDetector(model_path=model_path, device=device)


def predict_audio(audio_path: str) -> Dict:
    """
    Predict whether an audio file is real or fake using the global model.
    """
    if _model_instance is None:
        raise RuntimeError("Audio model is not loaded.")

    return _model_instance.predict(audio_path)