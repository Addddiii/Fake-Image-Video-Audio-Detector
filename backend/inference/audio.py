"""
Audio Deepfake Detection Inference

This module loads a trained audio classification model and predicts
whether an uploaded audio file is real or fake.
"""

import logging
from typing import Dict

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from architectures.audio_model import AudioClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Audio preprocessing settings
TARGET_SR = 16000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256
DURATION = 4.0
FIXED_SAMPLES = int(TARGET_SR * DURATION)


class AudioDeepfakeDetector:
    """
    Handles audio loading, spectrogram generation, and model prediction.
    """
    
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize the detector and load model weights.

        Args:
            model_path: Path to trained model file
            device: 'cuda' or 'cpu'. Automatically selected if None
        """
        self.model_path = model_path
        
        # Select computation device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Loading model on {self.device}")
        
        self.model = self._load_model()
        self.model.eval()
        
        logger.info("Model ready")
    
    def _load_model(self) -> AudioClassifier:
        """
        Load model weights into the audio classifier.

        Returns:
            AudioClassifier model ready for inference
        """
        try:
            model = AudioClassifier(num_classes=2)
            
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Support both checkpoint dictionaries and direct state_dict files
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(self.device)
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def load_and_fix_length(self, audio_path: str) -> np.ndarray:
        """
        Load audio and force it to a fixed duration.

        Args:
            audio_path: Path to audio file

        Returns:
            Audio waveform with fixed number of samples
        """
        try:
            audio, _ = librosa.load(audio_path, sr=TARGET_SR, mono=True)
            
            # Pad short audio or trim long audio
            if len(audio) < FIXED_SAMPLES:
                pad = FIXED_SAMPLES - len(audio)
                audio = np.pad(audio, (0, pad), mode='constant')
            else:
                audio = audio[:FIXED_SAMPLES]
            
            return audio
            
        except Exception as e:
            logger.error(f"Error loading audio file: {e}")
            raise ValueError(f"Failed to load audio file: {e}")
    
    def make_log_mel(self, audio: np.ndarray) -> np.ndarray:
        """
        Convert waveform audio into a log-mel spectrogram.

        Args:
            audio: Audio waveform array

        Returns:
            Log-mel spectrogram as a float32 array
        """
        try:
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
            
        except Exception as e:
            logger.error(f"Error creating spectrogram: {e}")
            raise ValueError(f"Failed to create spectrogram: {e}")
    
    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """
        Convert an audio file into the tensor format expected by the model.

        Args:
            audio_path: Path to audio file

        Returns:
            Tensor with shape (1, 3, 224, 224)
        """
        try:
            audio = self.load_and_fix_length(audio_path)
            log_mel = self.make_log_mel(audio)
            
            # Normalize each spectrogram independently
            mean = np.mean(log_mel)
            std = np.std(log_mel)
            log_mel = (log_mel - mean) / (std + 1e-6)
            
            # Add batch and channel dimensions: (1, 1, H, W)
            x = torch.from_numpy(log_mel).unsqueeze(0).unsqueeze(0)
            
            # Resize spectrogram to image-model input size
            x = F.interpolate(
                x,
                size=(224, 224),
                mode='bilinear',
                align_corners=False
            )
            
            # Convert from 1 channel to 3 channels
            x = x.squeeze(0)
            x = x.repeat(3, 1, 1)
            
            # Add batch dimension: (1, 3, 224, 224)
            x = x.unsqueeze(0)
            
            return x
            
        except Exception as e:
            logger.error(f"Error preprocessing audio: {e}")
            raise ValueError(f"Failed to preprocess audio: {e}")
    
    def predict(self, audio_path: str) -> Dict:
        """
        Perform fake vs real prediction on an audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary containing prediction, confidence, probabilities, and analyzed duration
        """
        try:
            audio_tensor = self.preprocess_audio(audio_path)
            audio_tensor = audio_tensor.to(self.device)
            
            with torch.no_grad():
                logits = self.model(audio_tensor)
                probabilities = F.softmax(logits, dim=1)
            
            # Class order used by the audio model
            real_prob = probabilities[0, 0].item() * 100
            fake_prob = probabilities[0, 1].item() * 100
            
            predicted_class = torch.argmax(probabilities, dim=1).item()
            prediction = "real" if predicted_class == 0 else "fake"
            confidence = max(real_prob, fake_prob)
            
            result = {
                "prediction": prediction,
                "confidence": round(confidence, 2),
                "probabilities": {
                    "fake": round(fake_prob, 2),
                    "real": round(real_prob, 2)
                },
                "duration_seconds": DURATION
            }
            
            logger.info(f"Prediction: {prediction} ({confidence:.2f}%)")
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise ValueError(f"Prediction failed: {e}")


def load_audio_detector(model_path: str = "models/audio_model.pth", device: str = None) -> AudioDeepfakeDetector:
    """
    Create and return an audio detector instance.

    Args:
        model_path: Path to trained model file
        device: Device for inference

    Returns:
        AudioDeepfakeDetector instance
    """
    return AudioDeepfakeDetector(model_path=model_path, device=device)

def predict_audio(audio_path: str) -> Dict:
    detector = load_audio_detector()
    return detector.predict(audio_path)