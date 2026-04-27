"""
Audio Deepfake Detection Inference
Handles audio file loading, preprocessing, and model inference
Matches preprocessing from preprocess_audio.py and train_audio.py
"""

import logging
import tempfile
from pathlib import Path
from typing import Dict

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from audio_model_architecture import AudioClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Preprocessing constants matching train_audio.py
TARGET_SR = 16000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256
DURATION = 4.0  # seconds
FIXED_SAMPLES = int(TARGET_SR * DURATION)


class AudioDeepfakeDetector:
    """
    Audio deepfake detection using trained EfficientNet-B0 model
    """
    
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize the audio detector
        
        Args:
            model_path: Path to the trained model (.pth file)
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.model_path = model_path
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Loading model on {self.device}")
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        logger.info("Model loaded successfully")
    
    def _load_model(self) -> AudioClassifier:
        """
        Load the trained model from checkpoint
        
        Returns:
            Loaded AudioClassifier model
        """
        try:
            # Create model instance
            model = AudioClassifier(num_classes=2)
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Load state dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Handle checkpoint format with metadata
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Handle direct state dict format
                model.load_state_dict(checkpoint)
            
            model.to(self.device)
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def load_and_fix_length(self, audio_path: str) -> np.ndarray:
        """
        Load audio file and fix length to FIXED_SAMPLES
        Matches preprocessing from preprocess_audio.py
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Audio array of fixed length
        """
        try:
            # Load audio using librosa
            audio, _ = librosa.load(audio_path, sr=TARGET_SR, mono=True)
            
            # Pad or truncate to fixed length
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
        Convert audio to log-mel spectrogram
        Matches preprocessing from preprocess_audio.py
        
        Args:
            audio: Audio array
            
        Returns:
            Log-mel spectrogram (H, W) array
        """
        try:
            # Create mel spectrogram
            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=TARGET_SR,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                n_mels=N_MELS,
                power=2.0,
            )
            
            # Convert to log scale
            log_mel = librosa.power_to_db(mel, ref=np.max)
            
            return log_mel.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error creating spectrogram: {e}")
            raise ValueError(f"Failed to create spectrogram: {e}")
    
    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """
        Preprocess audio file for model input
        Matches preprocessing from train_audio.py NpyDataset
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Preprocessed tensor of shape (1, 3, 224, 224)
        """
        try:
            # Load and fix length
            audio = self.load_and_fix_length(audio_path)
            
            # Create log-mel spectrogram
            log_mel = self.make_log_mel(audio)
            
            # Normalize (per-sample normalization)
            mean = np.mean(log_mel)
            std = np.std(log_mel)
            log_mel = (log_mel - mean) / (std + 1e-6)
            
            # Convert to tensor and add batch/channel dimensions
            x = torch.from_numpy(log_mel).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            
            # Interpolate to 224x224
            x = F.interpolate(
                x,
                size=(224, 224),
                mode='bilinear',
                align_corners=False
            )
            
            # Remove batch dimension and repeat to 3 channels
            x = x.squeeze(0)  # (1, 224, 224)
            x = x.repeat(3, 1, 1)  # (3, 224, 224)
            
            # Add batch dimension back
            x = x.unsqueeze(0)  # (1, 3, 224, 224)
            
            return x
            
        except Exception as e:
            logger.error(f"Error preprocessing audio: {e}")
            raise ValueError(f"Failed to preprocess audio: {e}")
    
    def predict(self, audio_path: str) -> Dict:
        """
        Predict if audio is real or fake
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with prediction results:
            {
                'prediction': 'fake' or 'real',
                'confidence': float (0-100),
                'probabilities': {
                    'fake': float (0-100),
                    'real': float (0-100)
                },
                'duration_seconds': float
            }
        """
        try:
            # Preprocess audio
            audio_tensor = self.preprocess_audio(audio_path)
            audio_tensor = audio_tensor.to(self.device)
            
            # Run inference
            with torch.no_grad():
                logits = self.model(audio_tensor)
                probabilities = F.softmax(logits, dim=1)
            
            # Get probabilities for each class
            # Class 0 = real, Class 1 = fake (from train_audio.py)
            real_prob = probabilities[0, 0].item() * 100
            fake_prob = probabilities[0, 1].item() * 100
            
            # Determine prediction
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
            
            logger.info(f"Prediction: {prediction} (confidence: {confidence:.2f}%)")
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise ValueError(f"Prediction failed: {e}")


def load_audio_detector(model_path: str, device: str = None) -> AudioDeepfakeDetector:
    """
    Helper function to load audio detector
    
    Args:
        model_path: Path to model checkpoint
        device: Device to run on
        
    Returns:
        AudioDeepfakeDetector instance
    """
    return AudioDeepfakeDetector(model_path=model_path, device=device)


if __name__ == "__main__":
    # Test audio detector
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python audio_inference.py <model_path> <audio_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    audio_path = sys.argv[2]
    
    detector = load_audio_detector(model_path)
    result = detector.predict(audio_path)
    
    print("\nPrediction Results:")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2f}%")
    print(f"Probabilities: Fake={result['probabilities']['fake']:.2f}%, Real={result['probabilities']['real']:.2f}%")
    print(f"Duration: {result['duration_seconds']}s")
