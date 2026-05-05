"""
Video Deepfake Detection Inference

This module loads a trained video model, extracts frames from a video,
and performs fake vs real classification.
"""

import os
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from architectures.video_model import VideoClassifier
import logging

# ✅ Add logger (same style as image)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoDeepfakeDetector:
    """
    Handles video preprocessing and prediction using a trained model.
    """
    
    def __init__(self, model_path, device=None, num_frames=20):
        """
        Initialize the detector and load model weights.

        Args:
            model_path: Path to the trained model file (.pth)
            device: 'cuda' or 'cpu'. Automatically selected if None
            num_frames: Number of frames sampled from each video
        """
        self.num_frames = num_frames
        
        # Select computation device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Loading model on {self.device}")
        
        # Create model architecture
        self.model = VideoClassifier(num_classes=2, dropout=0.3)
        
        # Ensure model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        
        # Support different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # 🔧 FIX: Handle classifier mismatch automatically
        new_state_dict = {}
        for k, v in state_dict.items():
            # Convert "classifier.1.1.weight" → "classifier.1.weight"
            if "classifier.1.1." in k:
                k = k.replace("classifier.1.1.", "classifier.1.")
            new_state_dict[k] = v

        # Load adjusted weights
        self.model.load_state_dict(new_state_dict, strict=True)
        
        # Prepare model for inference
        self.model.to(self.device)
        self.model.eval()
        
        # Define frame preprocessing (must match training setup)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print("Model ready")
    
    def extract_frames(self, video_path):
        """
        Extract evenly spaced frames from a video.

        Args:
            video_path: Path to video file

        Returns:
            List of frames as PIL images
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            raise ValueError(f"Video has no frames: {video_path}")
        
        frames = []
        
        # Select frame indices evenly across video
        if total_frames >= self.num_frames:
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            frame_indices = list(range(total_frames))
            frame_indices += [total_frames - 1] * (self.num_frames - total_frames)
        
        # Extract frames
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR (OpenCV) to RGB (PIL)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
            else:
                # If read fails, reuse last valid frame
                if frames:
                    frames.append(frames[-1])
        
        cap.release()
        
        # Ensure correct number of frames
        if len(frames) < self.num_frames:
            frames += [frames[-1]] * (self.num_frames - len(frames))
        
        return frames[:self.num_frames]
    
    def preprocess_frames(self, frames):
        """
        Convert frames into a tensor suitable for model input.

        Args:
            frames: List of PIL images

        Returns:
            Tensor of shape [1, num_frames, 3, 224, 224]
        """
        processed_frames = []
        
        for frame in frames:
            tensor = self.transform(frame)
            processed_frames.append(tensor)
        
        video_tensor = torch.stack(processed_frames)
        video_tensor = video_tensor.unsqueeze(0)
        
        return video_tensor
    
    def predict(self, video_path):
        """
        Perform prediction on a video file.

        Args:
            video_path: Path to video

        Returns:
            Dictionary with prediction results
        """
        frames = self.extract_frames(video_path)
        video_tensor = self.preprocess_frames(frames)
        video_tensor = video_tensor.to(self.device)
        
        with torch.no_grad():
            logits = self.model(video_tensor)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
        
        # Convert probabilities to percentages
        real_prob = probabilities[0, 0].item() * 100
        fake_prob = probabilities[0, 1].item() * 100
        
        # Determine label
        if fake_prob > real_prob:
            prediction = "fake"
            confidence = fake_prob
        else:
            prediction = "real"
            confidence = real_prob
        
        # ✅ SAME AS IMAGE LOGGING
        logger.info(f"Prediction: {prediction} ({confidence:.2f}%)")
        
        return {
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "probabilities": {
                "fake": round(fake_prob, 2),
                "real": round(real_prob, 2)
            },
            "frames_analyzed": self.num_frames
        }


def load_video_detector(model_path="models/video_model.pth", device=None):
    """
    Create and return a video detector instance.

    Args:
        model_path: Path to model file
        device: Device for inference

    Returns:
        VideoDeepfakeDetector instance
    """
    return VideoDeepfakeDetector(model_path=model_path, device=device)