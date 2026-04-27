"""
Video inference module for deepfake detection
Handles loading videos, extracting frames, and running model predictions
"""

import os
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from video_model_architecture import VideoClassifier


class VideoDeepfakeDetector:
    """
    Video deepfake detector that processes video files and returns predictions
    """
    
    def __init__(self, model_path, device=None, num_frames=20):
        """
        Initialize the video deepfake detector
        
        Args:
            model_path: Path to the trained model weights (.pth file)
            device: Device to run inference on (cpu or cuda). Auto-detects if None
            num_frames: Number of frames to extract from each video (default 20)
        """
        self.num_frames = num_frames
        
        # Set device for inference
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Loading model on {self.device}")
        
        # Initialize the model architecture
        self.model = VideoClassifier(num_classes=2, dropout=0.3)
        
        # Load trained weights
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Checkpoint saved with additional metadata
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Checkpoint is just the state dict
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Define image preprocessing transforms
        # These should match the transforms used during training
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print("Model loaded successfully")
    
    def extract_frames(self, video_path):
        """
        Extract evenly spaced frames from a video file
        
        Args:
            video_path: Path to the video file
        
        Returns:
            List of PIL Image objects representing extracted frames
        """
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get total number of frames in the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            raise ValueError(f"Video has no frames: {video_path}")
        
        frames = []
        
        # Calculate which frames to extract (evenly spaced)
        if total_frames >= self.num_frames:
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            # If video has fewer frames than needed, repeat the last frame
            frame_indices = list(range(total_frames))
            frame_indices += [total_frames - 1] * (self.num_frames - total_frames)
        
        # Extract the selected frames
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert from BGR (OpenCV format) to RGB (PIL format)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)
            else:
                # If frame reading fails, use the last successful frame
                if frames:
                    frames.append(frames[-1])
        
        cap.release()
        
        # Ensure we have exactly the right number of frames
        if len(frames) < self.num_frames:
            frames += [frames[-1]] * (self.num_frames - len(frames))
        
        return frames[:self.num_frames]
    
    def preprocess_frames(self, frames):
        """
        Preprocess extracted frames for model input
        
        Args:
            frames: List of PIL Image objects
        
        Returns:
            Tensor of shape [1, num_frames, 3, 224, 224]
        """
        # Apply transforms to each frame
        processed_frames = []
        for frame in frames:
            tensor = self.transform(frame)
            processed_frames.append(tensor)
        
        # Stack frames into a single tensor
        video_tensor = torch.stack(processed_frames)
        
        # Add batch dimension
        video_tensor = video_tensor.unsqueeze(0)
        
        return video_tensor
    
    def predict(self, video_path):
        """
        Run deepfake detection on a video file
        
        Args:
            video_path: Path to the video file
        
        Returns:
            Dictionary containing:
                - prediction: 'fake' or 'real'
                - confidence: Confidence score (0-100%)
                - probabilities: {fake: %, real: %}
                - frames_analyzed: Number of frames processed
        """
        # Extract frames from video
        frames = self.extract_frames(video_path)
        
        # Preprocess frames
        video_tensor = self.preprocess_frames(frames)
        video_tensor = video_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            logits = self.model(video_tensor)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
        
        # Extract probabilities for each class
        fake_prob = probabilities[0, 0].item() * 100
        real_prob = probabilities[0, 1].item() * 100
        
        # Determine prediction and confidence
        if predicted_class == 0:
            prediction = "fake"
            confidence = fake_prob
        else:
            prediction = "real"
            confidence = real_prob
        
        return {
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "probabilities": {
                "fake": round(fake_prob, 2),
                "real": round(real_prob, 2)
            },
            "frames_analyzed": self.num_frames
        }


# Helper function to create a detector instance
def load_video_detector(model_path="deepfake_video_model.pth", device=None):
    """
    Convenience function to load the video detector
    
    Args:
        model_path: Path to the model weights
        device: Device to use for inference
    
    Returns:
        VideoDeepfakeDetector instance
    """
    return VideoDeepfakeDetector(model_path=model_path, device=device)
