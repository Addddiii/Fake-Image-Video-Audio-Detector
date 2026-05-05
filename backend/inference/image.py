"""
Image Deepfake Detection Inference

This module loads a trained image classification model and performs
fake vs real prediction on input images.
"""

import torch
from torchvision import transforms
from PIL import Image
import os
from typing import Dict
import logging

# Import the same model architecture used during training
from architectures.image_model import create_fake_detection_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FakeImageDetector:
    """
    Handles model loading, image preprocessing, and prediction.
    The architecture must match the one used during training.
    """
    
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize the detector with model weights and device.

        Args:
            model_path: Path to the trained .pth file
            device: 'cuda' or 'cpu'. If None, automatically selects available device
        """
        self.model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = None

        # Class order must match training dataset
        self.class_names = ['fake', 'real']
        
        self._load_model()
        self._setup_transforms()
        
    def _load_model(self):
        """
        Load the trained model weights into the architecture.
        """
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            # Create model architecture
            self.model = create_fake_detection_model(num_classes=2, pretrained=False)
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
            
            # Support both plain state_dict and checkpoint dict formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Checkpoint loaded (epoch: {checkpoint.get('epoch', 'unknown')})")
            else:
                self.model.load_state_dict(checkpoint)
            
            # Move model to device and set evaluation mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model ready on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _setup_transforms(self):
        """
        Define image preprocessing steps.
        These must match the transformations used during training.
        """
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Load and preprocess an image for model input.

        Args:
            image_path: Path to image file

        Returns:
            Tensor ready for model inference
        """
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image)

            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0)

            return image_tensor.to(self.device)

        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
    
    def predict(self, image_path: str) -> Dict:
        """
        Perform prediction on a single image.

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with prediction, confidence, and probabilities
        """
        try:
            # Prepare input
            image_tensor = self.preprocess_image(image_path)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
            
            # Convert probabilities to percentages
            fake_prob = probabilities[0][0].item() * 100
            real_prob = probabilities[0][1].item() * 100

            predicted_label = self.class_names[predicted_class.item()]
            confidence_percent = confidence.item() * 100
            
            result = {
                'prediction': predicted_label,
                'confidence': round(confidence_percent, 2),
                'probabilities': {
                    'fake': round(fake_prob, 2),
                    'real': round(real_prob, 2)
                }
            }
            
            logger.info(f"Prediction: {predicted_label} ({confidence_percent:.2f}%)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise


# Global model instance (loaded once at startup)
_model_instance = None


def initialize_model(model_path: str):
    """
    Load the model once and store it globally.

    Args:
        model_path: Path to model file

    Returns:
        True if loaded successfully, False otherwise
    """
    global _model_instance
    
    if not os.path.exists(model_path):
        logger.warning(f"Model file not found at {model_path}")
        logger.warning("Image predictions will not be available.")
        return False
    
    try:
        _model_instance = FakeImageDetector(model_path)
        logger.info("Image model initialized")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        return False


def get_model() -> FakeImageDetector:
    """
    Retrieve the global model instance.

    Returns:
        Model instance or None if not initialized
    """
    return _model_instance


def predict_image(image_path: str) -> Dict:
    """
    Run prediction using the global model instance.

    Args:
        image_path: Path to image file

    Returns:
        Prediction result dictionary

    Raises:
        RuntimeError if model is not initialized
    """
    if _model_instance is None:
        raise RuntimeError("Model not initialized. Call initialize_model() first.")
    
    return _model_instance.predict(image_path)