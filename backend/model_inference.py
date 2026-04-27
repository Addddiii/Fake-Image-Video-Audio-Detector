"""
Model Inference Module for Fake Image Detection
Loads the trained model and makes predictions on uploaded images
"""

import torch
from torchvision import transforms
from PIL import Image
import os
from typing import Dict
import logging

# Import shared model architecture (same as used in training)
from model_architecture import create_fake_detection_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FakeImageDetector:
    """
    Wrapper class for the PyTorch fake image detection model
    Uses the same architecture as train_image.py for compatibility
    """
    
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize the model
        
        Args:
            model_path: Path to the .pth model file (trained weights)
            device: Device to run inference on ('cuda' or 'cpu'). Auto-detect if None.
        """
        self.model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = None
        self.class_names = ['fake', 'real']  # Based on ImageFolder alphabetical order
        
        self._load_model()
        self._setup_transforms()
        
    def _load_model(self):
        """Load the trained PyTorch model"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            # Create model with same architecture as training (EfficientNet-B0)
            # pretrained=False because we're loading our own trained weights
            self.model = create_fake_detection_model(num_classes=2, pretrained=False)
            
            # Load the trained weights from .pth file
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model = self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode (disables dropout, etc.)
            
            logger.info(f"✓ Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _setup_transforms(self):
        """Setup image preprocessing transforms (same as training)"""
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
        Preprocess an image for model input
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image tensor
        """
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image)
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
            return image_tensor.to(self.device)
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
    
    def predict(self, image_path: str) -> Dict:
        """
        Make a prediction on an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing prediction results:
            {
                'prediction': 'fake' or 'real',
                'confidence': float (0-100),
                'probabilities': {
                    'fake': float (0-100),
                    'real': float (0-100)
                }
            }
        """
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image_path)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
            
            # Convert to percentages
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
            
            logger.info(f"Prediction: {predicted_label} ({confidence_percent:.2f}% confidence)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise


# Global model instance (loaded once when server starts)
_model_instance = None


def initialize_model(model_path: str):
    """
    Initialize the global model instance
    
    Args:
        model_path: Path to the .pth model file
    """
    global _model_instance
    
    if not os.path.exists(model_path):
        logger.warning(f"Model file not found at {model_path}")
        logger.warning("Predictions will not be available. Place your trained model at this path.")
        return False
    
    try:
        _model_instance = FakeImageDetector(model_path)
        logger.info("✓ Model initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        return False


def get_model() -> FakeImageDetector:
    """
    Get the global model instance
    
    Returns:
        FakeImageDetector instance or None if not initialized
    """
    return _model_instance


def predict_image(image_path: str) -> Dict:
    """
    Make a prediction on an image using the global model
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary containing prediction results
        
    Raises:
        RuntimeError: If model is not initialized
    """
    if _model_instance is None:
        raise RuntimeError("Model not initialized. Call initialize_model() first.")
    
    return _model_instance.predict(image_path)
