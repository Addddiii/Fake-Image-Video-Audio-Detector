"""
Shared model architecture definition for training and inference
This ensures the model structure is consistent between training and deployment
"""

import torch.nn as nn
from torchvision import models


def create_fake_detection_model(num_classes=2, pretrained=True):
    """
    Create the fake image detection model architecture
    
    Args:
        num_classes: Number of output classes (default: 2 for fake/real)
        pretrained: Whether to load pretrained ImageNet weights (use True for training, False for inference)
    
    Returns:
        Model instance
    """
    # Use EfficientNet-B0 architecture (same as train_image.py)
    if pretrained:
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
    else:
        model = models.efficientnet_b0(weights=None)
    
    # Modify classifier for binary classification (fake vs real)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    return model
