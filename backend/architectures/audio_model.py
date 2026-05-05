"""
Audio Model Architecture

Defines the neural network used for audio-based deepfake detection.
Audio is converted into a spectrogram and treated as an image input.
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class AudioClassifier(nn.Module):
    """
    Audio classification model for fake vs real detection.

    The model uses EfficientNet-B0 as a feature extractor and applies
    a custom classification head for binary prediction.
    """
    
    def __init__(self, num_classes=2):
        """
        Initialize the audio classification model.

        Args:
            num_classes: Number of output classes (default is 2: real and fake)
        """
        super(AudioClassifier, self).__init__()
        
        # Load EfficientNet-B0 with pretrained weights
        efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        
        # Use convolutional layers as feature extractor
        self.features = efficientnet.features
        
        # Global pooling layer
        self.avgpool = efficientnet.avgpool
        
        # Replace classifier head for binary classification
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(efficientnet.classifier[1].in_features, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            Tensor of shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def build_audio_model(num_classes=2):
    """
    Create and return the audio classification model.

    Args:
        num_classes: Number of output classes

    Returns:
        AudioClassifier instance
    """
    return AudioClassifier(num_classes=num_classes)