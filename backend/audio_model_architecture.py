"""
Audio Deepfake Detection Model Architecture
Matches the architecture from train_audio.py
Uses EfficientNet-B0 backbone with custom classifier head for binary classification
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class AudioClassifier(nn.Module):
    """
    Audio deepfake detection model using EfficientNet-B0
    Input: Spectrogram converted to (3, 224, 224) tensor
    Output: 2-class logits (real=0, fake=1)
    
    Note: This class directly extends EfficientNet-B0 to match the training architecture
    """
    
    def __init__(self, num_classes=2):
        super(AudioClassifier, self).__init__()
        
        # Load pretrained EfficientNet-B0 and get its components
        efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        
        # Copy the features backbone
        self.features = efficientnet.features
        
        # Copy and modify the classifier head
        self.avgpool = efficientnet.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(efficientnet.classifier[1].in_features, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape (B, 3, 224, 224) - spectrogram as RGB image
        Returns:
            logits: Output tensor of shape (B, 2) - class logits
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def build_audio_model(num_classes=2):
    """
    Build and return the audio classifier model
    
    Args:
        num_classes: Number of output classes (default: 2 for real/fake)
    
    Returns:
        AudioClassifier model instance
    """
    return AudioClassifier(num_classes=num_classes)


if __name__ == "__main__":
    # Test model creation and forward pass
    model = build_audio_model()
    print(f"Model created successfully")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output logits: {output}")
