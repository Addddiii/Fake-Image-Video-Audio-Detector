"""
Video Model Architecture

Defines the neural network used for video-based deepfake detection.
Each video is processed as multiple frames, then frame predictions are
averaged into one final video-level prediction.
"""

import torch
import torch.nn as nn
from torchvision import models


class VideoClassifier(nn.Module):
    """
    Video classification model for fake vs real detection.

    This structure keeps EfficientNet-B0 layer names similar to the original
    torchvision model, which helps match saved training checkpoints.
    """

    def __init__(self, num_classes=2, dropout=0.3):
        """
        Initialize the video classification model.

        Args:
            num_classes: Number of output classes
            dropout: Dropout rate before final classifier
        """
        super().__init__()

        # Load EfficientNet-B0 backbone
        efficientnet = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )

        # Keep layer names compatible with EfficientNet checkpoints
        self.features = efficientnet.features
        self.avgpool = efficientnet.avgpool

        # Replace final classifier for fake/real classification
        in_features = efficientnet.classifier[1].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Tensor with shape [batch_size, num_frames, channels, height, width]

        Returns:
            Tensor with shape [batch_size, num_classes]
        """
        batch_size, num_frames, channels, height, width = x.shape

        # Process all frames through EfficientNet together
        x = x.view(batch_size * num_frames, channels, height, width)

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)

        # Convert frame-level predictions back to video-level predictions
        logits = logits.view(batch_size, num_frames, -1)
        logits = logits.mean(dim=1)

        return logits