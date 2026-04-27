"""
Video deepfake detection model architecture
Uses EfficientNet-B0 as backbone with temporal aggregation for frame-level predictions
"""

import torch
import torch.nn as nn
from torchvision import models


class VideoClassifier(nn.Module):
    """
    Video classification model for deepfake detection
    
    Architecture:
    - EfficientNet-B0 backbone extracts features from individual frames
    - Temporal aggregation combines predictions across all frames
    - Final classification: fake (0) or real (1)
    """
    
    def __init__(self, num_classes=2, dropout=0.3):
        """
        Initialize the video classifier
        
        Args:
            num_classes: Number of output classes (default 2: fake/real)
            dropout: Dropout rate for regularization (default 0.3)
        """
        super().__init__()
        
        # Load pre-trained EfficientNet-B0 backbone
        weights = models.EfficientNet_B0_Weights.DEFAULT
        backbone = models.efficientnet_b0(weights=weights)
        
        # Get the number of features from the backbone
        in_features = backbone.classifier[1].in_features
        
        # Remove the original classifier head
        backbone.classifier = nn.Identity()
        
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape [batch_size, num_frames, channels, height, width]
        
        Returns:
            logits: Output tensor of shape [batch_size, num_classes]
        """
        # Input shape: [batch_size, num_frames, channels, height, width]
        batch_size, num_frames, channels, height, width = x.shape
        
        # Reshape to process all frames at once
        # New shape: [batch_size * num_frames, channels, height, width]
        x = x.view(batch_size * num_frames, channels, height, width)
        
        # Extract features using the backbone
        features = self.backbone(x)
        
        # Apply dropout for regularization
        features = self.dropout(features)
        
        # Get classification logits for each frame
        logits = self.classifier(features)
        
        # Reshape back to separate batch and time dimensions
        # Shape: [batch_size, num_frames, num_classes]
        logits = logits.view(batch_size, num_frames, -1)
        
        # Temporal aggregation: average predictions across all frames
        # This gives us a single prediction per video
        logits = logits.mean(dim=1)
        
        return logits
