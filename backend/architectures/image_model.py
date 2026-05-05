"""
Image Model Architecture

Defines the neural network structure used for fake vs real image classification.
This file is shared between training and inference to ensure consistency.
"""

import torch.nn as nn
from torchvision import models


def create_fake_detection_model(num_classes=2, pretrained=True):
    """
    Create the image classification model.

    Args:
        num_classes: Number of output classes (default is 2: fake and real)
        pretrained: If True, load ImageNet pretrained weights (useful for training).
                    If False, initialize weights from scratch (used during inference loading).

    Returns:
        Configured EfficientNet-B0 model
    """

    # Initialize EfficientNet-B0 backbone
    if pretrained:
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
    else:
        model = models.efficientnet_b0(weights=None)

    # Replace the final classification layer to match the number of classes
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model