"""
Audio model architecture.

EfficientNet-B0 is used to classify audio spectrograms, with optional
handcrafted audio features added before the final classification layer.
"""

import torch
import torch.nn as nn
from torchvision import models


class AudioClassifier(nn.Module):
    """
    EfficientNet-B0 based classifier for real/fake audio detection.
    """

    def __init__(
        self,
        num_classes=2,
        dropout=0.4,
        extra_feature_dim=13,
        use_extra_features=True,
        pretrained=True,
    ):
        super().__init__()

        self.use_extra_features = use_extra_features
        self.extra_feature_dim = extra_feature_dim

        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        efficientnet = models.efficientnet_b0(weights=weights)

        self.features = efficientnet.features
        self.avgpool = efficientnet.avgpool

        audio_feature_dim = efficientnet.classifier[1].in_features

        if use_extra_features:
            self.extra_encoder = nn.Sequential(
                nn.Linear(extra_feature_dim, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(64, 32),
                nn.ReLU(inplace=True),
            )
            classifier_input_dim = audio_feature_dim + 32
        else:
            self.extra_encoder = None
            classifier_input_dim = audio_feature_dim

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(classifier_input_dim, num_classes),
        )

    def forward(self, x, extra_features=None):
        """
        Run the spectrogram and optional handcrafted features through the model.
        """
        batch_size = x.size(0)

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if self.use_extra_features:
            if extra_features is None:
                extra_features = torch.zeros(
                    batch_size,
                    self.extra_feature_dim,
                    device=x.device,
                    dtype=x.dtype,
                )

            extra_features = extra_features.to(device=x.device, dtype=x.dtype)
            extra_encoded = self.extra_encoder(extra_features)

            x = torch.cat([x, extra_encoded], dim=1)

        return self.classifier(x)


def create_audio_detection_model(
    num_classes=2,
    pretrained=True,
    dropout=0.4,
    extra_feature_dim=13,
    use_extra_features=True,
):
    """
    Create and return the audio detection model.
    """
    return AudioClassifier(
        num_classes=num_classes,
        dropout=dropout,
        extra_feature_dim=extra_feature_dim,
        use_extra_features=use_extra_features,
        pretrained=pretrained,
    )