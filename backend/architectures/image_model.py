"""
Image model architecture.
EfficientNet-B0 with handcrafted image features.
"""

import torch
import torch.nn as nn
from torchvision import models


class ImageClassifier(nn.Module):
    def __init__(
        self,
        num_classes=2,
        dropout=0.4,
        extra_feature_dim=4,
        use_extra_features=True,
        pretrained=True,
    ):
        super().__init__()

        self.use_extra_features = use_extra_features
        self.extra_feature_dim = extra_feature_dim

        if pretrained:
            weights = models.EfficientNet_B0_Weights.DEFAULT
            efficientnet = models.efficientnet_b0(weights=weights)
        else:
            efficientnet = models.efficientnet_b0(weights=None)

        self.features = efficientnet.features
        self.avgpool = efficientnet.avgpool

        image_feature_dim = efficientnet.classifier[1].in_features

        if self.use_extra_features:
            self.extra_encoder = nn.Sequential(
                nn.Linear(extra_feature_dim, 32),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(32, 32),
                nn.ReLU(inplace=True),
            )

            classifier_input_dim = image_feature_dim + 32
        else:
            self.extra_encoder = None
            classifier_input_dim = image_feature_dim

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(classifier_input_dim, num_classes),
        )

    def forward(self, x, extra_features=None):
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

            extra_features = extra_features.to(
                device=x.device,
                dtype=x.dtype,
            )

            extra_encoded = self.extra_encoder(extra_features)
            x = torch.cat([x, extra_encoded], dim=1)

        return self.classifier(x)


def create_fake_detection_model(
    num_classes=2,
    pretrained=True,
    dropout=0.4,
    extra_feature_dim=4,
    use_extra_features=True,
):
    return ImageClassifier(
        num_classes=num_classes,
        dropout=dropout,
        extra_feature_dim=extra_feature_dim,
        use_extra_features=use_extra_features,
        pretrained=pretrained,
    )