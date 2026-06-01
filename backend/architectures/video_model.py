"""
Video model architecture.
Uses EfficientNet-B0, motion channels, BiLSTM, and video features.
"""

import torch
import torch.nn as nn
from torchvision import models


def adapt_efficientnet_input_channels(model: nn.Module, in_channels: int) -> nn.Module:
    if in_channels == 3:
        return model

    first_block = model.features[0]
    old_conv = first_block[0] if isinstance(first_block, nn.Sequential) else first_block

    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )

    with torch.no_grad():
        new_conv.weight[:, :3, :, :] = old_conv.weight.data.clone()

        if in_channels > 3:
            nn.init.kaiming_normal_(
                new_conv.weight[:, 3:, :, :],
                mode="fan_out",
                nonlinearity="relu",
            )

    if isinstance(first_block, nn.Sequential):
        model.features[0][0] = new_conv
    else:
        model.features[0] = new_conv

    return model


class VideoClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3, in_channels=3):
        super().__init__()

        efficientnet = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT,
        )

        efficientnet = adapt_efficientnet_input_channels(
            efficientnet,
            in_channels,
        )

        self.features = efficientnet.features
        self.avgpool = efficientnet.avgpool

        in_features = efficientnet.classifier[1].in_features

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.shape

        x = x.view(batch_size * num_frames, channels, height, width)

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        logits = self.classifier(x)
        logits = logits.view(batch_size, num_frames, -1)

        return logits.mean(dim=1)


class VideoClassifierLSTM(nn.Module):
    def __init__(
        self,
        num_classes=2,
        lstm_hidden=256,
        lstm_layers=2,
        dropout=0.5,
        in_channels=3,
        extra_feature_dim=4,
        use_extra_features=True,
    ):
        super().__init__()

        self.use_extra_features = use_extra_features
        self.extra_feature_dim = extra_feature_dim

        efficientnet = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT,
        )

        efficientnet = adapt_efficientnet_input_channels(
            efficientnet,
            in_channels,
        )

        self.features = efficientnet.features
        self.avgpool = efficientnet.avgpool

        encoder_dim = efficientnet.classifier[1].in_features

        self.frame_classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(encoder_dim, num_classes),
        )

        self.lstm = nn.LSTM(
            input_size=encoder_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        lstm_output_dim = lstm_hidden * 2

        if self.use_extra_features:
            self.extra_encoder = nn.Sequential(
                nn.Linear(extra_feature_dim, 32),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(32, 32),
                nn.ReLU(inplace=True),
            )

            temporal_input_dim = lstm_output_dim + 32
        else:
            self.extra_encoder = None
            temporal_input_dim = lstm_output_dim

        self.temporal_classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(temporal_input_dim, num_classes),
        )

    def forward(self, x, extra_features=None):
        batch_size, num_frames, channels, height, width = x.shape

        x_flat = x.view(batch_size * num_frames, channels, height, width)

        features = self.features(x_flat)
        features = self.avgpool(features)
        features = torch.flatten(features, 1)

        per_frame_logits = self.frame_classifier(features)
        per_frame_logits = per_frame_logits.view(batch_size, num_frames, -1)

        spatial_logits = per_frame_logits.mean(dim=1)

        sequence_features = features.view(batch_size, num_frames, -1)

        lstm_output, _ = self.lstm(sequence_features)
        temporal_pooled = lstm_output.mean(dim=1)

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
            temporal_pooled = torch.cat([temporal_pooled, extra_encoded], dim=1)

        temporal_logits = self.temporal_classifier(temporal_pooled)

        return spatial_logits, temporal_logits, per_frame_logits