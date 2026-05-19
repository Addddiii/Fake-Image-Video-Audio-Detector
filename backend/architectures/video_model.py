"""
Video Model Architecture

Defines the neural network used for video-based deepfake detection.

Includes:
1. VideoClassifier
   - Frame-averaging EfficientNet-B0 model
   - Kept for backward compatibility

2. VideoClassifierLSTM
   - EfficientNet-B0 frame feature extractor
   - BiLSTM temporal model
   - Optional 4 handcrafted video features:
        1. mouth/lip movement score
        2. face motion consistency score
        3. eye/blink movement score
        4. artifact/compression inconsistency score
"""

import torch
import torch.nn as nn
from torchvision import models


def _adapt_efficientnet_input_channels(model: nn.Module, in_channels: int) -> nn.Module:
    """
    Adapt EfficientNet-B0 first convolution layer to support custom input channels.

    Default EfficientNet expects 3 channels:
        RGB

    Your video training uses 6 channels:
        RGB + motion difference

    Args:
        model: torchvision EfficientNet model
        in_channels: number of input channels

    Returns:
        model with adapted first convolution layer
    """

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
        # Copy pretrained RGB weights into first 3 channels
        new_conv.weight[:, :3, :, :] = old_conv.weight.data.clone()

        # Initialise extra channels, for example motion channels
        if in_channels > 3:
            nn.init.kaiming_normal_(
                new_conv.weight[:, 3:, :, :],
                mode="fan_out",
                nonlinearity="relu"
            )

    if isinstance(first_block, nn.Sequential):
        model.features[0][0] = new_conv
    else:
        model.features[0] = new_conv

    return model


class VideoClassifier(nn.Module):
    """
    Frame-averaging video classifier for fake vs real detection.

    This model processes every frame independently with EfficientNet-B0,
    then averages the frame predictions.

    Kept for backward compatibility.
    """

    def __init__(self, num_classes=2, dropout=0.3, in_channels=3):
        super().__init__()

        efficientnet = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )

        efficientnet = _adapt_efficientnet_input_channels(
            efficientnet,
            in_channels
        )

        self.features = efficientnet.features
        self.avgpool = efficientnet.avgpool

        in_features = efficientnet.classifier[1].in_features

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x:
                Tensor shaped [batch, frames, channels, height, width]

        Returns:
            logits:
                Tensor shaped [batch, num_classes]
        """

        batch_size, num_frames, channels, height, width = x.shape

        x = x.view(batch_size * num_frames, channels, height, width)

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        logits = self.classifier(x)
        logits = logits.view(batch_size, num_frames, -1)

        # Average predictions across frames
        logits = logits.mean(dim=1)

        return logits


class VideoClassifierLSTM(nn.Module):
    """
    Temporal video classifier for fake vs real detection.

    This model uses:

    1. EfficientNet-B0
       Extracts spatial features from each frame.

    2. BiLSTM
       Learns temporal motion patterns across frames.

    3. Optional 4 handcrafted features:
       - mouth/lip movement score
       - face motion consistency score
       - eye/blink movement score
       - artifact/compression inconsistency score

    Input video can be:
        RGB only:
            in_channels = 3

        RGB + motion channels:
            in_channels = 6

    Extra features are passed separately as:
        extra_features shaped [batch, 4]
    """

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

        # =========================
        # EfficientNet-B0 backbone
        # =========================

        efficientnet = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )

        efficientnet = _adapt_efficientnet_input_channels(
            efficientnet,
            in_channels
        )

        self.features = efficientnet.features
        self.avgpool = efficientnet.avgpool

        encoder_dim = efficientnet.classifier[1].in_features  # 1280

        # =========================
        # Frame-level spatial head
        # =========================

        self.frame_classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(encoder_dim, num_classes)
        )

        # =========================
        # BiLSTM temporal model
        # =========================

        self.lstm = nn.LSTM(
            input_size=encoder_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        lstm_out_dim = lstm_hidden * 2

        # =========================
        # 4-feature encoder
        # =========================

        if self.use_extra_features:
            self.extra_encoder = nn.Sequential(
                nn.Linear(extra_feature_dim, 32),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(32, 32),
                nn.ReLU(inplace=True),
            )

            temporal_input_dim = lstm_out_dim + 32
        else:
            self.extra_encoder = None
            temporal_input_dim = lstm_out_dim

        # =========================
        # Temporal final classifier
        # =========================

        self.temporal_classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(temporal_input_dim, num_classes)
        )

    def forward(self, x, extra_features=None):
        """
        Args:
            x:
                Tensor shaped:
                    [batch, frames, channels, height, width]

            extra_features:
                Optional tensor shaped:
                    [batch, 4]

                Feature order:
                    0 = mouth/lip movement score
                    1 = face motion consistency score
                    2 = eye/blink movement score
                    3 = artifact/compression inconsistency score

        Returns:
            spatial_logits:
                Tensor shaped [batch, num_classes]

            temporal_logits:
                Tensor shaped [batch, num_classes]

            per_frame_logits:
                Tensor shaped [batch, frames, num_classes]
        """

        batch_size, num_frames, channels, height, width = x.shape

        # =========================
        # EfficientNet per-frame features
        # =========================

        x_flat = x.view(batch_size * num_frames, channels, height, width)

        features = self.features(x_flat)
        features = self.avgpool(features)
        features = torch.flatten(features, 1)  # [batch * frames, 1280]

        # =========================
        # Spatial prediction head
        # =========================

        per_frame_logits = self.frame_classifier(features)
        per_frame_logits = per_frame_logits.view(batch_size, num_frames, -1)

        spatial_logits = per_frame_logits.mean(dim=1)

        # =========================
        # Temporal LSTM head
        # =========================

        seq_features = features.view(batch_size, num_frames, -1)

        lstm_out, _ = self.lstm(seq_features)

        # Average temporal features over all frames
        temporal_pooled = lstm_out.mean(dim=1)

        # =========================
        # Add 4 handcrafted features
        # =========================

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
                dtype=x.dtype
            )

            extra_encoded = self.extra_encoder(extra_features)

            temporal_pooled = torch.cat(
                [temporal_pooled, extra_encoded],
                dim=1
            )

        # =========================
        # Final temporal prediction
        # =========================

        temporal_logits = self.temporal_classifier(temporal_pooled)

        return spatial_logits, temporal_logits, per_frame_logits