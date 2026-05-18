# """
# Audio Model Architecture

# Defines the neural network used for audio-based deepfake detection.
# Audio is converted into a spectrogram and treated as an image input.
# """

# import torch
# import torch.nn as nn
# from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


# class AudioClassifier(nn.Module):
#     """
#     Audio classification model for fake vs real detection.

#     The model uses EfficientNet-B0 as a feature extractor and applies
#     a custom classification head for binary prediction.
#     """
    
#     def __init__(self, num_classes=2):
#         """
#         Initialize the audio classification model.

#         Args:
#             num_classes: Number of output classes (default is 2: real and fake)
#         """
#         super(AudioClassifier, self).__init__()
        
#         # Load EfficientNet-B0 with pretrained weights
#         efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        
#         # Use convolutional layers as feature extractor
#         self.features = efficientnet.features
        
#         # Global pooling layer
#         self.avgpool = efficientnet.avgpool
        
#         # Replace classifier head for binary classification
#         self.classifier = nn.Sequential(
#             nn.Dropout(p=0.2, inplace=True),
#             nn.Linear(efficientnet.classifier[1].in_features, num_classes)
#         )
    
#     def forward(self, x):
#         """
#         Forward pass of the model.

#         Args:
#             x: Input tensor of shape (batch_size, 3, 224, 224)

#         Returns:
#             Tensor of shape (batch_size, num_classes)
#         """
#         x = self.features(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x


# def build_audio_model(num_classes=2):
#     """
#     Create and return the audio classification model.

#     Args:
#         num_classes: Number of output classes

#     Returns:
#         AudioClassifier instance
#     """
#     return AudioClassifier(num_classes=num_classes)


"""
Audio Model Architecture

Defines the neural network used for audio-based deepfake detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_src = nn.Linear(out_dim, 1, bias=False)
        self.attn_dst = nn.Linear(out_dim, 1, bias=False)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x):
        h = self.fc(x)
        e = self.leaky(self.attn_src(h) + self.attn_dst(h).transpose(1, 2))
        alpha = F.softmax(e, dim=-1)
        return F.elu(torch.bmm(alpha, h))


class SpectroTemporalBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.spectral_gat = GraphAttentionLayer(channels, channels)
        self.temporal_gat = GraphAttentionLayer(channels, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        B, C, Fq, T = x.shape
        spec = x.permute(0, 3, 2, 1).reshape(B * T, Fq, C)
        spec = self.spectral_gat(spec).reshape(B, T, Fq, C).permute(0, 3, 2, 1)
        temp = x.permute(0, 2, 3, 1).reshape(B * Fq, T, C)
        temp = self.temporal_gat(temp).reshape(B, Fq, T, C).permute(0, 3, 1, 2)
        out = self.norm((spec + temp).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return out + x


class SincConv(nn.Module):
    def __init__(self, out_channels=70, kernel_size=513, sample_rate=16000):
        super().__init__()
        self.kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.out_channels = out_channels
        self.sample_rate = sample_rate

        low_hz, high_hz = 30.0, sample_rate / 2 - 200.0
        mel_pts = torch.linspace(self._hz2mel(low_hz), self._hz2mel(high_hz), out_channels + 2)
        hz_pts = self._mel2hz(mel_pts)
        self.freq_low  = nn.Parameter(hz_pts[:-2].unsqueeze(1))
        self.freq_band = nn.Parameter((hz_pts[1:-1] - hz_pts[:-2]).unsqueeze(1))

        half = self.kernel_size // 2
        t = torch.arange(1, half + 1, dtype=torch.float32)
        self.register_buffer('t_', t)
        win = 0.54 - 0.46 * torch.cos(math.pi * t / (half + 1))
        self.register_buffer('window_', win)

    @staticmethod
    def _hz2mel(hz): return 2595.0 * math.log10(1.0 + hz / 700.0)

    @staticmethod
    def _mel2hz(mel): return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def forward(self, x):
        sr = float(self.sample_rate)
        f1 = torch.abs(self.freq_low) / (sr / 2)
        f2 = torch.clamp(f1 + torch.abs(self.freq_band) / (sr / 2), max=1.0)
        t = self.t_.unsqueeze(0)
        band = (2 * f2 * torch.sinc(2 * f2 * t) - 2 * f1 * torch.sinc(2 * f1 * t)) * self.window_
        zero = torch.zeros(self.out_channels, 1, device=band.device)
        kernel = torch.cat([band.flip(dims=[1]), zero, band], dim=1)
        kernel = kernel / (kernel.abs().max(dim=1, keepdim=True)[0] + 1e-8)
        return F.conv1d(x, kernel.unsqueeze(1), padding=self.kernel_size // 2)


class AudioClassifier(nn.Module):
    """
    Audio classification model for fake vs real detection.

    # Architecture based on AASIST (Jung et al., CLOVA AI, ICASSP 2022),
    # finetuned and adapted to fit our inference pipeline and codebase.
    """

    def __init__(self, num_classes=2, sinc_channels=70, base_channels=64):
        super().__init__()
        self.sinc    = SincConv(out_channels=sinc_channels)
        self.sinc_bn = nn.BatchNorm1d(sinc_channels)
        self.encoder = nn.Sequential(
            nn.Conv1d(sinc_channels, base_channels, 3, padding=1),
            nn.BatchNorm1d(base_channels), nn.GELU(),
            nn.Conv1d(base_channels, base_channels, 3, padding=1, groups=base_channels),
            nn.Conv1d(base_channels, base_channels * 2, 1),
            nn.BatchNorm1d(base_channels * 2), nn.GELU(),
        )
        self.embed_dim  = base_channels * 2
        self.freq_bins  = 16
        self.st_block1  = SpectroTemporalBlock(self.embed_dim)
        self.st_block2  = SpectroTemporalBlock(self.embed_dim)
        self.pool       = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(self.embed_dim, 64), nn.GELU(),
            nn.Linear(64, num_classes),
        )

    def _spectrogram_to_waveform(self, x):
        x = x.mean(dim=1, keepdim=True)
        x = x.mean(dim=2)
        return F.interpolate(x, size=64000, mode='linear', align_corners=False)

    def forward(self, x):
        wav = self._spectrogram_to_waveform(x)
        out = F.gelu(self.sinc_bn(self.sinc(wav)))
        out = self.encoder(out)
        B, C, L = out.shape
        T = L // self.freq_bins
        out = out[:, :, :T * self.freq_bins].reshape(B, C, self.freq_bins, T)
        out = self.st_block2(self.st_block1(out))
        return self.classifier(self.pool(out).flatten(1))


def build_audio_model(num_classes=2):
    return AudioClassifier(num_classes=num_classes)