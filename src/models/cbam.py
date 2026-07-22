"""CBAM (Convolutional Block Attention Module) and a wrapper for inserting
it after a YOLOv8/Ultralytics C2f block."""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        mx = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg + mx)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg, mx], dim=1)
        return x * self.sigmoid(self.conv(attn))


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class C2fWithCBAM(nn.Module):
    """Wraps an existing Ultralytics C2f layer, applying CBAM to its output."""

    def __init__(self, c2f_layer, in_channels):
        super().__init__()
        self.c2f = c2f_layer
        self.cbam = CBAM(in_channels)

    def forward(self, x):
        return self.cbam(self.c2f(x))


def add_cbam_to_backbone(model, num_layers=2):
    """Inserts CBAM after the last `num_layers` C2f blocks of a YOLO model's backbone.

    Args:
        model: an ultralytics.YOLO instance (already loaded).
        num_layers: how many of the final C2f blocks to wrap with CBAM.
    """
    from ultralytics.nn.modules import C2f

    backbone = model.model.model
    c2f_indices = [i for i, layer in enumerate(backbone) if isinstance(layer, C2f)]
    target_indices = c2f_indices[-num_layers:] if len(c2f_indices) >= num_layers else c2f_indices

    for idx in target_indices:
        layer = backbone[idx]
        out_channels = layer.cv2.conv.out_channels if hasattr(layer, "cv2") else 256
        backbone[idx] = C2fWithCBAM(layer, out_channels)

    return model
