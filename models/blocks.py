from typing import Literal, Optional

import torch.nn as nn


NormType = Literal["instance", "group", "none"]
ActType = Literal["leaky_relu", "silu", "relu", "none"]


def make_norm_3d(channels: int, norm: NormType = "instance", num_groups: int = 8) -> nn.Module:
    """Create a configurable 3D normalization layer."""

    if norm == "instance":
        return nn.InstanceNorm3d(channels)
    if norm == "group":
        groups = min(num_groups, channels)
        while channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if norm == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported norm type: {norm}")


def make_activation(act: ActType = "leaky_relu", negative_slope: float = 0.2) -> nn.Module:
    """Create a configurable activation function."""

    if act == "leaky_relu":
        return nn.LeakyReLU(negative_slope, inplace=True)
    if act == "silu":
        return nn.SiLU(inplace=True)
    if act == "relu":
        return nn.ReLU(inplace=True)
    if act == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported activation: {act}")


class ConvNormAct3D(nn.Module):
    """Conv3d + normalization + activation block used across LRDP modules."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        norm: NormType = "instance",
        activation: ActType = "leaky_relu",
        num_groups: int = 8,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            make_norm_3d(out_channels, norm=norm, num_groups=num_groups),
            make_activation(activation),
        )

    def forward(self, x):
        return self.block(x)


class ConvBlock3D(nn.Module):
    """Two stacked configurable 3D convolution blocks."""

    def __init__(self, in_channels: int, out_channels: int, norm: NormType = "instance", activation: ActType = "leaky_relu"):
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct3D(in_channels, out_channels, norm=norm, activation=activation),
            ConvNormAct3D(out_channels, out_channels, norm=norm, activation=activation),
        )

    def forward(self, x):
        return self.block(x)


class ResBlock3D(nn.Module):
    """Residual 3D convolution block for feature extraction and refinement."""

    def __init__(self, channels: int, norm: NormType = "instance", activation: ActType = "leaky_relu"):
        super().__init__()
        self.conv1 = ConvNormAct3D(channels, channels, norm=norm, activation=activation)
        self.conv2 = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1),
            make_norm_3d(channels, norm=norm),
        )
        self.out_act = make_activation(activation)

    def forward(self, x):
        return self.out_act(x + self.conv2(self.conv1(x)))


class DownsampleBlock3D(nn.Module):
    """Stride-2 downsampling block for 3D feature pyramids."""

    def __init__(self, in_channels: int, out_channels: int, norm: NormType = "instance", activation: ActType = "leaky_relu"):
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct3D(in_channels, out_channels, kernel_size=3, stride=2, norm=norm, activation=activation),
            ResBlock3D(out_channels, norm=norm, activation=activation),
        )

    def forward(self, x):
        return self.block(x)


class UpsampleBlock3D(nn.Module):
    """Trilinear upsampling followed by convolution."""

    def __init__(self, in_channels: int, out_channels: int, norm: NormType = "instance", activation: ActType = "leaky_relu"):
        super().__init__()
        self.conv = ConvNormAct3D(in_channels, out_channels, norm=norm, activation=activation)

    def forward(self, x, target_size=None):
        if target_size is not None:
            x = nn.functional.interpolate(x, size=target_size, mode="trilinear", align_corners=True)
        else:
            x = nn.functional.interpolate(x, scale_factor=2, mode="trilinear", align_corners=True)
        return self.conv(x)
