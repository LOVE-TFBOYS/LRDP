from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvNormAct3D


class FeaturePyramid3D(nn.Module):
    """
    Optional top-down feature pyramid fusion.

    The module keeps scale count and spatial sizes aligned with encoder output
    and can be disabled in LRDP ablations with use_pyramid=False.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: Optional[Sequence[int]] = None,
        use_top_down: bool = True,
        norm: str = "instance",
        activation: str = "leaky_relu",
    ):
        super().__init__()
        self.in_channels = tuple(in_channels)
        self.out_channels = tuple(out_channels or in_channels)
        if len(self.in_channels) != len(self.out_channels):
            raise ValueError("in_channels and out_channels must have the same length")
        self.use_top_down = use_top_down
        self.lateral = nn.ModuleList(
            [
                ConvNormAct3D(in_ch, out_ch, kernel_size=1, padding=0, norm=norm, activation=activation)
                for in_ch, out_ch in zip(self.in_channels, self.out_channels)
            ]
        )
        # 平滑卷积模块列表
        self.smooth = nn.ModuleList(
            [ConvNormAct3D(ch, ch, kernel_size=3, norm=norm, activation=activation) for ch in self.out_channels]
        )
        # 上采样模块列表
        self.top_down = nn.ModuleList(
            [
                nn.Identity() if high_ch == low_ch else nn.Conv3d(high_ch, low_ch, kernel_size=1)
                for low_ch, high_ch in zip(self.out_channels[:-1], self.out_channels[1:])
            ]
        )

    def forward(self, features):
        projected = [proj(feat) for proj, feat in zip(self.lateral, features)]
        if self.use_top_down:
            fused = list(projected)
            for idx in range(len(fused) - 2, -1, -1):
                top = self.top_down[idx](fused[idx + 1])
                up = F.interpolate(top, size=fused[idx].shape[2:], mode="trilinear", align_corners=True)
                fused[idx] = fused[idx] + up
        else:
            fused = projected
        return [smooth(feat) for smooth, feat in zip(self.smooth, fused)]


class IdentityFeaturePyramid3D(nn.Module):
    """A no-op pyramid used when pyramid fusion is disabled."""

    def forward(self, features):
        return features
