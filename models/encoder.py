from collections.abc import Sequence

import torch.nn as nn

from .blocks import ConvBlock3D, DownsampleBlock3D


class SingleStreamEncoder3D(nn.Module):
    """
    3D CNN feature encoder for one volume.

    Given an input [B, C, D, H, W], outputs four aligned scales by default:
    F1: [B, base, D, H, W]
    F2: [B, 2*base, D/2, H/2, W/2]
    F3: [B, 4*base, D/4, H/4, W/4]
    F4: [B, 8*base, D/8, H/8, W/8]
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
        channel_mults: Sequence[int] = (1, 2, 4, 8),
        norm: str = "instance",
        activation: str = "leaky_relu",
    ):
        super().__init__()
        self.channels = tuple(base_channels * mult for mult in channel_mults)
        self.stem = ConvBlock3D(in_channels, self.channels[0], norm=norm, activation=activation)
        self.down_blocks = nn.ModuleList(
            [
                DownsampleBlock3D(self.channels[i - 1], self.channels[i], norm=norm, activation=activation)
                for i in range(1, len(self.channels))
            ]
        )

    def forward(self, x):
        features = [self.stem(x)]
        for block in self.down_blocks:
            features.append(block(features[-1]))
        return features


class DualStreamEncoder3D(nn.Module):
    """
    Dual-stream encoder for LRDP.

    This module follows the RDP/LDM-Morph pattern of extracting fixed and
    moving latent pyramids independently. Set shared_encoder=True for ablation.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
        channel_mults: Sequence[int] = (1, 2, 4, 8),
        num_scales: int = 4,
        shared_encoder: bool = False,
        norm: str = "instance",
        activation: str = "leaky_relu",
    ):
        super().__init__()
        if num_scales != len(channel_mults):
            raise ValueError("num_scales must match len(channel_mults)")
        self.out_channels = tuple(base_channels * mult for mult in channel_mults)
        self.fixed_encoder = SingleStreamEncoder3D(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
            norm=norm,
            activation=activation,
        )
        self.moving_encoder = (
            self.fixed_encoder
            if shared_encoder
            else SingleStreamEncoder3D(
                in_channels=in_channels,
                base_channels=base_channels,
                channel_mults=channel_mults,
                norm=norm,
                activation=activation,
            )
        )

    def forward(self, fixed, moving):
        """Return fixed_feats, moving_feats ordered high-to-low resolution."""

        return self.fixed_encoder(fixed), self.moving_encoder(moving)
