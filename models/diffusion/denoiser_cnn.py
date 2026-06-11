import math

import torch
import torch.nn as nn

from ..blocks import ConvNormAct3D


class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal timestep embedding used by DDPM denoisers."""

    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("time embedding dimension must be even")
        self.dim = dim

    def forward(self, t):
        t = t.float()
        half = self.dim // 2
        freq = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        emb = t[:, None] * freq[None]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)


class TimeInjectedConvBlock3D(nn.Module):
    """Convolution block with additive timestep embedding injection."""

    def __init__(self, channels: int, time_dim: int, norm: str = "group", activation: str = "silu"):
        super().__init__()
        self.conv = ConvNormAct3D(channels, channels, norm=norm, activation=activation)
        self.time_proj = nn.Linear(time_dim, channels)

    def forward(self, x, time_emb):
        return self.conv(x + self.time_proj(time_emb)[:, :, None, None, None])


class CNNDenoiser3D(nn.Module):
    """
    CNN fallback denoiser for GaussianDiffusionFlow.

    Interface is identical to SwinDenoiser3D:
        forward(z_t, t, z_cond) -> predicted_noise
    """

    def __init__(
        self,
        condition_channels: int,
        flow_channels: int = 16,
        hidden_channels: int = 32,
        time_dim: int = 64,
        depth: int = 3,
        norm: str = "group",
        activation: str = "silu",
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(inplace=True),
            nn.Linear(time_dim, time_dim),
        )
        self.in_proj = ConvNormAct3D(flow_channels + condition_channels, hidden_channels, norm=norm, activation=activation)
        self.blocks = nn.ModuleList(
            [TimeInjectedConvBlock3D(hidden_channels, time_dim, norm=norm, activation=activation) for _ in range(depth)]
        )
        self.out_proj = nn.Conv3d(hidden_channels, flow_channels, 3, padding=1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z_t, t, z_cond):
        if t.dim() == 0:
            t = t.expand(z_t.shape[0])
        time_emb = self.time_mlp(t.to(z_t.device))
        x = self.in_proj(torch.cat([z_t, z_cond], dim=1))
        for block in self.blocks:
            x = block(x, time_emb)
        return self.out_proj(x)
