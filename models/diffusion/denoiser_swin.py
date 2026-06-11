import torch
import torch.nn as nn
import torch.nn.functional as F

from ..blocks import ConvNormAct3D
from .denoiser_cnn import SinusoidalTimeEmbedding


class PatchEmbed3D(nn.Module):
    """3D patch embedding implemented as Conv3d projection."""

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 1):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x)


class PatchExpand3D(nn.Module):
    """Optional patch expansion hook; identity when patch_size=1."""

    def __init__(self, embed_dim: int, out_channels: int, patch_size: int = 1):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(embed_dim, out_channels, kernel_size=1)

    def forward(self, x, target_size):
        if self.patch_size > 1:
            x = F.interpolate(x, size=target_size, mode="trilinear", align_corners=True)
        return self.proj(x)


class SwinBlock3D(nn.Module):
    """Lightweight 3D shifted-window self-attention block."""

    def __init__(self, dim: int, num_heads: int = 4, window_size: int = 4, shifted: bool = False):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.window_size = window_size
        self.shifted = shifted
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x):
        b, c, d, h, w = x.shape
        win = min(self.window_size, d, h, w)
        pd, ph, pw = (win - d % win) % win, (win - h % win) % win, (win - w % win) % win
        x = F.pad(x, (0, pw, 0, ph, 0, pd))
        if self.shifted and win > 1:
            shift = win // 2
            x = torch.roll(x, shifts=(-shift, -shift, -shift), dims=(2, 3, 4))
        bp, cp, dp, hp, wp = x.shape
        tokens = x.view(bp, cp, dp // win, win, hp // win, win, wp // win, win)
        tokens = tokens.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous().view(-1, win**3, cp)
        residual = tokens
        attn_in = self.norm1(tokens)
        tokens, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        tokens = residual + tokens
        tokens = tokens + self.mlp(self.norm2(tokens))
        x = tokens.view(bp, dp // win, hp // win, wp // win, win, win, win, cp)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous().view(bp, cp, dp, hp, wp)
        if self.shifted and win > 1:
            x = torch.roll(x, shifts=(shift, shift, shift), dims=(2, 3, 4))
        return x[:, :, :d, :h, :w]


class SwinDenoiser3D(nn.Module):
    """
    Main LRDP denoising network.

    It implements a compact but real 3D window-attention design:
    concat [z_t, z_cond] -> patch embedding -> shifted-window Swin blocks ->
    projection head -> epsilon prediction with same shape as z_t.
    """

    def __init__(
        self,
        condition_channels: int,
        flow_channels: int = 16,
        hidden_channels: int = 48,
        time_dim: int = 64,
        depth: int = 4,
        window_size: int = 4,
        num_heads: int = 4,
        patch_size: int = 1,
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.patch_embed = PatchEmbed3D(flow_channels + condition_channels, hidden_channels, patch_size=patch_size)
        self.blocks = nn.ModuleList(
            [
                SwinBlock3D(hidden_channels, num_heads=num_heads, window_size=window_size, shifted=bool(i % 2))
                for i in range(depth)
            ]
        )
        self.norm = ConvNormAct3D(hidden_channels, hidden_channels, norm="group", activation="silu")
        self.patch_expand = PatchExpand3D(hidden_channels, flow_channels, patch_size=patch_size)
        self.patch_size = patch_size
        nn.init.zeros_(self.patch_expand.proj.weight)
        nn.init.zeros_(self.patch_expand.proj.bias)

    def forward(self, z_t, t, z_cond):
        if t.dim() == 0:
            t = t.expand(z_t.shape[0])
        target_size = z_t.shape[2:]
        x = self.patch_embed(torch.cat([z_t, z_cond], dim=1))
        time_emb = self.time_mlp(t.to(z_t.device))
        x = x + time_emb[:, :, None, None, None]
        skip = x
        for block in self.blocks:
            x = block(x)
        x = self.norm(x + skip)
        return self.patch_expand(x, target_size)
