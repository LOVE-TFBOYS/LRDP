from math import prod

import torch.nn as nn
import torch.nn.functional as F


class MSESimilarityLoss(nn.Module):
    """Mean-squared image similarity loss."""

    def forward(self, warped, fixed):
        return F.mse_loss(warped, fixed)


class LocalNCCLoss3D(nn.Module):
    """Windowed local normalized cross-correlation loss for 3D registration."""

    def __init__(self, window_size=9, eps=1e-5):
        super().__init__()
        if isinstance(window_size, int):
            window_size = (window_size, window_size, window_size)
        self.window_size = tuple(window_size)
        self.eps = eps

    def forward(self, warped, fixed):
        if warped.shape != fixed.shape:
            raise ValueError(f"warped and fixed shape mismatch: {warped.shape} vs {fixed.shape}")
        channels = warped.shape[1]
        kernel = warped.new_ones(channels, 1, *self.window_size)
        padding = tuple(size // 2 for size in self.window_size)
        warped_sum = F.conv3d(warped, kernel, padding=padding, groups=channels)
        fixed_sum = F.conv3d(fixed, kernel, padding=padding, groups=channels)
        warped_sq_sum = F.conv3d(warped * warped, kernel, padding=padding, groups=channels)
        fixed_sq_sum = F.conv3d(fixed * fixed, kernel, padding=padding, groups=channels)
        cross_sum = F.conv3d(warped * fixed, kernel, padding=padding, groups=channels)
        window_volume = float(prod(self.window_size))
        warped_mean = warped_sum / window_volume
        fixed_mean = fixed_sum / window_volume
        cross = cross_sum - warped_mean * fixed_sum - fixed_mean * warped_sum + warped_mean * fixed_mean * window_volume
        warped_var = warped_sq_sum - 2 * warped_mean * warped_sum + warped_mean * warped_mean * window_volume
        fixed_var = fixed_sq_sum - 2 * fixed_mean * fixed_sum + fixed_mean * fixed_mean * window_volume
        warped_var = warped_var.clamp_min(self.eps)
        fixed_var = fixed_var.clamp_min(self.eps)
        ncc = (cross * cross) / (warped_var * fixed_var)
        ncc = ncc.clamp(0.0, 1.0)
        return 1.0 - ncc.mean()


class NCCLoss3D(LocalNCCLoss3D):
    """Alias for local NCC, kept for experiment config readability."""


MSELoss = MSESimilarityLoss
