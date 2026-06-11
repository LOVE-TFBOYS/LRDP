import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformer3D(nn.Module):
    """
    3D spatial transformer for voxel displacement fields.

    src shape:  [B, C, D, H, W]
    flow shape: [B, 3, D, H, W]
    flow channels are D/H/W displacements, equivalent to z/y/x voxel axes.
    Internally the sampling grid is converted to grid_sample order x/y/z.
    """

    def __init__(self, mode: str = "bilinear", padding_mode: str = "border", align_corners: bool = True):
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    @staticmethod
    def identity_grid(shape, device, dtype):
        d, h, w = shape
        vectors = (
            torch.arange(d, device=device, dtype=dtype),
            torch.arange(h, device=device, dtype=dtype),
            torch.arange(w, device=device, dtype=dtype),
        )
        grid = torch.meshgrid(vectors, indexing="ij")
        return torch.stack(grid, dim=0).unsqueeze(0)

    @staticmethod
    def normalize_grid(grid):
        _, _, d, h, w = grid.shape
        sizes = (d, h, w)
        normalized = grid.clone()
        for axis, size in enumerate(sizes):
            if size > 1:
                normalized[:, axis] = 2.0 * (normalized[:, axis] / (size - 1) - 0.5)
            else:
                normalized[:, axis] = 0.0
        return normalized

    def forward(self, src, flow):
        grid = self.identity_grid(flow.shape[2:], flow.device, flow.dtype)
        sampling_grid = self.normalize_grid(grid + flow)
        sampling_grid = sampling_grid.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]
        return F.grid_sample(
            src,
            sampling_grid,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )
