import torch


def ensure_5d(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is [B, C, D, H, W]."""

    if tensor.dim() == 3:
        return tensor.unsqueeze(0).unsqueeze(0)
    if tensor.dim() == 4:
        return tensor.unsqueeze(0)
    if tensor.dim() != 5:
        raise ValueError(f"Expected 3D/4D/5D tensor, got shape {tuple(tensor.shape)}")
    return tensor


def same_spatial_shape(a: torch.Tensor, b: torch.Tensor) -> bool:
    return tuple(a.shape[2:]) == tuple(b.shape[2:])
