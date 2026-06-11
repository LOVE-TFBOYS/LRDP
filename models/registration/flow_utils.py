import torch
import torch.nn.functional as F


def identity_grid_3d(shape, device=None, dtype=None):
    """Create an identity voxel grid in D/H/W channel order."""

    d, h, w = shape
    vectors = (
        torch.arange(d, device=device, dtype=dtype),
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
    )
    grid = torch.meshgrid(vectors, indexing="ij")
    return torch.stack(grid, dim=0).unsqueeze(0)


def normalize_flow_for_grid_sample(flow):
    """
    Convert a D/H/W voxel displacement field to normalized grid_sample offsets.

    The returned tensor is still channel-first [B, 3, D, H, W] in D/H/W order.
    SpatialTransformer3D later permutes it to x/y/z order for grid_sample.
    """

    normalized = flow.clone()
    for axis, size in enumerate(flow.shape[2:]):
        normalized[:, axis] = 2.0 * normalized[:, axis] / max(size - 1, 1)
    return normalized


def resize_flow(flow, target_size):
    """Resize flow and scale voxel displacement magnitudes for D/H/W axes."""

    if tuple(flow.shape[2:]) == tuple(target_size):
        return flow
    source_size = flow.shape[2:]
    scale = flow.new_tensor([target_size[i] / source_size[i] for i in range(3)]).view(1, 3, 1, 1, 1)
    return F.interpolate(flow, size=target_size, mode="trilinear", align_corners=True) * scale


def upsample_flow(flow, target_size):
    """Alias for resize_flow used in recursive pyramid code."""

    return resize_flow(flow, target_size)


def compose_flows(flow_a, flow_b, transformer=None, mode="warp"):
    """
    Compose two displacement fields.

    mode="add" uses the approximation flow_a + flow_b.
    mode="warp" uses RDP/VoxelMorph-style composition:
        flow_a + flow_b o (Id + flow_a)
    """

    if flow_b.shape[2:] != flow_a.shape[2:]:
        flow_b = resize_flow(flow_b, flow_a.shape[2:])
    if mode == "add":
        return flow_a + flow_b
    if mode != "warp":
        raise ValueError(f"Unsupported composition mode: {mode}")
    if transformer is None:
        from .spatial_transformer import SpatialTransformer3D

        transformer = SpatialTransformer3D()
    return flow_a + transformer(flow_b, flow_a)


def detach_flow(flow):
    """Detach a flow state for ablations that block recursive gradients."""

    return None if flow is None else flow.detach()
