import torch


def gradient_smoothness_loss(flow, penalty="l2"):
    """First-order flow smoothness loss over D/H/W axes."""

    dz = flow[:, :, 1:] - flow[:, :, :-1]
    dy = flow[:, :, :, 1:] - flow[:, :, :, :-1]
    dx = flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]
    if penalty == "l1":
        return (dx.abs().mean() + dy.abs().mean() + dz.abs().mean()) / 3.0
    if penalty != "l2":
        raise ValueError("penalty must be 'l1' or 'l2'")
    return ((dx * dx).mean() + (dy * dy).mean() + (dz * dz).mean()) / 3.0


def bending_energy_loss(flow):
    """Second-order bending energy regularizer."""

    ddz = flow[:, :, 2:] - 2 * flow[:, :, 1:-1] + flow[:, :, :-2]
    ddy = flow[:, :, :, 2:] - 2 * flow[:, :, :, 1:-1] + flow[:, :, :, :-2]
    ddx = flow[:, :, :, :, 2:] - 2 * flow[:, :, :, :, 1:-1] + flow[:, :, :, :, :-2]
    return (ddx.square().mean() + ddy.square().mean() + ddz.square().mean()) / 3.0


def jacobian_determinant_3d(flow):
    """
    Jacobian determinant of Id + flow.

    Assumes flow is displacement in [B, 3, D, H, W] with D/H/W = z/y/x axes.
    """

    dz = flow[:, :, 1:, 1:-1, 1:-1] - flow[:, :, :-1, 1:-1, 1:-1]
    dy = flow[:, :, 1:-1, 1:, 1:-1] - flow[:, :, 1:-1, :-1, 1:-1]
    dx = flow[:, :, 1:-1, 1:-1, 1:] - flow[:, :, 1:-1, 1:-1, :-1]
    min_d = min(dz.shape[2], dy.shape[2], dx.shape[2])
    min_h = min(dz.shape[3], dy.shape[3], dx.shape[3])
    min_w = min(dz.shape[4], dy.shape[4], dx.shape[4])
    dz = dz[:, :, :min_d, :min_h, :min_w]
    dy = dy[:, :, :min_d, :min_h, :min_w]
    dx = dx[:, :, :min_d, :min_h, :min_w]

    j00, j01, j02 = 1.0 + dz[:, 0], dy[:, 0], dx[:, 0]
    j10, j11, j12 = dz[:, 1], 1.0 + dy[:, 1], dx[:, 1]
    j20, j21, j22 = dz[:, 2], dy[:, 2], 1.0 + dx[:, 2]
    return j00 * (j11 * j22 - j12 * j21) - j01 * (j10 * j22 - j12 * j20) + j02 * (j10 * j21 - j11 * j20)


def jacobian_folding_penalty(flow):
    """Penalty for non-positive Jacobian determinants."""

    return torch.relu(-jacobian_determinant_3d(flow)).mean()


jacobian_determinant = jacobian_determinant_3d
folding_penalty = jacobian_folding_penalty
