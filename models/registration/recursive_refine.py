import torch
import torch.nn as nn
from typing import Optional

from ..blocks import ConvNormAct3D
from .flow_utils import resize_flow


class CNNRefineBlock(nn.Module):
    """
    Fine-scale residual flow refinement.

    Used at scale 2 and scale 1 where local boundary correction is preferred
    over expensive diffusion.
    """

    def __init__(
        self,
        feature_channels: int,
        hidden_channels: Optional[int] = None,
        use_initial_flow: bool = False,
        residual_scale: float = 1.0,
        norm: str = "group",
        activation: str = "silu",
    ):
        super().__init__()
        self.use_initial_flow = use_initial_flow
        self.residual_scale = residual_scale
        hidden_channels = hidden_channels or max(feature_channels, 16)
        in_channels = feature_channels * 3 + 3 + (3 if use_initial_flow else 0)
        self.net = nn.Sequential(
            ConvNormAct3D(in_channels, hidden_channels, norm=norm, activation=activation),
            ConvNormAct3D(hidden_channels, hidden_channels, norm=norm, activation=activation),
            nn.Conv3d(hidden_channels, 3, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, fixed_feat, moving_feat, warped_moving_feat, previous_flow, initial_flow=None):
        if previous_flow.shape[2:] != fixed_feat.shape[2:]:
            previous_flow = resize_flow(previous_flow, fixed_feat.shape[2:])
        inputs = [fixed_feat, moving_feat, warped_moving_feat, previous_flow]
        if self.use_initial_flow:
            if initial_flow is None:
                initial_flow = fixed_feat.new_zeros(fixed_feat.shape[0], 3, *fixed_feat.shape[2:])
            elif initial_flow.shape[2:] != fixed_feat.shape[2:]:
                initial_flow = resize_flow(initial_flow, fixed_feat.shape[2:])
            inputs.append(initial_flow)
        return self.residual_scale * self.net(torch.cat(inputs, dim=1))
