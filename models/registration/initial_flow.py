import torch
import torch.nn as nn
from typing import Optional

from ..blocks import ConvNormAct3D
from .flow_utils import resize_flow


class InitialFlowNet3D(nn.Module):
    """
    VTN-like initial flow estimator.

    It provides a stable current-scale residual estimate and does not replace
    recursive residual correction. When use_previous_flow=True, previous flow
    is resized and concatenated as a condition.
    """

    def __init__(
        self,
        fixed_channels: int,
        moving_channels: int,
        hidden_channels: Optional[int] = None,
        use_previous_flow: bool = False,
        residual: bool = True,
        norm: str = "group",
        activation: str = "silu",
    ):
        super().__init__()
        self.use_previous_flow = use_previous_flow
        self.residual = residual
        hidden_channels = hidden_channels or max(fixed_channels, moving_channels, 16)
        in_channels = fixed_channels + moving_channels + (3 if use_previous_flow else 0)
        self.net = nn.Sequential(
            ConvNormAct3D(in_channels, hidden_channels, norm=norm, activation=activation),
            ConvNormAct3D(hidden_channels, hidden_channels, norm=norm, activation=activation),
            nn.Conv3d(hidden_channels, 3, 3, padding=1),
        )
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1e-5)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, fixed_feature, moving_feature, previous_flow=None):
        inputs = [fixed_feature, moving_feature]
        if self.use_previous_flow:
            if previous_flow is None:
                previous_flow = fixed_feature.new_zeros(fixed_feature.shape[0], 3, *fixed_feature.shape[2:])
            elif previous_flow.shape[2:] != fixed_feature.shape[2:]:
                previous_flow = resize_flow(previous_flow, fixed_feature.shape[2:])
            inputs.append(previous_flow)
        return self.net(torch.cat(inputs, dim=1))


InitialFlowNet = InitialFlowNet3D
VTNLite = InitialFlowNet3D
