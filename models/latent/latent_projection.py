import torch
import torch.nn as nn

from ..blocks import ConvNormAct3D
from ..registration.flow_utils import resize_flow


class ConditionFusion3D(nn.Module):
    """
    LDM-Morph-style latent conditioning for LRDP.

    The module fuses fixed feature, moving feature, warped moving feature,
    previous flow, initial flow, optional anatomical prior, and optional scale
    embedding into z_cond for diffusion/refinement.
    """

    def __init__(
        self,
        feature_channels: int,
        condition_channels: int,
        anatomical_prior_channels: int = 0,
        scale_embedding_channels: int = 0,
        include_initial_flow: bool = True,
        norm: str = "group",
        activation: str = "silu",
    ):
        super().__init__()
        self.anatomical_prior_channels = anatomical_prior_channels
        self.scale_embedding_channels = scale_embedding_channels
        self.include_initial_flow = include_initial_flow
        flow_channels = 3 + (3 if include_initial_flow else 0)
        in_channels = feature_channels * 3 + flow_channels + anatomical_prior_channels + scale_embedding_channels
        self.project = nn.Sequential(
            ConvNormAct3D(in_channels, condition_channels, kernel_size=1, padding=0, norm=norm, activation=activation),
            ConvNormAct3D(condition_channels, condition_channels, kernel_size=3, norm=norm, activation=activation),
        )

    def _resize_optional(self, tensor, target_size):
        if tensor is None:
            return None
        if tensor.shape[2:] != target_size:
            return nn.functional.interpolate(tensor, size=target_size, mode="trilinear", align_corners=True)
        return tensor

    def forward(
        self,
        fixed_feat,
        moving_feat,
        warped_moving_feat,
        previous_flow,
        initial_flow=None,
        anatomical_prior=None,
        scale_embedding=None,
    ):
        target_size = fixed_feat.shape[2:]
        previous_flow = resize_flow(previous_flow, target_size)
        inputs = [fixed_feat, moving_feat, warped_moving_feat, previous_flow]

        if self.include_initial_flow:
            if initial_flow is None:
                initial_flow = fixed_feat.new_zeros(fixed_feat.shape[0], 3, *target_size)
            else:
                initial_flow = resize_flow(initial_flow, target_size)
            inputs.append(initial_flow)

        anatomical_prior = self._resize_optional(anatomical_prior, target_size)
        if anatomical_prior is not None:
            inputs.append(anatomical_prior)

        scale_embedding = self._resize_optional(scale_embedding, target_size)
        if scale_embedding is not None:
            inputs.append(scale_embedding)

        return self.project(torch.cat(inputs, dim=1))


class LatentProjection3D(ConditionFusion3D):
    """Backward-compatible alias for condition fusion."""


class LatentFlowEncoder3D(nn.Module):
    """Encode clean residual flow and z_cond into latent residual-flow space."""

    def __init__(self, condition_channels: int, latent_channels: int = 16, norm: str = "group", activation: str = "silu"):
        super().__init__()
        hidden_channels = max(condition_channels, latent_channels, 16)
        self.encoder = nn.Sequential(
            ConvNormAct3D(condition_channels + 3, hidden_channels, norm=norm, activation=activation),
            ConvNormAct3D(hidden_channels, hidden_channels, norm=norm, activation=activation),
            nn.Conv3d(hidden_channels, latent_channels, 1),
        )

    def forward(self, residual_flow, z_cond):
        return self.encoder(torch.cat([residual_flow, z_cond], dim=1))


class CorrectionFlowDecoder3D(nn.Module):
    """Decode denoised latent residual flow into voxel displacement residual."""

    def __init__(self, latent_channels: int = 16, condition_channels: int = 0, norm: str = "group", activation: str = "silu"):
        super().__init__()
        hidden_channels = max(latent_channels + condition_channels, 16)
        self.decoder = nn.Sequential(
            ConvNormAct3D(latent_channels + condition_channels, hidden_channels, norm=norm, activation=activation),
            ConvNormAct3D(hidden_channels, hidden_channels, norm=norm, activation=activation),
            nn.Conv3d(hidden_channels, 3, 3, padding=1),
        )
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, latent_flow, z_cond):
        return self.decoder(torch.cat([latent_flow, z_cond], dim=1))
