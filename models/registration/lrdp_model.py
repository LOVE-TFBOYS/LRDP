from collections.abc import Sequence
from typing import Literal, Optional

import torch
import torch.nn as nn

from ..diffusion.denoiser_cnn import CNNDenoiser3D
from ..diffusion.denoiser_swin import SwinDenoiser3D
from ..diffusion.diffusion import GaussianDiffusionFlow
from ..encoder import DualStreamEncoder3D
from ..latent.latent_projection import ConditionFusion3D
from ..pyramid import FeaturePyramid3D, IdentityFeaturePyramid3D
from .flow_utils import compose_flows, upsample_flow
from .initial_flow import InitialFlowNet3D
from .recursive_refine import CNNRefineBlock
from .spatial_transformer import SpatialTransformer3D


class DiffusionResidualBlock3D(nn.Module):
    """
    Diffusion block for residual deformation generation.

    It encodes clean residual flow into latent flow space during training and
    uses GaussianDiffusionFlow to model uncertainty over residual deformation.
    During inference, diffusion.sample generates residual latent flow from
    Gaussian noise conditioned on z_cond.
    """

    def __init__(
        self,
        condition_channels: int,
        latent_flow_channels: int,
        denoiser_type: Literal["swin", "cnn"] = "swin",
        diffusion_timesteps: int = 1000,
        diffusion_sample_steps: Optional[int] = None,
        beta_schedule: Literal["linear", "cosine"] = "linear",
    ):
        super().__init__()
        self.flow_encoder = nn.Sequential(
            nn.Conv3d(condition_channels + 3, max(condition_channels, latent_flow_channels, 16), 3, padding=1),
            nn.GroupNorm(8 if max(condition_channels, latent_flow_channels, 16) % 8 == 0 else 1, max(condition_channels, latent_flow_channels, 16)),
            nn.SiLU(inplace=True),
            nn.Conv3d(max(condition_channels, latent_flow_channels, 16), latent_flow_channels, 1),
        )
        self.flow_decoder = nn.Sequential(
            nn.Conv3d(condition_channels + latent_flow_channels, max(condition_channels, latent_flow_channels, 16), 3, padding=1),
            nn.GroupNorm(8 if max(condition_channels, latent_flow_channels, 16) % 8 == 0 else 1, max(condition_channels, latent_flow_channels, 16)),
            nn.SiLU(inplace=True),
            nn.Conv3d(max(condition_channels, latent_flow_channels, 16), 3, 3, padding=1),
        )
        nn.init.zeros_(self.flow_decoder[-1].weight)
        nn.init.zeros_(self.flow_decoder[-1].bias)

        denoiser_cls = SwinDenoiser3D if denoiser_type == "swin" else CNNDenoiser3D
        denoiser = denoiser_cls(condition_channels=condition_channels, flow_channels=latent_flow_channels)
        self.diffusion = GaussianDiffusionFlow(
            denoiser=denoiser,
            timesteps=diffusion_timesteps,
            beta_schedule=beta_schedule,
            sample_steps=diffusion_sample_steps,
        )

    def encode_residual(self, residual_flow, z_cond):
        return self.flow_encoder(torch.cat([residual_flow, z_cond], dim=1))

    def decode_residual(self, latent_flow, z_cond):
        return self.flow_decoder(torch.cat([latent_flow, z_cond], dim=1))

    def forward(self, z_cond, residual_target=None):
        losses = {}
        if self.training and residual_target is not None:
            z_start = self.encode_residual(residual_target, z_cond)
            t = torch.randint(0, self.diffusion.num_timesteps, (z_start.shape[0],), device=z_start.device)
            loss_out = self.diffusion.p_losses(z_start, t, z_cond)
            z_denoised = self.diffusion.predict_start_from_noise(loss_out["x_t"], t, loss_out["predicted_noise"])
            losses = {"loss": loss_out["loss"], "predicted_noise": loss_out["predicted_noise"]}
        else:
            latent_shape = (z_cond.shape[0], self.flow_encoder[-1].out_channels, *z_cond.shape[2:])
            z_denoised = self.diffusion.sample(z_cond, latent_shape)
        return self.decode_residual(z_denoised, z_cond), losses


class LRDPRegistrationModel(nn.Module):
    """
    Paper-grade Latent Recursive Diffusion Pyramid registration model.

    Flow convention: all dense flows are voxel displacement fields shaped
    [B, 3, D, H, W], channels are D/H/W (z/y/x) displacement.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
        channel_mults: Sequence[int] = (1, 2, 4, 8),
        num_scales: int = 4,
        use_pyramid: bool = True,
        use_initial_flow: bool = True,
        use_diffusion_scales: Sequence[int] = (3, 4),
        denoiser_type: Literal["swin", "cnn"] = "swin",
        diffusion_timesteps: int = 1000,
        diffusion_sample_steps: Optional[int] = None,
        diffusion_beta_schedule: Literal["linear", "cosine"] = "linear",
        latent_flow_channels: int = 16,
        condition_channels: Optional[Sequence[int]] = None,
        flow_representation: Literal["displacement", "velocity"] = "displacement",
        use_diffeomorphic: bool = False,
        return_intermediates: bool = True,
        shared_encoder: bool = False,
        composition_mode: Literal["warp", "add"] = "warp",
    ):
        super().__init__()
        if num_scales != 4:
            raise ValueError("Current LRDP implementation expects four scales")
        self.use_initial_flow = use_initial_flow
        self.use_diffusion_scales = set(use_diffusion_scales)
        self.flow_representation = flow_representation
        self.use_diffeomorphic = use_diffeomorphic
        self.default_return_intermediates = return_intermediates
        self.composition_mode = composition_mode

        self.encoder = DualStreamEncoder3D(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
            num_scales=num_scales,
            shared_encoder=shared_encoder,
        )
        feature_channels = self.encoder.out_channels
        condition_channels = tuple(condition_channels or feature_channels)
        if len(condition_channels) != num_scales:
            raise ValueError("condition_channels must match num_scales")

        pyramid_cls = FeaturePyramid3D if use_pyramid else IdentityFeaturePyramid3D
        self.fixed_pyramid = pyramid_cls(feature_channels, feature_channels) if use_pyramid else pyramid_cls()
        self.moving_pyramid = pyramid_cls(feature_channels, feature_channels) if use_pyramid else pyramid_cls()
        self.transformer = SpatialTransformer3D()

        self.initial4 = InitialFlowNet3D(feature_channels[3], feature_channels[3], use_previous_flow=False, residual=True)
        self.initial3 = InitialFlowNet3D(feature_channels[2], feature_channels[2], use_previous_flow=True, residual=True)

        self.condition4 = ConditionFusion3D(feature_channels[3], condition_channels[3], include_initial_flow=True)
        self.condition3 = ConditionFusion3D(feature_channels[2], condition_channels[2], include_initial_flow=True)
        self.condition2 = ConditionFusion3D(feature_channels[1], condition_channels[1], include_initial_flow=False)
        self.condition1 = ConditionFusion3D(feature_channels[0], condition_channels[0], include_initial_flow=False)

        self.diffusion4 = DiffusionResidualBlock3D(
            condition_channels[3],
            latent_flow_channels,
            denoiser_type=denoiser_type,
            diffusion_timesteps=diffusion_timesteps,
            diffusion_sample_steps=diffusion_sample_steps,
            beta_schedule=diffusion_beta_schedule,
        )
        self.diffusion_target4 = CNNRefineBlock(feature_channels[3], use_initial_flow=True)
        self.diffusion3 = DiffusionResidualBlock3D(
            condition_channels[2],
            latent_flow_channels,
            denoiser_type=denoiser_type,
            diffusion_timesteps=diffusion_timesteps,
            diffusion_sample_steps=diffusion_sample_steps,
            beta_schedule=diffusion_beta_schedule,
        )
        self.diffusion_target3 = CNNRefineBlock(feature_channels[2], use_initial_flow=True)

        self.refine2 = CNNRefineBlock(feature_channels[1])
        self.refine1 = CNNRefineBlock(feature_channels[0])

    def _compose(self, base_flow, residual_flow):
        return compose_flows(base_flow, residual_flow, transformer=self.transformer, mode=self.composition_mode)

    def _zero_flow(self, feature):
        return feature.new_zeros(feature.shape[0], 3, *feature.shape[2:])

    def forward(self, fixed, moving, return_intermediates=None):
        if return_intermediates is None:
            return_intermediates = self.default_return_intermediates

        fixed_feats, moving_feats = self.encoder(fixed, moving)
        fixed_feats = self.fixed_pyramid(fixed_feats)
        moving_feats = self.moving_pyramid(moving_feats)
        f1, f2, f3, f4 = fixed_feats
        m1, m2, m3, m4 = moving_feats

        diffusion_losses = {}
        warped_features = {}

        # Scale 4:
        # phi4_init = InitialFlowNet(F4, M4)
        # delta_phi4 = DiffusionFlow(z_cond4)
        # phi4 = phi4_init + delta_phi4
        phi4_init = self.initial4(f4, m4) if self.use_initial_flow else self._zero_flow(f4)
        m4_warp = self.transformer(m4, phi4_init)
        warped_features["M4"] = m4_warp
        z_cond4 = self.condition4(f4, m4, m4_warp, self._zero_flow(f4), initial_flow=phi4_init)
        if 4 in self.use_diffusion_scales:
            residual_target4 = (
                self.diffusion_target4(f4, m4, m4_warp, self._zero_flow(f4), initial_flow=phi4_init)
                if self.training
                else None
            )
            delta_phi4, diff4 = self.diffusion4(z_cond4, residual_target=residual_target4)
            diffusion_losses["scale4"] = diff4.get("loss") if diff4 else None
            diffusion_losses["diff_loss4"] = diff4.get("loss") if diff4 else None
        else:
            residual_target4 = None
            delta_phi4 = self._zero_flow(f4)
            diffusion_losses["scale4"] = None
            diffusion_losses["diff_loss4"] = None
        phi4 = self._compose(phi4_init, delta_phi4)

        # Scale 3:
        # up_phi4 = upsample_flow(phi4)
        # M3_warp = STN(M3, up_phi4)
        # phi3_init = InitialFlowNet(F3, M3_warp, up_phi4)
        # delta_phi3 = DiffusionFlow(z_cond3)
        # phi3 = up_phi4 + phi3_init + delta_phi3
        up_phi4 = upsample_flow(phi4, f3.shape[2:])
        m3_warp = self.transformer(m3, up_phi4)
        warped_features["M3"] = m3_warp
        phi3_init = self.initial3(f3, m3_warp, previous_flow=up_phi4) if self.use_initial_flow else self._zero_flow(f3)
        z_cond3 = self.condition3(f3, m3, m3_warp, up_phi4, initial_flow=phi3_init)
        phi3_pre = self._compose(up_phi4, phi3_init)
        if 3 in self.use_diffusion_scales:
            residual_target3 = (
                self.diffusion_target3(f3, m3, m3_warp, up_phi4, initial_flow=phi3_init)
                if self.training
                else None
            )
            delta_phi3, diff3 = self.diffusion3(z_cond3, residual_target=residual_target3)
            diffusion_losses["scale3"] = diff3.get("loss") if diff3 else None
            diffusion_losses["diff_loss3"] = diff3.get("loss") if diff3 else None
        else:
            residual_target3 = None
            delta_phi3 = self._zero_flow(f3)
            diffusion_losses["scale3"] = None
            diffusion_losses["diff_loss3"] = None
        phi3 = self._compose(phi3_pre, delta_phi3)

        # Scale 2:
        up_phi3 = upsample_flow(phi3, f2.shape[2:])
        m2_warp = self.transformer(m2, up_phi3)
        warped_features["M2"] = m2_warp
        delta_phi2 = self.refine2(f2, m2, m2_warp, up_phi3)
        phi2 = self._compose(up_phi3, delta_phi2)

        # Scale 1:
        up_phi2 = upsample_flow(phi2, f1.shape[2:])
        m1_warp = self.transformer(m1, up_phi2)
        warped_features["M1"] = m1_warp
        delta_phi1 = self.refine1(f1, m1, m1_warp, up_phi2)
        phi1 = self._compose(up_phi2, delta_phi1)

        final_flow = upsample_flow(phi1, fixed.shape[2:])
        if self.flow_representation == "velocity" or self.use_diffeomorphic:
            # Interface reserved for scaling-and-squaring integration.
            # The default paper-code path uses displacement fields directly.
            pass
        warped = self.transformer(moving, final_flow)

        diff_values = [value for key, value in diffusion_losses.items() if key.startswith("scale") and value is not None]
        diff_loss_total = torch.stack(diff_values).mean() if diff_values else None

        outputs = {
            "warped": warped,
            "flow": final_flow,
            "multi_scale_flows": [phi1, phi2, phi3, phi4],
            "residual_flows": [delta_phi1, delta_phi2, delta_phi3, delta_phi4],
            "initial_flows": [None, None, phi3_init, phi4_init],
            "diffusion_losses": diffusion_losses,
            "diffusion_enabled": len(self.use_diffusion_scales) > 0,
            "diffusion_train_scales": sorted(self.use_diffusion_scales),
            "losses": {
                "diff": diff_loss_total,
                "diff_loss_total": diff_loss_total,
                "diff_loss4": diffusion_losses.get("diff_loss4"),
                "diff_loss3": diffusion_losses.get("diff_loss3"),
            },
            "diff_loss": diff_loss_total,
        }
        if return_intermediates:
            outputs["features"] = {"fixed": fixed_feats, "moving": moving_feats}
            outputs["warped_features"] = warped_features
            outputs["conditions"] = {"scale4": z_cond4, "scale3": z_cond3}
            outputs["preliminary_flows"] = {"scale3": phi3_pre, "scale4": phi4_init}
            outputs["diffusion_targets"] = {"scale4": residual_target4, "scale3": residual_target3}
        return outputs


def build_lrdp_model(**kwargs):
    return LRDPRegistrationModel(**kwargs)
