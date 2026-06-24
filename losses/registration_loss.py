import torch
import torch.nn as nn
import torch.nn.functional as F

from .regularization import gradient_smoothness_loss, jacobian_folding_penalty
from .similarity import LocalNCCLoss3D, MSESimilarityLoss
from models.registration import SpatialTransformer3D


class RegistrationLoss(nn.Module):
    """
    Aggregate LRDP registration objective for paper experiments.

    Returns total and every weighted component input used by the trainer logs.
    """

    def __init__(
        self,
        similarity: str = "ncc",
        sim_weight: float = None,
        smooth_weight: float = None,
        jac_weight: float = None,
        diff_weight: float = None,
        ms_weight: float = None,
        lambda_sim: float = None,
        lambda_smooth: float = None,
        lambda_jac: float = None,
        lambda_diff: float = None,
        lambda_multiscale: float = None,
        ncc_window_size: int = 9,
        smooth_penalty: str = "l2",
        multi_scale: dict = None,
    ):
        super().__init__()
        self.sim_weight = 1.0 if sim_weight is None and lambda_sim is None else float(sim_weight if sim_weight is not None else lambda_sim)
        self.smooth_weight = 0.01 if smooth_weight is None and lambda_smooth is None else float(smooth_weight if smooth_weight is not None else lambda_smooth)
        self.jac_weight = 0.0 if jac_weight is None and lambda_jac is None else float(jac_weight if jac_weight is not None else lambda_jac)
        self.diff_weight = 0.01 if diff_weight is None and lambda_diff is None else float(diff_weight if diff_weight is not None else lambda_diff)
        self.ms_weight = 0.0 if ms_weight is None and lambda_multiscale is None else float(ms_weight if ms_weight is not None else lambda_multiscale)
        self.smooth_penalty = smooth_penalty
        self.multi_scale = dict(multi_scale or {})
        self.multi_scale_enabled = bool(self.multi_scale.get("enabled", False))
        self.multi_scale_scales = list(self.multi_scale.get("scales", [4, 3, 2, 1]))
        self.multi_scale_weights = list(self.multi_scale.get("weights", [0.4, 0.3, 0.2, 0.1]))
        self.transformer = SpatialTransformer3D()
        if similarity == "mse":
            self.similarity = MSESimilarityLoss()
        elif similarity == "ncc":
            self.similarity = LocalNCCLoss3D(window_size=ncc_window_size)
        else:
            raise ValueError("similarity must be 'ncc' or 'mse'")

    def loss_weights(self):
        return {
            "sim_weight": self.sim_weight,
            "smooth_weight": self.smooth_weight,
            "jac_weight": self.jac_weight,
            "diff_weight": self.diff_weight,
            "ms_weight": self.ms_weight,
        }

    def _diffusion_loss(self, diffusion_losses, reference):
        if torch.is_tensor(diffusion_losses):
            values = [diffusion_losses]
        elif isinstance(diffusion_losses, dict):
            if torch.is_tensor(diffusion_losses.get("diff")):
                values = [diffusion_losses["diff"]]
            else:
                values = [
                    value
                    for key, value in diffusion_losses.items()
                    if key.startswith("scale") and value is not None
                ]
        else:
            values = []
        if not values:
            return reference.new_zeros(())
        return torch.stack(values).mean()

    def _component_statuses(self, outputs, diff):
        if not outputs.get("diffusion_enabled", False):
            diff_status = "not_used"
        elif torch.is_tensor(outputs.get("diff_loss")) or diff.detach().abs().item() > 0.0:
            diff_status = "computed"
        else:
            diff_status = "not_computed"
        ms_status = "computed" if self.multi_scale_enabled and self.ms_weight != 0 else "not_used"
        return {"diff": diff_status, "ms": ms_status}

    def _scale_flow_pairs(self, outputs):
        flows = outputs.get("multi_scale_flows") or []
        flow_by_scale = {scale: flow for scale, flow in zip([1, 2, 3, 4], flows)}
        for scale, weight in zip(self.multi_scale_scales, self.multi_scale_weights):
            flow = flow_by_scale.get(int(scale))
            if flow is not None:
                yield int(scale), float(weight), flow

    def _regularize_flows(self, outputs, final_flow, loss_fn):
        value = loss_fn(final_flow)
        for _, weight, flow in self._scale_flow_pairs(outputs):
            if flow is final_flow:
                continue
            value = value + weight * loss_fn(flow)
        return value

    def _multiscale_loss(self, outputs, fixed, moving):
        if not self.multi_scale_enabled or self.ms_weight == 0 or moving is None:
            return fixed.new_zeros(())
        values = []
        for _, weight, flow in self._scale_flow_pairs(outputs):
            target_size = flow.shape[2:]
            fixed_s = F.interpolate(fixed, size=target_size, mode="trilinear", align_corners=True)
            moving_s = F.interpolate(moving, size=target_size, mode="trilinear", align_corners=True)
            warped_s = self.transformer(moving_s, flow)
            values.append(weight * self.similarity(warped_s, fixed_s))
        if not values:
            return fixed.new_zeros(())
        return torch.stack(values).sum()

    def forward(self, outputs, fixed, moving=None):
        sim = self.similarity(outputs["warped"], fixed)
        smooth = self._regularize_flows(
            outputs,
            outputs["flow"],
            lambda flow: gradient_smoothness_loss(flow, penalty=self.smooth_penalty),
        )
        # Jacobian terms are computed on displacement flows, including the final
        # composed deformation and available pyramid flows.
        jac = self._regularize_flows(outputs, outputs["flow"], jacobian_folding_penalty)
        diff = self._diffusion_loss(outputs.get("losses", outputs.get("diffusion_losses")), fixed)
        ms = self._multiscale_loss(outputs, fixed, moving)
        statuses = self._component_statuses(outputs, diff)
        total = (
            self.sim_weight * sim
            + self.smooth_weight * smooth
            + self.jac_weight * jac
            + self.diff_weight * diff
            + self.ms_weight * ms
        )
        return {
            "total": total,
            "sim": sim,
            "smooth": smooth,
            "jac": jac,
            "diff": diff,
            "ms": ms,
            "multiscale": ms,
            "statuses": statuses,
        }
