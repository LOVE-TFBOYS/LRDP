import torch
import torch.nn as nn

from .regularization import gradient_smoothness_loss, jacobian_folding_penalty
from .similarity import LocalNCCLoss3D, MSESimilarityLoss


class RegistrationLoss(nn.Module):
    """
    Aggregate LRDP registration objective for paper experiments.

    Returns a dict containing total, sim, smooth, jac, diff, and multiscale.
    """

    def __init__(
        self,
        similarity: str = "ncc",
        lambda_sim: float = 1.0,
        lambda_smooth: float = 0.01,
        lambda_jac: float = 0.0,
        lambda_diff: float = 0.01,
        lambda_multiscale: float = 0.0,
        ncc_window_size: int = 9,
        smooth_penalty: str = "l2",
    ):
        super().__init__()
        self.lambda_sim = lambda_sim
        self.lambda_smooth = lambda_smooth
        self.lambda_jac = lambda_jac
        self.lambda_diff = lambda_diff
        self.lambda_multiscale = lambda_multiscale
        self.smooth_penalty = smooth_penalty
        if similarity == "mse":
            self.similarity = MSESimilarityLoss()
        elif similarity == "ncc":
            self.similarity = LocalNCCLoss3D(window_size=ncc_window_size)
        else:
            raise ValueError("similarity must be 'ncc' or 'mse'")

    def _diffusion_loss(self, diffusion_losses, reference):
        if isinstance(diffusion_losses, dict):
            values = [value for value in diffusion_losses.values() if value is not None]
        elif torch.is_tensor(diffusion_losses):
            values = [diffusion_losses]
        else:
            values = []
        if not values:
            return reference.new_zeros(())
        return torch.stack(values).mean()

    def _multiscale_loss(self, outputs, fixed):
        # Reserved for experiments that add multi-scale warped images.
        return fixed.new_zeros(())

    def forward(self, outputs, fixed, moving=None):
        sim = self.similarity(outputs["warped"], fixed)
        smooth = gradient_smoothness_loss(outputs["flow"], penalty=self.smooth_penalty)
        jac = jacobian_folding_penalty(outputs["flow"])
        diff = self._diffusion_loss(outputs.get("diffusion_losses"), fixed)
        multiscale = self._multiscale_loss(outputs, fixed)
        total = (
            self.lambda_sim * sim
            + self.lambda_smooth * smooth
            + self.lambda_jac * jac
            + self.lambda_diff * diff
            + self.lambda_multiscale * multiscale
        )
        return {
            "total": total,
            "sim": sim,
            "smooth": smooth,
            "jac": jac,
            "diff": diff,
            "multiscale": multiscale,
        }
