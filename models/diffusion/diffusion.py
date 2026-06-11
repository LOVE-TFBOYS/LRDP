import math
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _extract(values, t, target_shape):
    gathered = values.gather(0, t)
    return gathered.view(t.shape[0], *((1,) * (len(target_shape) - 1)))


def make_beta_schedule(schedule: Literal["linear", "cosine"], timesteps: int, beta_start=1e-4, beta_end=2e-2):
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, timesteps)
    if schedule == "cosine":
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(1e-5, 0.999)
    raise ValueError(f"Unsupported beta schedule: {schedule}")


class GaussianDiffusionFlow(nn.Module):
    """
    DiffuseReg-style Gaussian diffusion over residual flow / velocity fields.

    x_start is clean residual flow in latent flow space. The denoiser predicts
    epsilon, never image noise.
    """

    def __init__(
        self,
        denoiser: nn.Module,
        timesteps: int = 1000,
        beta_schedule: Literal["linear", "cosine"] = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        sample_steps: Optional[int] = None,
    ):
        super().__init__()
        self.denoiser = denoiser
        self.num_timesteps = timesteps
        self.sample_steps = sample_steps or timesteps

        betas = make_beta_schedule(beta_schedule, timesteps, beta_start, beta_end)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        alpha_cumprod_prev = torch.cat([torch.ones(1), alpha_cumprod[:-1]])

        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas", alphas.float())
        self.register_buffer("alpha_cumprod", alpha_cumprod.float())
        self.register_buffer("alpha_cumprod_prev", alpha_cumprod_prev.float())
        self.register_buffer("sqrt_alpha_cumprod", torch.sqrt(alpha_cumprod).float())
        self.register_buffer("sqrt_one_minus_alpha_cumprod", torch.sqrt(1.0 - alpha_cumprod).float())
        self.register_buffer("sqrt_recip_alpha_cumprod", torch.sqrt(1.0 / alpha_cumprod).float())
        self.register_buffer("sqrt_recipm1_alpha_cumprod", torch.sqrt(1.0 / alpha_cumprod - 1).float())
        posterior_variance = betas * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)
        self.register_buffer("posterior_variance", posterior_variance.float())
        self.register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)).float())
        self.register_buffer(
            "posterior_mean_coef1",
            (betas * torch.sqrt(alpha_cumprod_prev) / (1.0 - alpha_cumprod)).float(),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            ((1.0 - alpha_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alpha_cumprod)).float(),
        )

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return _extract(self.sqrt_alpha_cumprod, t, x_start.shape) * x_start + _extract(
            self.sqrt_one_minus_alpha_cumprod, t, x_start.shape
        ) * noise

    def predict_start_from_noise(self, x_t, t, noise):
        return _extract(self.sqrt_recip_alpha_cumprod, t, x_t.shape) * x_t - _extract(
            self.sqrt_recipm1_alpha_cumprod, t, x_t.shape
        ) * noise

    def q_posterior(self, x_start, x_t, t):
        mean = _extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + _extract(
            self.posterior_mean_coef2, t, x_t.shape
        ) * x_t
        var = _extract(self.posterior_variance, t, x_t.shape)
        log_var = _extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var

    def p_losses(self, x_start, t, z_cond, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)
        predicted_noise = self.denoiser(x_t, t, z_cond)
        return {
            "loss": F.mse_loss(predicted_noise, noise),
            "predicted_noise": predicted_noise,
            "target_noise": noise,
            "x_t": x_t,
        }

    def p_mean_variance(self, x_t, t, z_cond):
        predicted_noise = self.denoiser(x_t, t, z_cond)
        x_start = self.predict_start_from_noise(x_t, t, predicted_noise)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start, x_t, t)
        return {
            "mean": model_mean,
            "variance": posterior_variance,
            "log_variance": posterior_log_variance,
            "x_start": x_start,
            "predicted_noise": predicted_noise,
        }

    @torch.no_grad()
    def p_sample(self, x_t, t, z_cond):
        out = self.p_mean_variance(x_t, t, z_cond)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(t.shape[0], *((1,) * (x_t.dim() - 1)))
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        out["sample"] = sample
        return out

    @torch.no_grad()
    def sample(self, z_cond, shape, steps=None):
        steps = self.sample_steps if steps is None else int(steps)
        steps = max(1, min(steps, self.num_timesteps))
        x_t = torch.randn(shape, device=z_cond.device, dtype=z_cond.dtype)
        indices = torch.linspace(self.num_timesteps - 1, 0, steps, device=z_cond.device).long()
        for index in indices:
            t = torch.full((shape[0],), int(index), device=z_cond.device, dtype=torch.long)
            x_t = self.p_sample(x_t, t, z_cond)["sample"]
        return x_t

    @torch.no_grad()
    def ddim_sample(self, z_cond, shape, steps=None, eta=0.0):
        """DDIM sampling interface reserved for future experiments."""

        raise NotImplementedError("DDIM sampling is reserved for LRDP ablation experiments.")
