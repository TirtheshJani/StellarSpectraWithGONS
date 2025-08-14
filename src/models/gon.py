import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Positional encodings
# -----------------------------

class SinusoidalPE(nn.Module):
	def __init__(self, num_frequencies: int = 16, log_space: bool = True):
		super().__init__()
		self.num_frequencies = num_frequencies
		self.log_space = log_space

	def forward(self, lam: torch.Tensor) -> torch.Tensor:
		# lam: (B, N) wavelengths
		x = lam
		if self.log_space:
			x = torch.log(lam.clamp(min=1e-6))
		freqs = 2.0 ** torch.arange(self.num_frequencies, device=lam.device).float()
		angles = x[..., None] * freqs[None, None, :] * math.pi
		enc = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
		return enc  # (B, N, 2*num_frequencies)


class SIRENLayer(nn.Module):
	def __init__(self, in_features: int, out_features: int, is_first: bool = False, w0: float = 30.0):
		super().__init__()
		self.in_features = in_features
		self.is_first = is_first
		self.w0 = w0
		self.linear = nn.Linear(in_features, out_features)
		self.reset_parameters()

	def reset_parameters(self) -> None:
		with torch.no_grad():
			if self.is_first:
				self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
			else:
				c = math.sqrt(6 / self.in_features) / self.w0
				self.linear.weight.uniform_(-c, c)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return torch.sin(self.w0 * self.linear(x))


# -----------------------------
# RV layer
# -----------------------------

class RadialVelocityLayer(nn.Module):
	def __init__(self, learn_per_example: bool = True):
		super().__init__()
		self.learn_per_example = learn_per_example
		# Placeholder; per-example v is supplied at forward via context. If learnable, keep a scalar bias.
		self.global_v = nn.Parameter(torch.zeros(()))
		self.c_kms = 299792.458

	def forward(self, lam_rest: torch.Tensor, flux_rest: torch.Tensor, v_kms: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
		# lam_rest: (B, N), flux_rest: (B, N)
		if v_kms is None:
			v_kms = self.global_v.expand(lam_rest.shape[0])
		beta = v_kms[..., None] / self.c_kms
		lam_obs = lam_rest * (1.0 + beta)
		return lam_obs, flux_rest


# -----------------------------
# Generator g(z; θ) mapping and GON core
# -----------------------------

class CoordinateGenerator(nn.Module):
	def __init__(self, latent_dim: int, hidden_dim: int = 256, depth: int = 4,
				 use_pe: bool = True, pe_frequencies: int = 16, use_siren: bool = False):
		super().__init__()
		self.latent_dim = latent_dim
		self.use_pe = use_pe
		self.use_siren = use_siren
		inp_dim = 1  # λ scalar input
		pe_dim = 2 * pe_frequencies if use_pe else 0
		self.pe = SinusoidalPE(pe_frequencies) if use_pe else None
		in_features = inp_dim + pe_dim + latent_dim
		layers = []
		if use_siren:
			layers.append(SIRENLayer(in_features, hidden_dim, is_first=True))
			for _ in range(depth - 1):
				layers.append(SIRENLayer(hidden_dim, hidden_dim, is_first=False))
			layers.append(nn.Linear(hidden_dim, 1))
		else:
			layers.append(nn.Linear(in_features, hidden_dim))
			layers.append(nn.SiLU())
			for _ in range(depth - 1):
				layers.append(nn.Linear(hidden_dim, hidden_dim))
				layers.append(nn.SiLU())
			layers.append(nn.Linear(hidden_dim, 1))
		self.net = nn.Sequential(*layers)

	def forward(self, lam: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
		# lam: (B, N), z: (B, latent_dim)
		B, N = lam.shape
		feat = [lam.unsqueeze(-1)]
		if self.pe is not None:
			feat.append(self.pe(lam))
		feat.append(z.unsqueeze(1).expand(B, N, -1))
		x = torch.cat(feat, dim=-1)
		out = self.net(x).squeeze(-1)
		return out  # (B, N)


class GON(nn.Module):
	def __init__(self, latent_dim: int = 32, hidden_dim: int = 256, depth: int = 4,
				 use_pe: bool = True, pe_frequencies: int = 16, use_siren: bool = False,
				 loss_huber_delta: float = 0.0, line_weight_mask: Optional[torch.Tensor] = None):
		super().__init__()
		self.gen = CoordinateGenerator(latent_dim, hidden_dim, depth, use_pe, pe_frequencies, use_siren)
		self.rv = RadialVelocityLayer()
		self.latent_dim = latent_dim
		self.loss_huber_delta = loss_huber_delta
		self.register_buffer('line_weight_mask', line_weight_mask if line_weight_mask is not None else torch.tensor(1.0))

	def reconstruct(self, lam_obs: torch.Tensor, v_kms: Optional[torch.Tensor], z: torch.Tensor) -> torch.Tensor:
		# Predict rest-frame flux, shift wavelengths, return predicted flux on observed λ by nearest neighbor interp
		flux_rest = self.gen(lam_obs, z)  # we can treat coord as observed for simplicity; RV layer outputs lam_obs again
		lam_shifted, flux_shifted = self.rv(lam_obs, flux_rest, v_kms)
		# For now, assume lam_shifted aligns with lam_obs (small v); otherwise re-evaluate gen at lam_rest
		return flux_shifted

	def infer_latent(self, lam_obs: torch.Tensor, flux_obs: torch.Tensor, noise_sigma: float = 0.02,
					 steps: int = 1, step_size: float = 1.0, v_kms: Optional[torch.Tensor] = None) -> torch.Tensor:
		# Empirical Bayes: start z=0, take gradient steps of log p(x|z) wrt z
		B = lam_obs.shape[0]
		z = torch.zeros(B, self.latent_dim, device=lam_obs.device, dtype=lam_obs.dtype, requires_grad=True)
		for _ in range(steps):
			beta = (v_kms[..., None] / self.rv.c_kms) if v_kms is not None else 0.0
			lam_rest = lam_obs / (1.0 + beta)
			pred = self.gen(lam_rest, z)
			resid = (flux_obs - pred)
			logp = -0.5 * torch.sum(resid * resid / (noise_sigma ** 2), dim=-1).mean()
			grad, = torch.autograd.grad(logp, z, retain_graph=False, create_graph=False)
			with torch.no_grad():
				z += step_size * grad
			z.requires_grad_(True)
		return z.detach()

	def loss(self, lam_obs: torch.Tensor, flux_obs: torch.Tensor, v_kms: Optional[torch.Tensor], z: torch.Tensor) -> torch.Tensor:
		pred = self.reconstruct(lam_obs, v_kms, z)
		res = pred - flux_obs
		if self.loss_huber_delta and self.loss_huber_delta > 0:
			loss = F.huber_loss(pred, flux_obs, delta=self.loss_huber_delta, reduction='none')
		else:
			loss = res * res
		# Optional line-window weighting
		if self.line_weight_mask.ndim > 0:
			w = F.interpolate(self.line_weight_mask[None, None, :], size=lam_obs.shape[1], mode='linear', align_corners=False).squeeze(1)
			loss = loss * w
		return loss.mean()