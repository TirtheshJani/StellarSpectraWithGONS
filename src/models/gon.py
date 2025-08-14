import math
from typing import List, Optional, Tuple, Literal, Dict

import torch
from torch import nn
from torch.nn import functional as F


C_LIGHT_KMS: float = 299792.458


def apply_rv_shift_to_wavelengths(observed_wavelengths: torch.Tensor,
									  radial_velocity_kms: Optional[torch.Tensor]) -> torch.Tensor:
	"""Convert observed wavelengths to rest-frame wavelengths.

	If radial_velocity_kms is None, returns observed_wavelengths unchanged.

	Args:
		observed_wavelengths: Tensor of shape (batch, num_points) or (batch, num_points, 1)
		radial_velocity_kms: Optional tensor of shape (batch,) giving velocity in km/s.

	Returns:
		Tensor of same shape as observed_wavelengths with rest-frame wavelengths.
	"""
	if radial_velocity_kms is None:
		return observed_wavelengths
	if observed_wavelengths.dim() == 3 and observed_wavelengths.size(-1) == 1:
		rest = observed_wavelengths[..., 0] / (1.0 + radial_velocity_kms[:, None] / C_LIGHT_KMS)
		return rest.unsqueeze(-1)
	return observed_wavelengths / (1.0 + radial_velocity_kms[:, None] / C_LIGHT_KMS)


class FourierPositionalEncoding(nn.Module):
	"""Fourier features for 1D coordinates (wavelengths).

	Encodes λ into [sin(2π f_k λ_scaled), cos(2π f_k λ_scaled)] for k=0..K-1.
	"""

	def __init__(self, num_frequencies: int = 8, max_frequency_log2: Optional[int] = None,
				 scale: float = 1.0, include_identity: bool = True, log_sampling: bool = True) -> None:
		super().__init__()
		if max_frequency_log2 is None:
			max_frequency_log2 = num_frequencies - 1
		self.num_frequencies = num_frequencies
		self.scale = scale
		self.include_identity = include_identity
		if log_sampling:
			frequencies = 2.0 ** torch.linspace(0.0, float(max_frequency_log2), steps=num_frequencies)
		else:
			frequencies = torch.linspace(1.0, 2.0 ** float(max_frequency_log2), steps=num_frequencies)
		self.register_buffer('frequencies', frequencies, persistent=False)

	def forward(self, wavelengths: torch.Tensor) -> torch.Tensor:
		# wavelengths: (B, N) or (B, N, 1)
		if wavelengths.dim() == 3 and wavelengths.size(-1) == 1:
			w = wavelengths[..., 0]
		else:
			w = wavelengths
		w_scaled = w * self.scale
		# (B, N, F)
		angles = 2.0 * math.pi * w_scaled[..., None] * self.frequencies
		encoded = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
		if self.include_identity:
			encoded = torch.cat([w_scaled[..., None], encoded], dim=-1)
		return encoded

	@property
	def out_dim(self) -> int:
		return (2 * self.num_frequencies) + (1 if self.include_identity else 0)


class SineLayer(nn.Linear):
	"""A linear layer followed by sin activation with SIREN initialization."""

	def __init__(self, in_features: int, out_features: int, is_first: bool = False, w0: float = 30.0) -> None:
		super().__init__(in_features, out_features)
		self.is_first = is_first
		self.w0 = w0
		self.reset_parameters()

	def reset_parameters(self) -> None:
		with torch.no_grad():
			if self.is_first:
				# First-layer SIREN init
				self.weight.uniform_(-1.0 / self.in_features, 1.0 / self.in_features)
			else:
				bound = math.sqrt(6.0 / self.in_features) / self.w0
				self.weight.uniform_(-bound, bound)
			self.bias.zero_()

	def forward(self, input: torch.Tensor) -> torch.Tensor:
		return torch.sin(self.w0 * super().forward(input))


class CoordinateMLP(nn.Module):
	"""A small MLP that maps [encoded_lambda, latent_z] -> normalized flux.

	Supports ReLU/GeLU activations or SIREN (sine) with proper initialization.
	"""

	def __init__(self,
			 in_dim: int,
			 hidden_dim: int = 128,
			 num_layers: int = 4,
			 activation: Literal['relu', 'gelu', 'sine'] = 'relu',
			 out_dim: int = 1,
			 siren_w0: float = 30.0) -> None:
		super().__init__()
		layers: List[nn.Module] = []
		if activation == 'sine':
			layers.append(SineLayer(in_dim, hidden_dim, is_first=True, w0=siren_w0))
			for _ in range(num_layers - 2):
				layers.append(SineLayer(hidden_dim, hidden_dim, is_first=False, w0=siren_w0))
			layers.append(nn.Linear(hidden_dim, out_dim))
		else:
			layers.append(nn.Linear(in_dim, hidden_dim))
			layers.append(nn.GELU() if activation == 'gelu' else nn.ReLU())
			for _ in range(num_layers - 2):
				layers.append(nn.Linear(hidden_dim, hidden_dim))
				layers.append(nn.GELU() if activation == 'gelu' else nn.ReLU())
			layers.append(nn.Linear(hidden_dim, out_dim))
		self.network = nn.Sequential(*layers)

	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		# inputs: (B*N, in_dim)
		return self.network(inputs)


def build_line_window_weights(wavelengths: torch.Tensor,
							   windows: Optional[List[Tuple[float, float, float]]],
							   base_weight: float = 1.0) -> Optional[torch.Tensor]:
	"""Create per-point weights emphasizing given spectral windows.

	Args:
		wavelengths: (B, N) wavelengths.
		windows: Optional list of (center, half_width, weight_multiplier).
		base_weight: Weight outside windows.
	Returns:
		weights tensor of shape (B, N) or None if windows is None/empty.
	"""
	if not windows:
		return None
	device = wavelengths.device
	weights = torch.full_like(wavelengths, fill_value=float(base_weight))
	for center, half_width, weight_multiplier in windows:
		in_window = (wavelengths >= center - half_width) & (wavelengths <= center + half_width)
		weights = torch.where(in_window, weights * float(weight_multiplier), weights)
	return weights


class GONModel(nn.Module):
	"""Gradient Origin Network for spectra.

	This model learns a generator g([λ_enc, z]; θ) -> normalized flux. During training,
	per-example latent z is not a learned parameter; it is obtained via the gradient of
	the log-likelihood wrt z at z=0 (empirical Bayes). Optional RV shift is supported.
	"""

	def __init__(self,
			 latent_dim: int = 16,
			 coord_encoding: Literal['none', 'pe', 'siren'] = 'pe',
			 pe_num_frequencies: int = 8,
			 pe_max_frequency_log2: Optional[int] = None,
			 pe_scale: float = 1.0,
			 mlp_hidden_dim: int = 256,
			 mlp_layers: int = 5,
			 mlp_activation: Literal['relu', 'gelu', 'sine'] = 'relu',
			 siren_w0: float = 30.0) -> None:
		super().__init__()
		self.latent_dim = latent_dim
		self.coord_encoding_kind = coord_encoding
		self.coord_encoder: Optional[nn.Module]
		if coord_encoding == 'pe':
			self.coord_encoder = FourierPositionalEncoding(
				num_frequencies=pe_num_frequencies,
				max_frequency_log2=pe_max_frequency_log2,
				scale=pe_scale,
				include_identity=True,
				log_sampling=True,
			)
			coord_feat_dim = self.coord_encoder.out_dim
		elif coord_encoding == 'siren':
			# No explicit encoding; SIREN handled inside the MLP.
			self.coord_encoder = None
			coord_feat_dim = 1
		else:
			self.coord_encoder = None
			coord_feat_dim = 1

		in_dim = coord_feat_dim + latent_dim
		self.generator = CoordinateMLP(
			in_dim=in_dim,
			hidden_dim=mlp_hidden_dim,
			num_layers=mlp_layers,
			activation='sine' if coord_encoding == 'siren' else mlp_activation,
			out_dim=1,
			siren_w0=siren_w0,
		)

	def encode_wavelengths(self, wavelengths: torch.Tensor) -> torch.Tensor:
		if self.coord_encoder is None:
			if wavelengths.dim() == 3 and wavelengths.size(-1) == 1:
				return wavelengths[..., 0]
			return wavelengths
		return self.coord_encoder(wavelengths)

	def predict_flux(self,
				   wavelengths_observed: torch.Tensor,
				   latent_z: torch.Tensor,
				   radial_velocity_kms: Optional[torch.Tensor] = None) -> torch.Tensor:
		"""Predict normalized flux at observed wavelengths, optionally correcting RV.

		Args:
			wavelengths_observed: (B, N) or (B, N, 1)
			latent_z: (B, D)
			radial_velocity_kms: optional (B,)
		Returns:
			predicted_flux: (B, N)
		"""
		device = wavelengths_observed.device
		B = wavelengths_observed.size(0)
		# Convert observed to rest-frame input coordinates
		rest_wavelengths = apply_rv_shift_to_wavelengths(wavelengths_observed, radial_velocity_kms)
		encoded_lambda = self.encode_wavelengths(rest_wavelengths)  # (B, N, F) or (B, N)
		if encoded_lambda.dim() == 2:
			encoded_lambda = encoded_lambda.unsqueeze(-1)
		N = encoded_lambda.size(1)
		if latent_z.dim() == 1:
			latent_z = latent_z.unsqueeze(0)
		# Repeat z across coordinates
		z_expanded = latent_z[:, None, :].expand(B, N, self.latent_dim)
		inputs = torch.cat([encoded_lambda, z_expanded], dim=-1)  # (B, N, F + D)
		outputs = self.generator(inputs.reshape(B * N, -1)).reshape(B, N, 1)
		return outputs[..., 0]

	@torch.no_grad()
	def init_zero_latent(self, batch_size: int, device: torch.device) -> torch.Tensor:
		return torch.zeros(batch_size, self.latent_dim, device=device)

	def compute_gon_latent(self,
						 wavelengths_observed: torch.Tensor,
						 target_flux: torch.Tensor,
						 radial_velocity_kms: Optional[torch.Tensor] = None,
						 loss_type: Literal['mse', 'huber'] = 'mse',
						 noise_sigma: float = 1.0,
						 line_windows: Optional[List[Tuple[float, float, float]]] = None,
						 base_weight: float = 1.0,
						 scale: float = 1.0) -> torch.Tensor:
		"""Compute per-example latent z from gradient of log-likelihood at z=0.

		This sets z_i = scale * (∇_{z_i} log p(x_i|z_i)) evaluated at z_i=0, using Gaussian
		likelihood (MSE) or Huber. For Gaussian: log p ∝ - (1/(2σ^2)) ||x - g(λ, z)||^2,
		so ∇_z log p = - (1/σ^2) ∇_z L_mse.
		"""
		device = wavelengths_observed.device
		B = wavelengths_observed.size(0)
		z0 = torch.zeros(B, self.latent_dim, device=device, requires_grad=True)
		pred = self.predict_flux(wavelengths_observed, z0, radial_velocity_kms)
		weights = build_line_window_weights(wavelengths_observed if wavelengths_observed.dim() == 2 else wavelengths_observed[..., 0],
												line_windows, base_weight=base_weight)
		if loss_type == 'mse':
			if weights is None:
				per_sample_losses = 0.5 * ((pred - target_flux) ** 2).mean(dim=1) / (noise_sigma ** 2)
			else:
				# Weighted MSE: average with weights, include 1/(2σ^2)
				weighted_sq = 0.5 * ((pred - target_flux) ** 2) * weights / (noise_sigma ** 2)
				per_sample_losses = weighted_sq.sum(dim=1) / (weights.sum(dim=1) + 1e-8)
		elif loss_type == 'huber':
			# SmoothL1 with beta=1.0 default; approximate Gaussian tails differently
			beta = 1.0
			diff = pred - target_flux
			abs_diff = diff.abs()
			if weights is None:
				per_point = torch.where(abs_diff < beta, 0.5 * (diff ** 2) / beta, abs_diff - 0.5 * beta)
				per_sample_losses = per_point.mean(dim=1) / (noise_sigma ** 2)
			else:
				per_point = torch.where(abs_diff < beta, 0.5 * (diff ** 2) / beta, abs_diff - 0.5 * beta)
				weighted = per_point * weights / (noise_sigma ** 2)
				per_sample_losses = weighted.sum(dim=1) / (weights.sum(dim=1) + 1e-8)
		else:
			raise ValueError(f"Unsupported loss_type: {loss_type}")

		# Negative log-likelihood per sample. We need gradient of sum wrt z0.
		loss_sum = per_sample_losses.sum()
		grads = torch.autograd.grad(loss_sum, z0, create_graph=False, retain_graph=False)[0]
		# ∇ log p = -∇ NLL, set z = scale * ∇ log p = -scale * grads
		latents = -scale * grads.detach()
		return latents

	def reconstruction_loss(self,
						  pred_flux: torch.Tensor,
						  target_flux: torch.Tensor,
						  wavelengths_observed: Optional[torch.Tensor] = None,
						  line_windows: Optional[List[Tuple[float, float, float]]] = None,
						  base_weight: float = 1.0,
						  loss_type: Literal['mse', 'huber'] = 'mse') -> torch.Tensor:
		"""Compute reconstruction loss with optional spectral window weighting."""
		if loss_type == 'mse':
			if not line_windows or wavelengths_observed is None:
				return F.mse_loss(pred_flux, target_flux)
			weights = build_line_window_weights(wavelengths_observed if wavelengths_observed.dim() == 2 else wavelengths_observed[..., 0], line_windows, base_weight)
			diff2 = (pred_flux - target_flux) ** 2
			weighted = diff2 * weights
			return weighted.sum() / (weights.sum() + 1e-8)
		elif loss_type == 'huber':
			beta = 1.0
			diff = pred_flux - target_flux
			abs_diff = diff.abs()
			per_point = torch.where(abs_diff < beta, 0.5 * (diff ** 2) / beta, abs_diff - 0.5 * beta)
			if not line_windows or wavelengths_observed is None:
				return per_point.mean()
			weights = build_line_window_weights(wavelengths_observed if wavelengths_observed.dim() == 2 else wavelengths_observed[..., 0], line_windows, base_weight)
			weighted = per_point * weights
			return weighted.sum() / (weights.sum() + 1e-8)
		else:
			raise ValueError(f"Unsupported loss_type: {loss_type}")

	def forward(self,
			  wavelengths_observed: torch.Tensor,
			  target_flux: Optional[torch.Tensor] = None,
			  radial_velocity_kms: Optional[torch.Tensor] = None,
			  infer_latent: bool = True,
			  initial_latent: Optional[torch.Tensor] = None,
			  loss_type: Literal['mse', 'huber'] = 'mse',
			  noise_sigma: float = 1.0,
			  line_windows: Optional[List[Tuple[float, float, float]]] = None,
			  base_weight: float = 1.0,
			  latent_scale: float = 1.0,
			  return_details: bool = False,
			  infer_rv: bool = True,
			  rv_init_kms: float = 0.0,
			  rv_scale: float = 1.0) -> Dict[str, torch.Tensor]:
		"""Full forward pass for training/eval.

		If infer_latent is True and target_flux is provided, compute z via gradient at z=0.
		Optionally infer per-example radial velocity via gradient at v=rv_init.
		Otherwise use initial_latent and/or provided radial_velocity_kms.
		"""
		B = wavelengths_observed.size(0)
		device = wavelengths_observed.device
		# Infer RV by one-step gradient of NLL wrt v if requested and target provided
		if infer_rv and target_flux is not None and radial_velocity_kms is None:
			rv0 = torch.full((B,), float(rv_init_kms), device=device, requires_grad=True)
			# Use zero latent for RV inference step to avoid coupling; then infer z with that v
			z_tmp = torch.zeros(B, self.latent_dim, device=device)
			pred_tmp = self.predict_flux(wavelengths_observed, z_tmp, rv0)
			weights = build_line_window_weights(wavelengths_observed if wavelengths_observed.dim() == 2 else wavelengths_observed[..., 0],
													line_windows, base_weight=base_weight)
			if loss_type == 'mse':
				if weights is None:
					loss_rv = F.mse_loss(pred_tmp, target_flux)
				else:
					diff2 = (pred_tmp - target_flux) ** 2
					loss_rv = (diff2 * weights).sum() / (weights.sum() + 1e-8)
			elif loss_type == 'huber':
				beta = 1.0
				diff = pred_tmp - target_flux
				abs_diff = diff.abs()
				per_point = torch.where(abs_diff < beta, 0.5 * (diff ** 2) / beta, abs_diff - 0.5 * beta)
				if weights is None:
					loss_rv = per_point.mean()
				else:
					loss_rv = (per_point * weights).sum() / (weights.sum() + 1e-8)
			else:
				raise ValueError(f"Unsupported loss_type: {loss_type}")
			grad_v = torch.autograd.grad(loss_rv, rv0, create_graph=False, retain_graph=False)[0]
			# v = -rv_scale * ∇ NLL (i.e., proportional to ∇ log p)
			radial_velocity_kms = (-rv_scale * grad_v).detach()
		# Infer latent
		if infer_latent and target_flux is not None:
			latent_z = self.compute_gon_latent(
				wavelengths_observed=wavelengths_observed,
				target_flux=target_flux,
				radial_velocity_kms=radial_velocity_kms,
				loss_type=loss_type,
				noise_sigma=noise_sigma,
				line_windows=line_windows,
				base_weight=base_weight,
				scale=latent_scale,
			)
		else:
			if initial_latent is None:
				latent_z = self.init_zero_latent(B, device)
			else:
				latent_z = initial_latent
		pred = self.predict_flux(wavelengths_observed, latent_z, radial_velocity_kms)
		result: Dict[str, torch.Tensor] = {'pred': pred, 'z': latent_z}
		if radial_velocity_kms is not None:
			result['rv_kms'] = radial_velocity_kms
		if target_flux is not None:
			loss = self.reconstruction_loss(pred, target_flux, wavelengths_observed, line_windows, base_weight, loss_type)
			result['loss'] = loss
		return result