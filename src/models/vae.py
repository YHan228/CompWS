"""
Variational Autoencoder with Monte Carlo Dropout for uncertainty estimation.

See IMPLEMENTATION_PLAN.md Section 1.1 for full specification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class KintsugiVAE(nn.Module):
    """
    Convolutional VAE with dropout layers retained at inference
    for Monte Carlo uncertainty estimation.

    Architecture:
        Encoder: Conv2d(1→32) → Conv2d(32→64) → Linear(64*7*7→256) → (μ, log_var)
        Decoder: Linear(z_dim→256) → Linear(256→64*7*7) → ConvT(64→32) → ConvT(32→1)

    All layers include Dropout(0.25) for MC sampling.
    """

    def __init__(self, z_dim: int = 20, dropout_p: float = 0.25):
        super().__init__()
        self.z_dim = z_dim
        # TODO: Implement encoder layers
        # TODO: Implement mu and log_var projection layers
        # TODO: Implement decoder layers
        raise NotImplementedError("Implement VAE architecture per IMPLEMENTATION_PLAN.md")

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Encode input to latent distribution parameters.

        Args:
            x: Input images, shape (B, 1, 28, 28)

        Returns:
            mu: Mean of latent distribution, shape (B, z_dim)
            log_var: Log variance of latent distribution, shape (B, z_dim)
        """
        raise NotImplementedError

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """
        Reparameterization trick: z = mu + std * epsilon

        Args:
            mu: Mean, shape (B, z_dim)
            log_var: Log variance, shape (B, z_dim)

        Returns:
            z: Sampled latent vector, shape (B, z_dim)
        """
        raise NotImplementedError

    def decode(self, z: Tensor) -> Tensor:
        """
        Decode latent vector to reconstructed image.

        Args:
            z: Latent vector, shape (B, z_dim)

        Returns:
            Reconstructed image, shape (B, 1, 28, 28)
        """
        raise NotImplementedError

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Full forward pass.

        Args:
            x: Input images, shape (B, 1, 28, 28)

        Returns:
            recon: Reconstructed images, shape (B, 1, 28, 28)
            mu: Latent mean, shape (B, z_dim)
            log_var: Latent log variance, shape (B, z_dim)
        """
        raise NotImplementedError

    def sample_with_uncertainty(
        self, x: Tensor, n_samples: int = 50
    ) -> tuple[Tensor, Tensor]:
        """
        Run n_samples forward passes with dropout enabled for MC uncertainty.

        Args:
            x: Input images, shape (B, 1, 28, 28)
            n_samples: Number of MC samples

        Returns:
            mean_recon: Mean reconstruction, shape (B, 1, 28, 28)
            variance: Per-pixel variance, shape (B, 1, 28, 28)
        """
        raise NotImplementedError


def vae_loss(
    recon_x: Tensor, x: Tensor, mu: Tensor, log_var: Tensor, beta: float = 1.0
) -> Tensor:
    """
    VAE ELBO loss: reconstruction + KL divergence.

    Loss = BCE(recon, x) + beta * KL(q(z|x) || p(z))

    Args:
        recon_x: Reconstructed images, shape (B, 1, 28, 28)
        x: Original images, shape (B, 1, 28, 28)
        mu: Latent mean, shape (B, z_dim)
        log_var: Latent log variance, shape (B, z_dim)
        beta: KL weight (default 1.0)

    Returns:
        Total loss (scalar)
    """
    raise NotImplementedError
