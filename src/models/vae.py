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
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256, z_dim)
        self.fc_log_var = nn.Linear(256, z_dim)

        self.decoder_input = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 64 * 7 * 7),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.ConvTranspose2d(
                32, 1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid(),
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Encode input to latent distribution parameters.

        Args:
            x: Input images, shape (B, 1, 28, 28)

        Returns:
            mu: Mean of latent distribution, shape (B, z_dim)
            log_var: Log variance of latent distribution, shape (B, z_dim)
        """
        features = self.encoder(x)
        mu = self.fc_mu(features)
        log_var = self.fc_log_var(features)
        log_var = torch.clamp(log_var, min=-20.0, max=20.0)
        return mu, log_var

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """
        Reparameterization trick: z = mu + std * epsilon

        Args:
            mu: Mean, shape (B, z_dim)
            log_var: Log variance, shape (B, z_dim)

        Returns:
            z: Sampled latent vector, shape (B, z_dim)
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor) -> Tensor:
        """
        Decode latent vector to reconstructed image.

        Args:
            z: Latent vector, shape (B, z_dim)

        Returns:
            Reconstructed image, shape (B, 1, 28, 28)
        """
        hidden = self.decoder_input(z)
        hidden = hidden.view(-1, 64, 7, 7)
        return self.decoder(hidden)

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
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var

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
        was_training = self.training
        self.train()
        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                recon, _, _ = self.forward(x)
                samples.append(recon)
        stacked = torch.stack(samples, dim=0)
        mean_recon = stacked.mean(dim=0)
        variance = stacked.var(dim=0, unbiased=False)
        if not was_training:
            self.eval()
        return mean_recon, variance


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
    bce = F.binary_cross_entropy(recon_x, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return bce + beta * kl
