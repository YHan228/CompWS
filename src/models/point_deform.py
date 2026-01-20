"""
Point Deformation Network with MC Dropout for per-vertex uncertainty.

Unlike PointNet VAE which generates new points from latent, this architecture
maintains point correspondence: each input vertex gets a predicted offset,
enabling meaningful per-vertex uncertainty estimation.

Architecture:
    1. Encode global context: PointNet → global feature
    2. Concatenate global feature with each point's local coordinates
    3. Predict per-point offset via shared MLP
    4. Output = input + offset

This is essentially an identity-residual architecture where the model
learns to predict deformations. For clean inputs, offsets should be ~0.
For OOD inputs (dented, occluded), the model is uncertain about offsets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np


class PointDeformNet(nn.Module):
    """
    Point deformation network with per-vertex correspondence.

    Input: (B, N, 3) point cloud
    Output: (B, N, 3) reconstructed point cloud (same N, same order)

    The model predicts per-point offsets and adds them to input.
    MC Dropout gives per-vertex uncertainty: high variance = model
    unsure how to "fix" that vertex.
    """

    def __init__(
        self,
        global_dim: int = 256,
        dropout_p: float = 0.3,
    ):
        super().__init__()

        # Global feature extractor (PointNet-style)
        self.global_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, global_dim),
            nn.ReLU(),
        )

        # Per-point deformation predictor
        # Input: point coords (3) + global feature (global_dim)
        self.deform_mlp = nn.Sequential(
            nn.Linear(3 + global_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # Predict offset
        )

        # Initialize final layer near zero for identity init
        nn.init.zeros_(self.deform_mlp[-1].weight)
        nn.init.zeros_(self.deform_mlp[-1].bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Point cloud, shape (B, N, 3)

        Returns:
            Reconstructed point cloud, shape (B, N, 3)
            Points are in same order as input (correspondence preserved)
        """
        B, N, _ = x.shape

        # Extract global feature
        h = self.global_encoder(x)  # (B, N, global_dim)
        global_feat = h.max(dim=1, keepdim=True)[0]  # (B, 1, global_dim)
        global_feat = global_feat.expand(-1, N, -1)  # (B, N, global_dim)

        # Concatenate point coords with global feature
        h = torch.cat([x, global_feat], dim=2)  # (B, N, 3 + global_dim)

        # Predict per-point offset
        offset = self.deform_mlp(h)  # (B, N, 3)

        # Output = input + offset (residual)
        return x + offset

    def sample_with_uncertainty(
        self, x: Tensor, n_samples: int = 50
    ) -> tuple[Tensor, Tensor]:
        """
        MC Dropout sampling for per-vertex uncertainty.

        Args:
            x: Point cloud, shape (B, N, 3)
            n_samples: Number of MC samples

        Returns:
            mean_output: Mean reconstruction, shape (B, N, 3)
            variance: Per-vertex variance, shape (B, N, 3)
                      High variance = model uncertain about this vertex
        """
        was_training = self.training
        self.train()  # Enable dropout

        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                output = self.forward(x)
                samples.append(output)

        stacked = torch.stack(samples, dim=0)  # (n_samples, B, N, 3)
        mean_output = stacked.mean(dim=0)
        variance = stacked.var(dim=0, unbiased=False)

        if not was_training:
            self.eval()

        return mean_output, variance


def deform_loss(output: Tensor, target: Tensor, offset_penalty: float = 0.01) -> Tensor:
    """
    Loss for point deformation network.

    Args:
        output: Predicted points, shape (B, N, 3)
        target: Target points, shape (B, N, 3) - same correspondence
        offset_penalty: Regularization to keep offsets small

    Returns:
        Loss value
    """
    # Point-wise L2 loss (correspondence preserved)
    recon_loss = F.mse_loss(output, target)

    # Offset regularization: encourage identity mapping
    offset = output - target
    reg_loss = (offset ** 2).mean()

    return recon_loss + offset_penalty * reg_loss


class PointDeformVAE(nn.Module):
    """
    VAE variant of point deformation network.

    Adds latent bottleneck while preserving point correspondence.
    Useful for learning a manifold of shapes while still getting
    per-vertex uncertainty.
    """

    def __init__(
        self,
        z_dim: int = 32,
        global_dim: int = 256,
        dropout_p: float = 0.3,
    ):
        super().__init__()
        self.z_dim = z_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, global_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(global_dim, z_dim)
        self.fc_log_var = nn.Linear(global_dim, z_dim)

        # Decoder: per-point offset prediction conditioned on z
        self.deform_mlp = nn.Sequential(
            nn.Linear(3 + z_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 3),
        )
        nn.init.zeros_(self.deform_mlp[-1].weight)
        nn.init.zeros_(self.deform_mlp[-1].bias)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.encoder(x)  # (B, N, global_dim)
        h = h.max(dim=1)[0]  # (B, global_dim)
        mu = self.fc_mu(h)
        log_var = torch.clamp(self.fc_log_var(h), -20, 20)
        return mu, log_var

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, x: Tensor, z: Tensor) -> Tensor:
        """Decode with original points for correspondence."""
        B, N, _ = x.shape
        z_exp = z.unsqueeze(1).expand(-1, N, -1)  # (B, N, z_dim)
        h = torch.cat([x, z_exp], dim=2)  # (B, N, 3 + z_dim)
        offset = self.deform_mlp(h)
        return x + offset

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(x, z)
        return recon, mu, log_var

    def sample_with_uncertainty(
        self, x: Tensor, n_samples: int = 50
    ) -> tuple[Tensor, Tensor]:
        was_training = self.training
        self.train()

        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                recon, _, _ = self.forward(x)
                samples.append(recon)

        stacked = torch.stack(samples, dim=0)
        mean_output = stacked.mean(dim=0)
        variance = stacked.var(dim=0, unbiased=False)

        if not was_training:
            self.eval()

        return mean_output, variance


def deform_vae_loss(
    output: Tensor,
    target: Tensor,
    mu: Tensor,
    log_var: Tensor,
    beta: float = 0.001
) -> Tensor:
    """VAE loss with point-wise reconstruction."""
    recon_loss = F.mse_loss(output, target, reduction='mean')
    kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + beta * kl
