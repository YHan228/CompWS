"""
PointNet-based VAE with Monte Carlo Dropout for 3D uncertainty estimation.

Extends the Kintsugi concept to 3D point clouds: high-variance vertices
become candidates for "golden repair" visualization.

Architecture:
    Encoder: PointNet (per-point MLP + max pooling) → (μ, log_var)
    Decoder: FoldingNet-style (latent + 2D grid → MLP → 3D points)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np


class PointNetEncoder(nn.Module):
    """
    PointNet encoder: per-point MLPs followed by symmetric max pooling.

    Input: (B, N, 3) point cloud
    Output: (B, global_dim) global feature
    """

    def __init__(self, global_dim: int = 256, dropout_p: float = 0.25):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 256),
            nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, global_dim),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Point cloud, shape (B, N, 3)
        Returns:
            Global feature, shape (B, global_dim)
        """
        # Per-point features
        h = self.mlp1(x)  # (B, N, 256)
        h = self.mlp2(h)  # (B, N, global_dim)
        # Symmetric max pooling
        global_feat = h.max(dim=1)[0]  # (B, global_dim)
        return global_feat


class FoldingDecoder(nn.Module):
    """
    FoldingNet-style decoder: folds a 2D grid into 3D using latent code.

    Input: (B, z_dim) latent code
    Output: (B, N, 3) reconstructed point cloud
    """

    def __init__(self, z_dim: int = 20, n_points: int = 1024, dropout_p: float = 0.25):
        super().__init__()
        self.z_dim = z_dim
        self.n_points = n_points

        # Create 2D grid for folding
        grid_size = int(np.ceil(np.sqrt(n_points)))
        u = torch.linspace(-1, 1, grid_size)
        v = torch.linspace(-1, 1, grid_size)
        grid_u, grid_v = torch.meshgrid(u, v, indexing='ij')
        grid = torch.stack([grid_u.flatten(), grid_v.flatten()], dim=1)[:n_points]
        self.register_buffer('grid', grid)  # (N, 2)

        # First folding: (z_dim + 2) → 3
        self.fold1 = nn.Sequential(
            nn.Linear(z_dim + 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 3),
        )

        # Second folding: (z_dim + 3) → 3 (refine)
        self.fold2 = nn.Sequential(
            nn.Linear(z_dim + 3, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, z: Tensor) -> Tensor:
        """
        Args:
            z: Latent code, shape (B, z_dim)
        Returns:
            Reconstructed point cloud, shape (B, N, 3)
        """
        B = z.shape[0]
        N = self.n_points

        # Expand latent and grid
        z_exp = z.unsqueeze(1).expand(-1, N, -1)  # (B, N, z_dim)
        grid_exp = self.grid.unsqueeze(0).expand(B, -1, -1)  # (B, N, 2)

        # First folding
        h = torch.cat([z_exp, grid_exp], dim=2)  # (B, N, z_dim + 2)
        points1 = self.fold1(h)  # (B, N, 3)

        # Second folding (refinement)
        h2 = torch.cat([z_exp, points1], dim=2)  # (B, N, z_dim + 3)
        points2 = self.fold2(h2)  # (B, N, 3)

        return points2


class PointNetVAE(nn.Module):
    """
    PointNet VAE with MC Dropout for 3D uncertainty estimation.

    Combines PointNet encoder with FoldingNet decoder.
    Dropout layers remain active during inference for Monte Carlo sampling.

    Usage:
        model = PointNetVAE(z_dim=20, n_points=1024)
        recon, mu, log_var = model(point_cloud)

        # For uncertainty estimation:
        mean_recon, variance = model.sample_with_uncertainty(point_cloud, n_samples=50)
        # variance shape: (B, N, 3) - per-vertex, per-coordinate variance
    """

    def __init__(
        self,
        z_dim: int = 20,
        n_points: int = 1024,
        global_dim: int = 256,
        dropout_p: float = 0.25
    ):
        super().__init__()
        self.z_dim = z_dim
        self.n_points = n_points

        self.encoder = PointNetEncoder(global_dim=global_dim, dropout_p=dropout_p)

        self.fc_mu = nn.Linear(global_dim, z_dim)
        self.fc_log_var = nn.Linear(global_dim, z_dim)

        self.decoder = FoldingDecoder(z_dim=z_dim, n_points=n_points, dropout_p=dropout_p)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Encode point cloud to latent distribution parameters.

        Args:
            x: Point cloud, shape (B, N, 3)
        Returns:
            mu: Mean, shape (B, z_dim)
            log_var: Log variance, shape (B, z_dim)
        """
        global_feat = self.encoder(x)
        mu = self.fc_mu(global_feat)
        log_var = self.fc_log_var(global_feat)
        log_var = torch.clamp(log_var, min=-20.0, max=20.0)
        return mu, log_var

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """Reparameterization trick: z = mu + std * epsilon"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent to point cloud."""
        return self.decoder(z)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Full forward pass.

        Args:
            x: Point cloud, shape (B, N, 3)
        Returns:
            recon: Reconstructed point cloud, shape (B, n_points, 3)
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
        Monte Carlo dropout sampling for uncertainty estimation.

        Args:
            x: Point cloud, shape (B, N, 3)
            n_samples: Number of MC samples

        Returns:
            mean_recon: Mean reconstruction, shape (B, n_points, 3)
            variance: Per-vertex variance, shape (B, n_points, 3)
                      High variance = model uncertain about vertex position
        """
        was_training = self.training
        self.train()  # Keep dropout active

        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                recon, _, _ = self.forward(x)
                samples.append(recon)

        stacked = torch.stack(samples, dim=0)  # (n_samples, B, N, 3)
        mean_recon = stacked.mean(dim=0)
        variance = stacked.var(dim=0, unbiased=False)

        if not was_training:
            self.eval()

        return mean_recon, variance


def chamfer_distance(pred: Tensor, target: Tensor) -> Tensor:
    """
    Chamfer distance between two point clouds.

    Args:
        pred: Predicted points, shape (B, N, 3)
        target: Target points, shape (B, M, 3)

    Returns:
        Chamfer distance (scalar)
    """
    # pred -> target
    diff = pred.unsqueeze(2) - target.unsqueeze(1)  # (B, N, M, 3)
    dist = (diff ** 2).sum(dim=-1)  # (B, N, M)
    min_pred_to_target = dist.min(dim=2)[0].mean()  # avg min distance

    # target -> pred
    min_target_to_pred = dist.min(dim=1)[0].mean()

    return min_pred_to_target + min_target_to_pred


def pointnet_vae_loss(
    recon: Tensor,
    target: Tensor,
    mu: Tensor,
    log_var: Tensor,
    beta: float = 0.001
) -> Tensor:
    """
    PointNet VAE loss: Chamfer distance + KL divergence.

    Args:
        recon: Reconstructed point cloud, shape (B, N, 3)
        target: Target point cloud, shape (B, M, 3)
        mu: Latent mean, shape (B, z_dim)
        log_var: Latent log variance, shape (B, z_dim)
        beta: KL weight (typically small for point clouds)

    Returns:
        Total loss
    """
    chamfer = chamfer_distance(recon, target)
    kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return chamfer + beta * kl
