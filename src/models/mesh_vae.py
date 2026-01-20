"""
Mesh VAE with fixed topology for true per-vertex correspondence.

Unlike point cloud VAEs, mesh VAE uses a template mesh where each vertex
has a fixed semantic meaning. This enables meaningful per-vertex uncertainty:
- Vertex #100 always represents "this location on the template"
- If vertex #100 is damaged (displaced), model is uncertain how to restore it
- Gold appears where model can't confidently predict the restoration

Architecture:
    Template: Icosphere with fixed vertices/faces
    Encoder: Graph convolutions → global latent z
    Decoder: z → per-vertex offset from template
    Output: template_vertices + predicted_offsets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from typing import Tuple


def create_icosphere(subdivisions: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create an icosphere mesh with fixed topology.

    Args:
        subdivisions: Number of subdivision iterations (0=icosahedron, 2=642 verts)

    Returns:
        vertices: (V, 3) vertex positions on unit sphere
        faces: (F, 3) triangle face indices
    """
    # Start with icosahedron
    t = (1.0 + np.sqrt(5.0)) / 2.0

    vertices = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1],
    ], dtype=np.float32)
    vertices /= np.linalg.norm(vertices[0])

    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int64)

    # Subdivide
    for _ in range(subdivisions):
        vertices, faces = _subdivide(vertices, faces)

    return vertices, faces


def _subdivide(vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Subdivide each triangle into 4 triangles."""
    edge_midpoints = {}
    new_vertices = list(vertices)
    new_faces = []

    def get_midpoint(i1, i2):
        key = (min(i1, i2), max(i1, i2))
        if key not in edge_midpoints:
            mid = (vertices[i1] + vertices[i2]) / 2
            mid = mid / np.linalg.norm(mid)  # Project to sphere
            edge_midpoints[key] = len(new_vertices)
            new_vertices.append(mid)
        return edge_midpoints[key]

    for v0, v1, v2 in faces:
        a = get_midpoint(v0, v1)
        b = get_midpoint(v1, v2)
        c = get_midpoint(v2, v0)
        new_faces.extend([
            [v0, a, c], [v1, b, a], [v2, c, b], [a, b, c]
        ])

    return np.array(new_vertices, dtype=np.float32), np.array(new_faces, dtype=np.int64)


def build_adjacency(faces: np.ndarray, num_vertices: int) -> Tensor:
    """
    Build normalized adjacency matrix for graph convolution.

    Args:
        faces: (F, 3) face indices
        num_vertices: Number of vertices

    Returns:
        Sparse adjacency matrix (V, V) with self-loops, normalized
    """
    edges = set()
    for f in faces:
        edges.add((f[0], f[1]))
        edges.add((f[1], f[0]))
        edges.add((f[1], f[2]))
        edges.add((f[2], f[1]))
        edges.add((f[2], f[0]))
        edges.add((f[0], f[2]))

    # Add self-loops
    for i in range(num_vertices):
        edges.add((i, i))

    edges = list(edges)
    row = [e[0] for e in edges]
    col = [e[1] for e in edges]

    # Compute degree for normalization
    degree = np.zeros(num_vertices)
    for r in row:
        degree[r] += 1

    # D^{-1/2} A D^{-1/2} normalization
    values = []
    for r, c in zip(row, col):
        values.append(1.0 / np.sqrt(degree[r] * degree[c]))

    indices = torch.tensor([row, col], dtype=torch.long)
    values = torch.tensor(values, dtype=torch.float32)
    adj = torch.sparse_coo_tensor(indices, values, (num_vertices, num_vertices))

    return adj.coalesce()


class GraphConv(nn.Module):
    """Simple graph convolution layer."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """
        Args:
            x: Node features (B, V, C) or (V, C)
            adj: Adjacency matrix (V, V) sparse
        Returns:
            Updated features, same shape as x
        """
        # Handle batched input
        if x.dim() == 3:
            B, V, C = x.shape
            x_flat = x.reshape(B * V, C)
            # Apply linear
            h = self.linear(x_flat)  # (B*V, out)
            h = h.reshape(B, V, -1)
            # Graph conv: aggregate neighbors
            # adj is (V, V), h is (B, V, C) -> need to handle batch
            out = torch.stack([torch.sparse.mm(adj, h[b]) for b in range(B)])
        else:
            h = self.linear(x)
            out = torch.sparse.mm(adj, h)
        return out


class MeshEncoder(nn.Module):
    """Graph convolutional encoder for mesh."""

    def __init__(self, in_dim: int = 3, hidden_dim: int = 64, global_dim: int = 256):
        super().__init__()
        self.gc1 = GraphConv(in_dim, hidden_dim)
        self.gc2 = GraphConv(hidden_dim, hidden_dim)
        self.gc3 = GraphConv(hidden_dim, hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, global_dim)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """
        Args:
            x: Vertex positions (B, V, 3)
            adj: Adjacency matrix (V, V)
        Returns:
            Global feature (B, global_dim)
        """
        h = F.relu(self.gc1(x, adj))
        h = F.relu(self.gc2(h, adj))
        h = F.relu(self.gc3(h, adj))
        # Global pooling
        h = h.mean(dim=1)  # (B, hidden*2)
        return F.relu(self.fc(h))


class MeshDecoder(nn.Module):
    """
    Decoder that predicts per-vertex offsets from template.

    Takes global latent + template vertex positions → offset for each vertex.
    """

    def __init__(self, z_dim: int = 32, hidden_dim: int = 128, dropout_p: float = 0.3):
        super().__init__()
        # Per-vertex MLP: (z, template_coord) → offset
        self.mlp = nn.Sequential(
            nn.Linear(z_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim // 2, 3),
        )
        # Initialize final layer near zero for identity-like init
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, z: Tensor, template: Tensor) -> Tensor:
        """
        Args:
            z: Global latent (B, z_dim)
            template: Template vertex positions (V, 3)
        Returns:
            Offsets (B, V, 3)
        """
        B = z.shape[0]
        V = template.shape[0]

        # Expand z to all vertices
        z_exp = z.unsqueeze(1).expand(-1, V, -1)  # (B, V, z_dim)
        template_exp = template.unsqueeze(0).expand(B, -1, -1)  # (B, V, 3)

        # Concatenate and predict offset
        h = torch.cat([z_exp, template_exp], dim=2)  # (B, V, z_dim + 3)
        offsets = self.mlp(h)  # (B, V, 3)

        return offsets


class MeshVAE(nn.Module):
    """
    Mesh VAE with fixed template topology.

    The model learns to encode meshes into a latent space and decode
    by predicting per-vertex offsets from a template icosphere.

    For Kintsugi:
    - Train on clean meshes (spheres with small variations)
    - Damaged mesh (dent) → vertices displaced from expected positions
    - MC Dropout → high variance at damaged vertices
    - Gold overlay on high-variance vertices
    """

    def __init__(
        self,
        z_dim: int = 32,
        hidden_dim: int = 64,
        subdivisions: int = 2,
        dropout_p: float = 0.3,
    ):
        super().__init__()
        self.z_dim = z_dim

        # Create template icosphere
        vertices, faces = create_icosphere(subdivisions)
        self.register_buffer('template', torch.tensor(vertices, dtype=torch.float32))
        self.register_buffer('faces', torch.tensor(faces, dtype=torch.long))

        # Build adjacency matrix
        adj = build_adjacency(faces, len(vertices))
        self.register_buffer('adj', adj)

        self.num_vertices = len(vertices)

        # Encoder
        self.encoder = MeshEncoder(in_dim=3, hidden_dim=hidden_dim, global_dim=hidden_dim * 4)
        self.fc_mu = nn.Linear(hidden_dim * 4, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 4, z_dim)

        # Decoder
        self.decoder = MeshDecoder(z_dim=z_dim, hidden_dim=hidden_dim * 2, dropout_p=dropout_p)

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encode mesh vertices to latent distribution.

        Args:
            x: Vertex positions (B, V, 3)
        Returns:
            mu, log_var: Latent distribution parameters (B, z_dim)
        """
        h = self.encoder(x, self.adj)
        mu = self.fc_mu(h)
        log_var = torch.clamp(self.fc_logvar(h), -20, 20)
        return mu, log_var

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor) -> Tensor:
        """
        Decode latent to vertex positions.

        Args:
            z: Latent code (B, z_dim)
        Returns:
            Vertex positions (B, V, 3) = template + offsets
        """
        offsets = self.decoder(z, self.template)
        return self.template.unsqueeze(0) + offsets

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Full forward pass.

        Args:
            x: Input mesh vertices (B, V, 3)
        Returns:
            recon: Reconstructed vertices (B, V, 3)
            mu: Latent mean (B, z_dim)
            log_var: Latent log variance (B, z_dim)
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var

    def sample_with_uncertainty(
        self, x: Tensor, n_samples: int = 50
    ) -> Tuple[Tensor, Tensor]:
        """
        MC Dropout sampling for per-vertex uncertainty.

        Args:
            x: Input mesh vertices (B, V, 3)
            n_samples: Number of MC samples

        Returns:
            mean_recon: Mean reconstruction (B, V, 3)
            variance: Per-vertex variance (B, V, 3)
        """
        was_training = self.training
        self.train()  # Enable dropout

        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                recon, _, _ = self.forward(x)
                samples.append(recon)

        stacked = torch.stack(samples, dim=0)  # (n_samples, B, V, 3)
        mean_recon = stacked.mean(dim=0)
        variance = stacked.var(dim=0, unbiased=False)

        if not was_training:
            self.eval()

        return mean_recon, variance

    def get_template(self) -> Tuple[Tensor, Tensor]:
        """Return template vertices and faces for visualization."""
        return self.template.clone(), self.faces.clone()


def mesh_vae_loss(
    recon: Tensor,
    target: Tensor,
    mu: Tensor,
    log_var: Tensor,
    beta: float = 0.001
) -> Tensor:
    """
    Mesh VAE loss: vertex MSE + KL divergence.

    Args:
        recon: Reconstructed vertices (B, V, 3)
        target: Target vertices (B, V, 3)
        mu, log_var: Latent distribution parameters
        beta: KL weight

    Returns:
        Total loss
    """
    # Per-vertex L2 loss (correspondence preserved due to fixed topology)
    recon_loss = F.mse_loss(recon, target, reduction='mean')

    # KL divergence
    kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

    return recon_loss + beta * kl
