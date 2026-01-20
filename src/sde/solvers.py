"""
Stochastic Differential Equation solvers for weathering simulation.

See IMPLEMENTATION_PLAN.md Section 2.2-2.3 for full specification.
"""

import numpy as np


def euler_maruyama_ou(
    x0: np.ndarray,
    theta: float,
    mu: np.ndarray,
    sigma_t: np.ndarray,
    dt: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate Ornstein-Uhlenbeck process using Euler-Maruyama method.

    SDE: dX_t = θ(μ - X_t)dt + σ(t)dW_t

    Args:
        x0: Initial state, shape (D,) or (N, D)
        theta: Mean reversion rate (higher = more resistant to change)
        mu: Long-term mean, shape (D,)
        sigma_t: Time-varying diffusion coefficients, shape (T,)
        dt: Time step size
        seed: Random seed for reproducibility

    Returns:
        Trajectory of shape (T, D) or (T, N, D)
    """
    if sigma_t.size == 0:
        raise ValueError("sigma_t must contain at least one timestep.")
    x0 = np.asarray(x0, dtype=float)
    mu = np.asarray(mu, dtype=float)
    trajectory = np.zeros((len(sigma_t), *x0.shape), dtype=float)
    trajectory[0] = x0
    rng = np.random.default_rng(seed)

    for t in range(1, len(sigma_t)):
        d_w = rng.standard_normal(size=x0.shape) * np.sqrt(dt)
        drift = theta * (mu - trajectory[t - 1]) * dt
        diffusion = sigma_t[t] * d_w
        trajectory[t] = trajectory[t - 1] + drift + diffusion

    return trajectory


def apply_sde_to_points(
    points: np.ndarray,
    sigma_t: np.ndarray,
    theta: float = 0.1,
    mu_mode: str = "centroid",
    seed: int | None = None,
) -> np.ndarray:
    """
    Apply OU weathering process to a 3D point cloud.

    Args:
        points: Point cloud, shape (N, 3)
        sigma_t: Diffusion coefficients from external data, shape (T,)
        theta: Mean reversion rate
        mu_mode: How to compute long-term mean:
            - "centroid": use point cloud centroid
            - "origin": use (0, 0, 0)
            - "self": each point reverts to itself (no drift)
        seed: Random seed

    Returns:
        Weathered point cloud, shape (N, 3) — final state after T steps
    """
    points = np.asarray(points, dtype=float)
    if mu_mode == "centroid":
        mu = points.mean(axis=0)
    elif mu_mode == "origin":
        mu = np.zeros(3, dtype=float)
    elif mu_mode == "self":
        mu = points
    else:
        raise ValueError(f"Unsupported mu_mode: {mu_mode}")

    trajectory = euler_maruyama_ou(
        points, theta=theta, mu=mu, sigma_t=sigma_t, seed=seed
    )
    return trajectory[-1]


def get_trajectory(
    points: np.ndarray,
    sigma_t: np.ndarray,
    theta: float = 0.1,
    mu_mode: str = "centroid",
    seed: int | None = None,
) -> np.ndarray:
    """
    Get full weathering trajectory for animation.

    Args:
        Same as apply_sde_to_points

    Returns:
        Full trajectory, shape (T, N, 3)
    """
    points = np.asarray(points, dtype=float)
    if mu_mode == "centroid":
        mu = points.mean(axis=0)
    elif mu_mode == "origin":
        mu = np.zeros(3, dtype=float)
    elif mu_mode == "self":
        mu = points
    else:
        raise ValueError(f"Unsupported mu_mode: {mu_mode}")

    return euler_maruyama_ou(points, theta=theta, mu=mu, sigma_t=sigma_t, seed=seed)


# --- Geometry Primitives ---


def create_sphere(n_points: int = 1000, radius: float = 1.0) -> np.ndarray:
    """
    Generate uniformly distributed points on sphere surface.

    Uses Fibonacci lattice for uniform distribution.

    Args:
        n_points: Number of points
        radius: Sphere radius

    Returns:
        Point cloud, shape (N, 3)
    """
    indices = np.arange(n_points, dtype=float)
    phi = (1 + np.sqrt(5)) / 2
    theta = 2 * np.pi * indices / phi
    z = 1 - 2 * (indices + 0.5) / n_points
    radius_xy = np.sqrt(1 - z**2)
    x = radius_xy * np.cos(theta)
    y = radius_xy * np.sin(theta)
    points = np.stack([x, y, z], axis=1) * radius
    return points


def create_cube(n_points: int = 1000, size: float = 1.0) -> np.ndarray:
    """
    Generate points on cube surface.

    Args:
        n_points: Number of points
        size: Cube side length

    Returns:
        Point cloud, shape (N, 3)
    """
    rng = np.random.default_rng()
    half = size / 2
    face_ids = rng.integers(0, 6, size=n_points)
    u = rng.uniform(-half, half, size=n_points)
    v = rng.uniform(-half, half, size=n_points)
    points = np.zeros((n_points, 3), dtype=float)

    for idx in range(n_points):
        face = face_ids[idx]
        if face == 0:
            points[idx] = [half, u[idx], v[idx]]
        elif face == 1:
            points[idx] = [-half, u[idx], v[idx]]
        elif face == 2:
            points[idx] = [u[idx], half, v[idx]]
        elif face == 3:
            points[idx] = [u[idx], -half, v[idx]]
        elif face == 4:
            points[idx] = [u[idx], v[idx], half]
        else:
            points[idx] = [u[idx], v[idx], -half]

    return points


def create_torus(
    n_points: int = 1000, R: float = 1.0, r: float = 0.3
) -> np.ndarray:
    """
    Generate points on torus surface.

    Args:
        n_points: Number of points
        R: Major radius (center of tube to center of torus)
        r: Minor radius (tube radius)

    Returns:
        Point cloud, shape (N, 3)
    """
    rng = np.random.default_rng()
    u = rng.uniform(0, 2 * np.pi, size=n_points)
    v = rng.uniform(0, 2 * np.pi, size=n_points)
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    return np.stack([x, y, z], axis=1)
