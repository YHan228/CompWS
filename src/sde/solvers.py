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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


def create_cube(n_points: int = 1000, size: float = 1.0) -> np.ndarray:
    """
    Generate points on cube surface.

    Args:
        n_points: Number of points
        size: Cube side length

    Returns:
        Point cloud, shape (N, 3)
    """
    raise NotImplementedError


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
    raise NotImplementedError
