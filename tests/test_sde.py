"""
Unit tests for SDE solvers.
"""

import numpy as np
import pytest


class TestEulerMaruyamaOU:
    """Tests for the Ornstein-Uhlenbeck Euler-Maruyama solver."""

    def test_output_shape_1d(self):
        """Test output shape for 1D input."""
        from src.sde.solvers import euler_maruyama_ou

        x0 = np.array([0.0])
        sigma_t = np.ones(100) * 0.01
        mu = np.array([0.0])

        result = euler_maruyama_ou(x0, theta=0.1, mu=mu, sigma_t=sigma_t, seed=42)

        assert result.shape == (100, 1), f"Expected (100, 1), got {result.shape}"

    def test_output_shape_3d_points(self):
        """Test output shape for 3D point cloud."""
        from src.sde.solvers import euler_maruyama_ou

        x0 = np.random.randn(50, 3)  # 50 points in 3D
        sigma_t = np.ones(100) * 0.01
        mu = np.zeros(3)

        result = euler_maruyama_ou(x0, theta=0.1, mu=mu, sigma_t=sigma_t, seed=42)

        assert result.shape == (100, 50, 3), f"Expected (100, 50, 3), got {result.shape}"

    def test_reproducibility(self):
        """Test that same seed gives same result."""
        from src.sde.solvers import euler_maruyama_ou

        x0 = np.array([1.0, 2.0, 3.0])
        sigma_t = np.random.rand(50) * 0.1
        mu = np.zeros(3)

        result1 = euler_maruyama_ou(x0, theta=0.1, mu=mu, sigma_t=sigma_t, seed=123)
        result2 = euler_maruyama_ou(x0, theta=0.1, mu=mu, sigma_t=sigma_t, seed=123)

        np.testing.assert_array_equal(result1, result2)

    def test_zero_diffusion_no_noise(self):
        """With sigma=0, should follow deterministic mean reversion."""
        from src.sde.solvers import euler_maruyama_ou

        x0 = np.array([10.0])
        sigma_t = np.zeros(100)
        mu = np.array([0.0])

        result = euler_maruyama_ou(x0, theta=0.5, mu=mu, sigma_t=sigma_t, seed=42)

        # Should decay toward mu=0
        assert result[-1, 0] < result[0, 0], "Should decay toward mean"


class TestGeometryPrimitives:
    """Tests for geometry primitive generators."""

    def test_sphere_shape(self):
        """Test sphere output shape."""
        from src.sde.solvers import create_sphere

        points = create_sphere(n_points=500, radius=2.0)

        assert points.shape == (500, 3)

    def test_sphere_radius(self):
        """Test that sphere points are at correct radius."""
        from src.sde.solvers import create_sphere

        radius = 2.5
        points = create_sphere(n_points=1000, radius=radius)
        distances = np.linalg.norm(points, axis=1)

        np.testing.assert_allclose(distances, radius, rtol=0.01)

    def test_cube_bounds(self):
        """Test that cube points are within bounds."""
        from src.sde.solvers import create_cube

        size = 2.0
        points = create_cube(n_points=1000, size=size)

        assert np.all(points >= -size / 2)
        assert np.all(points <= size / 2)

    def test_torus_shape(self):
        """Test torus output shape."""
        from src.sde.solvers import create_torus

        points = create_torus(n_points=800)

        assert points.shape == (800, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
