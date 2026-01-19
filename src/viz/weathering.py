"""
3D visualization for SDE weathering using Open3D.

See IMPLEMENTATION_PLAN.md Section 2.4 for full specification.
"""

import numpy as np


def visualize_point_cloud(
    points: np.ndarray,
    colors: np.ndarray | None = None,
    window_name: str = "Point Cloud",
) -> None:
    """
    Display point cloud in interactive Open3D window.

    Args:
        points: Point cloud, shape (N, 3)
        colors: RGB colors, shape (N, 3), values [0, 1]. If None, use uniform gray.
        window_name: Window title
    """
    raise NotImplementedError


def visualize_weathering_sequence(
    trajectories: np.ndarray,
    sample_frames: list[int] | None = None,
    n_frames: int = 5,
) -> None:
    """
    Display multiple frames of weathering process.

    Args:
        trajectories: Full trajectory, shape (T, N, 3)
        sample_frames: Specific timesteps to show. If None, sample uniformly.
        n_frames: Number of frames if sample_frames is None
    """
    raise NotImplementedError


def create_weathering_animation(
    trajectories: np.ndarray,
    output_path: str,
    fps: int = 10,
    resolution: tuple[int, int] = (800, 600),
) -> None:
    """
    Export weathering process as GIF animation.

    Uses Open3D OffscreenRenderer for headless rendering.

    Args:
        trajectories: Full trajectory, shape (T, N, 3)
        output_path: Output file path (.gif)
        fps: Frames per second
        resolution: Image resolution (width, height)
    """
    raise NotImplementedError


def color_by_displacement(
    original: np.ndarray,
    weathered: np.ndarray,
    colormap: str = "coolwarm",
) -> np.ndarray:
    """
    Color points by displacement magnitude.

    Args:
        original: Original point cloud, shape (N, 3)
        weathered: Weathered point cloud, shape (N, 3)
        colormap: Matplotlib colormap name

    Returns:
        RGB colors, shape (N, 3), values [0, 1]
        Blue = minimal displacement, Red = maximum displacement
    """
    raise NotImplementedError


def save_point_cloud(
    points: np.ndarray,
    output_path: str,
    colors: np.ndarray | None = None,
) -> None:
    """
    Save point cloud to PLY file.

    Args:
        points: Point cloud, shape (N, 3)
        output_path: Output file path (.ply)
        colors: RGB colors, shape (N, 3), values [0, 1]
    """
    raise NotImplementedError
