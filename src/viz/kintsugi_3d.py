"""
3D Kintsugi visualization: apply gold material to high-uncertainty vertices.

Extends the 2D kintsugi concept to point clouds:
- Vertices with high variance (model uncertainty) → gold color
- Vertices with low variance (confident reconstruction) → original/neutral color
"""

import numpy as np
from matplotlib import colormaps


def compute_vertex_uncertainty(variance: np.ndarray) -> np.ndarray:
    """
    Compute scalar uncertainty per vertex from 3D variance.

    Args:
        variance: Per-vertex variance, shape (N, 3)

    Returns:
        Scalar uncertainty per vertex, shape (N,)
    """
    # Use magnitude of variance vector as uncertainty
    return np.linalg.norm(variance, axis=1)


def render_kintsugi_3d(
    points: np.ndarray,
    variance: np.ndarray,
    gold_color: tuple[int, int, int] = (212, 175, 55),
    base_color: tuple[int, int, int] = (200, 200, 200),
    percentile: float = 85.0,
    gold_intensity: float = 0.9,
    smooth_blend: bool = True,
) -> np.ndarray:
    """
    Apply gold color to high-uncertainty vertices.

    Args:
        points: Point cloud, shape (N, 3) - used for context, not modified
        variance: Per-vertex variance from MC dropout, shape (N, 3)
        gold_color: RGB tuple for gold (0-255)
        base_color: RGB tuple for low-uncertainty vertices (0-255)
        percentile: Vertices above this percentile get gold
        gold_intensity: Blending factor [0, 1]
        smooth_blend: If True, blend gold proportional to uncertainty

    Returns:
        Vertex colors, shape (N, 3), values [0, 1]
    """
    N = points.shape[0]
    uncertainty = compute_vertex_uncertainty(variance)

    # Normalize uncertainty to [0, 1]
    u_min, u_max = uncertainty.min(), uncertainty.max()
    if np.isclose(u_max, u_min):
        norm_uncertainty = np.zeros(N)
    else:
        norm_uncertainty = (uncertainty - u_min) / (u_max - u_min)

    # Compute threshold
    threshold = np.percentile(norm_uncertainty, percentile)

    # Convert colors to [0, 1]
    gold = np.array(gold_color, dtype=np.float32) / 255.0
    base = np.array(base_color, dtype=np.float32) / 255.0

    # Initialize with base color
    colors = np.tile(base, (N, 1))

    if smooth_blend:
        # Smooth blending: intensity proportional to how much above threshold
        blend_factor = np.clip((norm_uncertainty - threshold) / (1 - threshold + 1e-8), 0, 1)
        blend_factor = blend_factor * gold_intensity
        colors = colors * (1 - blend_factor[:, None]) + gold * blend_factor[:, None]
    else:
        # Hard threshold
        mask = norm_uncertainty >= threshold
        colors[mask] = gold * gold_intensity + base * (1 - gold_intensity)

    return colors


def render_uncertainty_heatmap_3d(
    variance: np.ndarray,
    colormap: str = "magma",
) -> np.ndarray:
    """
    Color vertices by uncertainty magnitude using a colormap.

    Args:
        variance: Per-vertex variance, shape (N, 3)
        colormap: Matplotlib colormap name

    Returns:
        Vertex colors, shape (N, 3), values [0, 1]
    """
    uncertainty = compute_vertex_uncertainty(variance)

    # Normalize to [0, 1]
    u_min, u_max = uncertainty.min(), uncertainty.max()
    if np.isclose(u_max, u_min):
        norm = np.zeros_like(uncertainty)
    else:
        norm = (uncertainty - u_min) / (u_max - u_min)

    cmap = colormaps.get_cmap(colormap)
    colors = cmap(norm)[:, :3]  # Drop alpha
    return colors.astype(np.float64)


def create_kintsugi_comparison_3d(
    original: np.ndarray,
    reconstruction: np.ndarray,
    variance: np.ndarray,
    percentile: float = 85.0,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Create a set of visualizations for 3D kintsugi.

    Args:
        original: Original point cloud, shape (N, 3)
        reconstruction: Reconstructed point cloud, shape (M, 3)
        variance: Per-vertex variance, shape (M, 3)
        percentile: Threshold for gold overlay

    Returns:
        Dict with keys:
            'original': (points, colors) - gray
            'reconstruction': (points, colors) - gray
            'uncertainty': (points, colors) - heatmap
            'kintsugi': (points, colors) - gold overlay
    """
    gray = np.array([0.7, 0.7, 0.7])

    return {
        'original': (
            original,
            np.tile(gray, (original.shape[0], 1))
        ),
        'reconstruction': (
            reconstruction,
            np.tile(gray, (reconstruction.shape[0], 1))
        ),
        'uncertainty': (
            reconstruction,
            render_uncertainty_heatmap_3d(variance)
        ),
        'kintsugi': (
            reconstruction,
            render_kintsugi_3d(reconstruction, variance, percentile=percentile)
        ),
    }


def save_kintsugi_ply(
    points: np.ndarray,
    variance: np.ndarray,
    output_path: str,
    percentile: float = 85.0,
) -> None:
    """
    Save point cloud with kintsugi coloring to PLY file.

    Args:
        points: Point cloud, shape (N, 3)
        variance: Per-vertex variance, shape (N, 3)
        output_path: Output PLY file path
        percentile: Threshold for gold overlay
    """
    try:
        import open3d as o3d

        colors = render_kintsugi_3d(points, variance, percentile=percentile)

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(output_path, cloud)
    except ImportError:
        # Fallback: save as simple PLY without Open3D
        _save_ply_simple(points, colors, output_path)


def _save_ply_simple(
    points: np.ndarray,
    colors: np.ndarray,
    output_path: str
) -> None:
    """Fallback PLY writer without Open3D dependency."""
    N = points.shape[0]
    colors_uint8 = (colors * 255).astype(np.uint8)

    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(N):
            f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} "
                    f"{colors_uint8[i, 0]} {colors_uint8[i, 1]} {colors_uint8[i, 2]}\n")
