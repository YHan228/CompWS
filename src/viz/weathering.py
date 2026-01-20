"""
3D visualization for SDE weathering using Open3D.

See IMPLEMENTATION_PLAN.md Section 2.4 for full specification.
"""

import numpy as np
from PIL import Image
from matplotlib import cm


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
    try:
        import open3d as o3d

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        if colors is None:
            colors = np.tile(np.array([[0.7, 0.7, 0.7]]), (points.shape[0], 1))
        cloud.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([cloud], window_name=window_name)
    except Exception:
        return


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
    total_frames = trajectories.shape[0]
    if sample_frames is None:
        sample_frames = np.linspace(0, total_frames - 1, n_frames, dtype=int).tolist()
    for frame in sample_frames:
        visualize_point_cloud(trajectories[frame])


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
    try:
        import open3d as o3d
        from open3d.visualization.rendering import OffscreenRenderer

        width, height = resolution
        renderer = OffscreenRenderer(width, height)
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultUnlit"

        images = []
        for frame in trajectories:
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(frame)
            cloud.paint_uniform_color([0.7, 0.7, 0.7])
            renderer.scene.clear_geometry()
            renderer.scene.add_geometry("cloud", cloud, material)
            image = renderer.render_to_image()
            images.append(Image.fromarray(np.asarray(image)))

        if images:
            duration = int(1000 / fps)
            images[0].save(
                output_path,
                save_all=True,
                append_images=images[1:],
                duration=duration,
                loop=0,
            )
        renderer.release_resources()
    except Exception:
        return


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
    displacement = np.linalg.norm(weathered - original, axis=1)
    if np.isclose(displacement.max(), displacement.min()):
        norm = np.zeros_like(displacement)
    else:
        norm = (displacement - displacement.min()) / (
            displacement.max() - displacement.min()
        )
    cmap = cm.get_cmap(colormap)
    colors = cmap(norm)[:, :3]
    return colors.astype(float)


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
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        cloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(output_path, cloud)
