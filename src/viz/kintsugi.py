"""
Kintsugi visualization: overlay gold on high-uncertainty regions.

See IMPLEMENTATION_PLAN.md Section 1.3-1.4 for full specification.
"""

import numpy as np
from PIL import Image
from torch import Tensor


def render_kintsugi(
    original: np.ndarray,
    reconstruction: np.ndarray,
    variance: np.ndarray,
    gold_color: tuple[int, int, int] = (212, 175, 55),
    variance_threshold: float | None = None,
    percentile: float = 90.0,
    gold_intensity: float = 0.8,
    blur_edges: bool = True,
) -> np.ndarray:
    """
    Overlay gold color on high-variance (uncertain) regions.

    Args:
        original: Original image, shape (H, W), values [0, 1]
        reconstruction: Mean reconstruction, shape (H, W), values [0, 1]
        variance: Per-pixel variance, shape (H, W)
        gold_color: RGB tuple for gold overlay
        variance_threshold: Absolute threshold (if None, use percentile)
        percentile: Use top X percentile as threshold (default 90th)
        gold_intensity: Blending factor for gold overlay [0, 1]
        blur_edges: Apply Gaussian blur to mask edges for smoother veins

    Returns:
        RGB image as uint8, shape (H, W, 3)
    """
    raise NotImplementedError


def create_kintsugi_grid(
    model,  # KintsugiVAE
    test_images: Tensor,
    n_mc_samples: int = 50,
    cols: int = 4,
) -> Image.Image:
    """
    Generate comparison grid: Original | Reconstruction | Variance | Kintsugi

    Args:
        model: Trained KintsugiVAE model
        test_images: Test images, shape (N, 1, 28, 28)
        n_mc_samples: Number of MC dropout samples
        cols: Number of image sets per row

    Returns:
        PIL Image of the grid
    """
    raise NotImplementedError


# --- OOD Perturbation Functions ---


def add_occlusion(img: Tensor, box_size: int = 10, value: float = 0.0) -> Tensor:
    """
    Add random rectangular occlusion to image.

    Args:
        img: Image tensor, shape (1, H, W) or (H, W)
        box_size: Size of occluding square
        value: Fill value (0=black, 1=white)

    Returns:
        Occluded image, same shape as input
    """
    raise NotImplementedError


def add_gaussian_noise(img: Tensor, std: float = 0.3) -> Tensor:
    """
    Add Gaussian noise to image.

    Args:
        img: Image tensor, shape (1, H, W) or (H, W)
        std: Noise standard deviation

    Returns:
        Noisy image, clamped to [0, 1]
    """
    raise NotImplementedError


def rotate_image(img: Tensor, angle: float = 45.0) -> Tensor:
    """
    Rotate image by specified angle.

    Args:
        img: Image tensor, shape (1, H, W)
        angle: Rotation angle in degrees

    Returns:
        Rotated image
    """
    raise NotImplementedError


def mix_digits(img1: Tensor, img2: Tensor, alpha: float = 0.5) -> Tensor:
    """
    Blend two images together.

    Args:
        img1: First image tensor
        img2: Second image tensor
        alpha: Blending factor (0=img1, 1=img2)

    Returns:
        Blended image
    """
    raise NotImplementedError
