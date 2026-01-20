"""
Kintsugi visualization: overlay gold on high-uncertainty regions.

See IMPLEMENTATION_PLAN.md Section 1.3-1.4 for full specification.
"""

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch import Tensor
from torchvision.transforms import functional as tvf
from matplotlib import cm


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
    variance = np.asarray(variance, dtype=np.float32)
    var_min = float(np.min(variance))
    var_max = float(np.max(variance))
    if np.isclose(var_max, var_min):
        norm_var = np.zeros_like(variance)
    else:
        norm_var = (variance - var_min) / (var_max - var_min)

    if variance_threshold is None:
        threshold = np.percentile(norm_var, percentile)
    else:
        threshold = variance_threshold

    mask = norm_var >= threshold
    if blur_edges:
        blend_mask = gaussian_filter(mask.astype(np.float32), sigma=1.0)
        blend_mask = np.clip(blend_mask, 0.0, 1.0)
    else:
        blend_mask = mask.astype(np.float32)

    base = np.clip(reconstruction, 0.0, 1.0)
    if base.ndim == 2:
        base_rgb = np.repeat(base[..., None], 3, axis=2)
    else:
        base_rgb = base

    gold = np.array(gold_color, dtype=np.float32) / 255.0
    overlay_strength = gold_intensity * blend_mask[..., None]
    output = base_rgb * (1.0 - overlay_strength) + gold * overlay_strength
    output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
    return output


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
    model_device = next(model.parameters()).device
    rows = int(np.ceil(test_images.shape[0] / cols))
    tile_size = 28
    grid_width = cols * 4 * tile_size
    grid_height = rows * tile_size
    grid = Image.new("RGB", (grid_width, grid_height))

    model.eval()
    with torch.no_grad():
        for idx, img in enumerate(test_images):
            batch = img.unsqueeze(0).to(model_device)
            mean_recon, variance = model.sample_with_uncertainty(
                batch, n_samples=n_mc_samples
            )
            mean_recon_np = mean_recon.squeeze().cpu().numpy()
            variance_np = variance.squeeze().cpu().numpy()
            original_np = img.squeeze().cpu().numpy()

            variance_norm = variance_np
            var_min = float(np.min(variance_norm))
            var_max = float(np.max(variance_norm))
            if not np.isclose(var_max, var_min):
                variance_norm = (variance_norm - var_min) / (var_max - var_min)
            else:
                variance_norm = np.zeros_like(variance_norm)
            variance_rgb = (cm.get_cmap("magma")(variance_norm)[:, :, :3] * 255).astype(
                np.uint8
            )

            kintsugi_img = render_kintsugi(
                original_np, mean_recon_np, variance_np
            )
            original_rgb = (
                np.repeat(original_np[..., None], 3, axis=2) * 255
            ).astype(np.uint8)
            recon_rgb = (
                np.repeat(mean_recon_np[..., None], 3, axis=2) * 255
            ).astype(np.uint8)

            images = [
                Image.fromarray(original_rgb),
                Image.fromarray(recon_rgb),
                Image.fromarray(variance_rgb),
                Image.fromarray(kintsugi_img),
            ]
            row = idx // cols
            col = idx % cols
            x_offset = col * 4 * tile_size
            y_offset = row * tile_size
            for j, tile in enumerate(images):
                grid.paste(tile, (x_offset + j * tile_size, y_offset))
    return grid


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
    if img.dim() == 2:
        img_out = img.clone()
        _, height, width = 1, img_out.shape[0], img_out.shape[1]
        y = torch.randint(0, max(1, height - box_size + 1), (1,)).item()
        x = torch.randint(0, max(1, width - box_size + 1), (1,)).item()
        img_out[y : y + box_size, x : x + box_size] = value
        return img_out

    img_out = img.clone()
    _, height, width = img_out.shape
    y = torch.randint(0, max(1, height - box_size + 1), (1,)).item()
    x = torch.randint(0, max(1, width - box_size + 1), (1,)).item()
    img_out[:, y : y + box_size, x : x + box_size] = value
    return img_out


def add_gaussian_noise(img: Tensor, std: float = 0.3) -> Tensor:
    """
    Add Gaussian noise to image.

    Args:
        img: Image tensor, shape (1, H, W) or (H, W)
        std: Noise standard deviation

    Returns:
        Noisy image, clamped to [0, 1]
    """
    noise = torch.randn_like(img) * std
    return torch.clamp(img + noise, 0.0, 1.0)


def rotate_image(img: Tensor, angle: float = 45.0) -> Tensor:
    """
    Rotate image by specified angle.

    Args:
        img: Image tensor, shape (1, H, W)
        angle: Rotation angle in degrees

    Returns:
        Rotated image
    """
    return tvf.rotate(img, angle=angle, interpolation=tvf.InterpolationMode.BILINEAR)


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
    return torch.clamp(img1 * (1.0 - alpha) + img2 * alpha, 0.0, 1.0)
