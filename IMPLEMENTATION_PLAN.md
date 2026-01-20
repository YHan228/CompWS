# Implementation Plan: Computational Wabi-Sabi

**Target Audience:** Coding agent / implementer
**Scope:** Module II (Bayesian Kintsugi) → Module I (SDE Weathering)
**Module III:** Deferred

---

## Progress Log

- Implemented VAE architecture, data fetchers, SDE solvers, visualization utilities, and notebook scaffolding for smoke tests. (2026-01-19)

---

## Phase 0: Project Setup

### 0.1 Directory Structure

```
compwabi/
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── vae.py              # VAE with MC Dropout
│   ├── noise/
│   │   ├── __init__.py
│   │   └── data_sources.py     # API fetchers (yfinance, openmeteo)
│   ├── sde/
│   │   ├── __init__.py
│   │   └── solvers.py          # Euler-Maruyama, OU process
│   └── viz/
│       ├── __init__.py
│       └── kintsugi.py         # Uncertainty overlay rendering
├── notebooks/
│   ├── 01_vae_training.ipynb
│   ├── 02_kintsugi_demo.ipynb
│   └── 03_sde_weathering.ipynb
├── tests/
│   └── test_sde.py
├── results/                    # Generated artifacts (gitignored)
├── data/                       # Cached datasets (gitignored)
├── requirements.txt
├── PLAN.md
└── IMPLEMENTATION_PLAN.md
```

### 0.2 Dependencies (`requirements.txt`)

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
pillow>=9.5.0
yfinance>=0.2.0
openmeteo-requests>=1.0.0
requests-cache>=1.0.0
biopython>=1.81
tqdm>=4.65.0
open3d>=0.17.0
```

---

## Phase 1: Module II — Bayesian Kintsugi

### 1.1 VAE Architecture (`src/models/vae.py`)

Implement a convolutional VAE with MC Dropout for uncertainty estimation.

**Class: `KintsugiVAE`**

```python
class KintsugiVAE(nn.Module):
    """
    Convolutional VAE with dropout layers retained at inference
    for Monte Carlo uncertainty estimation.
    """
```

**Architecture Spec:**

| Component | Layers |
|-----------|--------|
| Encoder | Conv2d(1→32, k=3, s=2) → ReLU → Dropout(0.25) → Conv2d(32→64, k=3, s=2) → ReLU → Dropout(0.25) → Flatten → Linear(64*7*7 → 256) → ReLU |
| Latent | Linear(256 → z_dim) for μ, Linear(256 → z_dim) for log_var. Default z_dim=20 |
| Decoder | Linear(z_dim → 256) → ReLU → Dropout(0.25) → Linear(256 → 64*7*7) → Reshape → ConvTranspose2d(64→32, k=3, s=2) → ReLU → Dropout(0.25) → ConvTranspose2d(32→1, k=3, s=2) → Sigmoid |

**Key Methods:**

```python
def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
    """Returns (mu, log_var)"""

def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
    """z = mu + std * epsilon"""

def decode(self, z: Tensor) -> Tensor:
    """Returns reconstructed image"""

def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Returns (reconstruction, mu, log_var)"""

def sample_with_uncertainty(self, x: Tensor, n_samples: int = 50) -> tuple[Tensor, Tensor]:
    """
    Run n_samples forward passes with dropout enabled.
    Returns (mean_reconstruction, pixel_variance)
    """
```

**Loss Function:**

```python
def vae_loss(recon_x, x, mu, log_var, beta=1.0):
    """
    ELBO = BCE(recon, x) + beta * KL(q(z|x) || p(z))
    """
    bce = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return bce + beta * kl
```

### 1.2 Training Pipeline (`notebooks/01_vae_training.ipynb`)

**Steps:**

1. Load MNIST via `torchvision.datasets.MNIST`
2. Normalize to [0, 1]
3. Train for 30 epochs, batch_size=128, lr=1e-3 (Adam)
4. Save model checkpoint to `results/vae_mnist.pt`

**Training Loop Pseudocode:**

```python
for epoch in range(30):
    model.train()
    for batch in dataloader:
        optimizer.zero_grad()
        recon, mu, log_var = model(batch)
        loss = vae_loss(recon, batch, mu, log_var)
        loss.backward()
        optimizer.step()
```

### 1.3 Kintsugi Renderer (`src/viz/kintsugi.py`)

**Function: `render_kintsugi`**

```python
def render_kintsugi(
    original: np.ndarray,        # Shape (H, W), values [0, 1]
    reconstruction: np.ndarray,  # Shape (H, W), mean of MC samples
    variance: np.ndarray,        # Shape (H, W), pixel-wise variance
    gold_color: tuple = (212, 175, 55),  # RGB gold
    variance_threshold: float = 0.02,    # Percentile or absolute
    gold_intensity: float = 0.8
) -> np.ndarray:
    """
    Overlay gold color on high-variance regions.

    Returns: RGB image (H, W, 3) as uint8
    """
```

**Algorithm:**

1. Normalize variance to [0, 1] range
2. Create binary mask: `mask = variance > threshold` (or use top 10th percentile)
3. Convert reconstruction to RGB (grayscale → 3 channels)
4. Apply gold color where mask is True: `output[mask] = lerp(output[mask], gold_color, gold_intensity)`
5. Optional: Apply Gaussian blur to mask edges for smoother gold "veins"

**Function: `create_kintsugi_grid`**

```python
def create_kintsugi_grid(
    model: KintsugiVAE,
    test_images: Tensor,      # (N, 1, 28, 28)
    n_mc_samples: int = 50
) -> PIL.Image:
    """
    Generate a grid showing: Original | Reconstruction | Variance Heatmap | Kintsugi
    """
```

### 1.4 OOD Perturbation Functions (`src/viz/kintsugi.py`)

To induce uncertainty, apply perturbations that push inputs out-of-distribution:

```python
def add_occlusion(img: Tensor, box_size: int = 10) -> Tensor:
    """Add random black square occlusion"""

def add_gaussian_noise(img: Tensor, std: float = 0.3) -> Tensor:
    """Add Gaussian noise"""

def rotate_image(img: Tensor, angle: float = 45) -> Tensor:
    """Rotate image by angle degrees"""

def mix_digits(img1: Tensor, img2: Tensor, alpha: float = 0.5) -> Tensor:
    """Blend two digit images"""
```

### 1.5 Demo Notebook (`notebooks/02_kintsugi_demo.ipynb`)

**Cells:**

1. Load trained VAE from checkpoint
2. Load 10 random MNIST test images
3. Apply each perturbation type
4. Run `sample_with_uncertainty(perturbed, n_samples=50)`
5. Render kintsugi images
6. Save grid to `results/kintsugi_gallery.png`

**Expected Output:** A grid where occluded/noised regions show gold "scars" where the model is uncertain.

---

## Phase 2: Module I — SDE Weathering

### 2.1 Data Source Fetchers (`src/noise/data_sources.py`)

**Class: `VolatilityFetcher`**

```python
class VolatilityFetcher:
    """Fetch historical financial volatility as noise source."""

    def __init__(self, ticker: str = "^VIX", start: str = "2014-01-01", end: str = "2024-01-01"):
        pass

    def fetch(self) -> pd.Series:
        """Returns daily values as pandas Series with datetime index."""

    def to_diffusion_coefficients(self, scale: float = 0.01) -> np.ndarray:
        """Normalize to [0, scale] range for use as sigma(t) in SDE."""
```

**Class: `WeatherFetcher`**

```python
class WeatherFetcher:
    """Fetch historical weather data (wind speed, temperature) as noise source."""

    def __init__(self, latitude: float, longitude: float, start: str, end: str):
        pass

    def fetch(self, variable: str = "wind_speed_10m_max") -> pd.Series:
        """Returns daily values."""

    def to_diffusion_coefficients(self, scale: float = 0.01) -> np.ndarray:
        """Normalize for SDE use."""
```

### 2.2 SDE Solvers (`src/sde/solvers.py`)

**Ornstein-Uhlenbeck Process:**

The SDE to implement:

```
dX_t = θ(μ - X_t)dt + σ(t)dW_t
```

Where:
- `θ` (theta): mean reversion rate
- `μ` (mu): long-term mean
- `σ(t)`: time-varying diffusion coefficient (from external data)
- `dW_t`: Wiener process increment

**Function: `euler_maruyama_ou`**

```python
def euler_maruyama_ou(
    x0: np.ndarray,              # Initial state, shape (D,) or (N, D)
    theta: float,                # Mean reversion rate
    mu: np.ndarray,              # Long-term mean, shape (D,)
    sigma_t: np.ndarray,         # Time-varying diffusion, shape (T,)
    dt: float = 1.0,             # Time step
    seed: int = None
) -> np.ndarray:
    """
    Simulate OU process using Euler-Maruyama method.

    Returns: Trajectory of shape (T, D) or (T, N, D)
    """
```

**Algorithm:**

```python
T = len(sigma_t)
X = np.zeros((T, *x0.shape))
X[0] = x0

for t in range(1, T):
    dW = np.random.randn(*x0.shape) * np.sqrt(dt)
    drift = theta * (mu - X[t-1]) * dt
    diffusion = sigma_t[t] * dW
    X[t] = X[t-1] + drift + diffusion

return X
```

**Function: `apply_sde_to_points`**

```python
def apply_sde_to_points(
    points: np.ndarray,          # Shape (N, 3) point cloud
    sigma_t: np.ndarray,         # Diffusion coefficients from external data
    theta: float = 0.1,
    mu_mode: str = "centroid"    # "centroid", "origin", or "self"
) -> np.ndarray:
    """
    Apply OU weathering to a 3D point cloud.

    Returns: Weathered points, shape (N, 3)
    """
```

### 2.3 Geometry Primitives (`src/sde/solvers.py`)

```python
def create_sphere(n_points: int = 1000, radius: float = 1.0) -> np.ndarray:
    """Generate uniform points on sphere surface. Returns (N, 3)."""

def create_cube(n_points: int = 1000, size: float = 1.0) -> np.ndarray:
    """Generate points on cube surface. Returns (N, 3)."""

def create_torus(n_points: int = 1000, R: float = 1.0, r: float = 0.3) -> np.ndarray:
    """Generate points on torus surface. Returns (N, 3)."""
```

### 2.4 3D Visualization (`src/viz/weathering.py`)

Use Open3D for point cloud rendering:

```python
import open3d as o3d

def visualize_weathering_sequence(
    trajectories: np.ndarray,    # Shape (T, N, 3)
    sample_frames: list[int],    # Which timesteps to show
    save_path: str = None
) -> None:
    """Display or save point cloud sequence."""

def create_weathering_animation(
    trajectories: np.ndarray,
    output_path: str,            # .gif or .mp4
    fps: int = 10
) -> None:
    """Export weathering process as animation."""

def color_by_displacement(
    original: np.ndarray,        # (N, 3)
    weathered: np.ndarray        # (N, 3)
) -> np.ndarray:
    """
    Color points by how much they moved.
    Returns: RGB colors (N, 3) - blue=static, red=moved
    """
```

### 2.5 Demo Notebook (`notebooks/03_sde_weathering.ipynb`)

**Cells:**

1. Fetch 10 years of VIX data using `VolatilityFetcher`
2. Plot the raw volatility time series
3. Create a sphere point cloud (N=2000)
4. Apply `euler_maruyama_ou` with VIX as `sigma_t`
5. Visualize: original sphere vs. weathered sphere
6. Color by displacement magnitude
7. Create animation showing gradual weathering
8. Save artifacts to `results/`

**Variations to Explore:**

- Different theta values (high θ = resistant to weathering, low θ = malleable)
- Different geometries (cube, torus)
- Different data sources (weather vs. financial)
- Accumulated weathering (run multiple passes)

---

## Phase 3: Integration & Gallery

### 3.1 Combined Artifacts

After Phases 1-2 are complete:

1. **2D Kintsugi Gallery:** Grid of MNIST reconstructions with gold uncertainty overlays
2. **3D Weathered Objects:** Point clouds aged by historical data, colored by displacement
3. **Comparative Visualizations:** Same object weathered by financial vs. climate data

### 3.2 Final Outputs

| Artifact | Format | Location |
|----------|--------|----------|
| Trained VAE | `.pt` | `results/vae_mnist.pt` |
| Kintsugi samples | `.png` | `results/kintsugi_*.png` |
| Weathered point clouds | `.ply` | `results/weathered_*.ply` |
| Weathering animations | `.gif` | `results/weathering_*.gif` |

---

## Implementation Order (Checklist)

```
[ ] Phase 0: Setup
    [ ] Create directory structure
    [ ] Create requirements.txt
    [ ] Initialize src/ packages with __init__.py

[ ] Phase 1: Bayesian Kintsugi
    [ ] Implement KintsugiVAE class
    [ ] Implement vae_loss function
    [ ] Create training notebook, train on MNIST
    [ ] Implement render_kintsugi function
    [ ] Implement OOD perturbation functions
    [ ] Create demo notebook with gallery output

[ ] Phase 2: SDE Weathering
    [ ] Implement VolatilityFetcher
    [ ] Implement WeatherFetcher
    [ ] Implement euler_maruyama_ou solver
    [ ] Implement geometry primitives
    [ ] Implement apply_sde_to_points
    [ ] Implement 3D visualization functions
    [ ] Create demo notebook with animations

[ ] Phase 3: Polish
    [ ] Generate final gallery artifacts
    [ ] Write results documentation
```

---

## Notes for Implementer

1. **Dropout at Inference:** Ensure `model.train()` is called before MC sampling to keep dropout active, or manually set dropout layers to training mode.

2. **Numerical Stability:** In VAE, clamp `log_var` to [-20, 20] to prevent NaN in KL divergence.

3. **SDE Scaling:** The `sigma_t` values from financial data will need normalization. VIX ranges ~10-80; normalize to ~0.001-0.05 for reasonable point displacement.

4. **Open3D Headless:** For server environments, use `o3d.visualization.rendering.OffscreenRenderer` instead of interactive visualization.

5. **Reproducibility:** Set random seeds (`torch.manual_seed`, `np.random.seed`) in notebooks for reproducible outputs.
