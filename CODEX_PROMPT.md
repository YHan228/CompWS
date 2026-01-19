# Codex Task: Implement Computational Wabi-Sabi

## Objective

Implement all stub functions in `src/` so that the codebase is fully functional. Do NOT run full training — only verify code correctness with smoke tests.

## Scope

### MUST Implement

1. **`src/models/vae.py`**
   - `KintsugiVAE.__init__`: Build encoder/decoder layers per spec
   - `KintsugiVAE.encode`: Return (mu, log_var)
   - `KintsugiVAE.reparameterize`: z = mu + std * eps
   - `KintsugiVAE.decode`: Reconstruct from latent
   - `KintsugiVAE.forward`: Full pass
   - `KintsugiVAE.sample_with_uncertainty`: MC dropout sampling (keep dropout active via `self.train()`)
   - `vae_loss`: BCE + beta * KL

2. **`src/noise/data_sources.py`**
   - `VolatilityFetcher`: Fetch VIX via `yfinance`, normalize to diffusion coefficients
   - `WeatherFetcher`: Fetch via `openmeteo_requests`, normalize

3. **`src/sde/solvers.py`**
   - `euler_maruyama_ou`: Implement the solver per algorithm in IMPLEMENTATION_PLAN.md
   - `apply_sde_to_points`: Apply OU process to point cloud
   - `get_trajectory`: Return full (T, N, 3) trajectory
   - `create_sphere`, `create_cube`, `create_torus`: Geometry generators

4. **`src/viz/kintsugi.py`**
   - `render_kintsugi`: Gold overlay on high-variance pixels
   - `create_kintsugi_grid`: Comparison grid (original | recon | variance | kintsugi)
   - `add_occlusion`, `add_gaussian_noise`, `rotate_image`, `mix_digits`: OOD perturbations

5. **`src/viz/weathering.py`**
   - `color_by_displacement`: Color points by movement magnitude
   - `save_point_cloud`: Export to PLY
   - `visualize_point_cloud`, `create_weathering_animation`: Open3D rendering (handle headless with try/except)

### MUST Update Notebooks

Update the three notebooks to be runnable:

1. **`notebooks/01_vae_training.ipynb`**
   - Add `SMOKE_TEST = True` flag at top
   - When `SMOKE_TEST=True`: train 2 epochs on 1000 samples only
   - When `SMOKE_TEST=False`: train 30 epochs on full MNIST
   - Save checkpoint to `results/vae_mnist.pt`

2. **`notebooks/02_kintsugi_demo.ipynb`**
   - Load checkpoint (or train if missing)
   - Apply each perturbation type to 4 test images
   - Generate and save `results/kintsugi_gallery.png`

3. **`notebooks/03_sde_weathering.ipynb`**
   - Fetch VIX data (use smaller date range for smoke test: 1 year instead of 10)
   - Create sphere, apply weathering
   - Save `results/weathered_sphere.ply`
   - Attempt animation export (gracefully skip if headless)

## Constraints

- **NO GPU available** — all code must run on CPU
- **Time limit** — keep smoke tests under 5 minutes total
- **No new dependencies** — use only what's in `requirements.txt`
- **Headless environment** — Open3D visualization may fail; wrap in try/except and fall back to saving files only

## Verification Checklist

After implementation, run these commands to verify:

```bash
# 1. Imports work
python -c "from src.models.vae import KintsugiVAE, vae_loss"
python -c "from src.noise.data_sources import VolatilityFetcher, WeatherFetcher"
python -c "from src.sde.solvers import euler_maruyama_ou, create_sphere"
python -c "from src.viz.kintsugi import render_kintsugi"

# 2. Unit tests pass
python -m pytest tests/test_sde.py -v

# 3. Quick integration test
python -c "
import torch
from src.models.vae import KintsugiVAE, vae_loss

model = KintsugiVAE(z_dim=20)
x = torch.rand(4, 1, 28, 28)
recon, mu, log_var = model(x)
assert recon.shape == x.shape, 'VAE forward pass failed'
print('VAE OK')

mean_recon, var = model.sample_with_uncertainty(x, n_samples=5)
assert var.shape == x.shape, 'MC sampling failed'
print('MC Dropout OK')
"

# 4. SDE integration test
python -c "
import numpy as np
from src.sde.solvers import euler_maruyama_ou, create_sphere

sphere = create_sphere(n_points=100)
sigma_t = np.ones(50) * 0.01
result = euler_maruyama_ou(sphere, theta=0.1, mu=np.zeros(3), sigma_t=sigma_t, seed=42)
assert result.shape == (50, 100, 3), 'SDE solver failed'
print('SDE OK')
"
```

## Do NOT

- Run full 30-epoch training (too slow without GPU)
- Commit large files (models, data) — they are gitignored
- Add interactive visualizations that block execution
- Modify `PLAN.md`, `IMPLEMENTATION_PLAN.md`, or `AGENTS.md`

## Success Criteria

1. All imports succeed without error
2. `pytest tests/test_sde.py` passes
3. Integration tests above pass
4. Notebooks run end-to-end in smoke-test mode
5. At least one artifact exists in `results/` (e.g., `kintsugi_gallery.png` or `weathered_sphere.ply`)

## Reference

Read `IMPLEMENTATION_PLAN.md` for:
- Exact layer dimensions for VAE
- Euler-Maruyama algorithm pseudocode
- Function signatures and expected shapes
- API usage for yfinance and openmeteo
