# Computational Wabi-Sabi: Progress Report

## Overview

This document summarizes the implementation progress and key findings from developing the Computational Wabi-Sabi project, which explores uncertainty visualization as aesthetic intervention.

---

## Module II: Bayesian Kintsugi

### 2D Implementation (MNIST) — ✅ Working

**Architecture:** Convolutional VAE with MC Dropout

**Results:**
- Gold overlay correctly localizes to high-uncertainty regions
- Perturbations (occlusion, noise, rotation) create distinct uncertainty patterns
- Fix applied: percentile threshold computed within foreground only (not whole image)

**Key files:**
- `src/models/vae.py` — KintsugiVAE with MC Dropout
- `src/viz/kintsugi.py` — Gold overlay renderer
- `results/kintsugi_detail_v3.png` — Canonical demo

**Artifacts:**
- `results/vae_mnist_full.pt` — Trained model (30 epochs, loss: 108.58)
- `results/kintsugi_gallery_v3.png` — 4×4 perturbation matrix

---

### 3D Implementation — Partially Working

#### Attempt 1: Point Cloud VAE (PointNetVAE)

**Problem:** No point correspondence. Encoder compresses to global latent, decoder generates NEW points. Variance measures "where decoder places points," not "uncertainty about specific input vertex."

**Finding:** Uncertainty is uniform across output — cannot localize to damaged region.

#### Attempt 2: Coordinate-Conditioned VAE (SpatialVAE)

**Architecture:** Decoder takes (global_z, input_coordinate) → output_coordinate

**Finding:**
- **Dents (inward):** Ratio 0.6-0.8x — LOWER uncertainty at dent (wrong direction)
- **Bulges (outward):** Ratio 1.3-1.8x — HIGHER uncertainty (correct direction)

**Insight:** Model trained on spheres r∈[0.9, 1.1]. Points pushed inward are still "familiar" coordinates. Points pushed outward are OOD → uncertainty.

**Philosophical implication:** This architecture detects "excess," not "lack."

#### Attempt 3: Mesh-Based (LocalMeshDenoiser)

**Architecture:**
- Fixed-topology icosphere (642 vertices)
- Graph convolutions for local feature aggregation
- Per-vertex offset prediction toward template
- No global latent — purely local reasoning

**Results:**
- Ratio 1.52x (dent region vs intact region)
- Direction is correct: damaged vertices have higher uncertainty
- However: Gold not precisely localized to dent region in visualization

**Limitation:** Absolute uncertainty values very small (~1e-6) due to model converging well. Relative difference exists but visualization is subtle.

**Key files:**
- `src/models/mesh_vae.py` — MeshVAE and LocalMeshDenoiser
- `src/models/point_deform.py` — Point deformation experiments
- `src/models/pointnet_vae.py` — PointNet VAE experiments

---

## Module I: SDE Weathering — ✅ Working

**Architecture:** Ornstein-Uhlenbeck process driven by external data

**Implementation:**
- Euler-Maruyama solver for SDE integration
- Data fetchers for VIX (yfinance) and weather (OpenMeteo)
- Geometry primitives (sphere, cube, torus)

**Results:**
- Subtle weathering preserves shape while showing erosion
- Displacement colored by magnitude (blue→red)
- Volatility spikes in time series → localized displacement

**Key files:**
- `src/sde/solvers.py` — Euler-Maruyama, geometry primitives
- `src/noise/data_sources.py` — VolatilityFetcher, WeatherFetcher
- `src/viz/weathering.py` — Open3D visualization

**Artifacts:**
- `results/weathering_v2.png` — Sphere weathering demo
- `results/sphere_weathered_v2.ply` — 3D point cloud

---

## Key Insights

### Why 2D Kintsugi Works

1. **Pixel correspondence:** Conv layers maintain spatial structure
2. **Local reconstruction:** Each pixel predicted from local features + global latent
3. **Clear OOD:** Occluded pixels have no information → model must guess → variance

### Why 3D Kintsugi is Hard

1. **No natural correspondence:** Point clouds are unordered sets
2. **Global collapse:** PointNet encodes to single vector, loses spatial specificity
3. **Coordinate familiarity:** Inward dents produce coordinates model has "seen" (interior of sphere)
4. **Reconstruction dominance:** Models converge too well → minimal dropout variance

### What Partially Works for 3D

1. **Bulges/protrusions:** Push points outside training distribution → uncertainty
2. **Mesh with local reasoning:** Fixed topology + no global latent → per-vertex differentiation
3. **Ratio is correct (1.52x):** Damaged region has higher uncertainty, just not dramatically so

---

## Future Directions

### To improve 3D Kintsugi:

1. **Ensemble methods** instead of MC Dropout — more variance signal
2. **Evidential deep learning** — explicit uncertainty quantification
3. **Contrastive training** — learn "clean vs damaged" explicitly
4. **Higher resolution meshes** — more vertices for localization
5. **Graph attention** — learn which neighbors matter for each vertex

### To unify Modules I + II:

1. Weather a mesh with SDE
2. Pass weathered mesh through MeshVAE
3. Gold appears where weathering created OOD geometry

---

## Repository Structure

```
compwabi/
├── src/
│   ├── models/
│   │   ├── vae.py              # 2D KintsugiVAE (working)
│   │   ├── pointnet_vae.py     # 3D PointNetVAE (partial)
│   │   ├── point_deform.py     # Point deformation experiments
│   │   └── mesh_vae.py         # Mesh-based VAE (best 3D)
│   ├── sde/
│   │   └── solvers.py          # Euler-Maruyama, geometries
│   ├── noise/
│   │   └── data_sources.py     # VIX, weather fetchers
│   └── viz/
│       ├── kintsugi.py         # 2D gold overlay
│       ├── kintsugi_3d.py      # 3D gold overlay
│       └── weathering.py       # Open3D visualization
├── results/                    # Generated artifacts
├── PLAN.md                     # Original philosophical plan
├── IMPLEMENTATION_PLAN.md      # Technical specification
└── PROGRESS.md                 # This document
```

---

## Conclusion

**Module II (2D):** Fully working. Demonstrates core thesis: epistemic uncertainty → aesthetic "scars."

**Module II (3D):** Partially working. Mesh-based approach shows correct direction (1.52x ratio) but localization not precise. Fundamental challenges with point cloud architectures documented.

**Module I:** Fully working. SDE weathering produces compelling "aged" geometries.

The project successfully demonstrates that computational uncertainty can be reframed as aesthetic intervention, though 3D kintsugi requires further architectural innovation for precise damage localization.
