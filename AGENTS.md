# Agent Instructions

This file provides instructions for AI coding agents (Codex, Claude, etc.) working on this repository.

## Primary Reference

**Read `IMPLEMENTATION_PLAN.md` first.** It contains:
- Exact directory structure to create
- Class and function signatures with type hints
- Algorithm pseudocode
- Implementation checklist

## Task

Implement the project according to `IMPLEMENTATION_PLAN.md`, following this order:

1. **Phase 0:** Create directory structure and `__init__.py` files
2. **Phase 1:** Implement Bayesian Kintsugi (VAE + uncertainty visualization)
3. **Phase 2:** Implement SDE Weathering (data fetchers + Euler-Maruyama solver)

## Constraints

- Python 3.9+ compatibility
- Use only dependencies listed in `requirements.txt`
- Follow existing code style (type hints, docstrings)
- Create working Jupyter notebooks that produce artifacts in `results/`

## Implementation Notes

### VAE Training
- Train on MNIST for 30 epochs
- Save checkpoint to `results/vae_mnist.pt`
- Ensure dropout layers remain active during MC sampling (call `model.train()` or set dropout layers explicitly)

### SDE Solver
- Normalize external data (VIX, weather) to appropriate scale (~0.001-0.05) for point displacement
- Use `np.random.default_rng(seed)` for reproducibility

### Visualization
- Use Open3D for 3D point clouds (not Plotly)
- For headless/CI environments, use `OffscreenRenderer`
- Save all generated artifacts to `results/`

## Testing

After implementation, verify:
1. `python -c "from src.models.vae import KintsugiVAE"` imports without error
2. `python -c "from src.sde.solvers import euler_maruyama_ou"` imports without error
3. Notebooks run end-to-end and produce output files

## Do Not

- Add dependencies not in `requirements.txt` without justification
- Modify `PLAN.md` or `IMPLEMENTATION_PLAN.md`
- Commit large files (models, datasets) — they are gitignored
