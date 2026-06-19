# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

pyMAISE is a Python library for benchmarking ML models on nuclear engineering applications. It wraps scikit-learn, PyTorch (via skorch), and optuna-based hyperparameter tuning into a unified workflow.

**Neural network backend**: PyTorch via skorch is the primary going-forward direction. TF/Keras references in older docs are legacy.

## Setup

```bash
pip install .[dev]
```

Python 3.9–3.11 required. Python 3.12 is **not** supported despite appearing in classifiers.

## Commands

**Run tests (CI-equivalent — skips dataset downloads):**
```bash
cd tests && pytest -m "not datasets"
```

**Run a single test:**
```bash
cd tests && pytest -k "test_name"
```

**Run dataset tests (requires Zenodo network access):**
```bash
cd tests && pytest -m "datasets"
```

**Lint (errors only):**
```bash
flake8 . --count --select=E9,F63,F7,F82
```

**Format:**
```bash
black .
```

**Build docs:**
```bash
cd docs && make html
```

## Test Structure

- `tests/unit/` — unit tests (run without network access)
- `tests/regression/` — regression tests for classical and NN models
- Tests marked `@pytest.mark.datasets` download from Zenodo; CI skips them by default

## Code Style

- Black, 88-character line length (flake8 max-line-length also 88)
- `__init__.py` files: F401 (unused import) is ignored — public re-exports are intentional
- Pre-commit: `pre-commit install` to enable Black + flake8 hooks

## Known Gotchas

- **numpy compatibility**: `tuner.py` patches `np.int = int` for numpy ≥ 1.24. Don't remove this patch.
- **scikit-optimize 0.9.0 is pinned**: do not upgrade — newer versions break numpy/scipy compatibility.
- **Dataset loading uses pooch**: benchmark datasets are fetched from Zenodo on first use and cached locally.

## Architecture

- `pyMAISE/tuner.py` — model fitting and hyperparameter search orchestrator
- `pyMAISE/postprocessor.py` — model evaluation, comparison, and plotting
- `pyMAISE/methods/` — wrappers for 15 classical models + PyTorch NN layers
- `pyMAISE/explain/` — SHAP explainability methods through Captum package (DeepLiftShap, GradientShap, KernelShap, ShapleyValues), also includes explainability plotting functions
- `pyMAISE/datasets/` — benchmark dataset loaders (pooch/Zenodo)
- `docs/source/benchmarks/` — 9 Jupyter notebooks (MIT Reactor, Reactor Physics, BWR, etc.) Stored hyperparameter tuning results from these notebooks are on Zenodo and loaded unless the user sets RETUNE=True or files are unaccessible.
