# Continuous-Discrete Projection Smoother — numerical validation

Numerical companion of the paper

> Md Shamim Akhter and Muhammad Fuady Emzir,
> *Continuous-Discrete Projection Smoothing on Exponential Family Manifolds*
> (draft; King Fahd University of Petroleum and Minerals).

Univariate linear-Gaussian (Ornstein–Uhlenbeck) validation of the proposed
continuous-discrete projection smoother against

* the **exact** continuous-discrete Kalman filter and Rauch–Tung–Striebel
  (RTS) smoother computed on the exactly discretized system, and
* the projection smoother of **Koyama** [Signal Processing 144 (2018)
  333–340, doi:10.1016/j.sigpro.2017.10.028], in its Gaussian moment form
  (Eqs. (30)–(31)) and its generic natural-parameter form (Eq. (20)),

with particular attention to **sparse measurement schedules** (measurement
interval up to five mean-reversion times).

## Layout

```
numerical_implementation.py     entry point: runs E1 + E2 + figure generation
cd_smoother/
  models.py                     OU model (24), exact discretization, natural-parameter maps, L(y) (Def. 1)
  kalman.py                     exact CD Kalman filter, RTS smoother, dense-grid reference
  projection_filter.py          projection filter, Eq. (22) + conjugate update (Prop. 1); Alg. 1 step 1
  projection_smoother.py        backward pass, Eq. (23)/(26) for theta_s (Alg. 1 step 2); tilting-form cross-check
  koyama_smoother.py            Koyama (2018) smoother; restarted variant and KGPS (method (ii))
  hdf_io.py                     HDF5 output utilities
  plotting.py                   journal-figure rcParams and palette
experiments/
  exp1_correctness.py           E1: single-realization correctness study
  exp2_interval_sweep.py        E2: Monte Carlo sweep over the measurement interval
  plot_results.py               figures (PDF, 252 pt column width) from the HDF5 outputs
tests/                          pytest suite (run before the experiments)
```

## Installation

Requires Python ≥ 3.11 (tested with 3.12, jax/jaxlib 0.10.2). With
[uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12 .venv
uv pip install -p .venv/bin/python -e ".[dev]"
```

## Reproducing the results

```bash
.venv/bin/python -m pytest          # unit tests (exactness gates)
.venv/bin/python numerical_implementation.py
```

Outputs: HDF5 archives under `simulation_result/<CASE>_<timestamp>/variables.hdf`
and figures (PDF) in `../Journal_Paper/Template_L_CSS/figures/` (adjust
`FIG_DIR` in `experiments/plot_results.py` when using this repository
standalone).  The figures show the projection filter, the proposed smoother
(Algorithm 1) and Koyama's Gaussian projection smoother (KGPS); the restarted Koyama
smoother is computed and stored in the HDF5 files but not plotted.

Equation numbers quoted in the docstrings refer to the compiled
`Journal_Paper/Template_L_CSS/LCSS_Latex_template.tex`; the number ↔ label
map is in the docstring of `cd_smoother/__init__.py`.  Every function carries
a numpy-style docstring with [jaxtyping](https://github.com/patrick-kidger/jaxtyping)
annotations; they can be checked at runtime with
`pip install beartype && pytest --jaxtyping-packages=cd_smoother,beartype.beartype`.

All experiments are seeded and deterministic (JAX in float64).


## Key numerical facts checked by the test suite

* The natural-parameter projection filter reproduces the exact
  continuous-discrete Kalman filter (linear-Gaussian exactness).
* The proposed backward pass (Algorithm 1: Eq. (23) for theta_s, integrated
  against the stored dense-output theta_p(t), with theta_s continuous across
  the measurement times by Prop. 1) reproduces the exact dense RTS smoothing
  moments to < 1e-6, and agrees with the closed-form tilting form (the ODE
  Eq. (26) minus Eq. (25) for lambda = theta_s - theta_p, with the jump
  lambda += L(y_k) at every measurement time, including t_N).
* Koyama's Gaussian form and natural-parameter form agree with each other and
  with the RTS reference when the backward integration is restarted at the
  discontinuities of the stored filter trajectory; integrating blindly across
  those discontinuities degrades the accuracy by more than an order of
  magnitude.

## License

GPL-3.0-or-later (see `pyproject.toml`).
