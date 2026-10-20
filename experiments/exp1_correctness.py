"""Experiment E1 — correctness of the projection smoother (paper, Sec. V-B).

Univariate OU model (24) with sparse linear-Gaussian measurements
(measurement interval Delta = 2 mean-reversion times, Sec. V-A).  The
proposed projection smoother (method (i) of Sec. V-A: Algorithm 1, i.e.
Eq. (26) for theta_s, continuous across the measurement times by Prop. 1),
its closed-form tilting version (cross-check), Koyama's Gaussian projection
smoother (27) with restarts at the anchor discontinuities and Koyama's
generic natural-parameter form (both computed but not shown in the paper),
and Koyama's Gaussian projection smoother integrated across the measurement
times (KGPS, method (ii) of Sec. V-A; by fixed-step RK4 as in the paper and,
for comparison, by adaptive Dormand--Prince with rtol 1e-11, atol 1e-13,
reading the forward pass's dense output as Algorithm 1 does) are
compared
against the exact dense-grid Kalman/RTS reference.

All simulation variables are saved to HDF5 in
``simulation_result/<CASE>_<timestamp>/variables.hdf``.
"""
import jax

jax.config.update("jax_enable_x64", True)

import pathlib
import sys
from datetime import datetime

import numpy as onp
from jaxtyping import Float

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from cd_smoother.hdf_io import save_to_hdf
from cd_smoother.kalman import dense_kalman_rts, kalman_filter, rts_smoother
from cd_smoother.models import LinearSDE, moments_from_theta, transition_params
from cd_smoother.koyama_smoother import koyama_gaussian_smoother, koyama_gaussian_smoother_dopri, koyama_theta_smoother
from cd_smoother.projection_filter import projection_filter
from cd_smoother.projection_smoother import projection_smoother, tilting_smoother

CASE = "E1_CORRECTNESS"

# model and experiment parameters (paper, Sec. V-A)
A_DRIFT = -1.0
SIGMA = 1.0
R_MEAS = 0.25
M0, P0 = 0.0, 0.5
DELTA = 2.0
N_MEAS = 20
N_DENSE = 21
SEED = 42
N_NAIVE_STEPS = 512


def simulate_truth(
    model: LinearSDE, grid: Float[onp.ndarray, " n_grid"], m0: float, P0: float, rng: onp.random.Generator
) -> Float[onp.ndarray, " n_grid"]:
    """Sample a trajectory of the OU model (24) on `grid` via the exact discretization.

    Parameters
    ----------
    model : LinearSDE
        Model parameters.
    grid : Float[onp.ndarray, " n_grid"]
        Increasing time grid; the state at grid[0] is drawn from the prior.
    m0, P0 : float
        Prior mean and variance, x_{t_0} ~ N(m0, P0).
    rng : onp.random.Generator
        Random number generator.

    Returns
    -------
    x : Float[onp.ndarray, " n_grid"]
        Sampled true state on `grid`.
    """
    x = onp.empty(len(grid))
    x[0] = m0 + onp.sqrt(P0) * rng.standard_normal()
    for i in range(1, len(grid)):
        A, Q = transition_params(model, grid[i] - grid[i - 1])
        x[i] = float(A) * x[i - 1] + onp.sqrt(float(Q)) * rng.standard_normal()
    return x


def moments_grid(
    theta_traj: Float[onp.ndarray, "N n_dense 2"],
) -> tuple[Float[onp.ndarray, "N n_dense"], Float[onp.ndarray, "N n_dense"]]:
    """Convert an interval-stacked natural-parameter trajectory to moments.

    Parameters
    ----------
    theta_traj : Float[onp.ndarray, "N n_dense 2"]
        Natural parameters theta = (m / P, -1 / (2 P)) on the per-interval
        grids.

    Returns
    -------
    m : Float[onp.ndarray, "N n_dense"]
        Means.
    P : Float[onp.ndarray, "N n_dense"]
        Variances.
    """
    N, n_dense, _ = theta_traj.shape
    m = onp.empty((N, n_dense))
    P = onp.empty((N, n_dense))
    for k in range(N):
        for j in range(n_dense):
            m_, P_ = moments_from_theta(theta_traj[k, j])
            m[k, j], P[k, j] = float(m_), float(P_)
    return m, P


def interval_to_flat(arr_2d: Float[onp.ndarray, "N n_dense"], n_dense: int) -> Float[onp.ndarray, " n_grid"]:
    """Map an interval-stacked array onto the dense-grid layout of :func:`dense_kalman_rts`.

    Interval k contributes its points t_{k-1} < t <= t_k, so each measurement
    time takes the value at the right end of its interval (t_k^-); for the
    smoothing quantities this is immaterial by continuity (Prop. 1).

    Parameters
    ----------
    arr_2d : Float[onp.ndarray, "N n_dense"]
        Values on the per-interval grids.
    n_dense : int
        Grid points per interval (both endpoints included).

    Returns
    -------
    flat : Float[onp.ndarray, " n_grid"]
        Values on the deduplicated grid, n_grid = N (n_dense - 1) + 1.
    """
    N = arr_2d.shape[0]
    flat = onp.empty(N * (n_dense - 1) + 1)
    flat[0] = arr_2d[0, 0]
    for k in range(N):
        flat[k * (n_dense - 1) + 1 : (k + 1) * (n_dense - 1) + 1] = arr_2d[k, 1:]
    return flat


def run() -> tuple[pathlib.Path, dict[str, tuple[float, float]]]:
    """Run experiment E1 and save all variables to HDF5.

    Returns
    -------
    out_dir : pathlib.Path
        Directory containing ``variables.hdf``.
    deviations : dict[str, tuple[float, float]]
        Per method, the maximum absolute deviation of the smoothing mean and
        variance from the exact dense RTS reference over the whole grid.
    """
    model = LinearSDE(a=A_DRIFT, sigma=SIGMA, R=R_MEAS)
    rng = onp.random.default_rng(SEED)
    t_meas = DELTA * onp.arange(1, N_MEAS + 1)

    # exact dense reference grid (same layout as the smoother output)
    grid, meas_idx, _, _, _, _ = dense_kalman_rts(model, t_meas, onp.zeros(N_MEAS), M0, P0, n_dense=N_DENSE)
    x_true = simulate_truth(model, grid, M0, P0, rng)
    ys = x_true[meas_idx] + onp.sqrt(R_MEAS) * rng.standard_normal(N_MEAS)

    # exact reference
    grid, meas_idx, m_rts, P_rts, _, _ = dense_kalman_rts(model, t_meas, ys, M0, P0, n_dense=N_DENSE)
    kf = kalman_filter(model, t_meas, ys, M0, P0)
    m_rts_meas, P_rts_meas = rts_smoother(model, t_meas, kf)

    # forward projection filter (Alg. 1, step 1) and the smoothers
    pf = projection_filter(model, t_meas, ys, M0, P0, n_dense=N_DENSE)
    ps = projection_smoother(model, pf)
    pt = tilting_smoother(model, ys, pf)
    ks = koyama_gaussian_smoother(model, pf, restarts=True)
    kt = koyama_theta_smoother(model, pf)
    kn = koyama_gaussian_smoother(model, pf, restarts=False, n_steps_naive=N_NAIVE_STEPS)
    kd = koyama_gaussian_smoother_dopri(model, pf)  # KGPS with adaptive Dormand--Prince

    m_proj, P_proj = moments_grid(ps.theta_s)
    m_tilt, P_tilt = moments_grid(pt.theta_s)
    m_pf, P_pf = moments_grid(pf.theta_p)
    # post-update filtering moments at t_1, ..., t_N: theta_f(t_k) = theta_p(t_k^-) + L(y_k) (Prop. 1)
    m_fpost, P_fpost = moments_grid(pf.theta_f[None, 1:])
    m_fpost, P_fpost = m_fpost[0], P_fpost[0]
    m_kt, P_kt = moments_grid(kt.theta)

    # flatten onto the dense grid layout
    flat = lambda arr: interval_to_flat(arr, N_DENSE)
    m_proj_f, P_proj_f = flat(m_proj), flat(P_proj)
    m_tilt_f, P_tilt_f = flat(m_tilt), flat(P_tilt)
    m_ks_f, P_ks_f = flat(ks.m), flat(ks.V)
    m_kt_f, P_kt_f = flat(m_kt), flat(P_kt)
    m_kn_f, P_kn_f = flat(kn.m), flat(kn.V)
    m_kd_f, P_kd_f = flat(kd.m), flat(kd.V)
    m_pf_f, P_pf_f = flat(m_pf), flat(P_pf)

    def dev(m: Float[onp.ndarray, " n_grid"], P: Float[onp.ndarray, " n_grid"]) -> tuple[float, float]:
        """Maximum absolute deviation of (m, P) from the exact dense RTS moments.

        Parameters
        ----------
        m, P : Float[onp.ndarray, " n_grid"]
            Smoothing mean and variance on the dense grid.

        Returns
        -------
        dev_m, dev_P : float
            max_t |m - m_rts| and max_t |P - P_rts|.
        """
        return float(onp.max(onp.abs(m - m_rts))), float(onp.max(onp.abs(P - P_rts)))

    deviations = {
        "projection_smoother": dev(m_proj_f, P_proj_f),
        "tilting_form": dev(m_tilt_f, P_tilt_f),
        "koyama_gaussian_restarts": dev(m_ks_f, P_ks_f),
        "koyama_theta_form": dev(m_kt_f, P_kt_f),
        "koyama_naive": dev(m_kn_f, P_kn_f),
        "koyama_dopri": dev(m_kd_f, P_kd_f),
    }

    out_dir = pathlib.Path(__file__).resolve().parents[1] / "simulation_result" / (
        f"{CASE}_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    save_to_hdf(
        out_dir / "variables.hdf",
        {
            "grid": grid,
            "meas_idx": meas_idx,
            "t_meas": t_meas,
            "x_true": x_true,
            "ys": ys,
            "m_rts": m_rts,
            "P_rts": P_rts,
            "m_rts_meas": m_rts_meas,
            "P_rts_meas": P_rts_meas,
            "m_filter": m_pf_f,
            "P_filter": P_pf_f,
            "m_filter_post": m_fpost,
            "P_filter_post": P_fpost,
            "m_proj": m_proj_f,
            "P_proj": P_proj_f,
            "m_tilting": m_tilt_f,
            "P_tilting": P_tilt_f,
            "m_koyama": m_ks_f,
            "P_koyama": P_ks_f,
            "m_koyama_theta": m_kt_f,
            "P_koyama_theta": P_kt_f,
            "m_koyama_naive": m_kn_f,
            "P_koyama_naive": P_kn_f,
            "m_koyama_dopri": m_kd_f,
            "P_koyama_dopri": P_kd_f,
            "dev_proj_m": deviations["projection_smoother"][0],
            "dev_proj_P": deviations["projection_smoother"][1],
            "dev_tilting_m": deviations["tilting_form"][0],
            "dev_tilting_P": deviations["tilting_form"][1],
            "dev_koyama_m": deviations["koyama_gaussian_restarts"][0],
            "dev_koyama_P": deviations["koyama_gaussian_restarts"][1],
            "dev_koyama_theta_m": deviations["koyama_theta_form"][0],
            "dev_koyama_theta_P": deviations["koyama_theta_form"][1],
            "dev_naive_m": deviations["koyama_naive"][0],
            "dev_naive_P": deviations["koyama_naive"][1],
            "dev_dopri_m": deviations["koyama_dopri"][0],
            "dev_dopri_P": deviations["koyama_dopri"][1],
            "delta": DELTA,
            "n_meas": N_MEAS,
            "seed": SEED,
            "a_drift": A_DRIFT,
            "sigma": SIGMA,
            "R_meas": R_MEAS,
            "m0": M0,
            "P0": P0,
            "n_naive_steps": N_NAIVE_STEPS,
        },
    )
    print(f"[E1] saved to {out_dir}")
    print("[E1] max |deviation| from exact dense RTS (mean, variance):")
    for name, (dm, dP) in deviations.items():
        print(f"     {name:28s} {dm:.3e}  {dP:.3e}")
    return out_dir, deviations


if __name__ == "__main__":
    run()
