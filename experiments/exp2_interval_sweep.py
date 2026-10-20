"""Experiment E2 — Monte Carlo sweep over the measurement interval (paper, Sec. V-C).

For measurement intervals Delta in {0.5, 1, 2, 5} (in units of the
mean-reversion time 1/|a| = 1) and M = 100 independent realizations each,
compares:

* RMSE against the true trajectory (dense grid) for the forward filter, the
  proposed projection smoother (Algorithm 1), Koyama's smoother with
  restarts, and KGPS (method (ii) of Sec. V-A) by fixed-step RK4 and by
  adaptive Dormand--Prince on the forward pass's dense output;
* the maximum absolute deviation of the smoothing moments from the exact
  dense Kalman/RTS reference, for the four smoother implementations.

All output arrays are saved to HDF5.
"""
import jax

jax.config.update("jax_enable_x64", True)

import pathlib
import sys
from datetime import datetime

import numpy as onp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from cd_smoother.hdf_io import save_to_hdf
from cd_smoother.kalman import dense_kalman_rts
from cd_smoother.models import LinearSDE
from cd_smoother.koyama_smoother import koyama_gaussian_smoother, koyama_gaussian_smoother_dopri
from cd_smoother.projection_filter import projection_filter
from cd_smoother.projection_smoother import projection_smoother

from exp1_correctness import interval_to_flat, moments_grid, simulate_truth

CASE = "E2_INTERVAL_SWEEP"

# model and experiment parameters (paper, Sec. V-A and V-C)
A_DRIFT = -1.0
SIGMA = 1.0
R_MEAS = 0.25
M0, P0 = 0.0, 0.5
DELTAS = onp.array([0.5, 1.0, 2.0, 5.0])
N_MEAS = 20
N_DENSE = 21
N_SEEDS = 100
N_NAIVE_STEPS = 512


def run() -> pathlib.Path:
    """Run experiment E2 and save all variables to HDF5.

    Returns
    -------
    out_dir : pathlib.Path
        Directory containing ``variables.hdf``.
    """
    model = LinearSDE(a=A_DRIFT, sigma=SIGMA, R=R_MEAS)
    methods = ["filter", "proj", "koyama", "naive", "dopri"]
    rmse = {m: onp.empty((len(DELTAS), N_SEEDS)) for m in methods}
    dev_m = {m: onp.empty((len(DELTAS), N_SEEDS)) for m in ["proj", "koyama", "naive", "dopri"]}
    dev_P = {m: onp.empty((len(DELTAS), N_SEEDS)) for m in ["proj", "koyama", "naive", "dopri"]}

    flat = lambda arr: interval_to_flat(arr, N_DENSE)
    for i, delta in enumerate(DELTAS):
        t_meas = delta * onp.arange(1, N_MEAS + 1)
        for s in range(N_SEEDS):
            rng = onp.random.default_rng(1000 * i + s)
            grid, meas_idx, _, _, _, _ = dense_kalman_rts(
                model, t_meas, onp.zeros(N_MEAS), M0, P0, n_dense=N_DENSE
            )
            x_true = simulate_truth(model, grid, M0, P0, rng)
            ys = x_true[meas_idx] + onp.sqrt(R_MEAS) * rng.standard_normal(N_MEAS)

            grid, meas_idx, m_rts, P_rts, _, _ = dense_kalman_rts(model, t_meas, ys, M0, P0, n_dense=N_DENSE)
            pf = projection_filter(model, t_meas, ys, M0, P0, n_dense=N_DENSE)
            ps = projection_smoother(model, pf)
            ks = koyama_gaussian_smoother(model, pf, restarts=True)
            kn = koyama_gaussian_smoother(model, pf, restarts=False, n_steps_naive=N_NAIVE_STEPS)
            kd = koyama_gaussian_smoother_dopri(model, pf)  # KGPS with adaptive Dormand--Prince

            m_proj, _P_proj = moments_grid(ps.theta_s)
            m_pf, _ = moments_grid(pf.theta_p)
            m_proj_f, P_proj_f = flat(m_proj), flat(_P_proj)
            m_pf_f = flat(m_pf)
            m_ks_f, P_ks_f = flat(ks.m), flat(ks.V)
            m_kn_f, P_kn_f = flat(kn.m), flat(kn.V)
            m_kd_f, P_kd_f = flat(kd.m), flat(kd.V)

            rmse["filter"][i, s] = onp.sqrt(onp.mean((m_pf_f - x_true) ** 2))
            rmse["proj"][i, s] = onp.sqrt(onp.mean((m_proj_f - x_true) ** 2))
            rmse["koyama"][i, s] = onp.sqrt(onp.mean((m_ks_f - x_true) ** 2))
            rmse["naive"][i, s] = onp.sqrt(onp.mean((m_kn_f - x_true) ** 2))
            rmse["dopri"][i, s] = onp.sqrt(onp.mean((m_kd_f - x_true) ** 2))

            for name, (mm, PP) in {
                "proj": (m_proj_f, P_proj_f),
                "koyama": (m_ks_f, P_ks_f),
                "naive": (m_kn_f, P_kn_f),
                "dopri": (m_kd_f, P_kd_f),
            }.items():
                dev_m[name][i, s] = onp.max(onp.abs(mm - m_rts))
                dev_P[name][i, s] = onp.max(onp.abs(PP - P_rts))
        print(f"[E2] Delta = {delta}: done ({N_SEEDS} seeds)")

    out_dir = pathlib.Path(__file__).resolve().parents[1] / "simulation_result" / (
        f"{CASE}_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    save_to_hdf(
        out_dir / "variables.hdf",
        {
            "deltas": DELTAS,
            "rmse_filter": rmse["filter"],
            "rmse_proj": rmse["proj"],
            "rmse_koyama": rmse["koyama"],
            "rmse_naive": rmse["naive"],
            "rmse_dopri": rmse["dopri"],
            "dev_m_proj": dev_m["proj"],
            "dev_m_koyama": dev_m["koyama"],
            "dev_m_naive": dev_m["naive"],
            "dev_m_dopri": dev_m["dopri"],
            "dev_P_proj": dev_P["proj"],
            "dev_P_koyama": dev_P["koyama"],
            "dev_P_naive": dev_P["naive"],
            "dev_P_dopri": dev_P["dopri"],
            "n_meas": N_MEAS,
            "n_seeds": N_SEEDS,
            "a_drift": A_DRIFT,
            "sigma": SIGMA,
            "R_meas": R_MEAS,
            "m0": M0,
            "P0": P0,
            "n_naive_steps": N_NAIVE_STEPS,
        },
    )
    print(f"[E2] saved to {out_dir}")
    for name in ("proj", "koyama", "naive", "dopri"):
        worst = onp.maximum(dev_m[name], dev_P[name])
        print(f"     {name:8s} median max-dev per Delta: " + "  ".join(f"{v:.2e}" for v in onp.median(worst, axis=1)))
    return out_dir


if __name__ == "__main__":
    run()
