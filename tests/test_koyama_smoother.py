"""Tests for the Koyama (2018) projection smoother implementation.

Koyama's Gaussian projection smoother [Eqs. (30)-(31)] reduces, for linear
dynamics, to the backward moment ODEs
    d mu/dt = a mu + (sigma^2 / V^f) (mu - mu^f),
    d V /dt = 2 a V - sigma^2 + (2 sigma^2 / V^f) V,
integrated backward from the filtered moments at t_N against the stored
forward trajectory (mu^f, V^f).  For linear-Gaussian systems this coincides
with the Type-I / RTS smoother [Koyama 2018, Sec. 2.3.1].

Tests:
1. koyama_gaussian_smoother (with per-interval restarts at the measurement
   discontinuities of the stored filter trajectory) matches the RTS smoother
   at measurement times.
2. The generic natural-parameter form [Eq. (20)] gives the same trajectory
   as the (mu, V) form.
3. KGPS (method (ii) of Sec. V-A: no restarts, fixed-step RK4) incurs a
   substantially larger error for large measurement intervals.
"""
import jax

jax.config.update("jax_enable_x64", True)

import numpy as onp
import numpy.testing as npt
import pytest

from cd_smoother.models import LinearSDE, moments_from_theta
from cd_smoother.kalman import kalman_filter, rts_smoother
from cd_smoother.projection_filter import PFOutput, projection_filter
from cd_smoother.koyama_smoother import KoyamaOutput, koyama_gaussian_smoother, koyama_theta_smoother

Setup = tuple[LinearSDE, float, float, onp.ndarray, onp.ndarray, PFOutput]


@pytest.fixture(scope="module")
def setup() -> Setup:
    """Six measurements with a sparse interval Delta = 2 and the forward pass.

    Returns
    -------
    model : LinearSDE
        OU model (24) with a = -1, sigma = 1, R = 0.25.
    m0, P0 : float
        Prior moments.
    t_meas : Float[onp.ndarray, " N"]
        Measurement times.
    ys : Float[onp.ndarray, " N"]
        Measurements.
    pf : PFOutput
        Projection-filter output with 41 stored points per interval.
    """
    rng = onp.random.default_rng(5)
    model = LinearSDE(a=-1.0, sigma=1.0, R=0.25)
    m0, P0 = 0.2, 0.5
    dt = 2.0
    N = 6
    t_meas = dt * onp.arange(1, N + 1)
    ys = rng.normal(size=N)
    pf = projection_filter(model, t_meas, ys, m0, P0, n_dense=41)
    return model, m0, P0, t_meas, ys, pf


def test_koyama_gaussian_matches_rts(setup: Setup) -> None:
    """Koyama's restarted Gaussian form reproduces the RTS smoother at t_k.

    Parameters
    ----------
    setup : Setup
        Module fixture.
    """
    model, m0, P0, t_meas, ys, pf = setup
    kf = kalman_filter(model, t_meas, ys, m0, P0)
    m_rts, P_rts = rts_smoother(model, t_meas, kf)
    ks = koyama_gaussian_smoother(model, pf, restarts=True)
    for k in range(len(t_meas)):
        npt.assert_allclose([ks.m[k, -1], ks.V[k, -1]], [m_rts[k], P_rts[k]], rtol=1e-6, atol=1e-9)


def test_koyama_theta_form_matches_gaussian_form(setup: Setup) -> None:
    """Koyama's natural-parameter form [Eq. (20)] equals the Gaussian form [Eqs. (30)-(31)].

    Parameters
    ----------
    setup : Setup
        Module fixture.
    """
    model, m0, P0, t_meas, ys, pf = setup
    ks = koyama_gaussian_smoother(model, pf, restarts=True)
    kt = koyama_theta_smoother(model, pf)
    for k in range(len(t_meas)):
        for j in (0, ks.m.shape[1] // 2, ks.m.shape[1] - 1):
            m_t, V_t = moments_from_theta(kt.theta[k, j])
            npt.assert_allclose([m_t, V_t], [ks.m[k, j], ks.V[k, j]], rtol=1e-6, atol=1e-8)


def test_naive_no_restart_is_worse_for_large_intervals(setup: Setup) -> None:
    """KGPS (no restarts) is > 10x less accurate than the restarted integration.

    Parameters
    ----------
    setup : Setup
        Module fixture.
    """
    model, m0, P0, t_meas, ys, pf = setup
    kf = kalman_filter(model, t_meas, ys, m0, P0)
    m_rts, P_rts = rts_smoother(model, t_meas, kf)

    ks_good = koyama_gaussian_smoother(model, pf, restarts=True)
    ks_naive = koyama_gaussian_smoother(model, pf, restarts=False, n_steps_naive=200)

    def err(ks: KoyamaOutput) -> float:
        """Worst moment error at the interior measurement times.

        Parameters
        ----------
        ks : KoyamaOutput
            Smoother output.

        Returns
        -------
        float
            max_k max(|m_k - m_k^RTS|, |V_k - P_k^RTS|), k = 1, ..., N - 1.
        """
        e = 0.0
        for k in range(len(t_meas) - 1):  # interior measurement times
            e = max(e, abs(ks.m[k, -1] - m_rts[k]), abs(ks.V[k, -1] - P_rts[k]))
        return e

    assert err(ks_naive) > 10.0 * err(ks_good)
