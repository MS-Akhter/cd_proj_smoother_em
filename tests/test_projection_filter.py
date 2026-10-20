"""Tests for the exponential-family projection filter (Alg. 1, step 1; Gaussian family, OU model).

For linear dynamics and the Gaussian family the projection is exact
(zero projection residual, Rem. 2), so the natural-parameter projection
filter must reproduce the continuous-discrete Kalman filter.

Tests:
1. filter_ode_rhs: the natural-parameter prediction ODE (22)
   d(theta)/dt = g(theta)^{-1} E_theta[L(c)] agrees with the chain-rule image
   of the Kalman moment ODEs dm/dt = a m, dP/dt = 2 a P + sigma^2 (Sec. V-A).
2. projection_filter: filtered/predicted moments at measurement times match
   kalman_filter to ODE-solver accuracy.
3. The stored dense prediction trajectory starts at the post-update parameter
   and ends at the pre-update parameter of each interval.
4. The dense-output interpolant of the stored theta_p(t), which the backward
   pass of Alg. 1 evaluates, agrees with the sampled grid values.
"""
import jax

jax.config.update("jax_enable_x64", True)

import numpy as onp
import numpy.testing as npt
import pytest

from cd_smoother.models import LinearSDE, moments_from_theta, theta_from_moments
from cd_smoother.kalman import kalman_filter
from cd_smoother.projection_filter import filter_ode_rhs, projection_filter

Setup = tuple[LinearSDE, float, float, onp.ndarray, onp.ndarray]


@pytest.fixture(scope="module")
def setup() -> Setup:
    """Six measurements with a sparse interval Delta = 2.

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
    """
    rng = onp.random.default_rng(1)
    model = LinearSDE(a=-1.0, sigma=1.0, R=0.25)
    m0, P0 = 0.2, 0.5
    dt = 2.0
    N = 6
    t_meas = dt * onp.arange(1, N + 1)
    ys = rng.normal(size=N)
    return model, m0, P0, t_meas, ys


def test_filter_ode_rhs_matches_moment_odes() -> None:
    """Eq. (22) is the chain-rule image of dm/dt = a m, dP/dt = 2 a P + sigma^2."""
    model = LinearSDE(a=-0.6, sigma=1.2, R=0.1)
    rng = onp.random.default_rng(2)
    for _ in range(5):
        m, P = rng.normal(), 0.2 + rng.random()
        theta = onp.asarray(theta_from_moments(m, P))
        dm = model.a * m
        dP = 2.0 * model.a * P + model.sigma**2
        # chain rule: theta1 = m/P, theta2 = -1/(2P)
        dtheta_ref = onp.array([dm / P - m * dP / P**2, dP / (2.0 * P**2)])
        npt.assert_allclose(onp.asarray(filter_ode_rhs(theta, model)), dtheta_ref, rtol=1e-11)


def test_projection_filter_matches_kalman(setup: Setup) -> None:
    """Predicted/filtered moments at t_k equal the exact Kalman filter (Rem. 2).

    Parameters
    ----------
    setup : Setup
        Module fixture.
    """
    model, m0, P0, t_meas, ys = setup
    kf = kalman_filter(model, t_meas, ys, m0, P0)
    pf = projection_filter(model, t_meas, ys, m0, P0, n_dense=41)
    for k in range(len(t_meas)):
        m_pk, P_pk = moments_from_theta(pf.theta_p[k, -1])   # end of interval k = pre-update
        npt.assert_allclose([m_pk, P_pk], [kf.m_p[k], kf.P_p[k]], rtol=1e-8)
        m_fk, P_fk = moments_from_theta(pf.theta_f[k + 1])   # post-update at t_k
        npt.assert_allclose([m_fk, P_fk], [kf.m_f[k], kf.P_f[k]], rtol=1e-8)


def test_projection_filter_interval_endpoints(setup: Setup) -> None:
    """Interval k starts at theta_f(t_{k-1}) = theta_p(t_{k-1}^+) (Alg. 1, step 1).

    Parameters
    ----------
    setup : Setup
        Module fixture.
    """
    model, m0, P0, t_meas, ys = setup
    pf = projection_filter(model, t_meas, ys, m0, P0, n_dense=11)
    # interval k starts at the post-update parameter of t_{k-1}
    npt.assert_allclose(pf.theta_p[0, 0], pf.theta_f[0], rtol=1e-12)
    for k in range(1, len(t_meas)):
        npt.assert_allclose(pf.theta_p[k, 0], pf.theta_f[k], rtol=1e-12)
    # grid endpoints
    npt.assert_allclose(pf.ts[0, 0], 0.0, atol=1e-15)
    npt.assert_allclose(pf.ts[:, -1], t_meas, rtol=1e-15)


def test_dense_output_matches_stored_grid(setup: Setup) -> None:
    """The dense interpolant of theta_p(t) used by Alg. 1, step 2 matches the stored grid.

    Parameters
    ----------
    setup : Setup
        Module fixture.
    """
    model, m0, P0, t_meas, ys = setup
    pf = projection_filter(model, t_meas, ys, m0, P0, n_dense=11)
    assert len(pf.dense) == len(t_meas)
    for k, sol in enumerate(pf.dense):
        vals = onp.stack([onp.asarray(sol.evaluate(t)) for t in pf.ts[k]])
        npt.assert_allclose(vals, pf.theta_p[k], rtol=1e-12, atol=1e-12)
