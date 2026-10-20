"""Tests for the continuous-discrete projection smoother (Algorithm 1 of the paper).

In the linear-Gaussian case the projection is exact (Rem. 2), so the
projection smoother must reproduce the exact (dense) Kalman/RTS smoothing
moments.

Tests:
1. Consistency of the right-hand sides (Thm. 2): Eq. (23) at
   (theta_s, theta_p) minus Eq. (22) at theta_p (for the OU model, Eq. (26)
   minus Eq. (25)) equals the closed-form tilting ODE at
   lambda = theta_s - theta_p.
2. End-to-end: Algorithm 1 matches the RTS smoother at the measurement times
   and the dense RTS reference everywhere (tolerance 1e-6, the paper's
   correctness gate).
3. Algorithm 1 (theta_s continuous, Prop. 1) and the tilting form (lambda
   jumps by L(y_k)) give the same smoothing trajectory.
4. Structure: lambda_2 <= 0 along the whole backward pass, hence P_s <= P_p
   pointwise (theta_s stays in Theta, Sec. V-A); theta_s = theta_p + lambda of the tilting form is
   continuous across the measurement times (Prop. 1).
"""
import jax

jax.config.update("jax_enable_x64", True)

import numpy as onp
import numpy.testing as npt
import pytest

from cd_smoother.models import LinearSDE, moments_from_theta, theta_from_moments
from cd_smoother.kalman import dense_kalman_rts, kalman_filter, rts_smoother
from cd_smoother.projection_filter import PFOutput, filter_ode_rhs, projection_filter
from cd_smoother.projection_smoother import (
    lambda_ode_rhs,
    projection_smoother,
    theta_s_ode_rhs,
    tilting_smoother,
)

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
    rng = onp.random.default_rng(3)
    model = LinearSDE(a=-1.0, sigma=1.0, R=0.25)
    m0, P0 = 0.2, 0.5
    dt = 2.0  # large measurement interval
    N = 6
    t_meas = dt * onp.arange(1, N + 1)
    ys = rng.normal(size=N)
    pf = projection_filter(model, t_meas, ys, m0, P0, n_dense=41)
    return model, m0, P0, t_meas, ys, pf


def test_rhs_consistency_theorem() -> None:
    """Eq. (23) minus Eq. (22) equals the closed-form tilting ODE at lambda = theta_s - theta_p."""
    model = LinearSDE(a=-0.8, sigma=1.3, R=0.2)
    rng = onp.random.default_rng(4)
    for _ in range(8):
        m_p, P_p = rng.normal(), 0.3 + rng.random()
        theta_p = onp.asarray(theta_from_moments(m_p, P_p))
        lam = onp.array([0.5 * rng.normal(), -0.4 * rng.random()])
        theta_s = theta_p + lam
        lhs = onp.asarray(theta_s_ode_rhs(theta_s, theta_p, model)) - onp.asarray(filter_ode_rhs(theta_p, model))
        rhs = onp.asarray(lambda_ode_rhs(lam, model))
        npt.assert_allclose(lhs, rhs, rtol=1e-10, atol=1e-12)


def test_projection_smoother_matches_rts(setup: Setup) -> None:
    """Algorithm 1 reproduces the RTS smoother at the measurement times (Rem. 2).

    Parameters
    ----------
    setup : Setup
        Module fixture.
    """
    model, m0, P0, t_meas, ys, pf = setup
    ps = projection_smoother(model, pf)
    kf = kalman_filter(model, t_meas, ys, m0, P0)
    m_rts, P_rts = rts_smoother(model, t_meas, kf)
    # at measurement times: theta_s at the end of each interval
    for k in range(len(t_meas)):
        m_sk, P_sk = moments_from_theta(ps.theta_s[k, -1])
        npt.assert_allclose([m_sk, P_sk], [m_rts[k], P_rts[k]], rtol=1e-6, atol=1e-9)


def test_projection_smoother_matches_dense_rts(setup: Setup) -> None:
    """Algorithm 1 reproduces the exact dense RTS moments at every grid point (Rem. 2).

    Parameters
    ----------
    setup : Setup
        Module fixture.
    """
    model, m0, P0, t_meas, ys, pf = setup
    ps = projection_smoother(model, pf)
    n_dense = pf.ts.shape[1]
    grid, meas_idx, m_sd, P_sd, _, _ = dense_kalman_rts(model, t_meas, ys, m0, P0, n_dense=n_dense)
    m_flat, P_flat = [], []
    for k in range(ps.theta_s.shape[0]):
        for j in range(n_dense):
            m_, P_ = moments_from_theta(ps.theta_s[k, j])
            m_flat.append(m_)
            P_flat.append(P_)
    # map interval-stacked values onto the dense grid (duplicated measurement
    # points in the interval representation are consistent by continuity)
    m_ref, P_ref = [], []
    for k in range(ps.theta_s.shape[0]):
        i0 = k * (n_dense - 1)
        for j in range(n_dense):
            m_ref.append(m_sd[i0 + j])
            P_ref.append(P_sd[i0 + j])
    npt.assert_allclose(m_flat, m_ref, rtol=1e-6, atol=1e-8)
    npt.assert_allclose(P_flat, P_ref, rtol=1e-6, atol=1e-8)


def test_algorithm1_matches_tilting_form(setup: Setup) -> None:
    """Algorithm 1 (Eq. (23)) and the tilting form (tilting ODE + jumps) agree (Prop. 1).

    Parameters
    ----------
    setup : Setup
        Module fixture.
    """
    model, m0, P0, t_meas, ys, pf = setup
    ps = projection_smoother(model, pf)
    pt = tilting_smoother(model, ys, pf)
    npt.assert_allclose(ps.theta_s, pt.theta_s, rtol=1e-8, atol=1e-9)
    npt.assert_allclose(ps.lam, pt.lam, rtol=1e-8, atol=1e-9)


def test_structure_properties(setup: Setup) -> None:
    """lambda_2 <= 0, P_s <= P_p, and continuity of theta_s across t_k (Prop. 1).

    Parameters
    ----------
    setup : Setup
        Module fixture.
    """
    model, m0, P0, t_meas, ys, pf = setup
    ps = projection_smoother(model, pf)
    # lambda_2 <= 0 everywhere along the backward pass
    assert onp.all(ps.lam[:, :, 1] <= 1e-12)
    # P_s <= P_p pointwise
    for k in range(ps.theta_s.shape[0]):
        for j in range(ps.theta_s.shape[1]):
            _, P_s = moments_from_theta(ps.theta_s[k, j])
            _, P_p = moments_from_theta(pf.theta_p[k, j])
            assert P_s <= P_p + 1e-10
    # theta_s = theta_p + lambda continuous across interior measurement times:
    # end of interval k equals start of interval k+1.  Non-trivial for the
    # tilting form, where theta_p and lambda both jump at t_k.
    pt = tilting_smoother(model, ys, pf)
    for k in range(pt.theta_s.shape[0] - 1):
        npt.assert_allclose(pt.theta_s[k, -1], pt.theta_s[k + 1, 0], rtol=1e-7, atol=1e-9)
