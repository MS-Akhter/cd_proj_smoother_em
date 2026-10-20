"""Tests for the univariate linear SDE model utilities (OU model, Eq. (24)).

Tests:
1. transition_params: exact discretization (A, Q) matches the closed form
   A = exp(a*dt), Q = sigma^2 * (exp(2*a*dt) - 1) / (2*a).
2. transition_params: the a -> 0 limit is finite and equals sigma^2 * dt.
3. theta_from_moments / moments_from_theta are mutually inverse and map to
   the natural parameters of the Gaussian exponential family with
   c(x) = (x, x^2).
4. conjugate_update_vector: the additive natural-parameter update
   L(y) = (y/R, -1/(2R)) of Def. 1 / Prop. 1 reproduces the Kalman
   measurement update.
"""
import jax

jax.config.update("jax_enable_x64", True)

import numpy as onp
import numpy.testing as npt

from cd_smoother.models import (
    LinearSDE,
    conjugate_update_vector,
    moments_from_theta,
    theta_from_moments,
    transition_params,
)


def _model(a: float = -1.0, sigma: float = 1.0, R: float = 0.25) -> LinearSDE:
    """Build an OU model (24).

    Parameters
    ----------
    a, sigma, R : float, optional
        Drift, diffusion and measurement-noise variance.

    Returns
    -------
    LinearSDE
        Model parameters.
    """
    return LinearSDE(a=a, sigma=sigma, R=R)


def test_transition_params_closed_form() -> None:
    """(A, Q) match exp(a dt) and sigma^2 (exp(2 a dt) - 1) / (2 a)."""
    model = _model(a=-0.7, sigma=1.3)
    dt = 0.8
    A, Q = transition_params(model, dt)
    npt.assert_allclose(A, onp.exp(-0.7 * dt), rtol=1e-14)
    npt.assert_allclose(Q, 1.3**2 * (onp.exp(2 * (-0.7) * dt) - 1.0) / (2 * (-0.7)), rtol=1e-14)
    assert Q > 0.0


def test_transition_params_zero_drift_limit() -> None:
    """The a -> 0 limit of Q is finite, equals sigma^2 dt, and is continuous."""
    dt = 0.5
    A, Q = transition_params(_model(a=0.0, sigma=2.0), dt)
    npt.assert_allclose(A, 1.0, rtol=1e-14)
    npt.assert_allclose(Q, 2.0**2 * dt, rtol=1e-10)
    # continuity of the formula near a = 0
    _, Q_eps = transition_params(_model(a=1e-12, sigma=2.0), dt)
    npt.assert_allclose(Q_eps, Q, rtol=1e-6)


def test_theta_moment_bijection() -> None:
    """theta = (m / P, -1 / (2 P)) (Sec. V-A) and its inverse are mutually inverse."""
    m, P = 0.3, 0.7
    theta = theta_from_moments(m, P)
    npt.assert_allclose(theta, onp.array([m / P, -1.0 / (2.0 * P)]), rtol=1e-14)
    m2, P2 = moments_from_theta(theta)
    npt.assert_allclose([m2, P2], [m, P], rtol=1e-14)


def test_conjugate_update_matches_kalman() -> None:
    """theta_f = theta_p + L(y) (Prop. 1) equals the Kalman measurement update."""
    model = _model(R=0.09)
    m_p, P_p = -0.4, 1.7
    y = 0.5
    # Kalman measurement update (H = 1)
    K = P_p / (P_p + model.R)
    m_f_ref = m_p + K * (y - m_p)
    P_f_ref = (1.0 - K) * P_p
    # Natural-parameter additive update
    theta_f = theta_from_moments(m_p, P_p) + conjugate_update_vector(model, y)
    m_f, P_f = moments_from_theta(theta_f)
    npt.assert_allclose([m_f, P_f], [m_f_ref, P_f_ref], rtol=1e-12)
