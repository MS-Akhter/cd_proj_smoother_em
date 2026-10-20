r"""Tests for the exact continuous-discrete Kalman filter and RTS smoother (Sec. V-A reference).

The reference is an independent batch linear-Gaussian computation: for the
Ornstein--Uhlenbeck process the joint law of the states on any time grid is
Gaussian with covariance
    Cov(x_s, x_t) = e^{a(s-t0)} e^{a(t-t0)} P0
                    + sigma^2 e^{a(s+t)} \int_{t0}^{min(s,t)} e^{-2 a u} du,
and conditioning on the (linear-Gaussian) measurements gives the exact
filtering/smoothing moments by block matrix algebra.

Tests:
1. kalman_filter: predicted and filtered moments at measurement times match
   the batch reference (conditioning on y_{1:k} for the k-th filtered moment).
2. rts_smoother: smoothed moments at measurement times match the batch
   posterior marginals (conditioning on all measurements).
3. dense_kalman_rts: smoothed moments on a dense grid (including
   off-measurement times) match the batch posterior on the augmented grid.
"""
import jax

jax.config.update("jax_enable_x64", True)

import numpy as onp
import numpy.testing as npt
import pytest
from jaxtyping import ArrayLike, Float

from cd_smoother.models import LinearSDE
from cd_smoother.kalman import dense_kalman_rts, kalman_filter, rts_smoother

Setup = tuple[LinearSDE, float, float, onp.ndarray, onp.ndarray]


# =====================================================================
# Batch reference (independent of the recursive implementations)
# =====================================================================

def _ou_joint_cov(
    times: Float[ArrayLike, " n"], a: float, sigma: float, m0: float, P0: float, t0: float = 0.0
) -> tuple[Float[onp.ndarray, " n"], Float[onp.ndarray, "n n"]]:
    """Joint mean and covariance of the OU process (24) at `times`, x_{t0} ~ N(m0, P0).

    Parameters
    ----------
    times : Float[ArrayLike, " n"]
        Evaluation times (>= t0).
    a, sigma : float
        Drift and diffusion coefficients.
    m0, P0 : float
        Prior mean and variance at t0.
    t0 : float, optional
        Initial time.

    Returns
    -------
    means : Float[onp.ndarray, " n"]
        E[x_t] = exp(a (t - t0)) m0.
    C : Float[onp.ndarray, "n n"]
        Cov(x_s, x_t).
    """
    times = onp.asarray(times)
    n = len(times)
    C = onp.empty((n, n))
    for i in range(n):
        for j in range(n):
            s, t = times[i], times[j]
            tmin = min(s, t)
            if a == 0.0:
                integ = tmin - t0
            else:
                integ = (onp.exp(-2 * a * t0) - onp.exp(-2 * a * tmin)) / (2 * a)
            C[i, j] = onp.exp(a * (s - t0)) * onp.exp(a * (t - t0)) * P0 + sigma**2 * onp.exp(a * (s + t)) * integ
    means = onp.exp(a * (times - t0)) * m0
    return means, C


def _batch_posterior(
    times: Float[ArrayLike, " n"],
    meas_idx: list[int],
    ys: Float[ArrayLike, " n_meas"],
    model: LinearSDE,
    m0: float,
    P0: float,
) -> tuple[Float[onp.ndarray, " n"], Float[onp.ndarray, " n"]]:
    """Exact Gaussian posterior of the states at `times` given ys at times[meas_idx].

    Parameters
    ----------
    times : Float[ArrayLike, " n"]
        State times.
    meas_idx : list[int]
        Indices into `times` of the measured states.
    ys : Float[ArrayLike, " n_meas"]
        Measurements y = x + xi, xi ~ N(0, R).
    model : LinearSDE
        Model parameters.
    m0, P0 : float
        Prior mean and variance at t0 = 0.

    Returns
    -------
    mean : Float[onp.ndarray, " n"]
        Posterior means.
    var : Float[onp.ndarray, " n"]
        Posterior marginal variances.
    """
    mu, C = _ou_joint_cov(times, model.a, model.sigma, m0, P0)
    H = onp.zeros((len(meas_idx), len(times)))
    for r, i in enumerate(meas_idx):
        H[r, i] = 1.0
    S = H @ C @ H.T + model.R * onp.eye(len(meas_idx))
    K = C @ H.T @ onp.linalg.solve(S, onp.eye(len(meas_idx)))
    mean = mu + K @ (onp.asarray(ys) - H @ mu)
    cov = C - K @ H @ C
    return mean, onp.diag(cov)


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
    rng = onp.random.default_rng(0)
    model = LinearSDE(a=-1.0, sigma=1.0, R=0.25)
    m0, P0 = 0.2, 0.5
    dt = 2.0  # large measurement interval
    N = 6
    t_meas = dt * onp.arange(1, N + 1)
    ys = rng.normal(size=N)  # measurement VALUES need not be realizable to test the algebra
    return model, m0, P0, t_meas, ys


def test_kalman_filter_matches_batch(setup: Setup) -> None:
    """Predicted/filtered moments equal the batch posterior given y_{1:k-1} / y_{1:k}.

    Parameters
    ----------
    setup : Setup
        Module fixture.
    """
    model, m0, P0, t_meas, ys = setup
    out = kalman_filter(model, t_meas, ys, m0, P0)
    for k in range(len(t_meas)):
        # filtered moment at t_k conditions on y_{1:k}
        mean, var = _batch_posterior(t_meas[: k + 1], list(range(k + 1)), ys[: k + 1], model, m0, P0)
        npt.assert_allclose(out.m_f[k], mean[k], rtol=1e-9, atol=1e-12)
        npt.assert_allclose(out.P_f[k], var[k], rtol=1e-9, atol=1e-12)
    # predicted moment at t_k conditions on y_{1:k-1}
    mean, var = _batch_posterior(t_meas, list(range(len(t_meas) - 1)), ys[:-1], model, m0, P0)
    npt.assert_allclose(out.m_p[-1], mean[-1], rtol=1e-9, atol=1e-12)
    npt.assert_allclose(out.P_p[-1], var[-1], rtol=1e-9, atol=1e-12)


def test_rts_smoother_matches_batch(setup: Setup) -> None:
    """RTS moments at the measurement times equal the batch posterior given y_{1:N}.

    Parameters
    ----------
    setup : Setup
        Module fixture.
    """
    model, m0, P0, t_meas, ys = setup
    out = kalman_filter(model, t_meas, ys, m0, P0)
    m_s, P_s = rts_smoother(model, t_meas, out)
    mean, var = _batch_posterior(t_meas, list(range(len(t_meas))), ys, model, m0, P0)
    npt.assert_allclose(m_s, mean, rtol=1e-9, atol=1e-12)
    npt.assert_allclose(P_s, var, rtol=1e-9, atol=1e-12)


def test_dense_kalman_rts_matches_batch(setup: Setup) -> None:
    """Dense-grid RTS moments equal the batch posterior on the augmented grid.

    Parameters
    ----------
    setup : Setup
        Module fixture.
    """
    model, m0, P0, t_meas, ys = setup
    n_dense = 7
    grid, meas_idx, m_s, P_s, m_p, P_p = dense_kalman_rts(model, t_meas, ys, m0, P0, n_dense=n_dense)
    assert onp.all(onp.isin(t_meas, grid))
    mean, var = _batch_posterior(grid, list(meas_idx), ys, model, m0, P0)
    npt.assert_allclose(m_s, mean, rtol=1e-8, atol=1e-11)
    npt.assert_allclose(P_s, var, rtol=1e-8, atol=1e-11)
