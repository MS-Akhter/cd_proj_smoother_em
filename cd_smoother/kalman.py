"""Exact continuous-discrete Kalman filter and RTS smoother for the OU model.

These are the exact references of the paper's numerical section (Sec. V-A):
for the linear-Gaussian model (24) the discrete-time Kalman filter on the
exactly discretized system, followed by the Rauch--Tung--Striebel backward
recursion, gives the exact filtering/smoothing moments at the grid points.
By Rem. 2, Algorithm 1 must reproduce them up to ODE-solver accuracy.
"""
import jax

jax.config.update("jax_enable_x64", True)

from typing import NamedTuple

import numpy as onp
from jaxtyping import ArrayLike, Float, Int

from cd_smoother.models import LinearSDE, transition_params


class KFOutput(NamedTuple):
    """Filtering output at the measurement times.

    Attributes
    ----------
    m_p, P_p : Float[onp.ndarray, " N"]
        Predicted mean/variance at t_k (before assimilating y_k).
    m_f, P_f : Float[onp.ndarray, " N"]
        Filtered mean/variance at t_k (after assimilating y_k).
    """

    m_p: Float[onp.ndarray, " N"]
    P_p: Float[onp.ndarray, " N"]
    m_f: Float[onp.ndarray, " N"]
    P_f: Float[onp.ndarray, " N"]


def kalman_filter(
    model: LinearSDE,
    t_meas: Float[ArrayLike, " N"],
    ys: Float[ArrayLike, " N"],
    m0: float,
    P0: float,
    t0: float = 0.0,
) -> KFOutput:
    """Exact continuous-discrete Kalman filter.

    Parameters
    ----------
    model : LinearSDE
        Model parameters.
    t_meas : Float[ArrayLike, " N"]
        Strictly increasing measurement times (> t0).
    ys : Float[ArrayLike, " N"]
        Measurements.
    m0, P0 : float
        Prior mean and variance at time t0.
    t0 : float, optional
        Initial time.

    Returns
    -------
    KFOutput
        Predicted and filtered moments at the measurement times.
    """
    t_meas = onp.asarray(t_meas)
    ys = onp.asarray(ys)
    N = len(t_meas)
    m_p = onp.empty(N)
    P_p = onp.empty(N)
    m_f = onp.empty(N)
    P_f = onp.empty(N)
    m, P, t_prev = m0, P0, t0
    for k in range(N):
        A, Q = transition_params(model, t_meas[k] - t_prev)
        A, Q = float(A), float(Q)
        m, P = A * m, A * P * A + Q
        m_p[k], P_p[k] = m, P
        S = P + model.R
        K = P / S
        m = m + K * (ys[k] - m)
        P = (1.0 - K) * P
        m_f[k], P_f[k] = m, P
        t_prev = t_meas[k]
    return KFOutput(m_p=m_p, P_p=P_p, m_f=m_f, P_f=P_f)


def rts_smoother(
    model: LinearSDE, t_meas: Float[ArrayLike, " N"], kf: KFOutput, t0: float = 0.0
) -> tuple[Float[onp.ndarray, " N"], Float[onp.ndarray, " N"]]:
    """Exact RTS smoother at the measurement times.

    Parameters
    ----------
    model : LinearSDE
        Model parameters.
    t_meas : Float[ArrayLike, " N"]
        Measurement times used by the filter.
    kf : KFOutput
        Output of :func:`kalman_filter`.
    t0 : float, optional
        Initial time (unused; kept for signature symmetry).

    Returns
    -------
    m_s : Float[onp.ndarray, " N"]
        Smoothed mean at the measurement times.
    P_s : Float[onp.ndarray, " N"]
        Smoothed variance at the measurement times.
    """
    t_meas = onp.asarray(t_meas)
    N = len(t_meas)
    m_s = onp.empty(N)
    P_s = onp.empty(N)
    m_s[-1], P_s[-1] = kf.m_f[-1], kf.P_f[-1]
    for k in range(N - 2, -1, -1):
        A, _ = transition_params(model, t_meas[k + 1] - t_meas[k])
        A = float(A)
        G = kf.P_f[k] * A / kf.P_p[k + 1]
        m_s[k] = kf.m_f[k] + G * (m_s[k + 1] - kf.m_p[k + 1])
        P_s[k] = kf.P_f[k] + G * (P_s[k + 1] - kf.P_p[k + 1]) * G
    return m_s, P_s


def dense_kalman_rts(
    model: LinearSDE,
    t_meas: Float[ArrayLike, " N"],
    ys: Float[ArrayLike, " N"],
    m0: float,
    P0: float,
    n_dense: int = 21,
    t0: float = 0.0,
) -> tuple[
    Float[onp.ndarray, " n_grid"],
    Int[onp.ndarray, " N"],
    Float[onp.ndarray, " n_grid"],
    Float[onp.ndarray, " n_grid"],
    Float[onp.ndarray, " n_grid"],
    Float[onp.ndarray, " n_grid"],
]:
    """Exact smoothing moments on a dense grid (including off-measurement times).

    The grid consists of `n_dense` equally spaced points on each
    inter-measurement interval [t_{k-1}, t_k] (endpoints included, shared
    endpoints deduplicated), so n_grid = N (n_dense - 1) + 1.  A discrete
    Kalman filter on the exactly discretized system with measurement updates
    only at the measurement indices, followed by a discrete RTS pass over the
    whole grid, is exact for the linear-Gaussian model at every grid point;
    this is the "exact reference" of Sec. V-A.

    Parameters
    ----------
    model : LinearSDE
        Model parameters.
    t_meas : Float[ArrayLike, " N"]
        Measurement times.
    ys : Float[ArrayLike, " N"]
        Measurements.
    m0, P0 : float
        Prior mean and variance at t0.
    n_dense : int, optional
        Grid points per interval (including both endpoints).
    t0 : float, optional
        Initial time.

    Returns
    -------
    grid : Float[onp.ndarray, " n_grid"]
        Dense time grid, starting at t0.
    meas_idx : Int[onp.ndarray, " N"]
        Indices of the measurement times in `grid`.
    m_s, P_s : Float[onp.ndarray, " n_grid"]
        Smoothed moments on `grid`.
    m_p, P_p : Float[onp.ndarray, " n_grid"]
        Predicted moments on `grid` (before the update where applicable).
    """
    t_meas = onp.asarray(t_meas)
    ys = onp.asarray(ys)
    N = len(t_meas)
    edges = onp.concatenate(([t0], t_meas))
    grid = [onp.array([t0])]
    for k in range(N):
        grid.append(onp.linspace(edges[k], edges[k + 1], n_dense)[1:])
    grid = onp.concatenate(grid)
    meas_idx = onp.array([k * (n_dense - 1) + (n_dense - 1) for k in range(N)])
    meas_set = {int(i): k for k, i in enumerate(meas_idx)}

    n = len(grid)
    m_p = onp.empty(n)
    P_p = onp.empty(n)
    m_fd = onp.empty(n)
    P_fd = onp.empty(n)
    m, P = m0, P0
    m_p[0], P_p[0] = m, P
    m_fd[0], P_fd[0] = m, P
    for i in range(1, n):
        A, Q = transition_params(model, grid[i] - grid[i - 1])
        A, Q = float(A), float(Q)
        m, P = A * m, A * P * A + Q
        m_p[i], P_p[i] = m, P
        if i in meas_set:
            y = ys[meas_set[i]]
            S = P + model.R
            K = P / S
            m = m + K * (y - m)
            P = (1.0 - K) * P
        m_fd[i], P_fd[i] = m, P

    m_s = onp.empty(n)
    P_s = onp.empty(n)
    m_s[-1], P_s[-1] = m_fd[-1], P_fd[-1]
    for i in range(n - 2, -1, -1):
        A, _ = transition_params(model, grid[i + 1] - grid[i])
        A = float(A)
        G = P_fd[i] * A / P_p[i + 1]
        m_s[i] = m_fd[i] + G * (m_s[i + 1] - m_p[i + 1])
        P_s[i] = P_fd[i] + G * (P_s[i + 1] - P_p[i + 1]) * G
    return grid, meas_idx, m_s, P_s, m_p, P_p
