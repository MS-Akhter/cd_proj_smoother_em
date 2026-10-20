"""Exponential-family projection filter (Gaussian family, OU model).

Implements the forward pass, step 1 of the paper's Algorithm 1: starting from
theta_f(t_0) = theta_0, on each I_k = (t_{k-1}, t_k) the natural-parameter
prediction ODE, Eq. (22),

    d(theta_p)/dt = g(theta_p)^{-1} E_{theta_p}[L(c)],

with theta_p(t_{k-1}^+) = theta_f(t_{k-1}) is integrated with an adaptive
solver and stored, and at t_k the conjugate update of Prop. 1,

    theta_f(t_k) = theta_p(t_k^-) + L(y_k),    theta_p(t_k^+) = theta_f(t_k),

is applied.  Here L is the generator (4) and L(y) the conjugate update vector
of Def. 1.  For the Gaussian family and linear dynamics the projection
residual vanishes (Rem. 2), so this filter coincides with the exact
continuous-discrete Kalman filter.
"""
import jax

jax.config.update("jax_enable_x64", True)

from typing import Any, NamedTuple

import diffrax
import equinox as eqx
import jax.numpy as jnp
import numpy as onp
from jaxtyping import Array, ArrayLike, Float, PyTree

from cd_smoother.models import (
    LinearSDE,
    conjugate_update_vector,
    moments_from_theta,
    theta_from_moments,
)

#: Step budget of the dense-output forward solves.  diffrax preallocates the
#: dense interpolation buffers to this length, so it is kept moderate (the
#: Sec. V runs need < 300 steps per interval); an interval that needs more
#: steps raises instead of silently truncating.
DENSE_MAX_STEPS = 4096


def fisher_metric(theta: Float[ArrayLike, "2"]) -> Float[Array, "2 2"]:
    """Fisher information matrix g(theta) of the Gaussian family, c(x) = (x, x^2).

    g(theta) = grad^2 psi(theta) = Cov_theta(c) (Sec. IV, after Eq. (21)).

    Parameters
    ----------
    theta : Float[ArrayLike, "2"]
        Natural parameter.

    Returns
    -------
    g : Float[Array, "2 2"]
        Fisher matrix Cov_theta(c) = [[P, 2 m P], [2 m P, 2 P^2 + 4 m^2 P]].
    """
    m, P = moments_from_theta(theta)
    return jnp.array([[P, 2.0 * m * P], [2.0 * m * P, 2.0 * P**2 + 4.0 * m**2 * P]])


def expected_generator_stats(theta: Float[ArrayLike, "2"], model: LinearSDE) -> Float[Array, "2"]:
    """E_theta[L(c)] for the OU generator and c(x) = (x, x^2).

    With the generator (4), L = a x d/dx + (sigma^2 / 2) d^2/dx^2, one has
    L(x) = a x and L(x^2) = 2 a x^2 + sigma^2, hence

        E_theta[L(c)] = (a m, 2 a (m^2 + P) + sigma^2).

    This is the first term of both Eq. (22) and Eq. (23).

    Parameters
    ----------
    theta : Float[ArrayLike, "2"]
        Natural parameter.
    model : LinearSDE
        Model parameters.

    Returns
    -------
    v : Float[Array, "2"]
        Expected generator statistics.
    """
    m, P = moments_from_theta(theta)
    return jnp.array([model.a * m, 2.0 * model.a * (m**2 + P) + model.sigma**2])


def filter_ode_rhs(theta: Float[ArrayLike, "2"], model: LinearSDE) -> Float[Array, "2"]:
    """Right-hand side of the projection-filter ODE, Eq. (22) (Eq. (25) for the OU model).

    Parameters
    ----------
    theta : Float[ArrayLike, "2"]
        Natural parameter theta_p.
    model : LinearSDE
        Model parameters.

    Returns
    -------
    dtheta : Float[Array, "2"]
        g(theta)^{-1} E_theta[L(c)].
    """
    return jnp.linalg.solve(fisher_metric(theta), expected_generator_stats(theta, model))


class PFOutput(NamedTuple):
    """Projection-filter output (Alg. 1, step 1) on the per-interval grids.

    Attributes
    ----------
    ts : Float[onp.ndarray, "N n_dense"]
        Time grid of interval k, from t_{k-1} to t_k (endpoints included).
    theta_p : Float[onp.ndarray, "N n_dense 2"]
        Predicted natural-parameter trajectory on interval k; the left
        endpoint equals the post-update parameter theta_f(t_{k-1}), the right
        endpoint is the pre-update parameter theta_p(t_k^-).
    theta_f : Float[onp.ndarray, "N+1 2"]
        Post-update parameters: theta_f[0] is the prior at t_0 and theta_f[k]
        (k >= 1) the parameter after assimilating y_k at t_k.
    dense : tuple[diffrax.Solution, ...]
        Length-N tuple; entry k is the dense-output solution of Eq. (22) on
        [t_{k-1}, t_k], i.e. the stored theta_p(t) of Alg. 1 that the
        backward pass evaluates at arbitrary t in I_k.
    """

    ts: Float[onp.ndarray, "N n_dense"]
    theta_p: Float[onp.ndarray, "N n_dense 2"]
    theta_f: Float[onp.ndarray, "N+1 2"]
    dense: tuple[diffrax.Solution, ...]


@eqx.filter_jit
def _jitted_solve(
    term: diffrax.ODETerm,
    y0: Float[Array, " d"],
    t0: Float[ArrayLike, ""],
    t1: Float[ArrayLike, ""],
    dt0: Float[ArrayLike, ""],
    save_ts: Float[Array, " n_save"],
    args: PyTree[Any],
) -> Float[Array, "n_save d"]:
    """JIT-compiled adaptive Dormand--Prince solve saving at `save_ts`.

    Compiled once per term/shape combination (Sec. V: relative tolerance
    1e-11).

    Parameters
    ----------
    term : diffrax.ODETerm
        Vector field f(t, y, args).
    y0 : Float[Array, " d"]
        Initial state at `t0`.
    t0, t1 : Float[ArrayLike, ""]
        Integration endpoints (t1 < t0 integrates backward in time).
    dt0 : Float[ArrayLike, ""]
        Initial step, with the sign of t1 - t0.
    save_ts : Float[Array, " n_save"]
        Save times, monotone in the integration direction.
    args : PyTree[Any]
        Static/dynamic arguments forwarded to the vector field.

    Returns
    -------
    ys : Float[Array, "n_save d"]
        Solution at `save_ts`.
    """
    sol = diffrax.diffeqsolve(
        term,
        diffrax.Dopri5(),
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        args=args,
        saveat=diffrax.SaveAt(ts=save_ts),
        stepsize_controller=diffrax.PIDController(rtol=1e-11, atol=1e-13),
        max_steps=1_000_000,
    )
    return sol.ys


@eqx.filter_jit
def _jitted_dense_solve(
    term: diffrax.ODETerm,
    y0: Float[Array, " d"],
    t0: Float[ArrayLike, ""],
    t1: Float[ArrayLike, ""],
    dt0: Float[ArrayLike, ""],
    save_ts: Float[Array, " n_save"],
    args: PyTree[Any],
) -> diffrax.Solution:
    """Forward adaptive solve returning both `save_ts` values and dense output.

    Parameters
    ----------
    term : diffrax.ODETerm
        Vector field f(t, y, args).
    y0 : Float[Array, " d"]
        Initial state at `t0`.
    t0, t1 : Float[ArrayLike, ""]
        Integration endpoints, t0 < t1.
    dt0 : Float[ArrayLike, ""]
        Initial step (positive).
    save_ts : Float[Array, " n_save"]
        Increasing save times in [t0, t1].
    args : PyTree[Any]
        Arguments forwarded to the vector field.

    Returns
    -------
    sol : diffrax.Solution
        Solution with ``sol.ys`` of shape (n_save, d) and a dense
        interpolant, ``sol.evaluate(t)`` for t in [t0, t1].
    """
    return diffrax.diffeqsolve(
        term,
        diffrax.Dopri5(),
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        args=args,
        saveat=diffrax.SaveAt(ts=save_ts, dense=True),
        stepsize_controller=diffrax.PIDController(rtol=1e-11, atol=1e-13),
        max_steps=DENSE_MAX_STEPS,
    )


def _solve_interval(
    term: diffrax.ODETerm,
    y0: Float[ArrayLike, " d"],
    t_left: float,
    t_right: float,
    ts: Float[onp.ndarray, " n_save"],
    args: PyTree[Any],
    backward: bool = False,
) -> Float[Array, "n_save d"]:
    """Solve dy/dt = f(t, y, args) over [t_left, t_right], saving at `ts`.

    `term` must be a module-level ``diffrax.ODETerm`` so that the underlying
    JIT compilation is reused across calls.

    Parameters
    ----------
    term : diffrax.ODETerm
        Vector field f(t, y, args).
    y0 : Float[ArrayLike, " d"]
        Initial value, at `t_left` (forward) or at `t_right` (backward).
    t_left, t_right : float
        Interval endpoints, t_left < t_right.
    ts : Float[onp.ndarray, " n_save"]
        Increasing save times in [t_left, t_right].
    args : PyTree[Any]
        Arguments forwarded to the vector field.
    backward : bool, optional
        If True, integrate from `t_right` down to `t_left`.

    Returns
    -------
    ys : Float[Array, "n_save d"]
        Solution at `ts`, in increasing time order in both directions.
    """
    if backward:
        t0, t1, dt0 = t_right, t_left, -(t_right - t_left) / 100.0
        save_ts = ts[::-1].copy()
    else:
        t0, t1, dt0 = t_left, t_right, (t_right - t_left) / 100.0
        save_ts = ts
    ys = _jitted_solve(term, jnp.asarray(y0), t0, t1, dt0, jnp.asarray(save_ts), args)
    if backward:
        ys = ys[::-1]
    return ys


def _prediction_rhs(t: Float[ArrayLike, ""], theta: Float[Array, "2"], args: LinearSDE) -> Float[Array, "2"]:
    """``diffrax.ODETerm`` wrapper of the prediction ODE, Eq. (22).

    Parameters
    ----------
    t : Float[ArrayLike, ""]
        Time (unused; Eq. (22) is autonomous for the OU model).
    theta : Float[Array, "2"]
        Natural parameter theta_p.
    args : LinearSDE
        Model parameters.

    Returns
    -------
    dtheta : Float[Array, "2"]
        Right-hand side of Eq. (22).
    """
    return filter_ode_rhs(theta, args)


PREDICTION_TERM = diffrax.ODETerm(_prediction_rhs)


def projection_filter(
    model: LinearSDE,
    t_meas: Float[ArrayLike, " N"],
    ys: Float[ArrayLike, " N"],
    m0: float,
    P0: float,
    n_dense: int = 21,
    t0: float = 0.0,
) -> PFOutput:
    """Forward pass of Algorithm 1: conjugate projection filter with stored trajectory.

    For k = 1, ..., N, integrates Eq. (22) on I_k from
    theta_p(t_{k-1}^+) = theta_f(t_{k-1}), stores theta_p(t) (sampled on a
    uniform grid and as a dense interpolant), and applies the conjugate update
    theta_f(t_k) = theta_p(t_k^-) + L(y_k) of Prop. 1.

    Parameters
    ----------
    model : LinearSDE
        Model parameters.
    t_meas : Float[ArrayLike, " N"]
        Measurement times t_1 < ... < t_N.
    ys : Float[ArrayLike, " N"]
        Measurements y_1, ..., y_N.
    m0, P0 : float
        Prior mean and variance at t_0, theta_0 = (m0 / P0, -1 / (2 P0)).
    n_dense : int, optional
        Stored grid points per interval (both endpoints included).
    t0 : float, optional
        Initial time t_0.

    Returns
    -------
    PFOutput
        Stored prediction trajectory and post-update parameters.
    """
    t_meas = onp.asarray(t_meas)
    ys_arr = onp.asarray(ys)
    N = len(t_meas)
    edges = onp.concatenate(([t0], t_meas))

    ts_out = onp.empty((N, n_dense))
    theta_p_out = onp.empty((N, n_dense, 2))
    theta_f_out = onp.empty((N + 1, 2))
    dense = []
    theta = jnp.asarray(theta_from_moments(m0, P0))
    theta_f_out[0] = onp.asarray(theta)
    for k in range(N):
        ts = onp.linspace(edges[k], edges[k + 1], n_dense)
        sol = _jitted_dense_solve(
            PREDICTION_TERM, theta, edges[k], edges[k + 1], (edges[k + 1] - edges[k]) / 100.0, jnp.asarray(ts), model
        )
        ts_out[k] = ts
        theta_p_out[k] = onp.asarray(sol.ys)
        dense.append(sol)
        # conjugate update at t_k (Prop. 1)
        theta = sol.ys[-1] + conjugate_update_vector(model, ys_arr[k])
        theta_f_out[k + 1] = onp.asarray(theta)
    return PFOutput(ts=ts_out, theta_p=theta_p_out, theta_f=theta_f_out, dense=tuple(dense))
