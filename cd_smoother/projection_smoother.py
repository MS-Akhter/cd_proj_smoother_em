"""Continuous-discrete projection smoother (backward pass of the paper's Algorithm 1).

Algorithm 1, step 2: set theta_s(t_N) = theta_f(t_N) and, for k = N, ..., 1,
integrate the projected smoother ODE, Eq. (23) (Thm. 2),

    d(theta_s)/dt = g(theta_s)^{-1} E_{theta_s}[L(c)]
                    + g(theta_s)^{-1} E_{theta_s}[J_c Sigma J_c^T] (theta_s - theta_p),

backward on I_k using the stored predicted trajectory theta_p(t), with
theta_s(t_{k-1}^-) = theta_s(t_{k-1}^+): by Prop. 1 the smoothing parameter
is continuous across the measurement times, so no jump is applied to it.

For the OU model (24), Eq. (23) is Eq. (26), which :func:`projection_smoother`
integrates against the stored theta_p(t).

Equivalent tilting form (cross-check; not printed in the current draft).
Writing lambda = theta_s - theta_p (the exponent of the ratio p_s / p_p in the
proof of Thm. 2), Eq. (23) minus Eq. (22) -- for the OU model, Eq. (26) minus
Eq. (25) -- gives the closed-form, autonomous ODE

    d(lambda_1)/dt = -a lambda_1 - 2 sigma^2 lambda_1 lambda_2,
    d(lambda_2)/dt = -2 a lambda_2 - 2 sigma^2 lambda_2^2.

Since theta_p jumps by L(y_k) at t_k while theta_s does not (Prop. 1), lambda
jumps by lambda(t_k^-) = lambda(t_k^+) + L(y_k), starting from
lambda(t_N^+) = 0.  :func:`tilting_smoother` implements this form; it does not
need the stored theta_p and serves as an independent check of Alg. 1.
"""
import jax

jax.config.update("jax_enable_x64", True)

from typing import NamedTuple

import diffrax
import jax.numpy as jnp
import numpy as onp
from jaxtyping import Array, ArrayLike, Float

from cd_smoother.models import LinearSDE, conjugate_update_vector, moments_from_theta
from cd_smoother.projection_filter import (
    PFOutput,
    _solve_interval,
    expected_generator_stats,
    fisher_metric,
)


def expected_jsj(theta: Float[ArrayLike, "2"], model: LinearSDE) -> Float[Array, "2 2"]:
    """E_theta[J_c Sigma J_c^T] of Eq. (23) for c(x) = (x, x^2) and Sigma = sigma^2.

    J_c = dc/dx = (1, 2x)^T, so J_c Sigma J_c^T = sigma^2 [[1, 2x], [2x, 4x^2]]
    and

        E_theta[J_c Sigma J_c^T] = sigma^2 [[1, 2 m], [2 m, 4 (m^2 + P)]].

    Parameters
    ----------
    theta : Float[ArrayLike, "2"]
        Natural parameter.
    model : LinearSDE
        Model parameters.

    Returns
    -------
    M : Float[Array, "2 2"]
        Expected outer product of the statistic gradients.
    """
    m, P = moments_from_theta(theta)
    s2 = model.sigma**2
    return s2 * jnp.array([[1.0, 2.0 * m], [2.0 * m, 4.0 * (m**2 + P)]])


def theta_s_ode_rhs(
    theta_s: Float[ArrayLike, "2"], theta_anchor: Float[ArrayLike, "2"], model: LinearSDE
) -> Float[Array, "2"]:
    """Right-hand side of the projected smoother ODE, Eq. (23).

        d(theta_s)/dt = g(theta_s)^{-1} E_{theta_s}[L(c)]
                        + g(theta_s)^{-1} E_{theta_s}[J_c Sigma J_c^T]
                          (theta_s - theta_anchor).

    With theta_anchor = theta_p this is Eq. (23) (Eq. (26) for the OU model);
    with theta_anchor = theta^f, Koyama's filtering trajectory, it is Koyama's
    Eq. (20), which coincides with Eq. (23) on each open interval I_k (Rem. 1).

    Parameters
    ----------
    theta_s : Float[ArrayLike, "2"]
        Smoothing natural parameter.
    theta_anchor : Float[ArrayLike, "2"]
        Anchor parameter (predicted or filtering trajectory value).
    model : LinearSDE
        Model parameters.

    Returns
    -------
    dtheta_s : Float[Array, "2"]
        Time derivative of theta_s.
    """
    b = expected_generator_stats(theta_s, model) + expected_jsj(theta_s, model) @ (
        jnp.asarray(theta_s) - jnp.asarray(theta_anchor)
    )
    return jnp.linalg.solve(fisher_metric(theta_s), b)


def lambda_ode_rhs(lam: Float[ArrayLike, "2"], model: LinearSDE) -> Float[Array, "2"]:
    """Closed-form OU tilting ODE (Eq. (26) minus Eq. (25)).

    d(lambda_1)/dt = -a lambda_1 - 2 sigma^2 lambda_1 lambda_2 and
    d(lambda_2)/dt = -2 a lambda_2 - 2 sigma^2 lambda_2^2: Eq. (23) minus
    Eq. (22) at theta_s = theta_p + lambda for the OU model (24); the closure
    is exact for linear dynamics (Rem. 2).

    Parameters
    ----------
    lam : Float[ArrayLike, "2"]
        Tilting parameter (lambda_1, lambda_2) = theta_s - theta_p.
    model : LinearSDE
        Model parameters.

    Returns
    -------
    dlam : Float[Array, "2"]
        Time derivative of lambda.
    """
    l1, l2 = lam[0], lam[1]
    a, s2 = model.a, model.sigma**2
    return jnp.array([-a * l1 - 2.0 * s2 * l1 * l2, -2.0 * a * l2 - 2.0 * s2 * l2**2])


def _smoother_rhs(
    t: Float[ArrayLike, ""], theta_s: Float[Array, "2"], args: tuple[LinearSDE, diffrax.Solution]
) -> Float[Array, "2"]:
    """``diffrax.ODETerm`` wrapper of Eq. (23) with the stored theta_p(t).

    Parameters
    ----------
    t : Float[ArrayLike, ""]
        Time in the closed interval [t_{k-1}, t_k].
    theta_s : Float[Array, "2"]
        Smoothing natural parameter.
    args : tuple[LinearSDE, diffrax.Solution]
        Model parameters and the dense forward solution of Eq. (22) on the
        current interval (``PFOutput.dense[k]``).

    Returns
    -------
    dtheta_s : Float[Array, "2"]
        Right-hand side of Eq. (23) with theta_anchor = theta_p(t).
    """
    model, theta_p_dense = args
    return theta_s_ode_rhs(theta_s, theta_p_dense.evaluate(t), model)


def _tilting_rhs(t: Float[ArrayLike, ""], lam: Float[Array, "2"], args: LinearSDE) -> Float[Array, "2"]:
    """``diffrax.ODETerm`` wrapper of the closed-form tilting ODE (:func:`lambda_ode_rhs`).

    Parameters
    ----------
    t : Float[ArrayLike, ""]
        Time (unused; the tilting ODE is autonomous).
    lam : Float[Array, "2"]
        Tilting parameter.
    args : LinearSDE
        Model parameters.

    Returns
    -------
    dlam : Float[Array, "2"]
        Right-hand side of the tilting ODE.
    """
    return lambda_ode_rhs(lam, args)


SMOOTHER_TERM = diffrax.ODETerm(_smoother_rhs)
TILTING_TERM = diffrax.ODETerm(_tilting_rhs)


class PSOutput(NamedTuple):
    """Projection-smoother output on the filter's per-interval grids.

    Attributes
    ----------
    theta_s : Float[onp.ndarray, "N n_dense 2"]
        Smoothing natural parameter on interval k (grid ``PFOutput.ts[k]``);
        the right endpoint holds theta_s(t_k) and the left endpoint
        theta_s(t_{k-1}), so that, by continuity (Prop. 1),
        ``theta_s[k, -1] == theta_s[k + 1, 0]``.
    lam : Float[onp.ndarray, "N n_dense 2"]
        Tilting parameter lambda = theta_s - theta_p on the same grid; the
        right endpoint holds lambda(t_k^-) and the left endpoint
        lambda(t_{k-1}^+).
    """

    theta_s: Float[onp.ndarray, "N n_dense 2"]
    lam: Float[onp.ndarray, "N n_dense 2"]


def projection_smoother(model: LinearSDE, pf: PFOutput) -> PSOutput:
    """Backward pass of Algorithm 1: integrate Eq. (23) for theta_s.

    Sets theta_s(t_N) = theta_f(t_N) (Thm. 2) and, for k = N, ..., 1,
    integrates Eq. (23) backward on I_k, evaluating the stored dense
    theta_p(t) of the forward pass; the result at t_{k-1}^+ is the terminal
    value on I_{k-1}, theta_s(t_{k-1}^-) = theta_s(t_{k-1}^+) (Prop. 1).

    Parameters
    ----------
    model : LinearSDE
        Model parameters.
    pf : PFOutput
        Forward-pass output; ``pf.dense`` supplies theta_p(t) and
        ``pf.theta_f[-1]`` the terminal value theta_f(t_N).

    Returns
    -------
    PSOutput
        Smoothing and tilting parameter trajectories.
    """
    N, n_dense = pf.ts.shape
    theta_s_out = onp.empty((N, n_dense, 2))
    theta_s = jnp.asarray(pf.theta_f[N])
    for k in range(N - 1, -1, -1):
        ts = pf.ts[k]
        traj = _solve_interval(SMOOTHER_TERM, theta_s, ts[0], ts[-1], ts, (model, pf.dense[k]), backward=True)
        theta_s_out[k] = onp.asarray(traj)
        # continuity across t_{k-1} (Prop. 1): no jump in theta_s
        theta_s = traj[0]
    return PSOutput(theta_s=theta_s_out, lam=theta_s_out - pf.theta_p)


def tilting_smoother(model: LinearSDE, ys: Float[ArrayLike, " N"], pf: PFOutput) -> PSOutput:
    """Backward pass in the closed-form OU tilting form (cross-check of Algorithm 1).

    Starts from lambda(t_N^+) = 0 and, for k = N, ..., 1, applies the jump
    lambda(t_k^-) = lambda(t_k^+) + L(y_k), i.e. lambda_1 += y_k / R and
    lambda_2 -= 1 / (2 R), then integrates :func:`lambda_ode_rhs` backward on I_k.  The
    smoothing parameter is recovered as theta_s = theta_p + lambda.  By
    Prop. 1 this is equivalent to :func:`projection_smoother`.

    Parameters
    ----------
    model : LinearSDE
        Model parameters.
    ys : Float[ArrayLike, " N"]
        Measurements y_1, ..., y_N.
    pf : PFOutput
        Forward-pass output (only the stored grid values ``pf.theta_p`` are
        used, for the recovery theta_s = theta_p + lambda).

    Returns
    -------
    PSOutput
        Smoothing and tilting parameter trajectories.
    """
    ys_arr = onp.asarray(ys)
    N, n_dense = pf.ts.shape
    lam_out = onp.empty((N, n_dense, 2))
    lam = jnp.zeros(2)
    for k in range(N - 1, -1, -1):
        # jump of the tilting parameter at the measurement time t_k
        lam = lam + conjugate_update_vector(model, ys_arr[k])
        ts = pf.ts[k]
        traj = _solve_interval(TILTING_TERM, lam, ts[0], ts[-1], ts, model, backward=True)
        lam_out[k] = onp.asarray(traj)
        lam = traj[0]
    return PSOutput(theta_s=pf.theta_p + lam_out, lam=lam_out)
