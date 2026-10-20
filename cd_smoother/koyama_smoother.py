"""Projection smoother of Koyama (2018) for the univariate linear-Gaussian model.

Koyama's Gaussian projection smoother [Koyama 2018, Eqs. (30)-(31)] reduces,
for linear drift f(x) = a x and constant diffusion, to the backward moment
ODEs

    d(mu)/dt = a mu + (sigma^2 / V^f) (mu - mu^f),
    d(V)/dt  = 2 a V - sigma^2 + (2 sigma^2 / V^f) V,

integrated backward from the post-update filtered moments at t_N against the
stored forward filter trajectory (mu^f(t), V^f(t)); the smoothing parameter is
continuous across measurement times, while the anchor trajectory jumps there.
The generic natural-parameter form [Koyama 2018, Eq. (20)] is

    d(theta)/dt = g(theta)^{-1} E_theta[L(c)]
                  + g(theta)^{-1} E_theta[J_c Sigma J_c^T] (theta - theta^f),

i.e. the paper's Eq. (23) with Koyama's filtering anchor theta^f in place of
theta_p; the two coincide on each open interval I_k (Rem. 1).

For the OU model the Gaussian form is the paper's Eq. (27).  Two integration
variants of the Gaussian form are provided:

* ``restarts=True``: each inter-measurement interval is integrated
  separately with the anchor re-initialized at the interval's pre-update
  value and carried backward by the prediction ODE, i.e. the discontinuities
  of the stored trajectory are respected.  It coincides with method (i) of
  Sec. V-A on each I_k (Rem. 1), so the paper does not show it separately;
  the experiments still compute and store it.
* ``restarts=False`` (method (ii) of Sec. V-A, KGPS): a single fixed-step RK4 sweep
  from t_N to t_0 that interpolates the stored, discontinuous anchor
  trajectory blindly across the measurement times.  This is the reading of
  [Koyama 2018, Sec. 2.2.4, step 2] in which the backward ODE is handed to an
  off-the-shelf integrator; it degrades for large measurement intervals.

``koyama_gaussian_smoother_dopri`` runs the same single sweep across the
measurement times with adaptive Dormand--Prince at the tolerances of the
proposed smoother, reading the forward pass's dense output as Algorithm 1
does (``dense=True``) or the linearly interpolated samples (``dense=False``).
"""
import jax

jax.config.update("jax_enable_x64", True)

from typing import Any, NamedTuple

import diffrax
import jax.numpy as jnp
import numpy as onp
from jaxtyping import Array, ArrayLike, Float, PyTree

from cd_smoother.models import LinearSDE, moments_from_theta
from cd_smoother.projection_filter import PFOutput, _jitted_solve, _solve_interval, filter_ode_rhs
from cd_smoother.projection_smoother import theta_s_ode_rhs


class KoyamaOutput(NamedTuple):
    """Gaussian-form output on the filter's per-interval grids.

    Attributes
    ----------
    m : Float[onp.ndarray, "N n_dense"]
        Smoothing mean on interval k (grid ``pf.ts[k]``).
    V : Float[onp.ndarray, "N n_dense"]
        Smoothing variance on the same grid.
    """

    m: Float[onp.ndarray, "N n_dense"]
    V: Float[onp.ndarray, "N n_dense"]


class KoyamaThetaOutput(NamedTuple):
    """Natural-parameter-form output on the filter's per-interval grids.

    Attributes
    ----------
    theta : Float[onp.ndarray, "N n_dense 2"]
        Smoothing natural parameter on interval k.
    """

    theta: Float[onp.ndarray, "N n_dense 2"]


def _gaussian_backward_rhs(
    mu: Float[ArrayLike, ""],
    V: Float[ArrayLike, ""],
    mu_f: Float[ArrayLike, ""],
    V_f: Float[ArrayLike, ""],
    model: LinearSDE,
) -> tuple[Float[ArrayLike, ""], Float[ArrayLike, ""]]:
    """Right-hand side of Koyama's Gaussian backward ODEs [Eqs. (30)-(31)] = paper Eq. (27).

    Parameters
    ----------
    mu, V : Float[ArrayLike, ""]
        Smoothing mean and variance.
    mu_f, V_f : Float[ArrayLike, ""]
        Anchor (filtering-trajectory) mean and variance.
    model : LinearSDE
        Model parameters.

    Returns
    -------
    dmu, dV : Float[ArrayLike, ""]
        Time derivatives of the smoothing mean and variance.
    """
    a, s2 = model.a, model.sigma**2
    dmu = a * mu + (s2 / V_f) * (mu - mu_f)
    dV = 2.0 * a * V - s2 + (2.0 * s2 / V_f) * V
    return dmu, dV


def _gaussian_augmented_rhs(t: Float[ArrayLike, ""], y: Float[Array, "4"], args: LinearSDE) -> Float[Array, "4"]:
    """``diffrax.ODETerm`` of the augmented (anchor + smoother) Gaussian system.

    The anchor moments follow the Kalman moment ODEs dm/dt = a m,
    dP/dt = 2 a P + sigma^2 (equivalent to Eq. (22), Sec. V-A).

    Parameters
    ----------
    t : Float[ArrayLike, ""]
        Time (unused; the system is autonomous).
    y : Float[Array, "4"]
        State (mu^f, V^f, mu, V).
    args : LinearSDE
        Model parameters.

    Returns
    -------
    dy : Float[Array, "4"]
        Time derivative of the augmented state.
    """
    mu_f, V_f, mu_, V_ = y
    a, s2 = args.a, args.sigma**2
    dmu_f = a * mu_f
    dV_f = 2.0 * a * V_f + s2
    dmu, dV = _gaussian_backward_rhs(mu_, V_, mu_f, V_f, args)
    return jnp.array([dmu_f, dV_f, dmu, dV])


GAUSSIAN_AUGMENTED_TERM = diffrax.ODETerm(_gaussian_augmented_rhs)


def _theta_augmented_rhs(t: Float[ArrayLike, ""], y: Float[Array, "4"], args: LinearSDE) -> Float[Array, "4"]:
    """``diffrax.ODETerm`` of the augmented (anchor + smoother) natural-parameter system.

    The anchor theta^f follows Eq. (22); the smoothing parameter follows
    Koyama's Eq. (20) (= Eq. (23) with anchor theta^f).

    Parameters
    ----------
    t : Float[ArrayLike, ""]
        Time (unused; the system is autonomous).
    y : Float[Array, "4"]
        State (theta^f, theta), each of length 2.
    args : LinearSDE
        Model parameters.

    Returns
    -------
    dy : Float[Array, "4"]
        Time derivative of the augmented state.
    """
    th_f, th = y[:2], y[2:]
    return jnp.concatenate([filter_ode_rhs(th_f, args), theta_s_ode_rhs(th, th_f, args)])


THETA_AUGMENTED_TERM = diffrax.ODETerm(_theta_augmented_rhs)


def koyama_gaussian_smoother(
    model: LinearSDE,
    pf: PFOutput,
    restarts: bool = True,
    n_steps_naive: int = 200,
) -> KoyamaOutput:
    """Koyama's Gaussian projection smoother [Eqs. (30)-(31)], linear case.

    With ``restarts=False`` this is method (ii) (KGPS) of the paper's
    Sec. V-A; ``restarts=True`` is the restarted variant that Sec. V-A notes
    coincides with method (i) (Rem. 1) and does not show.

    Parameters
    ----------
    model : LinearSDE
        Model parameters.
    pf : PFOutput
        Forward projection-filter output (stored anchor trajectory).
    restarts : bool, optional
        If True, integrate interval-by-interval with the anchor
        re-initialized at each measurement discontinuity (correct handling).
        If False, use a single fixed-step RK4 sweep interpolating the stored
        anchor across the jumps without restarting (KGPS).
    n_steps_naive : int, optional
        Number of RK4 steps for KGPS.

    Returns
    -------
    KoyamaOutput
        Smoothing moments on the ``pf.ts`` grids.
    """
    N, n_dense = pf.ts.shape
    m_out = onp.empty((N, n_dense))
    V_out = onp.empty((N, n_dense))

    # terminal condition: theta(t_N) = theta^f(t_N) after the update
    mu_T, V_T = moments_from_theta(pf.theta_f[N])
    mu_T, V_T = float(mu_T), float(V_T)

    if restarts:
        mu, V = mu_T, V_T
        for k in range(N - 1, -1, -1):
            # anchor at the right end of interval k: pre-update predicted moments
            mu_fk, V_fk = moments_from_theta(pf.theta_p[k, -1])
            y0 = jnp.array([float(mu_fk), float(V_fk), mu, V])
            ts = pf.ts[k]
            traj = _solve_interval(GAUSSIAN_AUGMENTED_TERM, y0, ts[0], ts[-1], ts, model, backward=True)
            traj = onp.asarray(traj)
            m_out[k] = traj[:, 2]
            V_out[k] = traj[:, 3]
            mu, V = float(traj[0, 2]), float(traj[0, 3])
        return KoyamaOutput(m=m_out, V=V_out)

    # ---- KGPS: single RK4 sweep across the anchor discontinuities
    ts_flat, mf_flat, Vf_flat = [], [], []
    for k in range(N):
        for j in range(n_dense):
            ts_flat.append(pf.ts[k, j])
            mf, Vf = moments_from_theta(pf.theta_p[k, j])
            mf_flat.append(float(mf))
            Vf_flat.append(float(Vf))
    ts_flat = onp.asarray(ts_flat)
    mf_flat = onp.asarray(mf_flat)
    Vf_flat = onp.asarray(Vf_flat)

    def anchor(t: float) -> tuple[float, float]:
        """Linearly interpolated anchor moments (mu^f(t), V^f(t)).

        Parameters
        ----------
        t : float
            Time; at a measurement time the duplicated grid point makes
            ``onp.interp`` pick one side of the jump arbitrarily.

        Returns
        -------
        mu_f, V_f : float
            Interpolated anchor mean and variance.
        """
        return onp.interp(t, ts_flat, mf_flat), onp.interp(t, ts_flat, Vf_flat)

    def f(t: float, y: Float[onp.ndarray, "2"]) -> Float[onp.ndarray, "2"]:
        """Gaussian backward vector field with the interpolated anchor.

        Parameters
        ----------
        t : float
            Time.
        y : Float[onp.ndarray, "2"]
            Smoothing moments (mu, V).

        Returns
        -------
        dy : Float[onp.ndarray, "2"]
            Time derivative of (mu, V).
        """
        mu_f, V_f = anchor(t)
        dmu, dV = _gaussian_backward_rhs(y[0], y[1], mu_f, V_f, model)
        return onp.array([float(dmu), float(dV)])

    t_hist = [pf.ts[-1, -1]]
    y_hist = [onp.array([mu_T, V_T])]
    h = (pf.ts[-1, -1] - pf.ts[0, 0]) / n_steps_naive
    t, y = t_hist[0], y_hist[0]
    for _ in range(n_steps_naive):
        k1 = f(t, y)
        k2 = f(t - h / 2.0, y - h / 2.0 * k1)
        k3 = f(t - h / 2.0, y - h / 2.0 * k2)
        k4 = f(t - h, y - h * k3)
        y = y - h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t = t - h
        t_hist.append(t)
        y_hist.append(y)
    t_hist = onp.asarray(t_hist)[::-1]
    y_hist = onp.asarray(y_hist)[::-1]
    for k in range(N):
        m_out[k] = onp.interp(pf.ts[k], t_hist, y_hist[:, 0])
        V_out[k] = onp.interp(pf.ts[k], t_hist, y_hist[:, 1])
    return KoyamaOutput(m=m_out, V=V_out)


def koyama_theta_smoother(model: LinearSDE, pf: PFOutput) -> KoyamaThetaOutput:
    """Koyama's generic natural-parameter smoother [Eq. (20)], linear-Gaussian case.

    Restarted natural-parameter variant (not shown in Sec. V).  Integrates the theta-form backward
    ODE interval-by-interval, with the anchor theta^f re-initialized at the
    interval's pre-update value theta_p(t_k^-) and carried backward by
    Eq. (22) within the interval (augmented system, no interpolation of the
    stored trajectory).

    Parameters
    ----------
    model : LinearSDE
        Model parameters.
    pf : PFOutput
        Forward projection-filter output.

    Returns
    -------
    KoyamaThetaOutput
        Smoothing natural-parameter trajectory on the ``pf.ts`` grids.
    """
    N, n_dense = pf.ts.shape
    theta_out = onp.empty((N, n_dense, 2))
    theta = jnp.asarray(pf.theta_f[N])
    for k in range(N - 1, -1, -1):
        theta_fk = jnp.asarray(pf.theta_p[k, -1])
        y0 = jnp.concatenate([theta_fk, theta])
        ts = pf.ts[k]
        traj = _solve_interval(THETA_AUGMENTED_TERM, y0, ts[0], ts[-1], ts, model, backward=True)
        traj = onp.asarray(traj)
        theta_out[k] = traj[:, 2:]
        theta = jnp.asarray(traj[0, 2:])
    return KoyamaThetaOutput(theta=theta_out)


def _kgps_blind_rhs(t: Float[ArrayLike, ""], y: Float[Array, "2"], args: tuple) -> Float[Array, "2"]:
    """``diffrax.ODETerm`` of KGPS, Eq. (27), with the stored trajectory interpolated across t_k.

    Parameters
    ----------
    t : Float[ArrayLike, ""]
        Time.
    y : Float[Array, "2"]
        Smoothing moments (mu, V).
    args : tuple
        ``(model, ts_flat, mf_flat, Vf_flat)``: model parameters and the stored
        filtering trajectory on the concatenated per-interval grids (each t_k
        appears twice, pre- and post-update), linearly interpolated in time.

    Returns
    -------
    dy : Float[Array, "2"]
        Time derivative of (mu, V).
    """
    model, ts_flat, mf_flat, Vf_flat = args
    dmu, dV = _gaussian_backward_rhs(y[0], y[1], jnp.interp(t, ts_flat, mf_flat), jnp.interp(t, ts_flat, Vf_flat), model)
    return jnp.array([dmu, dV])


KGPS_BLIND_TERM = diffrax.ODETerm(_kgps_blind_rhs)


def _stack_dense(pf: PFOutput) -> PyTree[Any]:
    """Stack the per-interval dense interpolants of the forward pass.

    Parameters
    ----------
    pf : PFOutput
        Forward projection-filter output; the entries of ``pf.dense`` share
        their buffer shapes (``max_steps=DENSE_MAX_STEPS``).

    Returns
    -------
    stacked : PyTree[Any]
        A ``diffrax`` dense interpolation whose array leaves carry a leading
        interval axis of length N.
    """
    return jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *[sol.interpolation for sol in pf.dense])


def _kgps_dense_rhs(t: Float[ArrayLike, ""], y: Float[Array, "2"], args: tuple) -> Float[Array, "2"]:
    """``diffrax.ODETerm`` of KGPS, Eq. (27), reading the forward pass's dense output.

    The filtering trajectory is evaluated as in Algorithm 1, from the dense
    output theta_p(t) of the forward pass, here piecewise over all intervals:
    the interval k with t_{k-1} < t <= t_k is used, so at t_k the pre-update
    value theta_p(t_k^-) is returned and just after it the post-update
    branch.  The backward sweep still crosses the measurement times without
    restarting.

    Parameters
    ----------
    t : Float[ArrayLike, ""]
        Time.
    y : Float[Array, "2"]
        Smoothing moments (mu, V).
    args : tuple
        ``(model, edges, stacked)``: model parameters, the measurement times
        (t_0, ..., t_N) and the stacked dense interpolants (``_stack_dense``).

    Returns
    -------
    dy : Float[Array, "2"]
        Time derivative of (mu, V).
    """
    model, edges, stacked = args
    k = jnp.clip(jnp.searchsorted(edges, t, side="left") - 1, 0, edges.shape[0] - 2)
    interp = jax.tree_util.tree_map(lambda leaf: leaf[k], stacked)
    # clamp to the interval: round-off can place a solver stage a few ulps
    # outside it, where the dense-output buffers are padded with inf
    mu_f, V_f = moments_from_theta(interp.evaluate(jnp.clip(t, edges[k], edges[k + 1])))
    dmu, dV = _gaussian_backward_rhs(y[0], y[1], mu_f, V_f, model)
    return jnp.array([dmu, dV])


KGPS_DENSE_TERM = diffrax.ODETerm(_kgps_dense_rhs)


def koyama_gaussian_smoother_dopri(model: LinearSDE, pf: PFOutput, dense: bool = True) -> KoyamaOutput:
    """KGPS (method (ii) of Sec. V-A) integrated by adaptive Dormand--Prince instead of RK4.

    One backward sweep over [t_0, t_N] of Eq. (27) that crosses the
    measurement times without restarting, solved by ``_jitted_solve``
    (Dopri5, rtol 1e-11, atol 1e-13, the tolerances of the proposed
    smoother).  By default the filtering trajectory is read from the forward
    pass's dense output, exactly as Algorithm 1 reads theta_p(t), so the
    two methods differ only in how the measurement times are treated.

    Parameters
    ----------
    model : LinearSDE
        Model parameters.
    pf : PFOutput
        Forward projection-filter output (stored filtering trajectory).
    dense : bool, optional
        If True (default), evaluate the filtering trajectory from
        ``pf.dense``.  If False, linearly interpolate the moments of the
        stored samples ``pf.theta_p`` on ``pf.ts``, as KGPS with RK4 does.

    Returns
    -------
    KoyamaOutput
        Smoothing moments on the ``pf.ts`` grids.
    """
    N, n_dense = pf.ts.shape
    mu_T, V_T = moments_from_theta(pf.theta_f[N])
    grid = onp.concatenate([pf.ts[0, :1]] + [pf.ts[k, 1:] for k in range(N)])
    t0, tN = grid[0], grid[-1]
    if dense:
        edges = onp.concatenate([pf.ts[:, 0], pf.ts[-1:, -1]])
        term, args = KGPS_DENSE_TERM, (model, jnp.asarray(edges), _stack_dense(pf))
    else:
        ts_flat = pf.ts.reshape(-1)
        mf, Vf = jnp.vectorize(lambda a, b: moments_from_theta(jnp.array([a, b])))(pf.theta_p[..., 0], pf.theta_p[..., 1])
        term, args = KGPS_BLIND_TERM, (model, jnp.asarray(ts_flat), jnp.asarray(mf).reshape(-1), jnp.asarray(Vf).reshape(-1))
    ys = onp.asarray(_jitted_solve(term, jnp.array([mu_T, V_T]), tN, t0, -(tN - t0) / 100.0,
                                   jnp.asarray(grid[::-1].copy()), args))[::-1]
    m_out = onp.stack([onp.interp(pf.ts[k], grid, ys[:, 0]) for k in range(N)])
    V_out = onp.stack([onp.interp(pf.ts[k], grid, ys[:, 1]) for k in range(N)])
    return KoyamaOutput(m=m_out, V=V_out)
