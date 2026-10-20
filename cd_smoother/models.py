"""Univariate linear continuous-discrete model and Gaussian natural-parameter maps.

The model is the Ornstein--Uhlenbeck process of the paper's Eq. (24), the
special case f(x, t) = a x, G = sigma, h(x, t) = x of Eqs. (1)--(2),

    dx_t = a x_t dt + sigma dw_t,      y_k = x_{t_k} + xi_k,  xi_k ~ N(0, R),

and the exponential family (21) is the univariate Gaussian with statistics
c(x) = (x, x^2) and natural parameter theta = (m / P, -1 / (2 P)) (Sec. V-A).
"""
import jax

jax.config.update("jax_enable_x64", True)

import chex
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float


@chex.dataclass(frozen=True)
class LinearSDE:
    """Univariate Ornstein--Uhlenbeck model with linear-Gaussian measurements, Eq. (24).

    Attributes
    ----------
    a : float
        Drift coefficient of dx = a x dt + sigma dw.
    sigma : float
        Diffusion coefficient, so that Sigma = G G^T = sigma^2.
    R : float
        Measurement noise variance of y_k = x_{t_k} + xi_k.
    """

    a: float
    sigma: float
    R: float


def transition_params(model: LinearSDE, dt: Float[ArrayLike, ""]) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Exact discretization of the OU process (24) over a time step.

    Parameters
    ----------
    model : LinearSDE
        Model parameters.
    dt : Float[ArrayLike, ""]
        Time step.

    Returns
    -------
    A : Float[Array, ""]
        State transition multiplier exp(a dt).
    Q : Float[Array, ""]
        Exact process noise variance sigma^2 (exp(2 a dt) - 1) / (2 a),
        continuously extended to sigma^2 dt at a = 0.

    Notes
    -----
    The a -> 0 limit is handled with expm1 through the phi-function
    phi(z) = expm1(z) / z, phi(0) = 1, so that Q = sigma^2 dt phi(2 a dt).
    """
    a, sigma = model.a, model.sigma
    A = jnp.exp(a * dt)
    z = 2.0 * a * dt
    phi = jnp.where(jnp.abs(z) < 1e-8, 1.0 + z / 2.0, jnp.expm1(z) / jnp.where(z == 0.0, 1.0, z))
    Q = sigma**2 * dt * phi
    return A, Q


def theta_from_moments(m: Float[ArrayLike, ""], P: Float[ArrayLike, ""]) -> Float[Array, "2"]:
    """Natural parameters of N(m, P) in the family (21) with c(x) = (x, x^2).

    Parameters
    ----------
    m : Float[ArrayLike, ""]
        Mean.
    P : Float[ArrayLike, ""]
        Variance (must be positive).

    Returns
    -------
    theta : Float[Array, "2"]
        Natural parameter (m / P, -1 / (2 P)) (Sec. V-A).
    """
    return jnp.array([m / P, -1.0 / (2.0 * P)])


def moments_from_theta(theta: Float[ArrayLike, "2"]) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Mean and variance of the Gaussian with natural parameter theta.

    Inverse of :func:`theta_from_moments`; the mean is the first component of
    eta(theta) = grad psi(theta) = E_theta[c] (Sec. IV, after Eq. (21)).

    Parameters
    ----------
    theta : Float[ArrayLike, "2"]
        Natural parameter; theta_2 must be negative.

    Returns
    -------
    m : Float[Array, ""]
        Mean -theta_1 / (2 theta_2).
    P : Float[Array, ""]
        Variance -1 / (2 theta_2).
    """
    theta = jnp.asarray(theta)
    P = -1.0 / (2.0 * theta[1])
    m = theta[0] * P
    return m, P


def conjugate_update_vector(model: LinearSDE, y: Float[ArrayLike, ""]) -> Float[Array, "2"]:
    """Additive natural-parameter measurement update L(y) of Definition 1.

    The log-likelihood log p(y | x) = -(y - x)^2 / (2 R) + const is affine in
    c(x) = (x, x^2), i.e. log p(y | x) = l_0(y) + L(y)^T c(x) (Def. 1), with
    L(y) = (y / R, -1 / (2 R)) (Sec. V-A).  The conjugate Bayes update is then
    theta_f(t_k) = theta_p(t_k^-) + L(y_k) (Prop. 1).

    Parameters
    ----------
    model : LinearSDE
        Model parameters (only R is used).
    y : Float[ArrayLike, ""]
        Measurement value.

    Returns
    -------
    L : Float[Array, "2"]
        Additive natural-parameter update.
    """
    return jnp.array([y / model.R, -1.0 / (2.0 * model.R)])
