"""Continuous-discrete projection smoother on exponential family manifolds.

Numerical companion of the paper draft ``Continuous-Discrete Projection
Smoothing on Exponential Family Manifolds`` (Akhter & Emzir),
``Journal_Paper/Template_L_CSS/LCSS_Latex_template.tex``.  Univariate
linear-Gaussian validation against the projection smoother of Koyama (2018)
and the exact Kalman/RTS reference.

Equation numbers quoted in the docstrings refer to the compiled paper:

==========  ============================  =====================================
Number      LaTeX label                   Content
==========  ============================  =====================================
(1)--(2)    eq:sde, eq:measurement        continuous-discrete model
(3)         eq:forward-op                 forward operator L^*
(4)         eq:backward-op                generator L
(8)         eq:fpe                        Fokker--Planck equation for p_p
(17)        eq:smoothing-pde              backward smoothing PDE
(21)        eq:exp-family                 exponential family p(x; theta)
(22)        eq:proj-filter                projection-filter ODE for theta_p
(23)        eq:projected-smoother         projected smoother ODE for theta_s
(24)        eq:ou                         Ornstein--Uhlenbeck test model
(25)        eq:ou-fwd                     Eq. (22) for the OU model
(26)        eq:ou-bwd                     Eq. (23) for the OU model
(27)        eq:ou-rts                     Eq. (26) in moments = Koyama Eqs. (30)-(31)
Def. 1      def:conjugate                 conjugate observation model, L(y)
Prop. 1     prop:conjugate-jump           theta_f = theta_p + L(y), theta_s continuous
Thm. 2      thm:projected-smoother        projection of (17) onto the family
Rem. 1      rem:koyama-coincidence        relation to Koyama's Eq. (20)
Rem. 2      rem:exactness                 linear-Gaussian exactness
Alg. 1      alg:projection-smoother       forward and backward passes
Sec. V      sec:numerics                  numerical experiments
Figs. 1-4   fig:e1-*, fig:e2-*            E1 / E2 figures
==========  ============================  =====================================

The closed-form OU tilting ODE used by :func:`cd_smoother.projection_smoother.tilting_smoother`
(lambda = theta_s - theta_p, i.e. Eq. (26) minus Eq. (25)) is not printed in the
current draft; it serves as an independent cross-check of Algorithm 1.
"""
import jax

jax.config.update("jax_enable_x64", True)

__version__ = "0.1.0"
