"""End-to-end numerical validation of the continuous-discrete projection smoother.

Runs, in order (paper, Sec. V):

1. Experiment E1 (single-realization correctness of Algorithm 1 against the
   exact Kalman/RTS reference and Koyama's projection smoother, Sec. V-B),
2. Experiment E2 (Monte Carlo sweep over the measurement interval, Sec. V-C),
3. Figure generation into ``../Journal_Paper/Template_L_CSS/figures/``.

Usage::

    .venv/bin/python numerical_implementation.py

All simulation outputs are stored as HDF5 under ``simulation_result/``.
"""
import jax

jax.config.update("jax_enable_x64", True)

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "experiments"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import exp1_correctness
import exp2_interval_sweep
import plot_results


def main() -> None:
    """Run the full experiment pipeline and generate the journal figures.

    Returns
    -------
    None
    """
    e1_dir, _ = exp1_correctness.run()
    e2_dir = exp2_interval_sweep.run()
    plot_results.run(e1_dir=e1_dir, e2_dir=e2_dir)


if __name__ == "__main__":
    main()
