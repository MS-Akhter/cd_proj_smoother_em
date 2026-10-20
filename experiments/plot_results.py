"""Generate the journal figures (paper, Figs. 1-4 of Sec. V) from the E1/E2 HDF5 outputs.

Figures are written as PDF into ``../Journal_Paper/Template_L_CSS/figures/``
(the directory used by ``\\includegraphics`` in the paper), drawn at the
L-CSS column width (252 pt) so that ``width=\\columnwidth`` places them at 1:1
scale.  Plotted methods: the forward projection filter, the proposed
projection smoother (Algorithm 1) and Koyama's Gaussian projection smoother
(KGPS, method (ii) of Sec. V-A, integrated across the measurement times) with
two integrators, fixed-step RK4 as in the paper ("KGPS (RK4)") and adaptive
Dormand--Prince with the tolerances of the proposed smoother, reading the
forward pass's dense output as Algorithm 1 does ("KGPS (Dormand--Prince)"); the restarted Koyama smoother is computed and stored by
the experiments but not plotted.
"""
import pathlib
import sys
from typing import Optional, Union

import h5py
import numpy as onp
from jaxtyping import Float, Int, Shaped

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from cd_smoother.plotting import COLORS, SINGLE_COL, STYLES, new_figure, use_journal_style

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG_DIR = ROOT.parent / "Journal_Paper" / "Template_L_CSS" / "figures"

#: Series labels, short enough for a one-row legend above a single-column figure.
LABELS = {
    "proj": r"Proposed (Alg.~1)",
    "naive": r"KGPS (RK4)",
    "dopri": r"KGPS (Dormand--Prince)",
    "filter": r"Filter",
}

#: Legend order shared by all figures (methods absent from a figure are skipped).
LEGEND_ORDER = ("filter", "proj", "naive", "dopri")

#: Lower plotting floor for absolute errors on log axes (exact zeros are clipped to it).
ERR_FLOOR = 1e-16

Results = dict[str, Shaped[onp.ndarray, "..."]]


def _top_legend(fig: plt.Figure, ax: plt.Axes, **kwargs) -> None:
    """Frameless legend above the axes, in the shared ``LEGEND_ORDER``.

    One row by default; pass ``ncols`` to wrap entries that would exceed the
    column width.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure (constrained layout) receiving the legend.
    ax : matplotlib.axes.Axes
        Axes whose labelled artists populate the legend.
    **kwargs
        Forwarded to ``Figure.legend`` (``ncols`` defaults to the entry count).

    Returns
    -------
    None
    """
    handles, labels = ax.get_legend_handles_labels()
    rank = {LABELS[k]: i for i, k in enumerate(LEGEND_ORDER)}
    order = sorted(range(len(labels)), key=lambda i: rank.get(labels[i], -1))
    kwargs.setdefault("ncols", len(labels))
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="outside upper center", frameon=False, **kwargs)


def latest_run(case: str) -> pathlib.Path:
    """Return the newest ``simulation_result`` directory for `case`.

    Parameters
    ----------
    case : str
        Case prefix, e.g. ``"E1_CORRECTNESS"``.

    Returns
    -------
    pathlib.Path
        Most recently modified run directory (the ``dd_mm_YYYY`` timestamps
        in the names do not sort chronologically).

    Raises
    ------
    FileNotFoundError
        If no run directory exists for `case`.
    """
    runs = sorted((ROOT / "simulation_result").glob(f"{case}_*"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"no run directory for case {case}")
    return runs[-1]


def load(path: Union[str, pathlib.Path]) -> Results:
    """Load all datasets of an HDF5 file into a dict of numpy arrays.

    Parameters
    ----------
    path : str or pathlib.Path
        HDF5 file.

    Returns
    -------
    dict[str, Shaped[onp.ndarray, "..."]]
        Dataset contents keyed by dataset path.
    """
    out = {}
    with h5py.File(path, "r") as f:

        def visit(name: str, obj: Union[h5py.Dataset, h5py.Group]) -> None:
            """Copy `obj` into `out` if it is a dataset.

            Parameters
            ----------
            name : str
                Dataset path.
            obj : h5py.Dataset or h5py.Group
                Visited HDF5 object.

            Returns
            -------
            None
            """
            if isinstance(obj, h5py.Dataset):
                out[name] = obj[()]

        f.visititems(visit)
    return out


def _log_axis(ax: plt.Axes, axis: str = "y", numticks: int = 5) -> None:
    """Sparse decade ticks on a log axis (labels at most `numticks` decades).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to modify.
    axis : str, optional
        ``"x"`` or ``"y"``.
    numticks : int, optional
        Maximum number of labelled decades.

    Returns
    -------
    None
    """
    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_locator(LogLocator(base=10.0, numticks=numticks))
    target.set_minor_locator(LogLocator(base=10.0, subs="auto", numticks=100))
    target.set_minor_formatter(NullFormatter())


def _with_updates(
    grid: Float[onp.ndarray, " n_grid"],
    meas_idx: Int[onp.ndarray, " N"],
    m: Float[onp.ndarray, " n_grid"],
    P: Float[onp.ndarray, " n_grid"],
    m_post: Float[onp.ndarray, " N"],
    P_post: Float[onp.ndarray, " N"],
) -> tuple[Float[onp.ndarray, " n_grid+N"], Float[onp.ndarray, " n_grid+N"], Float[onp.ndarray, " n_grid+N"]]:
    """Filter trajectory with both one-sided values at every measurement time.

    On the flattened grid the filter is stored with its pre-update value
    theta_p(t_k^-) at each t_k; the post-update value theta_f(t_k) (Prop. 1)
    is inserted right after it at the same time, so that the measurement
    update appears as a vertical jump and the curve ends at theta_f(t_N).

    Parameters
    ----------
    grid : Float[onp.ndarray, " n_grid"]
        Evaluation grid.
    meas_idx : Int[onp.ndarray, " N"]
        Indices of the measurement times in `grid`.
    m, P : Float[onp.ndarray, " n_grid"]
        Filter mean and variance on `grid` (pre-update values at t_k).
    m_post, P_post : Float[onp.ndarray, " N"]
        Post-update mean and variance at t_1, ..., t_N.

    Returns
    -------
    t, m_out, P_out : Float[onp.ndarray, " n_grid+N"]
        Times (each t_k repeated) and the corresponding moments.
    """
    post = {int(i): k for k, i in enumerate(meas_idx)}
    t_out, m_out, P_out = [], [], []
    for i, t in enumerate(grid):
        t_out.append(t)
        m_out.append(m[i])
        P_out.append(P[i])
        if i in post:
            t_out.append(t)
            m_out.append(m_post[post[i]])
            P_out.append(P_post[post[i]])
    return onp.asarray(t_out), onp.asarray(m_out), onp.asarray(P_out)


def fig_e1_trajectory(d: Results) -> None:
    """Fig. 1 (E1): truth, measurements, filter and smoother means with 95% bands.

    Parameters
    ----------
    d : dict[str, Shaped[onp.ndarray, "..."]]
        E1 results (see ``exp1_correctness.run``).

    Returns
    -------
    None
    """
    fig, ax = new_figure(width=SINGLE_COL, height=1.5)
    grid = d["grid"]
    ax.plot(grid, d["x_true"], color=COLORS["truth"], linewidth=0.5, label="True state")
    ax.plot(grid[d["meas_idx"]], d["ys"], linestyle="none", marker="x", markersize=3.5, markeredgewidth=0.7,
            color=COLORS["truth"], label="Measurements")
    t_f, m_f, P_f = _with_updates(grid, d["meas_idx"], d["m_filter"], d["P_filter"],
                                  d["m_filter_post"], d["P_filter_post"])
    for key, t, m, P in (("filter", t_f, m_f, P_f), ("proj", grid, d["m_proj"], d["P_proj"])):
        st = STYLES[key]
        ax.plot(t, m, color=st["color"], linestyle=st["linestyle"], label=LABELS[key])
        ax.fill_between(t, m - 1.96 * onp.sqrt(P), m + 1.96 * onp.sqrt(P),
                        color=st["color"], alpha=0.15, linewidth=0)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$x_t$")
    ax.set_xlim(grid[0], grid[-1] + 0.015 * (grid[-1] - grid[0]))  # margin: keeps the update at t_N visible
    _top_legend(fig, ax, handlelength=1.6, columnspacing=0.9)
    fig.savefig(FIG_DIR / "fig_e1_trajectory.pdf")
    plt.close(fig)


def fig_e1_trajectory_kgps(d: Results) -> None:
    """Alternative Fig. 1 (E1): proposed smoother and KGPS, means with 95% bands.

    Same realization and layout as :func:`fig_e1_trajectory`, with KGPS
    (method (ii) of Sec. V-A) in place of the filter. The two smoothers differ
    by at most about 0.04 in the mean, so the band of the proposed smoother is
    shaded and the mean and band edges of KGPS are drawn as dotted lines.
    Not used in the paper.

    Parameters
    ----------
    d : dict[str, Shaped[onp.ndarray, "..."]]
        E1 results (see ``exp1_correctness.run``).

    Returns
    -------
    None
    """
    fig, ax = new_figure(width=SINGLE_COL, height=1.6)
    grid = d["grid"]
    ax.plot(grid, d["x_true"], color=COLORS["truth"], linewidth=0.5, label="True state")
    ax.plot(grid[d["meas_idx"]], d["ys"], linestyle="none", marker="x", markersize=3.5, markeredgewidth=0.7,
            color=COLORS["truth"], label="Measurements")
    st = STYLES["proj"]
    m, P = d["m_proj"], d["P_proj"]
    ax.plot(grid, m, color=st["color"], linestyle=st["linestyle"], label=LABELS["proj"])
    ax.fill_between(grid, m - 1.96 * onp.sqrt(P), m + 1.96 * onp.sqrt(P), color=st["color"], alpha=0.15, linewidth=0)
    st = STYLES["naive"]
    m, P = d["m_koyama_naive"], d["P_koyama_naive"]
    ax.plot(grid, m, color=st["color"], linestyle=st["linestyle"], label=LABELS["naive"])
    for edge in (m - 1.96 * onp.sqrt(P), m + 1.96 * onp.sqrt(P)):
        ax.plot(grid, edge, color=st["color"], linestyle=st["linestyle"], linewidth=0.5)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$x_t$")
    ax.set_xlim(grid[0], grid[-1] + 0.015 * (grid[-1] - grid[0]))
    _top_legend(fig, ax, ncols=2, handlelength=1.6, columnspacing=1.2)
    fig.savefig(FIG_DIR / "fig_e1_trajectory_kgps.pdf")
    plt.close(fig)


def fig_e1_moment_error(d: Results) -> None:
    """Fig. 2 (E1): absolute moment deviations from the exact dense RTS reference.

    Left panel: smoothing mean; right panel: smoothing variance (shared
    logarithmic y-axis).

    Parameters
    ----------
    d : dict[str, Shaped[onp.ndarray, "..."]]
        E1 results (see ``exp1_correctness.run``).

    Returns
    -------
    None
    """
    fig, axes = new_figure(width=SINGLE_COL, height=1.4, ncols=2, sharex=True, sharey=True)
    grid = d["grid"]
    series = [
        ("naive", d["m_koyama_naive"], d["P_koyama_naive"]),
        ("dopri", d["m_koyama_dopri"], d["P_koyama_dopri"]),
        ("proj", d["m_proj"], d["P_proj"]),
    ]
    for key, m, P in series:
        st = STYLES[key]
        for ax, err in zip(axes, (onp.abs(m - d["m_rts"]), onp.abs(P - d["P_rts"]))):
            ax.semilogy(grid, onp.maximum(err, ERR_FLOOR), color=st["color"], linestyle=st["linestyle"],
                        label=LABELS[key])
    axes[0].set_title(r"mean, $|m_s - m_s^{\mathrm{RTS}}|$", pad=3)
    axes[1].set_title(r"variance, $|P_s - P_s^{\mathrm{RTS}}|$", pad=3)
    axes[0].set_ylabel(r"absolute deviation")
    for ax in axes:
        ax.set_xlabel(r"$t$")
        ax.set_xlim(grid[0], grid[-1])
        ax.set_ylim(bottom=ERR_FLOOR)
        _log_axis(ax, "y", numticks=5)
    _top_legend(fig, axes[0], handlelength=1.3, columnspacing=0.6, handletextpad=0.3)
    fig.savefig(FIG_DIR / "fig_e1_moment_error.pdf")
    plt.close(fig)


def fig_e2_rmse(d: Results) -> None:
    """Fig. 3 (E2): RMSE against the true state versus the measurement interval.

    Mean and 95% confidence interval (1.96 standard errors) over the Monte
    Carlo realizations.

    Parameters
    ----------
    d : dict[str, Shaped[onp.ndarray, "..."]]
        E2 results (see ``exp2_interval_sweep.run``).

    Returns
    -------
    None
    """
    fig, ax = new_figure(width=SINGLE_COL, height=1.6)
    deltas = d["deltas"]
    for key in ("filter", "naive", "dopri", "proj"):
        st = STYLES[key]
        vals = d[f"rmse_{key}"]
        mean = vals.mean(axis=1)
        se = vals.std(axis=1) / onp.sqrt(vals.shape[1])
        ax.errorbar(deltas, mean, yerr=1.96 * se, color=st["color"], linestyle=st["linestyle"],
                    marker=st["marker"], capsize=2, elinewidth=0.6, label=LABELS[key])
    ax.set_xlabel(r"Measurement interval $\Delta$")
    ax.set_ylabel(r"RMSE")
    ax.set_xscale("log")
    ax.set_xticks(deltas, [f"{v:g}" for v in deltas])
    ax.minorticks_off()
    _top_legend(fig, ax, ncols=2, handlelength=1.5, columnspacing=1.2)
    fig.savefig(FIG_DIR / "fig_e2_rmse.pdf")
    plt.close(fig)


def fig_e2_moment_error(d: Results) -> None:
    """Fig. 4 (E2): worst-case moment deviation from the exact RTS versus Delta.

    Median and 10-90% band over the Monte Carlo realizations of
    max_t max(|m_s - m_s^RTS|, |P_s - P_s^RTS|).

    Parameters
    ----------
    d : dict[str, Shaped[onp.ndarray, "..."]]
        E2 results (see ``exp2_interval_sweep.run``).

    Returns
    -------
    None
    """
    fig, ax = new_figure(width=SINGLE_COL, height=1.4)
    deltas = d["deltas"]
    for key in ("naive", "dopri", "proj"):
        st = STYLES[key]
        vals = onp.maximum(d[f"dev_m_{key}"], d[f"dev_P_{key}"])
        med = onp.median(vals, axis=1)
        lo, hi = onp.quantile(vals, [0.1, 0.9], axis=1)
        ax.fill_between(deltas, lo, hi, color=st["color"], alpha=0.15, linewidth=0)
        ax.plot(deltas, med, color=st["color"], linestyle=st["linestyle"], marker=st["marker"],
                label=LABELS[key])
    ax.set_xlabel(r"Measurement interval $\Delta$")
    ax.set_ylabel(r"max.\ deviation")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(deltas, [f"{v:g}" for v in deltas])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    _log_axis(ax, "y", numticks=4)
    _top_legend(fig, ax, handlelength=1.3, columnspacing=0.6, handletextpad=0.3)
    fig.savefig(FIG_DIR / "fig_e2_moment_error.pdf")
    plt.close(fig)


def run(e1_dir: Optional[Union[str, pathlib.Path]] = None, e2_dir: Optional[Union[str, pathlib.Path]] = None) -> None:
    """Produce all figures from the latest (or given) run directories.

    Parameters
    ----------
    e1_dir, e2_dir : str or pathlib.Path, optional
        Run directories of E1 and E2; default to the newest ones.

    Returns
    -------
    None
    """
    use_journal_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    d1 = load((pathlib.Path(e1_dir) if e1_dir else latest_run("E1_CORRECTNESS")) / "variables.hdf")
    d2 = load((pathlib.Path(e2_dir) if e2_dir else latest_run("E2_INTERVAL_SWEEP")) / "variables.hdf")
    fig_e1_trajectory(d1)
    fig_e1_trajectory_kgps(d1)
    fig_e1_moment_error(d1)
    fig_e2_rmse(d2)
    fig_e2_moment_error(d2)
    print(f"[plots] figures written to {FIG_DIR}")


if __name__ == "__main__":
    run()
