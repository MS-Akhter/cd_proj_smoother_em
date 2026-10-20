"""Journal-figure plotting configuration (IEEE Control Systems Letters sizing).

Provides LaTeX-font rcParams, figure-size helpers and the categorical color
palette used across all figures.  Figures are drawn at the exact L-CSS column
width (252 pt) so that ``\\includegraphics[width=\\columnwidth]`` places them at
1:1 scale and the 8 pt figure text matches the 8 pt captions.

The palette uses fixed slots of a colorblind-validated categorical scheme
(validated with the dataviz six-check script: lightness band, chroma floor,
adjacent-pair CVD separation >= 8, normal-vision floor; aqua and yellow are
below 3:1 contrast against white, so every series also carries a legend entry,
a distinct line style and a distinct marker, which also keeps the figures
legible in grayscale print).  Colors follow the method, never its rank: a
method that is not plotted leaves its slot empty instead of repainting others.
"""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

#: Fixed categorical assignment (never re-ordered between figures).
COLORS = {
    "proj": "#2a78d6",  # slot 1, blue  — proposed projection smoother (Alg. 1)
    "dopri": "#eb6834",  # slot 2, orange — KGPS integrated by adaptive Dormand--Prince
    "filter": "#1baf7a",  # slot 3, aqua — forward projection filter
    "naive": "#eda100",  # slot 4, yellow — KGPS (Koyama, integrated across t_k)
    "truth": "#333333",  # reference ink, not a series color
    "exact": "#777777",  # exact RTS reference ink
}

#: Line styles / markers as secondary (grayscale/CVD-safe) encoding.
STYLES = {
    "proj": {"color": COLORS["proj"], "linestyle": "-", "marker": "o"},
    "dopri": {"color": COLORS["dopri"], "linestyle": "--", "marker": "s"},
    "filter": {"color": COLORS["filter"], "linestyle": "-.", "marker": "^"},
    "naive": {"color": COLORS["naive"], "linestyle": (0, (1.2, 1.2)), "marker": "v"},
}

PT_PER_INCH = 72.27
SINGLE_COL = 252.0 / PT_PER_INCH  # inches, L-CSS \columnwidth (252 pt)
FULL_WIDTH = 516.0 / PT_PER_INCH  # inches, L-CSS \textwidth (516 pt)


def use_journal_style() -> None:
    """Apply LaTeX-serif rcParams sized for 1:1 placement in an L-CSS column.

    Returns
    -------
    None
    """
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.handlelength": 2.0,
            "legend.handletextpad": 0.4,
            "legend.columnspacing": 1.0,
            "legend.borderaxespad": 0.2,
            "axes.grid": True,
            "grid.color": "#dddddd",
            "grid.linewidth": 0.4,
            "axes.axisbelow": True,
            "axes.linewidth": 0.5,
            "axes.labelpad": 2.0,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.major.pad": 2.0,
            "ytick.major.pad": 2.0,
            "lines.linewidth": 0.8,
            "lines.markeredgewidth": 0.7,
            "lines.markersize": 3.5,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "#cccccc",
            "figure.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.01,
        }
    )


def new_figure(
    width: float = SINGLE_COL, height: float = 1.5, ncols: int = 1, nrows: int = 1, **kwargs
) -> tuple[Figure, "Axes | list[Axes]"]:
    """Create a journal-sized figure with constrained layout.

    Parameters
    ----------
    width, height : float, optional
        Figure size in inches.
    ncols, nrows : int, optional
        Subplot grid.
    **kwargs
        Forwarded to ``matplotlib.pyplot.subplots``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    axes : matplotlib.axes.Axes or array of Axes
        The subplot axes.
    """
    kwargs.setdefault("layout", "constrained")
    return plt.subplots(nrows, ncols, figsize=(width, height), **kwargs)
