import matplotlib.pyplot as plt
from cycler import Cycler

from analysis.utils.colors import TangoColors


def setup(color_cycler: Cycler = TangoColors.cycler, errorbar_size: int = 2) -> None:
    """Setup matplotlib font size, error capsize, legend and colors"""
    plt.rcParams.update(
        {
            "font.size": 13,
            "errorbar.capsize": errorbar_size,
            "legend.frameon": False,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "DejaVu Sans"],
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}",
        }
    )
    plt.rc("axes", prop_cycle=color_cycler)


def add_legend(ax: plt.Axes, outside: bool = False, **kwargs) -> None:
    """Plot legend containing only unique labels"""
    if kwargs is None:
        kwargs = {}
    if outside:
        kwargs.update({"loc": "center left", "bbox_to_anchor": (1, 0.5)})
    handles, labels = ax.get_legend_handles_labels()
    unique = [
        (h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]
    ]
    ax.legend(*zip(*unique), **kwargs)
