import hist
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep
import pandas as pd

from analysis.utils.geometry import Geometry
from analysis.utils.truth import TruthParticle


def cleanup_dct(dct: dict, bad_keys: list[str] = ["xlim", "ylim", "zlim"]) -> dict:
    return {k: v for k, v in dct.items() if k not in bad_keys}


def scatter_plot(
    df: pd.DataFrame,
    xvar: str,
    yvar: str,
    xlabel: str,
    ylabel: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    ax: plt.Axes | None = None,
    **kwargs,
) -> None:
    if ax is None:
        fig, _ax = plt.subplots(figsize=(6, 4))
    else:
        _ax = ax

    plot_dict = {"color": "black", "s": 1, "marker": "."}
    kwargs = cleanup_dct(kwargs)
    plot_dict.update(kwargs)
    _ax.scatter(df[xvar], df[yvar], **plot_dict)

    _ax.set_xlabel(xlabel)
    _ax.set_ylabel(ylabel)
    _ax.set_xlim(xlim)
    _ax.set_ylim(ylim)
    if ax is None:
        plt.tight_layout()
        plt.show()


def xy_scatter_plot(
    df: pd.DataFrame,
    truth: TruthParticle,
    geo: Geometry,
    range: float | None = None,
    xvar: str = "x",
    yvar: str = "y",
    xlabel: str = r"$x$ [mm]",
    ylabel: str = r"$y$ [mm]",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    **kwargs,
) -> None:
    if xlim is None:
        if range is None:
            xlim = (geo.xmin, geo.xmax)
        else:
            xlim = (truth.x - range, truth.x + range)
    if ylim is None:
        if range is None:
            ylim = (geo.ymin, geo.ymax)
        else:
            ylim = (truth.y - range, truth.y + range)
    return scatter_plot(
        df=df,
        xvar=xvar,
        yvar=yvar,
        xlabel=xlabel,
        ylabel=ylabel,
        xlim=xlim,
        ylim=ylim,
        **kwargs,
    )


def zx_scatter_plot(
    df: pd.DataFrame,
    truth: TruthParticle,
    geo: Geometry,
    range: float | None = None,
    zvar: str = "z",
    xvar: str = "x",
    zlabel: str = r"$z$ [mm]",
    xlabel: str = r"$x$ [mm]",
    zlim: tuple[float, float] | None = None,
    xlim: tuple[float, float] | None = None,
    **kwargs,
) -> None:
    if zlim is None:
        zlim = (geo.zmin, geo.zmax)
    if xlim is None:
        if range is None:
            xlim = (geo.xmin, geo.xmax)
        else:
            xlim = (truth.x - range, truth.x + range)
    return scatter_plot(
        df=df,
        xvar=zvar,
        yvar=xvar,
        xlabel=zlabel,
        ylabel=xlabel,
        xlim=zlim,
        ylim=xlim,
        **kwargs,
    )


def zy_scatter_plot(
    df: pd.DataFrame,
    truth: TruthParticle,
    geo: Geometry,
    range: float | None = None,
    zvar: str = "z",
    yvar: str = "y",
    zlabel: str = r"$z$ [mm]",
    ylabel: str = r"$y$ [mm]",
    zlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    **kwargs,
) -> None:
    if zlim is None:
        zlim = (geo.zmin, geo.zmax)
    if ylim is None:
        if range is None:
            ylim = (geo.ymin, geo.ymax)
        else:
            ylim = (truth.y - range, truth.y + range)
    return scatter_plot(
        df=df,
        xvar=zvar,
        yvar=yvar,
        xlabel=zlabel,
        ylabel=ylabel,
        xlim=zlim,
        ylim=ylim,
        **kwargs,
    )


def show_scatter_plots(
    df: pd.DataFrame,
    truth: TruthParticle,
    geo: Geometry,
    range: float | None = None,
    axs: list[plt.Axes] | None = None,
    **kwargs,
):
    if axs is None:
        fig, axs = plt.subplots(1, 3, figsize=(6 * 3, 4))
        fig.suptitle(truth.get_latex_title())
    zx_scatter_plot(
        df=df,
        truth=truth,
        geo=geo,
        range=range,
        ax=axs[0],
        **kwargs,
    )
    zy_scatter_plot(
        df=df,
        truth=truth,
        geo=geo,
        range=range,
        ax=axs[1],
        **kwargs,
    )
    xy_scatter_plot(
        df=df,
        truth=truth,
        geo=geo,
        range=range,
        ax=axs[2],
        **kwargs,
    )


def hist_plot(
    df: pd.DataFrame,
    xvar: str,
    yvar: str,
    xlabel: str,
    ylabel: str,
    nxbins: int,
    nybins: int,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    weight: str | float = 1.0,
    ax: plt.Axes | None = None,
):
    if ax is None:
        fig, _ax = plt.subplots(figsize=(6, 4))
    else:
        _ax = ax
    if isinstance(weight, str):
        _weight = df[weight]
    else:
        _weight = weight

    h = hist.Hist(
        hist.axis.Regular(nxbins, xlim[0], xlim[1]),
        hist.axis.Regular(nybins, ylim[0], ylim[1]),
        storage=hist.storage.Weight(),
    )
    h.fill(df[xvar], df[yvar], weight=_weight)

    mplhep.hist2dplot(
        h, cmap="viridis", norm=mpl.colors.LogNorm(), cbar=False, ax=_ax, flow=False
    )
    _ax.set_xlabel(xlabel)
    _ax.set_ylabel(ylabel)
    if ax is None:
        plt.tight_layout()
        plt.show()


def xy_hist_plot(
    df: pd.DataFrame,
    truth: TruthParticle,
    geo: Geometry,
    range: float | None = None,
    xvar: str = "x",
    yvar: str = "y",
    xlabel: str = r"$x$ [mm]",
    ylabel: str = r"$y$ [mm]",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    ax: plt.Axes | None = None,
    **kwargs,
):
    if xlim is None:
        if range is None:
            xlim = (geo.xmin, geo.xmax)
        else:
            xlim = (truth.x - range, truth.x + range)
    if ylim is None:
        if range is None:
            ylim = (geo.ymin, geo.ymax)
        else:
            ylim = (truth.y - range, truth.y + range)
    nx_bins = int((ylim[1] - ylim[0]) // geo.pixel_y_size)
    ny_bins = int((ylim[1] - ylim[0]) // geo.pixel_y_size)
    return hist_plot(
        df=df,
        xvar=xvar,
        yvar=yvar,
        xlabel=xlabel,
        ylabel=ylabel,
        nxbins=nx_bins,
        nybins=ny_bins,
        xlim=xlim,
        ylim=ylim,
        ax=ax,
    )


def zx_hist_plot(
    df: pd.DataFrame,
    truth: TruthParticle,
    geo: Geometry,
    range: float | None = None,
    zvar: str = "z",
    xvar: str = "x",
    zlabel: str = r"$z$ [mm]",
    xlabel: str = r"$x$ [mm]",
    zlim: tuple[float, float] | None = None,
    xlim: tuple[float, float] | None = None,
    ax: plt.Axes | None = None,
    **kwargs,
):
    if zlim is None:
        zlim = (geo.zmin, geo.zmax)
    if xlim is None:
        if range is None:
            xlim = (geo.xmin, geo.xmax)
        else:
            xlim = (truth.x - range, truth.x + range)
    nz_bins = len(
        geo.pixel_zpos[(geo.pixel_zpos >= zlim[0]) & (geo.pixel_zpos < zlim[1])]
    )
    nx_bins = int((xlim[1] - xlim[0]) // geo.pixel_x_size)
    return hist_plot(
        df=df,
        xvar=zvar,
        yvar=xvar,
        xlabel=zlabel,
        ylabel=xlabel,
        nxbins=nz_bins,
        nybins=nx_bins,
        xlim=zlim,
        ylim=xlim,
        ax=ax,
    )


def zy_hist_plot(
    df: pd.DataFrame,
    truth: TruthParticle,
    geo: Geometry,
    range: float | None = None,
    zvar: str = "z",
    yvar: str = "y",
    zlabel: str = r"$z$ [mm]",
    ylabel: str = r"$y$ [mm]",
    zlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    ax: plt.Axes | None = None,
    **kwargs,
):
    if zlim is None:
        zlim = (geo.zmin, geo.zmax)
    if ylim is None:
        if range is None:
            ylim = (geo.ymin, geo.ymax)
        else:
            ylim = (truth.y - range, truth.y + range)
    nz_bins = len(
        geo.pixel_zpos[(geo.pixel_zpos >= zlim[0]) & (geo.pixel_zpos < zlim[1])]
    )
    ny_bins = int((ylim[1] - ylim[0]) // geo.pixel_y_size)
    return hist_plot(
        df=df,
        xvar=zvar,
        yvar=yvar,
        xlabel=zlabel,
        ylabel=ylabel,
        nxbins=nz_bins,
        nybins=ny_bins,
        xlim=zlim,
        ylim=ylim,
        ax=ax,
    )


def show_hist_plots(
    df: pd.DataFrame,
    truth: TruthParticle,
    geo: Geometry,
    range: float | None = None,
    **kwargs,
):
    fig, axs = plt.subplots(1, 3, figsize=(6 * 3, 4))
    fig.suptitle(truth.get_latex_title())
    zx_hist_plot(
        df=df,
        truth=truth,
        geo=geo,
        range=range,
        ax=axs[0],
        **kwargs,
    )
    zy_hist_plot(
        df=df,
        truth=truth,
        geo=geo,
        range=range,
        ax=axs[1],
        **kwargs,
    )
    xy_hist_plot(
        df=df,
        truth=truth,
        geo=geo,
        range=range,
        ax=axs[2],
        **kwargs,
    )
    plt.tight_layout()
    plt.show()
