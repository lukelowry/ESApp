"""
Geographic visualization utilities for power system analysis.

Provides plotting functions for transmission lines, tesselation grids,
vector fields, and geographic boundaries using matplotlib and geopandas.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
from numpy.typing import NDArray

from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, ListedColormap, rgb_to_hsv, hsv_to_rgb
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pandas import DataFrame

__all__ = [
    'format_plot',
    'darker_hsv_colormap',
    'border',
    'plot_lines',
    'plot_mesh',
    'plot_tiles',
    'plot_vecfield',
]

_SHAPES_DIR = Path(__file__).resolve().parent / 'shapes'


def format_plot(
    ax: Axes,
    title: str = 'Chart Title',
    xlabel: str = 'X Axis Label',
    ylabel: str = 'Y Axis Label',
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    grid: bool = True,
    plotarea: str = 'linen',
    spine_color: str = 'black',
    xticksep: float | None = None,
    yticksep: float | None = None,
) -> None:
    """
    Format a matplotlib axes with standard styling.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to format.
    title : str, default 'Chart Title'
        Plot title.
    xlabel, ylabel : str
        Axis labels.
    xlim, ylim : tuple of float, optional
        Axis limits as (min, max).
    grid : bool, default True
        Whether to show grid lines.
    plotarea : str, default 'linen'
        Background face color.
    spine_color : str, default 'black'
        Color for axis spines and ticks.
    xticksep, yticksep : float, optional
        Tick separation for x and y axes.
    """
    ax.set_facecolor(plotarea)
    ax.grid(grid)

    if grid:
        ax.set_axisbelow(True)

    ax.tick_params(color=spine_color, labelcolor=spine_color)
    for spine in ax.spines.values():
        spine.set_edgecolor(spine_color)

    if xlim:
        ax.set_xlim(xlim)
        if xticksep:
            ax.set_xticks(np.arange(*xlim, xticksep))
    if ylim:
        ax.set_ylim(ylim)
        if yticksep:
            ax.set_yticks(np.arange(*ylim, yticksep))

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)


def darker_hsv_colormap(scale_factor: float = 0.5) -> ListedColormap:
    """
    Create a darker version of the HSV colormap.

    Parameters
    ----------
    scale_factor : float, default 0.5
        Factor to scale the value (brightness). 1 means no change,
        0 means complete darkness.

    Returns
    -------
    matplotlib.colors.ListedColormap
        A darker version of the HSV colormap.
    """
    hsv_cmap = plt.cm.hsv(np.linspace(0, 1, 256))[:, :3]
    hsv_colors = rgb_to_hsv(hsv_cmap)

    hsv_colors[:, 2] *= scale_factor
    hsv_colors[:, 2] = np.clip(hsv_colors[:, 2], 0, 1)

    darker_rgb = hsv_to_rgb(hsv_colors)
    return ListedColormap(darker_rgb)


def border(ax: Axes, shape: str = 'Texas') -> None:
    """
    Plot a geographic boundary on a matplotlib axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    shape : str, default 'Texas'
        Name of the shape directory under ``esapp/utils/shapes/``.
    """
    shapepath = _SHAPES_DIR / shape / 'Shape.shp'
    shapeobj = gpd.read_file(shapepath)
    shapeobj.plot(ax=ax, edgecolor='black', facecolor='none')


def plot_lines(ax: Axes, lines: DataFrame, ms: float = 50, lw: float = 1) -> None:
    """
    Draw transmission lines geographically.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    lines : pandas.DataFrame
        DataFrame with 'Longitude', 'Longitude:1', 'Latitude', 'Latitude:1'.
    ms : float, default 50
        Marker size for bus endpoints.
    lw : float, default 1
        Line width for transmission lines.
    """
    cX = lines[['Longitude', 'Longitude:1']].to_numpy()
    cY = lines[['Latitude', 'Latitude:1']].to_numpy()

    for i in range(cX.shape[0]):
        ax.plot(cX[i], cY[i], zorder=4, c='k', linewidth=lw)
        ax.scatter(cX[i], cY[i], c='k', zorder=2, s=ms)


def plot_mesh(
    ax: Axes,
    gt,
    include_lines: bool = True,
    color: str = 'grey',
    tcolor: str = 'red',
    talpha: float = 0.3,
) -> None:
    """
    Plot a GIC tool tesselation grid.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    gt : object
        GIC tool object with ``tile_info``, ``tile_ids``, and ``lines``.
    include_lines : bool, default True
        Whether to overlay transmission lines.
    color : str, default 'grey'
        Grid line color.
    tcolor : str, default 'red'
        Tile face color.
    talpha : float, default 0.3
        Tile transparency.
    """
    if include_lines:
        plot_lines(ax, gt.lines, ms=2)

    X, Y, W = gt.tile_info

    for x in X:
        ax.plot([x, x], [Y.min(), Y.max()], c=color, zorder=1)
    for y in Y:
        ax.plot([X.min(), X.max()], [y, y], c=color, zorder=1)

    tile_ids = gt.tile_ids
    refpnt = np.array([[X.min(), Y.min()]]).T
    tiles_unique = np.unique(tile_ids[:, ~np.isnan(tile_ids[0])], axis=1)
    tile_pos = tiles_unique * W + refpnt

    for tile in tile_pos.T:
        ax.add_patch(Rectangle((tile[0], tile[1]), W, W, facecolor=tcolor, alpha=talpha))

    format_plot(ax, xlabel=r'Longitude ($^\circ$E)', ylabel=r'Latitude ($^\circ$N)',
                title='Geographic Line Plot', plotarea='white', grid=False)


def plot_tiles(
    ax: Axes,
    gt,
    colors: NDArray | None = None,
) -> None:
    """
    Plot colored tiles on a tesselation grid.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    gt : object
        GIC tool object with ``tile_info``.
    colors : np.ndarray, optional
        2D array of tile colors. If None, uses red.
    """
    X, Y, W = gt.tile_info

    for i in np.arange(len(X) - 1):
        for j in np.arange(len(Y) - 1):
            fc = colors[j, i] if colors is not None else 'red'
            ax.add_patch(Rectangle((X[i] * W, Y[j] * W), W, W, facecolor=fc, alpha=0.3))

    format_plot(ax, xlabel=r'Longitude ($^\circ$E)', ylabel=r'Latitude ($^\circ$N)',
                title='Tile Plot', plotarea='white', grid=False)


def plot_vecfield(
    ax: Axes,
    X: NDArray,
    Y: NDArray,
    U: NDArray,
    V: NDArray,
    cmap: ListedColormap | None = None,
    pivot: str = 'mid',
    scale: float = 70,
    width: float = 0.001,
    title: str = '',
) -> ScalarMappable:
    """
    Plot a vector field colored by angle.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    X, Y : np.ndarray
        Coordinates of vector origins.
    U, V : np.ndarray
        Vector components.
    cmap : matplotlib colormap, optional
        Colormap for angle encoding. Defaults to a darker HSV.
    pivot : str, default 'mid'
        Quiver pivot point.
    scale : float, default 70
        Quiver arrow scaling.
    width : float, default 0.001
        Quiver arrow width.
    title : str, default ''
        Plot title.

    Returns
    -------
    matplotlib.cm.ScalarMappable
        Mappable for creating colorbars.
    """
    if cmap is None:
        cmap = darker_hsv_colormap(0.8)

    norm = Normalize(vmin=-np.pi, vmax=np.pi)
    colors = np.arctan2(U, V)
    colors[np.isnan(colors)] = 0

    ax.quiver(X, Y, U, V, colors, norm=norm, pivot=pivot, scale=scale, width=width, cmap=cmap)

    format_plot(ax, xlabel=r'Longitude ($^\circ$E)', ylabel=r'Latitude ($^\circ$N)',
                title=title, plotarea='white', grid=False)

    return ScalarMappable(norm, cmap)
