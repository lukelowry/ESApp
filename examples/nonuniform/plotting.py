"""
Plotting helpers for non-uniform GIC analysis.

Provides standardized functions for visualizing gridded electric fields
and transformer GIC results on geographic maps.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import griddata

from examples.map import format_plot, border, plot_lines

__all__ = [
    'plot_efield',
    'plot_gic_map',
    'plot_gic_heatmap',
]


def plot_efield(ax, lons, lats, Ex, Ey, shape=None, lines=None,
                cmap='viridis', vmax=None, title=None, **fmt_kw):
    """Plot a gridded electric field with one arrow per cell.

    Renders the field magnitude as a filled heatmap and overlays a
    quiver arrow at every cell center.  Arrow length is proportional
    to field magnitude so both direction and strength are visible.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    lons, lats : np.ndarray
        1-D grid node coordinates (length nx, ny).
    Ex, Ey : np.ndarray
        Electric field components on the (ny, nx) node grid.
    shape : str or None
        Border shape name passed to ``border()``.
    lines : DataFrame or None
        Branch coordinate DataFrame for ``plot_lines()``.
    cmap : str
        Colormap for the magnitude heatmap and arrows.
    vmax : float or None
        Upper limit for the color scale.  If None, uses data max.
    title : str or None
        Axes title.
    **fmt_kw
        Extra keyword arguments forwarded to ``format_plot()``.

    Returns
    -------
    im : QuadMesh
        The pcolormesh artist (for external colorbar creation).
    """
    LON, LAT = np.meshgrid(lons, lats)
    mag = np.sqrt(Ex**2 + Ey**2)
    if vmax is None:
        vmax = float(np.nanmax(mag))

    # Magnitude heatmap at nodes
    im = ax.pcolormesh(LON, LAT, mag, cmap=cmap, shading='auto',
                       vmin=0, vmax=vmax)

    # Cell-center coordinates and cell-average field
    lon_c = 0.5 * (lons[:-1] + lons[1:])
    lat_c = 0.5 * (lats[:-1] + lats[1:])
    LONc, LATc = np.meshgrid(lon_c, lat_c)

    Ex_c = 0.25 * (Ex[:-1, :-1] + Ex[:-1, 1:] + Ex[1:, :-1] + Ex[1:, 1:])
    Ey_c = 0.25 * (Ey[:-1, :-1] + Ey[:-1, 1:] + Ey[1:, :-1] + Ey[1:, 1:])
    mag_c = np.sqrt(Ex_c**2 + Ey_c**2)

    # Quiver: arrows proportional to magnitude, colored by magnitude
    ax.quiver(LONc, LATc, Ex_c, Ey_c, mag_c,
              cmap=cmap, clim=(0, vmax),
              scale_units='xy', angles='xy',
              scale=vmax / (0.8 * (lons[1] - lons[0])),
              width=0.003, headwidth=3, headlength=3.5,
              linewidth=0.3, edgecolors='k', zorder=5)

    if shape is not None:
        border(ax, shape)
    if lines is not None:
        plot_lines(ax, lines, ms=1.5, lw=0.3)

    ax.set_xlim(lons[0], lons[-1])
    ax.set_ylim(lats[0], lats[-1])
    ax.set_aspect('equal')

    defaults = dict(plotarea='white', grid=False,
                    titlesize=11, labelsize=9, ticksize=8)
    defaults.update(fmt_kw)
    if title is not None:
        defaults['title'] = title
    format_plot(ax, xlabel=r'Longitude ($^\circ$E)',
                ylabel=r'Latitude ($^\circ$N)', **defaults)
    return im


def plot_gic_map(ax, lons, lats, Ex, Ey, xf_lons, xf_lats, gic,
                 shape=None, lines=None, cmap_field='viridis',
                 cmap_gic='YlOrRd', vmax_field=None, title=None,
                 **fmt_kw):
    """Plot transformer |GIC| bubbles over an E-field background.

    GIC magnitudes are shown via both bubble size and colour on a
    sequential (all-positive) colour scale.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    lons, lats : np.ndarray
        1-D grid node coordinates.
    Ex, Ey : np.ndarray
        Electric field on the (ny, nx) node grid.
    xf_lons, xf_lats : array-like
        Transformer geographic coordinates.
    gic : np.ndarray
        Absolute transformer GIC magnitudes (non-negative).
    shape : str or None
        Border shape name.
    lines : DataFrame or None
        Branch coordinate DataFrame.
    cmap_field : str
        Colormap for the E-field magnitude background.
    cmap_gic : str
        Sequential colormap for GIC bubbles.
    vmax_field : float or None
        Upper colour limit for E-field magnitude.
    title : str or None
        Axes title.
    **fmt_kw
        Extra keyword arguments forwarded to ``format_plot()``.

    Returns
    -------
    (im, sc) : tuple
        The pcolormesh and scatter artists (for external colorbars).
    """
    LON, LAT = np.meshgrid(lons, lats)
    mag = np.sqrt(Ex**2 + Ey**2)
    if vmax_field is None:
        vmax_field = float(np.nanmax(mag))

    # E-field magnitude background (faded)
    im = ax.pcolormesh(LON, LAT, mag, cmap=cmap_field, shading='auto',
                       vmin=0, vmax=vmax_field, alpha=0.4)

    if shape is not None:
        border(ax, shape)
    if lines is not None:
        plot_lines(ax, lines, ms=2, lw=0.3)

    # GIC bubbles: size AND colour encode |GIC|
    gic = np.asarray(gic).ravel()
    gic_max = max(float(gic.max()), 1e-6)
    sizes = 10 + 250 * (gic / gic_max)
    sc = ax.scatter(xf_lons, xf_lats,
                    s=sizes, c=gic, cmap=cmap_gic,
                    vmin=0, vmax=gic_max,
                    zorder=8, edgecolors='black', linewidth=0.4)

    ax.set_xlim(lons[0], lons[-1])
    ax.set_ylim(lats[0], lats[-1])
    ax.set_aspect('equal')

    defaults = dict(plotarea='white', grid=False,
                    titlesize=11, labelsize=9, ticksize=8)
    defaults.update(fmt_kw)
    if title is not None:
        defaults['title'] = title
    format_plot(ax, xlabel=r'Longitude ($^\circ$E)',
                ylabel=r'Latitude ($^\circ$N)', **defaults)
    return im, sc


def plot_gic_heatmap(ax, lons, lats, bus_df, shape=None, lines=None,
                     cmap='YlOrRd', method='cubic', title=None, **fmt_kw):
    """Plot bus-level |GIC| as a heatmap interpolated onto the grid.

    Interpolates sparse bus GIC values onto the regular (lons, lats)
    grid using ``scipy.interpolate.griddata``, producing a continuous
    heatmap of GIC magnitude across the geographic domain.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    lons, lats : np.ndarray
        1-D grid node coordinates (length nx, ny).
    bus_df : pandas.DataFrame
        Output of ``bus_gic()`` with columns
        ``Longitude``, ``Latitude``, ``GIC``.
    shape : str or None
        Border shape name passed to ``border()``.
    lines : DataFrame or None
        Branch coordinate DataFrame for ``plot_lines()``.
    cmap : str
        Colormap for the GIC heatmap.
    method : str
        Interpolation method for ``griddata``
        (``'nearest'``, ``'linear'``, or ``'cubic'``).
    title : str or None
        Axes title.
    **fmt_kw
        Extra keyword arguments forwarded to ``format_plot()``.

    Returns
    -------
    im : QuadMesh
        The pcolormesh artist (for external colorbar creation).
    """
    LON, LAT = np.meshgrid(lons, lats)

    points = np.column_stack([bus_df['Longitude'].to_numpy(),
                              bus_df['Latitude'].to_numpy()])
    values = bus_df['GIC'].to_numpy()
    gic_max = max(float(values.max()), 1e-6)

    gic_grid = griddata(points, values, (LON, LAT), method=method)
    # Fill NaN regions (outside convex hull) with nearest-neighbour
    mask = np.isnan(gic_grid)
    if mask.any():
        fill = griddata(points, values, (LON[mask], LAT[mask]),
                        method='nearest')
        gic_grid[mask] = fill

    im = ax.pcolormesh(LON, LAT, gic_grid, cmap=cmap, shading='auto',
                       vmin=0, vmax=gic_max)

    if shape is not None:
        border(ax, shape)
    if lines is not None:
        plot_lines(ax, lines, ms=2, lw=0.3)

    # Overlay bus locations as small markers
    ax.scatter(bus_df['Longitude'], bus_df['Latitude'],
               s=8, c='black', zorder=6, alpha=0.4)

    ax.set_xlim(lons[0], lons[-1])
    ax.set_ylim(lats[0], lats[-1])
    ax.set_aspect('equal')

    defaults = dict(plotarea='white', grid=False,
                    titlesize=11, labelsize=9, ticksize=8)
    defaults.update(fmt_kw)
    if title is not None:
        defaults['title'] = title
    format_plot(ax, xlabel=r'Longitude ($^\circ$E)',
                ylabel=r'Latitude ($^\circ$N)', **defaults)
    return im
