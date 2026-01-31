"""
Non-uniform GIC model formulation helpers.

Provides the line integration operator **L** and E-field vector
assembly for the non-uniform GIC computation:

    I_gic = abs(H @ L @ E)

where:
- **H** is the transformer-to-branch transfer matrix from ``wb.gic.model()``
- **L** maps a gridded E-field vector to branch induced voltages
- **E** = [Ex.ravel(), Ey.ravel()] is the stacked master E-field vector

The operator L discretizes the line integral of E along each branch by
tracing every transmission line through the 2-D grid and accumulating
the directed segment length (dx, dy) within each cell.  The E-field
inside a cell is taken as the average of its four corner nodes.

Example
-------
>>> from examples.nonuniform.nonuniform import build_L_matrix, stack_efield
>>>
>>> L = build_L_matrix(wb, lons, lats, H.shape[1])
>>> E = stack_efield(Ex, Ey)
>>> gic = np.abs(H @ L @ E)
"""

import numpy as np
from scipy.sparse import coo_matrix

from esapp.components import Branch, Bus, GICXFormer

__all__ = ['build_L_matrix', 'stack_efield', 'compute_gic', 'bus_gic']


def stack_efield(Ex, Ey):
    """Stack 2-D Ex and Ey grids into a single master E-field vector.

    Parameters
    ----------
    Ex : np.ndarray
        East-component of the electric field on a (ny, nx) grid.
    Ey : np.ndarray
        North-component of the electric field on a (ny, nx) grid.

    Returns
    -------
    np.ndarray
        1-D vector of length 2*N where N = nx*ny, ordered as
        [Ex.ravel(), Ey.ravel()].
    """
    return np.concatenate([Ex.ravel(), Ey.ravel()])


def _trace_line_through_grid(x0, y0, x1, y1, xs, ys):
    """Trace a straight line segment through a regular grid.

    Finds every cell the segment passes through and returns the
    directed (dx, dy) length within each cell in **grid-coordinate
    units** (i.e. fractional cell widths).

    Parameters
    ----------
    x0, y0 : float
        Start point in continuous grid coordinates
        (x = (lon - lon0) / dlon, y = (lat - lat0) / dlat).
    x1, y1 : float
        End point in continuous grid coordinates.
    xs : int
        Number of grid nodes in the x-direction (nx).
    ys : int
        Number of grid nodes in the y-direction (ny).

    Yields
    ------
    (ix, iy, frac_dx, frac_dy) : tuple
        Cell indices (ix, iy) and the directed length of the line
        segment within that cell, in fractional grid units.
    """
    # Clamp endpoints into valid grid range [0, size-1]
    x0c = np.clip(x0, 0, xs - 1)
    y0c = np.clip(y0, 0, ys - 1)
    x1c = np.clip(x1, 0, xs - 1)
    y1c = np.clip(y1, 0, ys - 1)

    dx_total = x1c - x0c
    dy_total = y1c - y0c

    if abs(dx_total) < 1e-12 and abs(dy_total) < 1e-12:
        # Zero-length segment (co-located buses)
        return

    # Collect all t-values where the line crosses vertical or horizontal
    # grid lines.  t parameterizes the clamped segment: r(t) = r0 + t*(r1-r0).
    crossings = [0.0, 1.0]

    if abs(dx_total) > 1e-12:
        # Vertical grid lines at x = 1, 2, ..., xs-2
        ix_lo = int(np.floor(min(x0c, x1c)))
        ix_hi = int(np.ceil(max(x0c, x1c)))
        for ix in range(max(ix_lo, 1), min(ix_hi, xs - 1) + 1):
            t = (ix - x0c) / dx_total
            if 0 < t < 1:
                crossings.append(t)

    if abs(dy_total) > 1e-12:
        # Horizontal grid lines at y = 1, 2, ..., ys-2
        iy_lo = int(np.floor(min(y0c, y1c)))
        iy_hi = int(np.ceil(max(y0c, y1c)))
        for iy in range(max(iy_lo, 1), min(iy_hi, ys - 1) + 1):
            t = (iy - y0c) / dy_total
            if 0 < t < 1:
                crossings.append(t)

    crossings.sort()

    # Walk through consecutive pairs of crossings
    for i in range(len(crossings) - 1):
        t_a = crossings[i]
        t_b = crossings[i + 1]
        if t_b - t_a < 1e-14:
            continue

        # Midpoint of this sub-segment -> determines which cell we're in
        t_mid = 0.5 * (t_a + t_b)
        mx = x0c + t_mid * dx_total
        my = y0c + t_mid * dy_total

        ix = int(np.clip(np.floor(mx), 0, xs - 2))
        iy = int(np.clip(np.floor(my), 0, ys - 2))

        # Directed length of this sub-segment in grid units
        frac_dx = (t_b - t_a) * dx_total
        frac_dy = (t_b - t_a) * dy_total

        yield ix, iy, frac_dx, frac_dy


def build_L_matrix(wb, lons, lats, n_branches_model):
    """Build the line integration operator L.

    L is a sparse matrix of shape ``(n_branches_model, 2*N)`` where
    ``N = len(lons) * len(lats)``.  Given the master E-field vector
    ``E = stack_efield(Ex, Ey)``, the product ``L @ E`` yields the
    induced voltage on each branch.

    **Discretization.**  Each transmission line is traced through the
    2-D grid cell by cell.  For every cell the line passes through, the
    directed segment length ``(dx_km, dy_km)`` within that cell is
    computed.  The E-field inside the cell is approximated as the
    average of its four corner nodes.  The voltage contribution from
    cell ``(ix, iy)`` for branch ``k`` is therefore:

        dV = E_x_avg * dx_km  +  E_y_avg * dy_km

    which translates to weight ``(1/4) * dx_km`` at each of the four
    corner nodes in the ``E_x`` block of L, and ``(1/4) * dy_km`` in
    the ``E_y`` block.

    Parameters
    ----------
    wb : GridWorkBench
        Live workbench instance (used to read branch coordinates).
    lons : np.ndarray
        1-D array of grid longitudes (length nx).
    lats : np.ndarray
        1-D array of grid latitudes (length ny).
    n_branches_model : int
        Total number of branch columns in the H-matrix
        (``H.shape[1]``).  Rows beyond the number of geographic
        branches are left as zeros (transformer windings, GSUs).

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse matrix of shape ``(n_branches_model, 2*N)``.
    """
    nx = len(lons)
    ny = len(lats)
    N = nx * ny

    dlon = lons[1] - lons[0]
    dlat = lats[1] - lats[0]

    # Branch endpoint coordinates
    br = wb[Branch, ['BusNum', 'BusNum:1', 'BranchDeviceType',
                     'Longitude', 'Longitude:1', 'Latitude', 'Latitude:1']]

    lon_a = br['Longitude'].to_numpy()
    lon_b = br['Longitude:1'].to_numpy()
    lat_a = br['Latitude'].to_numpy()
    lat_b = br['Latitude:1'].to_numpy()

    n_br = min(len(br), n_branches_model)

    rows, cols, data = [], [], []

    for k in range(n_br):
        # Continuous grid coordinates for endpoints
        gx0 = (lon_a[k] - lons[0]) / dlon
        gy0 = (lat_a[k] - lats[0]) / dlat
        gx1 = (lon_b[k] - lons[0]) / dlon
        gy1 = (lat_b[k] - lats[0]) / dlat

        # Midpoint latitude for the cos correction
        mid_lat = 0.5 * (lat_a[k] + lat_b[k])
        cos_lat = np.cos(np.radians(mid_lat))

        # Conversion: 1 grid-unit in x = dlon degrees = dlon * 111 * cos(lat) km
        #             1 grid-unit in y = dlat degrees = dlat * 111 km
        km_per_gx = dlon * 111.0 * cos_lat
        km_per_gy = dlat * 111.0

        for ix, iy, fdx, fdy in _trace_line_through_grid(gx0, gy0, gx1, gy1, nx, ny):
            dx_km = fdx * km_per_gx
            dy_km = fdy * km_per_gy

            # Four corner nodes of cell (ix, iy), each gets weight 1/4
            corners = [
                iy       * nx + ix,
                iy       * nx + (ix + 1),
                (iy + 1) * nx + ix,
                (iy + 1) * nx + (ix + 1),
            ]
            w = 0.25
            for idx in corners:
                # Ex block: columns [0, N)
                rows.append(k)
                cols.append(idx)
                data.append(w * dx_km)
                # Ey block: columns [N, 2N)
                rows.append(k)
                cols.append(N + idx)
                data.append(w * dy_km)

    L = coo_matrix((data, (rows, cols)), shape=(n_branches_model, 2 * N))
    # COO allows duplicate entries; converting to CSR sums them automatically
    return L.tocsr()


def bus_gic(wb, gic):
    """Aggregate absolute transformer GICs to bus-level totals.

    Each transformer in the H-matrix corresponds to a row in the
    GICXFormer table.  This function sums |GIC| contributions at each
    bus (using ``BusNum3W`` as the transformer's primary bus) and
    returns a DataFrame with bus coordinates for geographic plotting.

    Parameters
    ----------
    wb : GridWorkBench
        Live workbench instance.
    gic : np.ndarray
        Absolute transformer GIC magnitudes (length n_transformers).

    Returns
    -------
    pandas.DataFrame
        Columns: ``BusNum``, ``Longitude``, ``Latitude``, ``GIC``.
        One row per bus that hosts at least one transformer, with
        ``GIC`` being the sum of |GIC| over all transformers at that bus.
    """
    import pandas as pd

    gic = np.asarray(gic).ravel()

    xf = wb[GICXFormer, ['BusNum3W', 'BusNum3W:1']]
    xf = xf.iloc[:len(gic)].copy()
    xf['GIC'] = gic

    # Aggregate by primary bus (BusNum3W)
    bus_total = xf.groupby('BusNum3W')['GIC'].sum().reset_index()
    bus_total.columns = ['BusNum', 'GIC']

    # Join with bus coordinates
    coords = wb[Bus, ['BusNum', 'Longitude', 'Latitude']]
    result = bus_total.merge(coords, on='BusNum', how='inner')
    return result


def compute_gic(H, L, Ex, Ey):
    """Compute absolute transformer GICs from a gridded E-field.

    Evaluates ``|H @ L @ E|`` where E is the stacked field vector.

    Parameters
    ----------
    H : sparse matrix or np.ndarray
        Transfer matrix (n_transformers x n_branches).
    L : sparse matrix
        Line integration operator (n_branches x 2N).
    Ex, Ey : np.ndarray
        Electric field components on the (ny, nx) node grid.

    Returns
    -------
    np.ndarray
        Absolute transformer GIC magnitudes (n_transformers,).
    """
    E = stack_efield(Ex, Ey)
    gic = H @ L @ E
    if hasattr(gic, 'A'):
        gic = np.asarray(gic).ravel()
    return np.abs(gic)
