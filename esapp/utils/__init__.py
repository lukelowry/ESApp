"""
ESAplus utilities module.

Provides tools for:
- Mathematical operations (linear algebra, spectral analysis)
- Mesh and grid handling (PLY files, finite differences)
- Power system utilities (injection vectors, Y-bus modifications)
- Visualization (geographic plotting, vector fields)
- Binary data formats (B3D electric field data)
"""

from .mesh import (
    MU0,
    takagi,
    eigmax,
    sorteig,
    pathlap,
    pathincidence,
    normlap,
    hermitify,
    Mesh,
    extract_unique_edges,
    Grid2D,
    GridSelector,
)

from .misc import (
    InjectionVector,
    timing,
)

from .map import (
    format_plot,
    darker_hsv_colormap,
    border,
    plot_lines,
    plot_mesh,
    plot_tiles,
    plot_vecfield,
)

from .b3d import B3D

__all__ = [
    # mathtools (in mesh.py)
    'MU0',
    'takagi',
    'eigmax',
    'sorteig',
    'pathlap',
    'pathincidence',
    'normlap',
    'hermitify',
    # mesh
    'Mesh',
    'extract_unique_edges',
    'Grid2D',
    'GridSelector',
    # misc
    'InjectionVector',
    'timing',
    # map
    'format_plot',
    'darker_hsv_colormap',
    'border',
    'plot_lines',
    'plot_mesh',
    'plot_tiles',
    'plot_vecfield',
    # b3d
    'B3D',
]
