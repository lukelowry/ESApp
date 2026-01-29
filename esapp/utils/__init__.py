"""
ESAplus utilities module.

Provides tools for:
- Mathematical operations (linear algebra, spectral analysis)
- Mesh and grid handling (PLY files, finite differences)
- Power system utilities (injection vectors, Y-bus modifications)
- Visualization (geographic plotting, vector fields)
- Binary data formats (B3D electric field data)
"""

from .mathtools import (
    MU0,
    takagi,
    eigmax,
    sorteig,
    periodiclap,
    pathlap,
    periodicincidence,
    pathincidence,
    normlap,
    hermitify,
)

from .mesh import (
    Mesh,
    extract_unique_edges,
    Grid2D,
    GridSelector,
)

from .misc import (
    InjectionVector,
    ybus_with_loads,
)

from .decorators import timing

from .map import (
    formatPlot,
    darker_hsv_colormap,
    border,
    plot_lines,
    plot_mesh,
    plot_tiles,
    plot_compass,
    plot_vecfield,
)

from .b3d import B3D

__all__ = [
    # mathtools
    'MU0',
    'takagi',
    'eigmax',
    'sorteig',
    'periodiclap',
    'pathlap',
    'periodicincidence',
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
    'ybus_with_loads',
    # decorators
    'timing',
    # map
    'formatPlot',
    'darker_hsv_colormap',
    'border',
    'plot_lines',
    'plot_mesh',
    'plot_tiles',
    'plot_compass',
    'plot_vecfield',
    # b3d
    'B3D',
]
