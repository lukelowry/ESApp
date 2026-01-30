"""
Specialized Applications (:mod:`esapp.apps`)
============================================

Higher-level analysis tools for power systems built on top of the
core ``esapp`` components.

Modules
-------
GIC
    Geomagnetically induced currents analysis with matrix generation
    and PowerWorld integration.
Dynamics
    Transient stability simulation with fluent contingency definition.
Statics
    Continuation power flow and static analysis methods.
Network
    Network matrix construction and graph analysis.
ForcedOscillation
    Forced oscillation detection (experimental).

Example
-------
>>> from esapp import GridWorkBench
>>> from esapp.apps import GIC, BranchType
>>> wb = GridWorkBench("case.pwb")
>>> model = wb.gic.model()
>>> L = wb.net.laplacian(BranchType.LENGTH)
"""

# GIC Analysis
from .gic import GIC

# Static Analysis
from .static import Statics

# Dynamics Analysis
from .dynamics import Dynamics, ContingencyBuilder, SimAction

# Network Analysis
from .network import Network, BranchType

__all__ = [
    # GIC
    "GIC",
    # Dynamics
    "Dynamics",
    "ContingencyBuilder",
    "SimAction",
    # Statics
    "Statics",
    # Network
    "Network",
    "BranchType",
]
