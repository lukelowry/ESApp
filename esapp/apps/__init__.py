"""
Specialized Applications (:mod:`esapp.apps`)
============================================

This package contains higher-level, specialized tools for advanced power
systems analysis tasks built on top of the core ``esapp`` components.
"""

# Applications
from .gic import GIC
from .network import Network, BranchType
from .modes import ForcedOscillation

__all__ = [
    "GIC",
    "Network",
    "BranchType",
    "ForcedOscillation",
]