Specialized Applications
========================

The ``apps`` package provides high-level analysis modules accessible through ``GridWorkBench``.
These modules abstract common workflows for network analysis, transient stability, GIC studies,
and modal analysis.

Dynamics
--------

Transient stability simulation with a fluent API for defining contingencies and recording results.

.. currentmodule:: esapp.apps.dynamics

.. autoclass:: Dynamics
   :members:
   :show-inheritance:

.. autoclass:: ContingencyBuilder
   :members:

GIC Analysis
------------

Geomagnetically Induced Current (GIC) analysis tools.

.. currentmodule:: esapp.apps.gic

.. autoclass:: GIC
   :members:
   :show-inheritance:

Modal Analysis
--------------

Forced oscillation and modal analysis utilities.

.. currentmodule:: esapp.apps.modes

.. automodule:: esapp.apps.modes
   :members:
   :exclude-members: Modes

.. autoclass:: Modes
   :members:
   :show-inheritance:

Network Topology
----------------

Network graph analysis including incidence matrices, Laplacians, and path calculations.

.. currentmodule:: esapp.apps.network

.. autoclass:: Network
   :members:
   :show-inheritance:

Static Analysis
---------------

Power flow and steady-state analysis helpers.

.. currentmodule:: esapp.apps.static

.. autoclass:: Static
   :members:
   :show-inheritance:
