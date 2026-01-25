SimAuto Wrapper (SAW)
=====================

The ``SAW`` (SimAuto Wrapper) class exposes the full PowerWorld API. It is organized into mixins
corresponding to PowerWorld functional areas (power flow, contingencies, optimization, sensitivity,
transient, GIC, ATC, topology, voltage analysis, data management). Use ``wb.esa`` to access SAW from
``GridWorkBench``. This page lists the complete API.

API Documentation
------------------

.. currentmodule:: esapp.saw

.. autoclass:: SAW
   :show-inheritance:
   :members:
   :undoc-members:

Mixin Modules
--------------

.. autosummary::
   :toctree: generated/

   atc
   base
   case_actions
   contingency
   fault
   general
   gic
   matrices
   modify
   oneline
   opf
   powerflow
   pv
   qv
   regions
   scheduled
   sensitivity
   timestep
   topology
   transient
   weather