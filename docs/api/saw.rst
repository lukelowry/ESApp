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

   esapp.saw.atc
   esapp.saw.base
   esapp.saw.case_actions
   esapp.saw.contingency
   esapp.saw.fault
   esapp.saw.general
   esapp.saw.gic
   esapp.saw.matrices
   esapp.saw.modify
   esapp.saw.oneline
   esapp.saw.opf
   esapp.saw.powerflow
   esapp.saw.pv
   esapp.saw.qv
   esapp.saw.regions
   esapp.saw.scheduled
   esapp.saw.sensitivity
   esapp.saw.timestep
   esapp.saw.topology
   esapp.saw.transient
   esapp.saw.weather