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
   :members:
   :undoc-members:

Mixin Modules
--------------

Power Flow
~~~~~~~~~~

.. automodule:: esapp.saw.powerflow
   :members:

Contingency Analysis
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: esapp.saw.contingency
   :members:

Optimal Power Flow
~~~~~~~~~~~~~~~~~~~

.. automodule:: esapp.saw.opf
   :members:

Sensitivity Analysis
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: esapp.saw.sensitivity
   :members:

Transient Stability
~~~~~~~~~~~~~~~~~~~~

.. automodule:: esapp.saw.transient
   :members:

GIC Analysis
~~~~~~~~~~~~

.. automodule:: esapp.saw.gic
   :members:

Available Transfer Capability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: esapp.saw.atc
   :members:

Network Topology
~~~~~~~~~~~~~~~~

.. automodule:: esapp.saw.topology
   :members:

Branch Operations
~~~~~~~~~~~~~~~~~

.. automodule:: esapp.saw.modify
   :members:

Case Management
~~~~~~~~~~~~~~~

.. automodule:: esapp.saw.case_actions
   :members:

Base Operations
~~~~~~~~~~~~~~~

.. automodule:: esapp.saw.base
   :members:

Voltage Analysis
~~~~~~~~~~~~~~~~

.. automodule:: esapp.saw.pv
   :members:

.. automodule:: esapp.saw.qv
   :members:

Matrix Operations
~~~~~~~~~~~~~~~~~

.. automodule:: esapp.saw.matrices
   :members:

One-Line Diagram
~~~~~~~~~~~~~~~~

.. automodule:: esapp.saw.oneline
   :members:

General Utilities
~~~~~~~~~~~~~~~~~

.. automodule:: esapp.saw.general
   :members:

Regional Analysis
~~~~~~~~~~~~~~~~~

.. automodule:: esapp.saw.regions
   :members:

Scheduled Operations
~~~~~~~~~~~~~~~~~~~~

.. automodule:: esapp.saw.scheduled
   :members:

Weather Effects
~~~~~~~~~~~~~~~

.. automodule:: esapp.saw.weather
   :members:

Time-Step Analysis
~~~~~~~~~~~~~~~~~~~

.. automodule:: esapp.saw.timestep
   :members: