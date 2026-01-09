SimAuto Wrapper (SAW)
=====================

The ``SAW`` (SimAuto Wrapper) class provides a comprehensive, object-oriented interface to the 
PowerWorld Simulator's SimAuto COM server. It abstracts away COM complexity while providing access 
to the full range of PowerWorld automation capabilities.

Overview
--------

SAW is organized into modular mixins, each covering a specific functional area:

Base
    Fundamental operations (open case, save, set data, get data, run scripts)
Power Flow
    AC/DC power flow solutions and analysis
Contingency
    Single and multi-element contingency analysis
Optimization
    Optimal power flow, SCOPF, and economic dispatch
Sensitivity
    PTDF, LODF, and other sensitivity calculations
Transient Stability
    Dynamic stability analysis and critical clearing time
GIC
    Geomagnetic induced current calculations
ATC
    Available transfer capability analysis
Topology
    Network modification and branching operations
Voltage Analysis
    P-V and Q-V curve generation
Data Management
    Export, import, and reporting
Advanced
    Matrices, regions, scheduled operations, weather effects

Typical Usage
~~~~~~~~~~~~~

While GridWorkBench provides high-level convenience methods for common tasks, the SAW interface 
allows access to the full PowerWorld API for advanced usage:

.. code-block:: python

    from esapp import GridWorkBench
    
    wb = GridWorkBench("case.pwb")
    
    wb.pflow()
    wb.auto_insert_contingencies()
    
    saw = wb.esa
    saw.SolveAC_OPF()

Error Handling
~~~~~~~~~~~~~~

SAW provides specialized exception types for different PowerWorld errors:

.. code-block:: python

    from esapp import PowerWorldError, COMError, SimAutoFeatureError
    
    try:
        wb.pflow()
    except PowerWorldError as e:
        print(f"PowerWorld error: {e}")

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