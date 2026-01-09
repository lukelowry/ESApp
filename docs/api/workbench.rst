GridWorkBench
=============

The ``GridWorkBench`` is the primary interface for interacting with PowerWorld Simulator through ESA++. 
It orchestrates the entire lifecycle of power system analysis, from case loading to simulation and results retrieval.

Overview
--------

The GridWorkBench inherits from the ``Indexable`` class, which provides the core data access mechanism via 
intuitive NumPy-style indexing (e.g., ``wb[Bus, "BusPUVolt"]``). This enables seamless read and write access 
to all PowerWorld grid objects and their parameters.

The GridWorkBench manages:

- **Case Management**: Loading, saving, and modifying PowerWorld cases (.pwb files)
- **Simulation Control**: Running power flow, contingency analysis, optimization, and other studies
- **Data Access**: Reading component data (buses, generators, lines, etc.) and writing modifications
- **Analysis Apps**: Specialized tools for network analysis, GIC calculations, and other advanced features
- **PowerWorld Interface**: Underlying connection to PowerWorld Simulator via the SAW (SimAuto Wrapper) class

Core Concepts
~~~~~~~~~~~~~

**Indexable Interface**
  The ``GridWorkBench`` inherits the ``Indexable`` mixin, which enables the signature data access pattern:
  
  .. code-block:: python
  
      from esapp.grid import Bus, Gen, Branch
      
      # Retrieve data
      buses = wb[Bus, ["BusNum", "BusPUVolt"]]
      generators = wb[Gen, :]
      
      # Modify data
      wb[Bus, "BusVoltSet"] = 1.02

**Apps for Specialized Analysis**
  GridWorkBench provides access to specialized analysis tools:
  
  - ``wb.network``: Network topology and matrix extraction
  - ``wb.gic``: Geomagnetically induced current analysis
  - ``wb.esa``: Low-level SAW (SimAuto Wrapper) interface for advanced usage

**Method Organization**
  Methods are organized by functionality:
  
  - Simulation: ``pflow()``, ``reset()``, ``mode()``
  - Modification: ``set_gen()``, ``set_load()``, ``open_branch()``, ``close_branch()``
  - Analysis: ``auto_insert_contingencies()``, ``solve_contingencies()``, ``find_violations()``
  - Matrices: ``ybus()``, ``ptdf()``, ``lodf()``
  - Data Export: ``save()``, ``load_aux()``, ``load_script()``

Common Workflows
~~~~~~~~~~~~~~~~

**Power Flow Analysis**
  .. code-block:: python
  
      # Solve power flow
      voltages = wb.pflow()
      
      # Check for violations
      violations = wb.find_violations(v_min=0.95, v_max=1.05)
      
      # Retrieve results
      buses = wb[Bus, ["BusPUVolt", "BusAngle"]]

**Contingency Study**
  .. code-block:: python
  
      from esapp.grid import ViolationCTG
      
      wb.pflow()
      wb.auto_insert_contingencies()
      wb.solve_contingencies()
      
      violations = wb[ViolationCTG, :]

**Network Sensitivity**
  .. code-block:: python
  
      # Power transfer distribution factor
      ptdf = wb.ptdf(seller="Area 1", buyer="Area 2")
      
      # Line outage distribution factor
      lodf = wb.lodf(branch=(1, 2, "1"))

API Documentation
------------------

.. currentmodule:: esapp.workbench

.. autoclass:: GridWorkBench
   :members:
   :undoc-members:
   :show-inheritance:
