Specialized Applications
========================

The ``apps`` module contains specialized tools for advanced analysis like GIC, Network topology analysis, 
and other domain-specific features.

Each app is accessible as an attribute on the GridWorkBench instance:

.. code-block:: python

    from esapp import GridWorkBench
    
    wb = GridWorkBench("case.pwb")
    
    # Access apps
    network = wb.network           # Network topology and matrix analysis
    gic = wb.gic                   # Geomagnetically induced current analysis
    saw = wb.esa                   # Low-level SimAuto Wrapper for advanced usage

Network Analysis
----------------

The Network app provides tools for analyzing power system topology and extracting system matrices.

Key Features:

- **Y-Bus Matrix**: Extract the sparse admittance matrix for power flow calculations
- **Incidence Matrix**: Get the bus-branch incidence matrix for network analysis
- **Laplacian Matrix**: Calculate the network Laplacian with various weight types (length, resistance distance, delay)
- **Bus Mapping**: Map bus numbers to matrix indices
- **Branch Lengths**: Extract branch length information for weighting

Example:

.. code-block:: python

    # Extract system matrices
    Y = wb.ybus()                          # Admittance matrix
    A = wb.network.incidence()             # Incidence matrix
    L = wb.network.laplacian(weights="LENGTH")  # Laplacian with length weighting
    
    # Get bus-to-index mapping
    busmap = wb.network.busmap()
    
    # Calculate branch lengths
    lengths = wb.network.lengths()

.. currentmodule:: esapp.apps

.. automodule:: esapp.apps.network
   :members:

GIC Analysis
------------

The GIC (Geomagnetically Induced Current) app calculates harmonic currents induced in transformers 
due to geomagnetic disturbances, which is critical for power grid resilience studies.

Key Features:

- **Uniform Field Modeling**: Model uniform electric field effects across the grid
- **Transformer GIC**: Calculate neutral currents induced in power transformers
- **System-Wide Assessment**: Evaluate GIC impact across the entire power system
- **Field Orientation**: Vary geomagnetic field direction for comprehensive analysis

Example:

.. code-block:: python

    from esapp.grid import GICXFormer
    
    # Calculate GIC for a uniform electric field
    wb.calculate_gic(max_field=1.0, direction=90.0)
    
    # Retrieve transformer GIC results
    gic_results = wb[GICXFormer, ["BusNum", "GICXFNeutralAmps"]]
    
    # Find maximum GIC
    max_gic = gic_results["GICXFNeutralAmps"].max()

.. automodule:: esapp.apps.gic
   :members:

Additional Apps
---------------

The full SAW interface provides access to many additional analysis capabilities beyond Network and GIC.
These include:

- **Power Flow Analysis** (powerflow.py): Transient and steady-state power flow
- **Contingency Analysis** (contingency.py): N-1 and multi-element contingency studies  
- **Optimal Power Flow** (opf.py): Economic dispatch and constrained optimization
- **Sensitivity Analysis** (sensitivity.py): PTDF, LODF, and other sensitivity factors
- **Transient Stability** (transient.py): Dynamic stability and critical clearing time calculations
- **Voltage Analysis** (pv.py, qv.py): P-V and Q-V curve generation
- **Available Transfer Capability** (atc.py): ATC calculation between areas
- **Branch Modification** (topology.py): Network topology changes and branch operations
- **Data Modification** (modify.py): Component parameter updates
- **Scheduled Operations** (scheduled.py): Time-step simulations and operational planning

For direct access to all SAW functionality:

.. code-block:: python

    # Access low-level SAW interface
    saw = wb.esa
    
    # Use SAW methods directly
    saw.SolvePowerFlow()
    saw.DetermineATC(seller, buyer)
    saw.TSInitialize()