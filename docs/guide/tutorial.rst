Tutorial
========

This tutorial will walk you through the basics of using ESA++ to interact with a PowerWorld case.

Getting Started
---------------

First, import the ``GridWorkBench`` and point it to your ``.pwb`` file:

.. code-block:: python

    from esapp import GridWorkBench
    from esapp.grid import Bus, Gen, Load, Branch, Area
    
    # Initialize the workbench with your PowerWorld case
    wb = GridWorkBench("path/to/your/case.pwb")

The ``GridWorkBench`` is the central interface that manages the PowerWorld Simulator instance and provides access to all analysis capabilities.

Accessing Data
--------------

ESA++ uses a unique indexing system inspired by NumPy to make data retrieval intuitive. You can access grid components using their class types:

.. code-block:: python

    # Get all bus numbers and names as a DataFrame
    buses = wb[Bus, ["BusNum", "BusName"]]
    
    # Get all generator data (all fields)
    generators = wb[Gen, :]
    
    # Access specific fields for branch data
    line_flows = wb[Branch, ["BusNum", "BusNum:1", "LineMW", "LinePercent"]]
    
    # Get data for loads
    loads = wb[Load, ["BusNum", "LoadID", "LoadMW", "LoadMVR"]]

The power of the indexing syntax is that it returns standard Pandas DataFrames, allowing you to use all of Pandas' filtering and analysis tools immediately.

Filtering and Selection
-----------------------

ESA++ allows you to filter data easily using standard Pandas operations on the returned DataFrames:

.. code-block:: python

    # Get only buses in Area 1
    all_buses = wb[Bus, :]
    area_1_buses = all_buses[all_buses['AreaNum'] == 1]
    
    # Find lines with loading above 90%
    lines = wb[Branch, ["LinePercent", "LineLimit"]]
    heavy_lines = lines[lines['LinePercent'] > 90]
    
    # Get data for a specific object by its primary key
    all_bus_data = wb[Bus, ["BusNum", "BusPUVolt", "BusAngle"]]
    bus_5_data = all_bus_data[all_bus_data['BusNum'] == 5]
    
    # Find offline generators
    gens = wb[Gen, ["BusNum", "GenID", "GenStatus", "GenMW"]]
    offline_gens = gens[gens['GenStatus'] == 'Open']

Running Analysis
----------------

You can solve power flow and retrieve results:

.. code-block:: python

    # Solve the power flow
    voltages = wb.pflow()  # Returns a Series of complex voltages at each bus
    
    # Retrieve voltage magnitudes
    voltage_magnitudes = abs(voltages)
    
    # Check for voltage violations
    low_voltage_buses = voltage_magnitudes[voltage_magnitudes < 0.95]
    high_voltage_buses = voltage_magnitudes[voltage_magnitudes > 1.05]

Find System Violations
~~~~~~~~~~~~~~~~~~~~~~

After running power flow, you can check for violations:

.. code-block:: python

    # Find buses with voltage violations
    violations = wb.find_violations(v_min=0.95, v_max=1.05, branch_max_pct=100.0)
    
    # This returns a dictionary with:
    # - 'buses_low': buses below minimum voltage
    # - 'buses_high': buses above maximum voltage  
    # - 'branches_overloaded': branches exceeding limit

Modifying Data
--------------

You can update grid parameters using the same indexing syntax:

.. code-block:: python

    # Set the setpoint for Generator at Bus 5 to 150 MW
    wb.set_gen(bus=5, id="1", mw=150.0)
    
    # Set all bus voltage setpoints to 1.02 pu
    wb[Bus, "BusVoltSet"] = 1.02
    
    # Set load to a specific value
    wb.set_load(bus=10, id="1", mw=50.0, mvar=10.0)
    
    # Open a transmission line branch
    wb.open_branch(from_bus=1, to_bus=2, id="1")
    
    # Close a transmission line
    wb.close_branch(from_bus=1, to_bus=2, id="1")

Bulk Data Updates
~~~~~~~~~~~~~~~~~

For multiple updates at once, use DataFrames:

.. code-block:: python

    import pandas as pd
    
    # Create a DataFrame with updates
    updates = pd.DataFrame({
        'BusNum': [1, 2, 3],
        'BusVoltSet': [1.02, 1.02, 1.01]
    })
    
    # Apply bulk updates
    wb[Bus] = updates

Contingency Analysis
---------------------

Perform N-1 contingency analysis:

.. code-block:: python

    # Solve base case first
    wb.pflow()
    
    # Create and solve all N-1 contingencies automatically
    wb.auto_insert_contingencies()
    wb.solve_contingencies()
    
    # Retrieve contingency violations
    from esapp.grid import ViolationCTG
    violations = wb[ViolationCTG, :]

Advanced Analysis
-----------------

Sensitivity Analysis
~~~~~~~~~~~~~~~~~~~~

Calculate power transfer distribution factors (PTDF) and line outage distribution factors (LODF):

.. code-block:: python

    # Get PTDF matrix for a transfer from Area 1 to Area 2
    ptdf = wb.ptdf(seller="Area 1", buyer="Area 2")
    
    # Get LODF for a specific branch
    lodf = wb.lodf(branch=(1, 2, "1"))  # (from_bus, to_bus, circuit_id)

Network Topology
~~~~~~~~~~~~~~~~

Analyze network topology and extract system matrices:

.. code-block:: python

    # Get the Y-Bus admittance matrix
    Y = wb.ybus()  # Returns a sparse matrix
    
    # Get the incidence matrix
    A = wb.network.incidence()  # Returns buses x branches incidence matrix
    
    # Get the Laplacian matrix
    L = wb.network.laplacian(weights=wb.network.BranchType.LENGTH)
    
    # Get bus coordinate mapping
    busmap = wb.network.busmap()  # Maps bus numbers to matrix indices

GIC Analysis
~~~~~~~~~~~~

Calculate geomagnetically induced currents:

.. code-block:: python

    # Calculate GIC for a 1.0 V/km electric field at 90 degrees
    wb.calculate_gic(max_field=1.0, direction=90.0)
    
    # Retrieve transformer GIC results
    from esapp.grid import GICXFormer
    gic_results = wb[GICXFormer, ["BusNum", "BusNum:1", "GICXFNeutralAmps"]]

Saving Changes
--------------

After making modifications, save your case:

.. code-block:: python

    # Save to the original file
    wb.save()
    
    # Or save to a new location
    wb.save(filename="path/to/new/case.pwb")

Closing the Workbench
---------------------

When finished, close the PowerWorld connection:

.. code-block:: python

    wb.close()

This will cleanly shut down the PowerWorld Simulator instance.