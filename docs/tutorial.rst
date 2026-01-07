Tutorial
========

This tutorial will walk you through the basics of using ESA++ to interact with a PowerWorld case.

Getting Started
---------------

First, import the ``GridWorkBench`` and point it to your ``.pwb`` file:

.. code-block:: python

    from gridwb import GridWorkBench
    
    # Initialize the workbench
    wb = GridWorkBench("path/to/your/case.pwb")

Accessing Data
--------------

ESA++ uses a unique indexing system to make data retrieval intuitive. You can access grid components using their class types:

.. code-block:: python

    from gridwb.grid.components import Bus, Gen, Line
    
    # Get all bus numbers and names as a DataFrame
    buses = wb[Bus, ['BusNum', 'BusName']]
    
    # Get all generator data (all fields)
    generators = wb[Gen, :]
    
    # Access specific fields for a subset of components using a list of keys
    line_flows = wb[Line, ['BusNum', 'BusNum:1', 'LineMW']]

The power of the indexing syntax is that it returns standard Pandas objects, allowing you to use all of Pandas' filtering and analysis tools immediately.

Filtering and Selection
-----------------------

ESA++ allows you to filter data easily using standard Pandas operations on the returned DataFrames:

.. code-block:: python

    # Get only buses in Area 1
    area_1_buses = wb[Bus, :][wb[Bus, :]['AreaNum'] == 1]
    
    # Find lines with loading above 90%
    heavy_lines = wb[Line, ['BusNum', 'BusNum:1', 'LinePercent']]
    heavy_lines = heavy_lines[heavy_lines['LinePercent'] > 90]
    
    # Get data for a specific object by its primary key
    # For a Bus, the key is the Bus Number
    bus_5_data = wb[Bus, 5, ['BusPUVolt', 'BusAngle']]

Running Analysis
----------------

You can solve power flow and retrieve results in one line:

.. code-block:: python

    # Solve power flow and get voltages
    voltages = wb.pflow()
    
    # Check if the power flow converged
    if wb.io.esa.is_converged():
        print("Power flow converged successfully!")
    else:
        print("Power flow failed to converge.")

Modifying Data
--------------

You can update grid parameters using the same indexing syntax:

.. code-block:: python

    # Set the setpoint for Generator at Bus 5 to 150 MW
    wb[Gen, 5, 'GenMW'] = 150.0
    
    # You can also set values for multiple objects at once
    # Set all bus voltage setpoints to 1.02 pu
    wb[Bus, 'BusVoltSet'] = 1.02

Saving Changes
--------------

After making modifications, save your case:

.. code-block:: python

    wb.save()