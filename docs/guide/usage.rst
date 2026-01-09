Usage Guide
===========

This guide covers advanced usage patterns and specialized features of the ESA++ toolkit.

Data Access with NumPy-Style Indexing
======================================

The cornerstone of ESA++ is its intuitive indexing syntax for data access and modification.

Getting Data
~~~~~~~~~~~~

Retrieving data is as simple as indexing the workbench with a component class:

.. code-block:: python

    from esapp import GridWorkBench
    from esapp.grid import Bus, Gen, Load, Branch, Area, Zone

**Get Primary Keys Only**

To get just the primary keys (identifiers) for all objects of a type:

.. code-block:: python

    bus_keys = wb[Bus]  # Returns DataFrame with primary key columns

**Get Specific Fields**

Pass a string or a list of strings to retrieve specific fields:

.. code-block:: python

    # Single field - returns a Series
    voltages = wb[Bus, "BusPUVolt"]

    # Multiple fields - returns a DataFrame
    bus_info = wb[Bus, ["BusName", "BusPUVolt", "BusAngle"]]
    
    # Specific generator fields
    gen_data = wb[Gen, ["GenMW", "GenMVR", "GenStatus"]]

**Get All Available Fields**

Use the slice operator ``:`` to retrieve every field defined for that component:

.. code-block:: python

    all_bus_data = wb[Bus, :]
    all_gen_data = wb[Gen, :]
    all_branch_data = wb[Branch, :]

**Using Component Attributes for IDE Support**

For better IDE autocomplete and to avoid typos, use the attributes defined on component classes:

.. code-block:: python

    # Type-safe field access with IDE hints
    data = wb[Bus, [Bus.BusName, Bus.BusPUVolt, Bus.BusAngle]]
    
    # Works with all component types
    gen_output = wb[Gen, [Gen.GenMW, Gen.GenMVR, Gen.GenStatus]]

Filtering Data
~~~~~~~~~~~~~~

Since returned data is in standard Pandas DataFrames, use all of Pandas' filtering capabilities:

.. code-block:: python

    import pandas as pd
    
    # Filter buses in a specific area
    all_buses = wb[Bus, ["BusNum", "BusName", "AreaNum", "BusPUVolt"]]
    area_1_buses = all_buses[all_buses['AreaNum'] == 1]
    
    # Find heavily loaded branches
    branches = wb[Branch, ["BusNum", "BusNum:1", "LinePercent", "LineLimit"]]
    overloaded = branches[branches['LinePercent'] > 100.0]
    
    # Get offline generators
    gens = wb[Gen, ["GenMW", "GenStatus"]]
    offline = gens[gens['GenStatus'] == 'Open']
    
    # Complex filtering
    buses_with_low_voltage = all_buses[all_buses['BusPUVolt'] < 0.95]

Data Modification
=================

The same indexing syntax is used to update values in the PowerWorld case.

Broadcasting a Scalar Value
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set a single value for all objects of a type:

.. code-block:: python

    # Set all bus voltages to 1.05 pu
    wb[Bus, "BusPUVolt"] = 1.05
    
    # Set all generator status to online
    wb[Gen, "GenStatus"] = "Closed"

Updating Multiple Fields
~~~~~~~~~~~~~~~~~~~~~~~~

Update multiple fields simultaneously:

.. code-block:: python

    # Update MW and MVAR for all generators
    wb[Gen, ["GenMW", "GenMVR"]] = [100.0, 20.0]

Bulk Update from DataFrame
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform bulk updates using a DataFrame with primary keys:

.. code-block:: python

    import pandas as pd
    
    # Create update DataFrame (must include primary key columns)
    updates = pd.DataFrame({
        'BusNum': [1, 2, 5, 10],
        'BusPUVolt': [1.02, 1.03, 0.99, 1.01]
    })
    
    # Apply bulk updates
    wb[Bus] = updates

Component-Specific Methods
===~~~~~~~~~~~~~~~~~~~~~~~~

ESA++ provides convenience methods for common modifications:

.. code-block:: python

    # Set generator output
    wb.set_gen(bus=5, id="1", mw=150.0, mvar=50.0, status="Closed")
    
    # Set load consumption
    wb.set_load(bus=10, id="1", mw=100.0, mvar=30.0, status="Closed")
    
    # Open/close branches
    wb.open_branch(from_bus=1, to_bus=2, id="1")
    wb.close_branch(from_bus=1, to_bus=2, id="1")
    
    # Scale generation and loads
    wb.scale_gen(scale_factor=1.1)  # Increase all generation by 10%
    wb.scale_load(scale_factor=0.9)  # Decrease all loads by 10%

Analysis and Simulation
=======================

Power Flow Solution
~~~~~~~~~~~~~~~~~~~

Solve the AC power flow and retrieve results:

.. code-block:: python

    # Solve the base case power flow
    voltages = wb.pflow()  # Returns complex voltages at each bus
    
    # Extract voltage magnitudes
    voltage_mags = abs(voltages)
    
    # Get all bus voltages as a DataFrame
    bus_voltage_df = wb[Bus, ["BusNum", "BusPUVolt", "BusAngle"]]
    
    # Check for convergence
    pf_converged = wb.esa.pflow_converged

Violation Detection
~~~~~~~~~~~~~~~~~~~

Automatically detect system violations:

.. code-block:: python

    # Find all violations in one call
    violations = wb.find_violations(
        v_min=0.95,        # Min voltage (pu)
        v_max=1.05,        # Max voltage (pu)
        branch_max_pct=100 # Max branch loading (%)
    )
    
    # Returns dictionary with:
    # violations['buses_low'] - buses below min voltage
    # violations['buses_high'] - buses above max voltage
    # violations['branches_overloaded'] - overloaded branches

Contingency Analysis
~~~~~~~~~~~~~~~~~~~~

Perform N-1 contingency studies:

.. code-block:: python

    # Solve base case first
    wb.pflow()
    
    # Create N-1 contingencies for all branches
    wb.auto_insert_contingencies()
    
    # Solve all contingencies
    wb.solve_contingencies()
    
    # Retrieve violation results
    from esapp.grid import ViolationCTG
    violations = wb[ViolationCTG, :]
    
    # Filter to critical contingencies
    critical = violations[violations['ViolatedRecord'] == 'Yes']

Optimization and Control
~~~~~~~~~~~~~~~~~~~~~~~~

Run optimal power flow and security-constrained optimization:

.. code-block:: python

    # Solve AC OPF
    wb.esa.SolveAC_OPF()
    
    # Solve Security-Constrained OPF (SCOPF)
    wb.esa.InitializePrimalLP()
    wb.auto_insert_contingencies()
    wb.esa.SolveFullSCOPF()
    
    # Get optimization results
    opf_cost = wb[Area, "GenProdCost"]

Sensitivity Analysis
~~~~~~~~~~~~~~~~~~~~

Calculate power transfer distribution factors and sensitivity:

.. code-block:: python

    # Calculate PTDF (Power Transfer Distribution Factor)
    ptdf = wb.ptdf(seller="Area 1", buyer="Area 2", method='DC')
    
    # Calculate LODF (Line Outage Distribution Factor)
    lodf = wb.lodf(branch=(1, 2, "1"), method='DC')

Transient Stability
~~~~~~~~~~~~~~~~~~~

Perform transient stability analysis:

.. code-block:: python

    # Initialize transient stability module
    wb.esa.TSInitialize()
    
    # Calculate Critical Clearing Time (CCT) for a fault
    from esapp.saw._helpers import create_object_string
    branch = create_object_string("Branch", 1, 2, "1")
    wb.esa.TSCalculateCriticalClearTime(branch)
    
    # Generate stability plots
    wb.esa.TSAutoSavePlots(
        plot_names=["Generator Frequencies", "Bus Voltages"],
        ctg_names=["Fault_at_Bus_1"]
    )

GIC Analysis
~~~~~~~~~~~~

Calculate geomagnetically induced currents:

.. code-block:: python

    # Calculate GIC for a uniform electric field
    wb.calculate_gic(max_field=1.0, direction=90.0)
    
    # Retrieve transformer GIC results
    from esapp.grid import GICXFormer
    gic_results = wb[GICXFormer, ["BusNum", "BusNum:1", "GICXFNeutralAmps"]]
    
    # Find transformers with highest GIC
    max_gic = gic_results['GICXFNeutralAmps'].max()

Available Transfer Capability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calculate available transfer capability between areas:

.. code-block:: python

    from esapp.saw._helpers import create_object_string
    
    # Setup ATC parameters
    wb.esa.SetData("ATC_Options", ["Method"], ["IteratedLinearThenFull"])
    
    # Determine ATC from seller to buyer area
    seller = create_object_string("Area", 1)
    buyer = create_object_string("Area", 2)
    wb.esa.DetermineATC(seller, buyer)
    
    # Get ATC results
    results = wb.esa.GetATCResults(["MaxFlow", "LimitingContingency", "LimitingElement"])

Network Topology and Matrices
==============================

System Matrix Extraction
~~~~~~~~~~~~~~~~~~~~~~~~

Extract system matrices for mathematical analysis:

.. code-block:: python

    # Get the sparse Y-Bus (admittance) matrix
    Y = wb.ybus()  # Returns scipy sparse matrix
    
    # Get bus-branch incidence matrix
    A = wb.network.incidence()
    
    # Get Laplacian matrix with branch weights
    from esapp.apps.network import Network
    L = wb.network.laplacian(weights=Network.BranchType.LENGTH)
    
    # Get bus number to matrix index mapping
    busmap = wb.network.busmap()

Topology Analysis
~~~~~~~~~~~~~~~~~~

Analyze network structure:

.. code-block:: python

    # Get bus coordinate mapping
    bus_coords = wb.network.busmap()
    
    # Calculate branch lengths
    branch_lengths = wb.network.lengths()
    
    # Identify network islands/zones
    from esapp.grid import Zone
    zones = wb[Zone, :]

Network Modification
~~~~~~~~~~~~~~~~~~~~

Programmatically modify the network topology:

.. code-block:: python

    from esapp.saw._helpers import create_object_string
    
    # Tap an existing transmission line
    new_bus_num = wb[Bus, 'BusNum'].max() + 100
    line = create_object_string("Branch", 1, 2, "1")
    wb.esa.TapTransmissionLine(
        line,
        50.0,          # Tap location (% of line length)
        new_bus_num,   # New bus number
        "CAPACITANCE"  # Shunt model type
    )
    
    # Split a bus into multiple buses
    target_bus = create_object_string("Bus", 1)
    wb.esa.SplitBus(
        target_bus,
        new_bus_num + 1,
        insert_tie=True,
        line_open=False,
        branch_device_type="Breaker"
    )

Data Export and Reporting
==========================

Export to CSV
~~~~~~~~~~~~~

Save data to CSV files:

.. code-block:: python

    import os
    
    # Export bus data to CSV
    report_path = os.path.abspath("buses.csv")
    wb.esa.SaveDataWithExtra(
        filename=report_path,
        filetype="CSVCOLHEADER",
        objecttype="Bus",
        fieldlist=["BusNum", "BusName", "BusPUVolt", "BusAngle"],
        header_list=["Report_Date"],
        header_value_list=["2026-01-08"]
    )

Export to Excel
~~~~~~~~~~~~~~~

Export data directly to Excel worksheets:

.. code-block:: python

    # Export branch loading data to Excel
    excel_path = os.path.abspath("branch_loading.xlsx")
    wb.esa.SendToExcelAdvanced(
        objecttype="Branch",
        fieldlist=["BusNum", "BusNum:1", "LineCircuit", "LineMVA", "LinePercent"],
        filter_name="",
        worksheet="Branch Loading Report",
        workbook=excel_path
    )

Run Custom Scripts
==================

PowerWorld Auxiliary Scripts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Execute raw PowerWorld auxiliary commands:

.. code-block:: python

    # Run PowerWorld AUX commands directly
    wb.esa.RunScriptCommand('SolvePowerFlow(RECTNEWT);')
    
    # Load auxiliary script from file
    wb.load_script("path/to/script.aux")
    
    # Run multiple commands in sequence
    commands = [
        'SolvePowerFlow(RECTNEWT);',
        'CalculateLODF(Branch, 1);'
    ]
    for cmd in commands:
        wb.esa.RunScriptCommand(cmd)