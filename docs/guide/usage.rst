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

Retrieve just the primary keys (identifiers) for all objects:

.. code-block:: python

    bus_keys = wb[Bus]

**Get Specific Fields**

Pass field names as strings to retrieve specific data.

Single field returns a Series:

.. code-block:: python

    voltages = wb[Bus, "BusPUVolt"]

Multiple fields return a DataFrame:

.. code-block:: python

    bus_info = wb[Bus, ["BusName", "BusPUVolt", "BusAngle"]]
    gen_data = wb[Gen, ["GenMW", "GenMVR", "GenStatus"]]

**Get All Available Fields**

Use the slice operator ``:`` to retrieve every field:

.. code-block:: python

    all_bus_data = wb[Bus, :]
    all_gen_data = wb[Gen, :]
    all_branch_data = wb[Branch, :]

**Using Component Attributes for IDE Support**

Use component class attributes for autocomplete and type safety:

.. code-block:: python

    data = wb[Bus, [Bus.BusName, Bus.BusPUVolt, Bus.BusAngle]]
    gen_output = wb[Gen, [Gen.GenMW, Gen.GenMVR, Gen.GenStatus]]

Filtering Data
~~~~~~~~~~~~~~

Since returned data is in standard Pandas DataFrames, use all of Pandas' filtering capabilities.

**Filter by area:**

.. code-block:: python

    all_buses = wb[Bus, ["BusNum", "BusName", "AreaNum", "BusPUVolt"]]
    area_1_buses = all_buses[all_buses['AreaNum'] == 1]

**Find overloaded branches:**

.. code-block:: python

    branches = wb[Branch, ["BusNum", "BusNum:1", "LinePercent", "LineLimit"]]
    overloaded = branches[branches['LinePercent'] > 100.0]

**Filter by status:**

.. code-block:: python

    gens = wb[Gen, ["GenMW", "GenStatus"]]
    offline = gens[gens['GenStatus'] == 'Open']
    
    buses_low_voltage = all_buses[all_buses['BusPUVolt'] < 0.95]

Data Modification
=================

The same indexing syntax is used to update values in the PowerWorld case.

Broadcasting a Scalar Value
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set a single value for all objects of a type:

.. code-block:: python

    wb[Bus, "BusPUVolt"] = 1.05
    wb[Gen, "GenStatus"] = "Closed"

Updating Multiple Fields
~~~~~~~~~~~~~~~~~~~~~~~~

Update multiple fields simultaneously:

.. code-block:: python

    wb[Gen, ["GenMW", "GenMVR"]] = [100.0, 20.0]

Bulk Update from DataFrame
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform bulk updates using a DataFrame with primary keys:

.. code-block:: python

    import pandas as pd
    
    updates = pd.DataFrame({
        'BusNum': [1, 2, 5, 10],
        'BusPUVolt': [1.02, 1.03, 0.99, 1.01]
    })
    
    wb[Bus] = updates

.. note::
   DataFrame must include primary key columns (e.g., ``BusNum`` for Bus objects).

Component-Specific Methods
===~~~~~~~~~~~~~~~~~~~~~~~~

ESA++ provides convenience methods for common modifications:

.. code-block:: python

    wb.set_gen(bus=5, id="1", mw=150.0, mvar=50.0, status="Closed")
    wb.set_load(bus=10, id="1", mw=100.0, mvar=30.0, status="Closed")
    
    wb.open_branch(from_bus=1, to_bus=2, id="1")
    wb.close_branch(from_bus=1, to_bus=2, id="1")
    
    wb.scale_gen(scale_factor=1.1)
    wb.scale_load(scale_factor=0.9)

Analysis and Simulation
=======================

Power Flow Solution
~~~~~~~~~~~~~~~~~~~

Solve the AC power flow and retrieve results:

.. code-block:: python

    voltages = wb.pflow()
    voltage_mags = abs(voltages)
    
    bus_voltage_df = wb[Bus, ["BusNum", "BusPUVolt", "BusAngle"]]
    pf_converged = wb.esa.pflow_converged

Violation Detection
~~~~~~~~~~~~~~~~~~~

Automatically detect system violations:

.. code-block:: python

    violations = wb.find_violations(
        v_min=0.95,
        v_max=1.05,
        branch_max_pct=100
    )

Returns a dictionary containing:

:buses_low: Buses below minimum voltage
:buses_high: Buses above maximum voltage
:branches_overloaded: Overloaded branches

Contingency Analysis
~~~~~~~~~~~~~~~~~~~~

Perform N-1 contingency studies:

.. code-block:: python

    wb.pflow()
    wb.auto_insert_contingencies()
    wb.solve_contingencies()
    
    from esapp.grid import ViolationCTG
    violations = wb[ViolationCTG, :]
    critical = violations[violations['ViolatedRecord'] == 'Yes']

Optimization and Control
~~~~~~~~~~~~~~~~~~~~~~~~

Run optimal power flow and security-constrained optimization:

.. code-block:: python

    wb.esa.SolveAC_OPF()
    
    wb.esa.InitializePrimalLP()
    wb.auto_insert_contingencies()
    wb.esa.SolveFullSCOPF()
    
    opf_cost = wb[Area, "GenProdCost"]

Sensitivity Analysis
~~~~~~~~~~~~~~~~~~~~

Calculate power transfer distribution factors and sensitivity:

.. code-block:: python

    ptdf = wb.ptdf(seller="Area 1", buyer="Area 2", method='DC')
    lodf = wb.lodf(branch=(1, 2, "1"), method='DC')

Transient Stability
~~~~~~~~~~~~~~~~~~~

Perform transient stability analysis:

.. code-block:: python

    wb.esa.TSInitialize()
    
    from esapp.saw._helpers import create_object_string
    branch = create_object_string("Branch", 1, 2, "1")
    wb.esa.TSCalculateCriticalClearTime(branch)
    
    wb.esa.TSAutoSavePlots(
        plot_names=["Generator Frequencies", "Bus Voltages"],
        ctg_names=["Fault_at_Bus_1"]
    )

GIC Analysis
~~~~~~~~~~~~

Calculate geomagnetically induced currents:

.. code-block:: python

    wb.calculate_gic(max_field=1.0, direction=90.0)
    
    from esapp.grid import GICXFormer
    gic_results = wb[GICXFormer, ["BusNum", "BusNum:1", "GICXFNeutralAmps"]]
    max_gic = gic_results['GICXFNeutralAmps'].max()

Available Transfer Capability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calculate available transfer capability between areas:

.. code-block:: python

    from esapp.saw._helpers import create_object_string
    
    wb.esa.SetData("ATC_Options", ["Method"], ["IteratedLinearThenFull"])
    
    seller = create_object_string("Area", 1)
    buyer = create_object_string("Area", 2)
    wb.esa.DetermineATC(seller, buyer)
    
    results = wb.esa.GetATCResults(
        ["MaxFlow", "LimitingContingency", "LimitingElement"]
    )

Network Topology and Matrices
==============================

System Matrix Extraction
~~~~~~~~~~~~~~~~~~~~~~~~

Extract system matrices for mathematical analysis:

.. code-block:: python

    Y = wb.ybus()
    A = wb.network.incidence()
    
    from esapp.apps.network import Network
    L = wb.network.laplacian(weights=Network.BranchType.LENGTH)
    
    busmap = wb.network.busmap()

Topology Analysis
~~~~~~~~~~~~~~~~~~

Analyze network structure:

.. code-block:: python

    bus_coords = wb.network.busmap()
    branch_lengths = wb.network.lengths()
    
    from esapp.grid import Zone
    zones = wb[Zone, :]

Network Modification
~~~~~~~~~~~~~~~~~~~~

Programmatically modify the network topology:

.. code-block:: python

    from esapp.saw._helpers import create_object_string
    
    new_bus_num = wb[Bus, 'BusNum'].max() + 100
    line = create_object_string("Branch", 1, 2, "1")
    wb.esa.TapTransmissionLine(
        line,
        50.0,
        new_bus_num,
        "CAPACITANCE"
    )
    
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

    wb.esa.RunScriptCommand('SolvePowerFlow(RECTNEWT);')
    wb.load_script("path/to/script.aux")
    
    commands = [
        'SolvePowerFlow(RECTNEWT);',
        'CalculateLODF(Branch, 1);'
    ]
    for cmd in commands:
        wb.esa.RunScriptCommand(cmd)