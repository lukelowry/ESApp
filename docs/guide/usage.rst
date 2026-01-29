Usage Guide
===========

This guide explains the core mechanics of ESA++—how to index, read, and write fields and when to drop down
to SAW. If you want goal-driven, end-to-end scripts, head to :doc:`examples`. Think of this page as the
reference for everyday interactions: get data, filter it, push edits back, and call lower-level SAW features
when you need to.

Quick start
-----------

Create a workbench, import the grid components you care about, and you are ready to query or modify the
case. Keep paths absolute when launching PowerWorld so SimAuto can resolve the file cleanly.

.. code-block:: python

    from esapp import GridWorkBench
    from esapp.components import Bus, Gen, Branch

    wb = GridWorkBench("path/to/case.pwb")

Indexing basics
---------------

Indexing always follows the same pattern: component class first, then the fields you want. Leaving the
second slot as ``:`` returns every available field for that component. Use specific fields for small payloads
and ``:`` when you need the full shape of the object.

**Primary keys only**

.. code-block:: python

    bus_keys = wb[Bus]

**Specific fields**

.. code-block:: python

    voltages = wb[Bus, "BusPUVolt"]
    bus_info = wb[Bus, ["BusName", "BusPUVolt"]]
    gen_info = wb[Gen, ["GenMW", "GenStatus"]]

**All fields**

.. code-block:: python

    branches = wb[Branch, :]

**Field attributes for autocomplete**

.. code-block:: python

    bus_data = wb[Bus, [Bus.BusName, Bus.BusPUVolt, Bus.BusAngle]]

Filtering and slicing
---------------------

Returned objects are Pandas DataFrames or Series, so filter and slice with normal Pandas operations. Keep
the heavy lifting in Pandas, then write only the results you need back to PowerWorld.

.. code-block:: python

    buses = wb[Bus, ["BusNum", "AreaNum", "BusPUVolt"]]
    area_1 = buses[buses["AreaNum"] == 1]
    low_v = buses[buses["BusPUVolt"] < 0.95]

Writing data
------------

Writes mirror reads: same indexing form, but assign on the right-hand side. Broadcasting works for scalars;
bulk operations use DataFrames that include primary keys. Start with small, targeted updates before applying
wider changes.

**Broadcast a scalar**

.. code-block:: python

    wb[Bus, "BusPUVolt"] = 1.05
    wb[Gen, "GenStatus"] = "Closed"

**Update multiple fields**

.. code-block:: python

    wb[Gen, ["GenMW", "GenMVR"]] = [120.0, 25.0]

**Bulk update with DataFrame**

.. code-block:: python

    import pandas as pd

    updates = pd.DataFrame({
        "BusNum": [1, 2, 5],
        "BusPUVolt": [1.02, 1.01, 0.99]
    })

    wb[Bus] = updates

.. note::
   Include primary key columns (e.g., ``BusNum``) in bulk updates.

Convenience helpers
-------------------

Shortcuts for common edits when you do not want to assemble DataFrames or craft SAW calls:

.. code-block:: python

    wb.set_gen(bus=5, id="1", mw=150.0, mvar=40.0, status="Closed")
    wb.set_load(bus=10, id="1", mw=90.0, mvar=25.0, status="Closed")

    wb.open_branch(bus1=1, bus2=2, ckt="1")
    wb.close_branch(bus1=1, bus2=2, ckt="1")

    wb.scale_gen(factor=1.05)
    wb.scale_load(factor=0.95)

Calling SAW directly
--------------------

Access the full SimAuto interface when you need lower-level operations or features not surfaced on the
workbench helpers. Prefer the helpers for routine tasks; reach for SAW when you need the complete API.

.. code-block:: python

    saw = wb.esa
    saw.SolveAC_OPF()
    saw.RunScriptCommand("SolvePowerFlow(RECTNEWT);")

Matrices and topology
---------------------

Extract matrices and mappings without building a full study workflow. Use these as building blocks for
linearized studies, external analytics, or custom contingency logic.

.. code-block:: python

    Y = wb.ybus()
    A = wb.network.incidence()
    busmap = wb.network.busmap()
    from esapp.apps.network import Network
    L = wb.network.laplacian(weights=Network.BranchType.LENGTH)

Transient stability simulation
------------------------------

ESA++ provides a fluent API for transient stability (TS) simulation through the ``Dynamics`` app, accessed
via ``wb.dyn``. The ``TS`` class offers IDE intellisense for all available result fields.

**Setup and field watching**

Import the ``TS`` class for autocomplete on transient stability result fields. Register which fields to
record during simulation using ``watch()``:

.. code-block:: python

    from esapp import GridWorkBench, TS
    from esapp.components import Gen, Bus

    wb = GridWorkBench("path/to/case.pwb")

    # Set simulation duration
    wb.dyn.runtime = 10.0

    # Watch generator power, speed, and angle
    wb.dyn.watch(Gen, [TS.Gen.P, TS.Gen.W, TS.Gen.Delta])

    # Watch bus voltage
    wb.dyn.watch(Bus, [TS.Bus.VPU])

**Define contingencies**

Build contingencies using the fluent builder API. Chain time-based events with ``.at()`` to set the
time cursor:

.. code-block:: python

    # Fluent contingency definition
    (wb.dyn.contingency("Bus_Fault")
           .at(1.0).fault_bus("101")        # 3-phase fault at t=1.0s
           .at(1.1).clear_fault("101"))     # Clear at t=1.1s

    # Shorthand for simple bus faults
    wb.dyn.bus_fault("SimpleFault", bus="5", fault_time=1.0, duration=0.0833)

**Run simulation and plot results**

Execute the simulation and visualize results grouped by object type and metric:

.. code-block:: python

    meta, results = wb.dyn.solve("Bus_Fault")
    wb.dyn.plot(meta, results)

**Available contingency actions**

The ``ContingencyBuilder`` supports these event types:

- ``fault_bus(bus)``: Apply 3-phase solid fault to a bus
- ``clear_fault(bus)``: Clear fault at a bus
- ``trip_gen(bus, gid)``: Trip (open) a generator
- ``trip_branch(from_bus, to_bus, ckt)``: Trip (open) a branch

**TS field constants**

The ``TS`` class provides organized access to all transient stability result fields with full IDE
autocomplete support:

.. code-block:: python

    from esapp import TS

    # Generator fields
    TS.Gen.P          # Active power output
    TS.Gen.W          # Rotor speed
    TS.Gen.Delta      # Rotor angle

    # Bus fields
    TS.Bus.VPU        # Voltage magnitude (per-unit)
    TS.Bus.Freq       # Bus frequency

    # Area fields
    TS.Area.Frequency # Area average frequency

**List available dynamic models**

Inspect which transient stability models are present in the case:

.. code-block:: python

    models = wb.dyn.list_models()
    print(models)
    # Returns DataFrame with columns: Category, Model, Object Type

