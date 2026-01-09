Usage Guide
===========

This guide explains the core mechanics of ESA++—how to index, read, and write fields and when to drop down
to SAW. If you want goal-driven, end-to-end scripts, head to :doc:`../examples`. Think of this page as the
reference for everyday interactions: get data, filter it, push edits back, and call lower-level SAW features
when you need to.

Quick start
-----------

Create a workbench, import the grid components you care about, and you are ready to query or modify the
case. Keep paths absolute when launching PowerWorld so SimAuto can resolve the file cleanly.

.. code-block:: python

    from esapp import GridWorkBench
    from esapp.grid import Bus, Gen, Branch

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

    wb.open_branch(from_bus=1, to_bus=2, id="1")
    wb.close_branch(from_bus=1, to_bus=2, id="1")

    wb.scale_gen(scale_factor=1.05)
    wb.scale_load(scale_factor=0.95)

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

Where to go next
----------------
- End-to-end scripts: :doc:`../examples`
- Full API reference: :doc:`../api`
- Development and tests: :doc:`../dev/tests`