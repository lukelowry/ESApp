Getting Started
===============

.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: https://opensource.org/licenses/Apache-2.0
   :alt: License

.. image:: https://img.shields.io/badge/python-3.9%2B-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.9+

.. image:: https://img.shields.io/badge/docs-Read%20the%20Docs-blue.svg
   :target: https://esapp.readthedocs.io/
   :alt: Documentation

ESA++ is an open-source Python toolkit for power system automation, providing a high-performance
wrapper for PowerWorld's Simulator Automation Server (SimAuto). It transforms complex COM calls
into intuitive, Pythonic operations.

Key Features
------------

- **Intuitive Indexing**: Access and modify grid components using familiar syntax (e.g., ``wb[Bus, "BusPUVolt"]``)
- **Comprehensive SimAuto Wrapper**: Full coverage of PowerWorld's API through modular mixins
- **Native Pandas Integration**: All data operations return DataFrames for immediate analysis
- **Transient Stability Support**: Fluent API for dynamic simulation with ``TS`` field intellisense
- **Advanced Analysis Apps**: Built-in modules for GIC, network topology, and forced oscillation detection

Authors
-------

ESA++ is developed and maintained by **Luke Lowery** and **Adam Birchfield** at Texas A&M University.

- `Birchfield Research Group <https://birchfield.engr.tamu.edu/>`_
- `Luke Lowery's Research <https://lukelowry.github.io/>`_

Citation
--------

If you use this toolkit in your research, please cite:

.. code-block:: bibtex

    @article{esa2020,
      title={Easy SimAuto (ESA): A Python Package for PowerWorld Simulator Automation},
      author={Mao, Zeyu and Thayer, Brandon and Liu, Yijing and Birchfield, Adam},
      year={2020}
    }

License
-------

Distributed under the `Apache License 2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_.

Installation
------------

**Prerequisites**

- PowerWorld Simulator with SimAuto (COM interface) enabled
- Python 3.9+ and ``pip`` available on your path

**Install the package**

.. code-block:: bash

    pip install esapp

For development against this repository:

.. code-block:: bash

    pip install -e .

**Verify the installation**

.. code-block:: python

    from esapp import GridWorkBench
    wb = GridWorkBench("path/to/your/case.pwb")
    print(wb)

Quick Start
-----------

Create a workbench, import the grid components you care about, and you're ready to query or modify the
case. Keep paths absolute when launching PowerWorld so SimAuto can resolve the file cleanly.

.. code-block:: python

    from esapp import GridWorkBench
    from esapp.components import Bus, Gen, Branch

    wb = GridWorkBench("path/to/case.pwb")

    # Read data
    voltages = wb[Bus, "BusPUVolt"]

    # Solve and analyze
    V = wb.pflow()
    violations = wb.violations(v_min=0.95)

    # Modify and save
    wb[Gen, "GenMW"] = 100.0
    wb.save()

Indexing Basics
---------------

Indexing always follows the same pattern: component class first, then the fields you want. Leaving the
second slot as ``:`` returns every available field for that component.

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

Writing Data
------------

Writes mirror reads: same indexing form, but assign on the right-hand side. Broadcasting works for scalars;
bulk operations use DataFrames that include primary keys.

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

Convenience Helpers
-------------------

Shortcuts for common edits when you do not want to assemble DataFrames or craft SAW calls:

.. code-block:: python

    wb.set_gen(bus=5, id="1", mw=150.0, mvar=40.0, status="Closed")
    wb.set_load(bus=10, id="1", mw=90.0, mvar=25.0, status="Closed")

    wb.open_branch(bus1=1, bus2=2, ckt="1")
    wb.close_branch(bus1=1, bus2=2, ckt="1")

    wb.scale_gen(factor=1.05)
    wb.scale_load(factor=0.95)

Calling SAW Directly
--------------------

Access the full SimAuto interface when you need lower-level operations or features not surfaced on the
workbench helpers.

.. code-block:: python

    saw = wb.esa
    saw.SolveAC_OPF()
    saw.RunScriptCommand("SolvePowerFlow(RECTNEWT);")

Matrices and Topology
---------------------

Extract matrices and mappings without building a full study workflow:

.. code-block:: python

    Y = wb.ybus()
    A = wb.network.incidence()
    busmap = wb.network.busmap()
    from esapp.apps.network import Network
    L = wb.network.laplacian(weights=Network.BranchType.LENGTH)

Transient Stability
-------------------

ESA++ provides a fluent API for transient stability (TS) simulation through the ``Dynamics`` app.

**Setup and field watching**

.. code-block:: python

    from esapp import GridWorkBench, TS
    from esapp.components import Gen, Bus

    wb = GridWorkBench("path/to/case.pwb")

    # Set simulation duration
    wb.dyn.runtime = 10.0

    # Watch generator power, speed, and angle
    wb.dyn.watch(Gen, [TS.Gen.P, TS.Gen.W, TS.Gen.Delta])

**Define contingencies**

.. code-block:: python

    (wb.dyn.contingency("Bus_Fault")
           .at(1.0).fault_bus("101")
           .at(1.1).clear_fault("101"))

**Run and plot**

.. code-block:: python

    meta, results = wb.dyn.solve("Bus_Fault")
    wb.dyn.plot(meta, results)
