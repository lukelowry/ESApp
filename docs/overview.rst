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

- **Intuitive Indexing** — Access grid data with ``wb[Bus, "BusPUVolt"]`` syntax
- **Full SimAuto Coverage** — All PowerWorld API functions through modular mixins
- **Pandas Integration** — Every query returns a DataFrame
- **Transient Stability** — Fluent API with ``TS`` field intellisense
- **Analysis Utilities** — Built-in GIC, network topology, and contingency tools

About
-----

Developed by **Luke Lowery** and **Adam Birchfield** at Texas A&M University
(`Birchfield Research Group <https://birchfield.engr.tamu.edu/>`_).
Licensed under `Apache 2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_.

If you use ESA++ in research, please cite:

.. code-block:: bibtex

    @article{esa2020,
      title={Easy SimAuto (ESA): A Python Package for PowerWorld Simulator Automation},
      author={Mao, Zeyu and Thayer, Brandon and Liu, Yijing and Birchfield, Adam},
      year={2020}
    }

Installation
------------

Requires PowerWorld Simulator with SimAuto enabled and Python 3.9+.

.. code-block:: bash

    pip install esapp

Verify with:

.. code-block:: python

    from esapp import GridWorkBench
    wb = GridWorkBench("path/to/case.pwb")
    print(wb)

Basic Usage
-----------

Open a case and import the component types you need:

.. code-block:: python

    from esapp import GridWorkBench
    from esapp.components import Bus, Gen, Branch

    wb = GridWorkBench("path/to/case.pwb")

Reading Data
~~~~~~~~~~~~

Use bracket notation with a component class and field name(s):

.. code-block:: python

    # Single field → Series
    voltages = wb[Bus, "BusPUVolt"]

    # Multiple fields → DataFrame
    bus_info = wb[Bus, ["BusNum", "BusName", "BusPUVolt"]]

    # All fields
    branches = wb[Branch, :]

    # Use class attributes for IDE autocomplete
    data = wb[Bus, [Bus.BusName, Bus.BusPUVolt, Bus.BusAngle]]

Writing Data
~~~~~~~~~~~~

Assign values using the same bracket syntax:

.. code-block:: python

    # Broadcast scalar to all rows
    wb[Bus, "BusPUVolt"] = 1.05
    wb[Gen, "GenStatus"] = "Closed"

    # Update specific rows with a DataFrame (must include keys)
    import pandas as pd
    updates = pd.DataFrame({
        "BusNum": [1, 2, 5],
        "BusPUVolt": [1.02, 1.01, 0.99]
    })
    wb[Bus] = updates

Solving & Analysis
~~~~~~~~~~~~~~~~~~

Run power flow and check results:

.. code-block:: python

    wb.pflow()                         # Solve power flow
    low_v = wb.violations(v_min=0.9)   # Find voltage violations
    wb.save()                          # Save case

Common Helpers
~~~~~~~~~~~~~~

Shortcuts for frequent operations:

.. code-block:: python

    # Switch branches
    wb.open_branch(bus1=1, bus2=2, ckt="1")
    wb.close_branch(bus1=1, bus2=2, ckt="1")

    # Mode control
    wb.edit_mode()                     # Enter EDIT mode
    wb.run_mode()                      # Enter RUN mode
    wb.flatstart()                     # Reset to 1.0 pu, 0 angle

Direct SAW Access
~~~~~~~~~~~~~~~~~

For operations not wrapped by GridWorkBench, use the underlying SAW interface:

.. code-block:: python

    saw = wb.esa
    saw.SolvePrimalLP()
    saw.RunScriptCommand("EnterMode(Edit);")

Network Matrices
~~~~~~~~~~~~~~~~

Extract system matrices for external analysis:

.. code-block:: python

    from esapp.utils import BranchType

    Y = wb.ybus()                              # Admittance matrix
    A = wb.network.incidence()                 # Incidence matrix
    L = wb.network.laplacian(BranchType.LENGTH)  # Graph Laplacian

Transient Stability
-------------------

Use ``TSWatch`` and ``ContingencyBuilder`` from ``esapp.utils`` for time-domain simulation.

.. code-block:: python

    from esapp import GridWorkBench, TS
    from esapp.components import Gen, Bus
    from esapp.utils import TSWatch, ContingencyBuilder, get_ts_results, process_ts_results

    wb = GridWorkBench("path/to/case.pwb")

    # Register fields to record
    tsw = TSWatch()
    tsw.watch(Gen, [TS.Gen.P, TS.Gen.W, TS.Gen.Delta])
    tsw.watch(Bus, [TS.Bus.VPU])

    # Define a fault contingency
    ctg = (ContingencyBuilder("BusFault")
        .at(1.0).fault_bus("101")
        .at(1.1).clear_fault("101"))
    ctg_df, elem_df = ctg.to_dataframes()

    # Write contingency to PowerWorld
    from esapp.components import TSContingency, TSContingencyElement
    wb[TSContingency] = ctg_df
    wb[TSContingencyElement] = elem_df

    # Prepare, solve, and retrieve results
    fields = tsw.prepare(wb)
    wb.esa.TSSolve("BusFault")
    meta, results = get_ts_results(wb.esa, "BusFault", fields)
    meta, results = process_ts_results(meta, results, "BusFault")
