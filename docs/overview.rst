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
- **Analysis Apps** — Built-in GIC, network topology, and modal analysis

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
    low_v = wb.violations(v_min=0.95)  # Find voltage violations
    wb.save()                          # Save case

Common Helpers
~~~~~~~~~~~~~~

Shortcuts for frequent operations:

.. code-block:: python

    # Modify individual elements
    wb.set_gen(bus=5, id="1", mw=150.0)
    wb.set_load(bus=10, id="1", mw=90.0)

    # Switch branches
    wb.open_branch(bus1=1, bus2=2, ckt="1")
    wb.close_branch(bus1=1, bus2=2, ckt="1")

    # Scale generation/load
    wb.scale_gen(factor=1.05)
    wb.scale_load(factor=0.95)

Direct SAW Access
~~~~~~~~~~~~~~~~~

For operations not wrapped by GridWorkBench, use the underlying SAW interface:

.. code-block:: python

    saw = wb.esa
    saw.SolveAC_OPF()
    saw.RunScriptCommand("EnterMode(Edit);")

Network Matrices
~~~~~~~~~~~~~~~~

Extract system matrices for external analysis:

.. code-block:: python

    Y = wb.ybus()                # Admittance matrix
    A = wb.network.incidence()   # Incidence matrix
    L = wb.network.laplacian()   # Graph Laplacian

Transient Stability
-------------------

The ``Dynamics`` app provides a fluent interface for time-domain simulation.

.. code-block:: python

    from esapp import GridWorkBench, TS
    from esapp.components import Gen, Bus

    wb = GridWorkBench("path/to/case.pwb")

    # Configure simulation
    wb.dyn.runtime = 10.0
    wb.dyn.watch(Gen, [TS.Gen.P, TS.Gen.W, TS.Gen.Delta])
    wb.dyn.watch(Bus, [TS.Bus.VPU])

    # Define a fault contingency
    (wb.dyn.contingency("BusFault")
           .at(1.0).fault_bus("101")
           .at(1.1).clear_fault("101"))

    # Solve and plot
    meta, results = wb.dyn.solve("BusFault")
    wb.dyn.plot(meta, results)
