Tutorial
========

This tutorial covers the fundamentals of using ESA++ to interact with PowerWorld cases.

Getting Started
---------------

Initialize the ``GridWorkBench`` with your PowerWorld case file:

.. code-block:: python

    from esapp import GridWorkBench
    from esapp.grid import Bus, Gen, Load, Branch
    
    wb = GridWorkBench("path/to/your/case.pwb")

Basic Data Access
-----------------

Retrieve component data using NumPy-style indexing.

**Get specific fields:**

.. code-block:: python

    buses = wb[Bus, ["BusNum", "BusName", "BusPUVolt"]]

**Get all fields:**

.. code-block:: python

    generators = wb[Gen, :]

**Single field returns a Series:**

.. code-block:: python

    voltages = wb[Bus, "BusPUVolt"]

Filtering Data
~~~~~~~~~~~~~~

Use Pandas operations on returned DataFrames.

**Filter by condition:**

.. code-block:: python

    all_buses = wb[Bus, ["BusNum", "AreaNum", "BusPUVolt"]]
    area_1_buses = all_buses[all_buses['AreaNum'] == 1]

**Find violations:**

.. code-block:: python

    low_voltage = all_buses[all_buses['BusPUVolt'] < 0.95]

Power Flow Analysis
-------------------

Solve the power flow and check for violations.

.. code-block:: python

    voltages = wb.pflow()
    violations = wb.find_violations(v_min=0.95, v_max=1.05)

.. note::
   ``pflow()`` returns a Series of complex voltages at each bus.

Modifying Data
--------------

Update component parameters.

**Set generator output:**

.. code-block:: python

    wb.set_gen(bus=5, id="1", mw=150.0)

**Broadcast to all components:**

.. code-block:: python

    wb[Bus, "BusVoltSet"] = 1.02

**Bulk update from DataFrame:**

.. code-block:: python

    import pandas as pd
    
    updates = pd.DataFrame({
        'BusNum': [1, 2],
        'BusPUVolt': [1.02, 1.01]
    })
    wb[Bus] = updates

Common Workflows
----------------

Contingency Analysis
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    wb.pflow()
    wb.auto_insert_contingencies()
    wb.solve_contingencies()
    
    from esapp.grid import ViolationCTG
    violations = wb[ViolationCTG, :]

System Matrices
~~~~~~~~~~~~~~~

.. code-block:: python

    Y = wb.ybus()
    A = wb.network.incidence()
    busmap = wb.network.busmap()

:Y: Admittance matrix (sparse)
:A: Bus-branch incidence matrix
:busmap: Bus number to matrix index mapping

Saving and Closing
------------------

.. code-block:: python

    wb.save()
    wb.save("new_case.pwb")
    wb.close()

Next Steps
----------

- See :doc:`usage` for advanced features (sensitivity analysis, GIC, optimization, etc.)
- Explore :doc:`../examples` for detailed examples
- Check :doc:`../api/workbench` for complete API reference