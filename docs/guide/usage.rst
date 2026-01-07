Usage Guide
===========

This guide covers more advanced usage patterns of the ESA++ toolkit.

Indexing with IndexTool
-----------------------

The ``IndexTool`` is the heart of ESA++, providing a powerful, Pythonic way to interact with PowerWorld data. Instead of calling verbose SimAuto functions, you use standard Python indexing syntax on the ``GridWorkBench`` object.

Data Retrieval
~~~~~~~~~~~~~~

Retrieving data is as simple as indexing the workbench with a component class (like ``Bus``, ``Gen``, or ``Line``).

**1. Get Primary Keys**

To get just the primary keys for all objects of a type:

.. code-block:: python

    from gridwb.grid.components import Bus
    bus_keys = wb[Bus]

**2. Get Specific Fields**

Pass a string or a list of strings to retrieve specific fields:

.. code-block:: python

    # Single field
    voltages = wb[Bus, 'BusPUVolt']

    # Multiple fields
    bus_info = wb[Bus, ['BusName', 'BusPUVolt', 'BusAngle']]

**3. Get All Fields**

Use the slice operator ``:`` to retrieve all fields defined for that component:

.. code-block:: python

    all_gen_data = wb[Gen, :]

**4. Using Component Attributes**

For better IDE support and to avoid typos, you can use the attributes defined on the component classes:

.. code-block:: python

    data = wb[Bus, [Bus.BusName, Bus.BusPUVolt]]

Data Modification
~~~~~~~~~~~~~~~~~

The same indexing syntax is used to update values in the PowerWorld case.

**1. Broadcasting a Scalar**

Set a single value for all objects of a type:

.. code-block:: python

    # Set all bus voltages to 1.05 pu
    wb[Bus, 'BusPUVolt'] = 1.05

**2. Updating Multiple Fields**

You can update multiple fields at once by passing a list of values:

.. code-block:: python

    # Update MW and MVAR for all generators
    wb[Gen, ['GenMW', 'GenMVR']] = [100.0, 20.0]

**3. Bulk Update from DataFrame**

If you have a DataFrame containing updated data (including the necessary primary keys), you can perform a bulk update:

.. code-block:: python

    # Assuming 'df' is a DataFrame with 'BusNum' and updated 'BusPUVolt'
    wb[Bus] = df


The Adapter
-----------

The ``Adapter`` (accessed via ``wb.func``) provides a collection of high-level helper functions for common tasks:

.. code-block:: python

    # Find voltage violations
    violations = wb.func.find_violations(v_min=0.95, v_max=1.05)
    
    # Calculate PTDF between two areas
    ptdf_df = wb.func.ptdf('[AREA 1]', '[AREA 2]')

    # Run a full N-1 contingency analysis
    wb.func.auto_insert_contingencies()
    wb.func.solve_contingencies()
    violations = wb.func.get_contingency_violations()


The App Ecosystem
-----------------

ESA++ includes specialized "Apps" for complex analysis. For example, the GIC tool:

.. code-block:: python

    # Access the GIC application
    gic_results = wb.app.gic.run_uniform_field(field_mag=1.0, angle=0)

    # Use the Network app for topology analysis
    is_connected = wb.app.network.is_connected()
    islands = wb.app.network.get_islands()


Working with Matrices
---------------------

ESA++ makes it easy to extract system matrices for mathematical analysis:

.. code-block:: python

    # Get the sparse Y-Bus matrix
    ybus = wb.ybus()
    
    # Get the bus-branch incidence matrix
    incidence = wb.network.incidence()
    
    # Get the Power Flow Jacobian
    jacobian = wb.io.esa.get_jacobian()


Custom Scripts
--------------

You can run raw PowerWorld auxiliary scripts directly:

.. code-block:: python

    wb.func.command('SolvePowerFlow(RECTNEWT);')