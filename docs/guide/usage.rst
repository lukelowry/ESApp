Usage Guide
===========

This guide covers more advanced usage patterns of the ESA++ toolkit.

I/O with Numpy-Style Indexing
~~~~~~~~~~~~~~

Retrieving data is as simple as indexing the workbench with a component class (like ``Bus``, ``Gen``, or ``Line``).

**Get Primary Keys**

To get just the primary keys for all objects of a type:

.. code-block:: python

    from esapp.components import Bus
    bus_keys = wb[Bus]

**Get Specific Fields**

Pass a string or a list of strings to retrieve specific fields:

.. code-block:: python

    # Single field
    voltages = wb[Bus, "BusPUVolt"]

    # Multiple fields
    bus_info = wb[Bus, ["BusName", "BusPUVolt", "BusAngle"]]

**Get All Fields**

Use the slice operator ``:`` to retrieve all fields defined for that component:

.. code-block:: python

    all_gen_data = wb[Gen, :]

**Using Component Attributes**

For better IDE support and to avoid typos, you can use the attributes defined on the component classes:

.. code-block:: python

    data = wb[Bus, [Bus.BusName, Bus.BusPUVolt]]

Data Modification
~~~~~~~~~~~~~~~~~

The same indexing syntax is used to update values in the PowerWorld case.

**Broadcasting a Scalar**

Set a single value for all objects of a type:

.. code-block:: python

    # Set all bus voltages to 1.05 pu
    wb[Bus, "BusPUVolt"] = 1.05

**Updating Multiple Fields**

You can update multiple fields at once by passing a list of values:

.. code-block:: python

    # Update MW and MVAR for all generators
    wb[Gen, ["GenMW", "GenMVR"]] = [100.0, 20.0]

**Bulk Update from DataFrame**

If you have a DataFrame containing updated data (including the necessary primary keys), you can perform a bulk update:

.. code-block:: python

    # Assuming 'df' is a DataFrame with 'BusNum' and updated 'BusPUVolt'
    wb[Bus] = df


Specific Applications
---------------------

ESA++ includes specialized "Apps" for complex analysis. For example, the GIC tool:

.. code-block:: python

    # Access the GIC application
    gic_results = wb.calculate_gic(max_field=1.0, direction=0)

    # Use the Network app for topology analysis
    is_connected = wb.network.is_connected()
    islands = wb.islands()


Power System Matricies
----------------------

ESA++ makes it easy to extract system matrices for mathematical analysis:

.. code-block:: python

    # Get the sparse Y-Bus matrix
    ybus = wb.ybus()
    
    # Get the bus-branch incidence matrix
    incidence = wb.network.incidence()
    
    # Get the Power Flow Jacobian
    jacobian = wb.esa.get_jacobian()


Custom AUX Scripts
------------------

You can run raw PowerWorld auxiliary scripts directly:

.. code-block:: python

    wb.command('SolvePowerFlow(RECTNEWT);')