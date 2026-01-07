Tutorial
========

This tutorial will walk you through the basics of using ESA++ to interact with a PowerWorld case.

Getting Started
---------------

First, import the ``GridWorkBench`` and point it to your ``.pwb`` file:

.. code-block:: python

    from gridwb import GridWorkBench
    
    # Initialize the workbench
    wb = GridWorkBench("path/to/your/case.pwb")

Accessing Data
--------------

ESA++ uses a unique indexing system to make data retrieval intuitive. You can access grid components using their class types:

.. code-block:: python

    from gridwb.grid.components import Bus, Gen
    
    # Get all bus numbers and names
    buses = wb[Bus, ['BusNum', 'BusName']]
    
    # Get all generator data
    generators = wb[Gen, :]

Running Analysis
----------------

You can solve power flow and retrieve results in one line:

.. code-block:: python

    # Solve power flow and get voltages
    voltages = wb.pflow()
    
    print(voltages)

Saving Changes
--------------

After making modifications, save your case:

.. code-block:: python

    wb.save()