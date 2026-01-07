Usage Guide
===========

This guide covers more advanced usage patterns of the ESA++ toolkit.

The IndexTool
-------------

The ``IndexTool`` (accessed via ``wb.io``) is the core engine for data I/O. It supports broadcasting values to multiple objects:

.. code-block:: python

    # Set all bus voltages to 1.05 pu
    wb[Bus, 'BusPUVolt'] = 1.05

    # Update multiple fields for a specific object
    wb[Gen, (1, '1'), ['GenMW', 'GenMVAR']] = [100.0, 20.0]


The Adapter
-----------

The ``Adapter`` (accessed via ``wb.func``) provides a collection of high-level helper functions for common tasks:

.. code-block:: python

    # Find voltage violations
    violations = wb.func.find_violations(v_min=0.95, v_max=1.05)
    
    # Calculate PTDF between two areas
    ptdf_df = wb.func.ptdf('[AREA 1]', '[AREA 2]')

The App Ecosystem
-----------------

ESA++ includes specialized "Apps" for complex analysis. For example, the GIC tool:

.. code-block:: python

    # Access the GIC application
    gic_results = wb.app.gic.run_uniform_field(field_mag=1.0, angle=0)

Working with Matrices
---------------------

ESA++ makes it easy to extract system matrices for mathematical analysis:

.. code-block:: python

    # Get the sparse Y-Bus matrix
    ybus = wb.ybus()
    
    # Get the bus-branch incidence matrix
    incidence = wb.network.incidence()

Custom Scripts
--------------

You can run raw PowerWorld auxiliary scripts directly:

.. code-block:: python

    wb.func.command('SolvePowerFlow(RECTNEWT);')