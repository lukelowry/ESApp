ESA++
====================================

.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: https://opensource.org/licenses/Apache-2.0
   :alt: License

An open-source Python toolkit for power system automation, providing a high-performance "syntax-sugar" fork of Easy SimAuto (ESA). This library streamlines interaction with PowerWorld's Simulator Automation Server (SimAuto), transforming complex COM calls into intuitive, Pythonic operations.

Key Features
------------

- **Intuitive Indexing Syntax**: Access and modify grid components using a unique indexing system (e.g., ``wb[Bus, 'BusPUVolt']``) that feels like native Python.
- **Comprehensive SimAuto Wrapper**: Full coverage of PowerWorld's API through the ``SAW`` class, organized into modular mixins for power flow, contingencies, transients, and more.
- **High-Level Adapter Interface**: A collection of simplified "one-liner" functions for common tasks like GIC calculation, fault analysis, and voltage violation detection.
- **Native Pandas Integration**: Every data retrieval operation returns a Pandas DataFrame or Series, enabling immediate analysis, filtering, and visualization.
- **Advanced Analysis Apps**: Built-in specialized modules for Network topology analysis, Geomagnetically Induced Currents (GIC), and Forced Oscillation detection.

Installation
------------

For local development and the latest features, install the package in editable mode from the root directory:

.. code-block:: bash

    python -m pip install gridwb -e .


Documentation
-------------

For a comprehensive tutorial, usage guides, and the full API reference, please visit our `documentation website <https://esapp.readthedocs.io/>`_.

Usage Example
-------------

Here is a quick example of how ESA++ simplifies data access and power flow analysis.

.. code-block:: python

    from gridwb import GridWorkBench
    from gridwb.grid.components import *

    # Open Case
    wb = GridWorkBench("my_grid_model.pwb")

    # 2. Retrieve data 
    bus_data = wb[Bus, ['BusName', 'BusPUVolt']]

    # 3. Solve power flow and get complex voltages
    V = wb.pflow()

    # 4. Perform high-level operations 
    violations = wb.func.find_violations(v_min=0.95)

    # 5. Modify data and save
    wb[Gen, 'GenMW'] = 100.0
    wb.save()

Why ESA++?
----------

Traditional automation of PowerWorld Simulator often involves verbose COM calls and manual data parsing. ESA++ abstracts these complexities:

*   **Speed**: Optimized data transfer between Python and SimAuto.
*   **Clarity**: Code that reads like the engineering operations it performs.
*   **Ecosystem**: Built on top of the proven ESA library, adding modern Python features and better integration with the SciPy stack.


More Examples
-------------

The `examples/ <https://github.com/lukelowry/ESApp/tree/main/docs/examples>`_ directory contains a gallery of demonstrations, including:

- **Basic Data I/O**: Efficiently reading and writing large sets of grid parameters.
- **Contingency Analysis**: Automating N-1 studies and processing violation matrices.
- **Matrix Extraction**: Retrieving Y-Bus and Jacobian matrices for external mathematical modeling.

Testing
-------

ESA++ includes an extensive test suite covering both offline mocks and live PowerWorld connections. To run the tests, install the test dependencies and execute pytest:

.. code-block:: bash

    pip install .[test]
    pytest tests/test_saw.py

Citation
--------

If you use this toolkit in your research or industrial projects, please cite the original ESA work and this fork:

.. code-block:: bibtex

    @article{esa2020,
      title={Easy SimAuto (ESA): A Python Package for PowerWorld Simulator Automation},
      author={Mao, Zeyu and Thayer, Brandon and Liu, Yijing and Birchfield, Adam},
      year={2020}
    }

Authors
-------

ESA++ is maintained by **Luke Lowery** and **Adam Birchfield** at Texas A&M University. You can explore more of our research at the `Birchfield Research Group <https://birchfield.engr.tamu.edu/>`_.

License
-------
Distributed under the `Apache License 2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_.
