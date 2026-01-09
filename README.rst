ESA++
====================================

.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: https://opensource.org/licenses/Apache-2.0
   :alt: License

.. image:: https://img.shields.io/badge/python-3.9%2B-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.9+

.. image:: https://img.shields.io/badge/docs-Read%20the%20Docs-blue.svg
   :target: https://esapp.readthedocs.io/
   :alt: Documentation

.. image:: https://img.shields.io/badge/coverage-92%25-brightgreen.svg
   :alt: Coverage 92%

An open-source Python toolkit for power system automation, providing a high-performance "syntax-sugar" fork of Easy SimAuto (ESA). This library streamlines interaction with PowerWorld's Simulator Automation Server (SimAuto), transforming complex COM calls into intuitive, Pythonic operations.

Key Features
------------

- **Intuitive Indexing Syntax**: Access and modify grid components using a unique indexing system (e.g., ``wb[Bus, "BusPUVolt"]``) that feels like native Python.
- **Comprehensive SimAuto Wrapper**: Full coverage of PowerWorld's API through the ``SAW`` class, organized into modular mixins for power flow, contingencies, transients, and more.
- **High-Level Adapter Interface**: A collection of simplified "one-liner" functions for common tasks like GIC calculation, fault analysis, and voltage violation detection.
- **Native Pandas Integration**: Every data retrieval operation returns a Pandas DataFrame or Series, enabling immediate analysis, filtering, and visualization.
- **Advanced Analysis Apps**: Built-in specialized modules for Network topology analysis, Geomagnetically Induced Currents (GIC), and Forced Oscillation detection.

Installation
------------

For local development and the latest features, install the package in editable mode from the root directory:

.. code-block:: bash

    python -m pip install esapp -e .


Documentation
-------------

For a comprehensive tutorial, usage guides, and the full API reference, please visit our `documentation website <https://esapp.readthedocs.io/>`_.

Usage Example
-------------

Here is a quick example of how ESA++ simplifies data access and power flow analysis.

.. code-block:: python

    from esapp import GridWorkBench
    from esapp.grid import *

    # Open Case
    wb = GridWorkBench("path/to/case.pwb")

    # Retrieve data 
    bus_data = wb[Bus, ["BusName", "BusPUVolt"]]

    # Solve power flow
    V = wb.pflow()

    # Do some action, write to PW
    violations = wb.find_violations(v_min=0.95)
    wb[Gen, "GenMW"] = 100.0

    # Save case
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

- **Object Field Access**: Reduce the time you spend searching for field names with ESA++ IDE typehints for objects and fields.
- **Matrix Extraction**: Retrieving Y-Bus, Jacobian, and GIC conductance matrices for external mathematical modeling.

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

Luke Lowery developed this module during his PhD studies at Texas A&M University. You can learn more on his `research page <https://lukelowry.github.io/>`_ or view his publications on `Google Scholar <https://scholar.google.com/citations?user=CTynuRMAAAAJ&hl=en>`_.

ESA++ is maintained by **Luke Lowery** and **Adam Birchfield** at Texas A&M University. You can explore more of our research at the `Birchfield Research Group <https://birchfield.engr.tamu.edu/>`_.

License
-------
Distributed under the `Apache License 2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_.
