ESA++
=====

.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: https://opensource.org/licenses/Apache-2.0
   :alt: License

.. image:: https://img.shields.io/badge/python-3.9%2B-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.9+

.. image:: https://img.shields.io/badge/docs-Read%20the%20Docs-blue.svg
   :target: https://esapp.readthedocs.io/
   :alt: Documentation

.. image:: https://img.shields.io/badge/coverage-98%25-brightgreen.svg
   :alt: Coverage 98%

An open-source Python toolkit for power system automation, providing a
high-performance wrapper for PowerWorld's Simulator Automation Server
(SimAuto). Transforms complex COM calls into intuitive, Pythonic operations.

- **Intuitive Indexing** -- Access grid data with ``pw[Bus, "BusPUVolt"]`` syntax
- **Full SimAuto Coverage** -- All PowerWorld API functions through modular mixins
- **Pandas Integration** -- Every query returns a DataFrame
- **Pythonic Settings** -- Solver and GIC options as descriptor attributes (``pw.max_iterations = 250``)
- **Convenience Methods** -- Flows, overloads, PTDF/LODF, snapshot context manager, case summary
- **Transient Stability** -- Fluent API with ``TS`` field intellisense
- **Analysis Utilities** -- Built-in GIC, network topology, and contingency tools

Installation
------------

Requires Windows with PowerWorld Simulator (SimAuto enabled) and Python 3.9+.

.. code-block:: bash

    pip install esapp

Quick Example
-------------

.. code-block:: python

    from esapp import PowerWorld
    from esapp.components import *

    pw = PowerWorld("path/to/case.pwb")

    # Read data
    bus_data = pw[Bus, ["BusName", "BusPUVolt"]]

    # Solve power flow
    V = pw.pflow()

    # Inspect and modify
    violations = pw.violations(v_min=0.95)
    pw[Gen, "GenMW"] = 100.0

Documentation
-------------

Full tutorials, API reference, and examples at `esapp.readthedocs.io <https://esapp.readthedocs.io/>`_.

Testing
-------

.. code-block:: bash

    pip install .[test]
    pytest tests/

Citation
--------

.. code-block:: bibtex

    @article{esa2020,
      title={Easy SimAuto (ESA): A Python Package for PowerWorld Simulator Automation},
      author={Mao, Zeyu and Thayer, Brandon and Liu, Yijing and Birchfield, Adam},
      year={2020}
    }

Authors
-------

Developed by **Luke Lowery** and **Adam Birchfield** at Texas A&M University
(`Birchfield Research Group <https://birchfield.engr.tamu.edu/>`_). See my personal (`research website <https://lukelowry.github.io/>`_).

License
-------

Distributed under the `Apache License 2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_.
