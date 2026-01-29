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

An open-source Python toolkit for power system automation, providing a high-performance wrapper for PowerWorld's Simulator Automation Server (SimAuto). ESA++ transforms complex COM calls into intuitive, Pythonic operations.

Key Features
------------

- **Intuitive Indexing**: Access and modify grid components using familiar syntax (e.g., ``wb[Bus, "BusPUVolt"]``)
- **Comprehensive SimAuto Wrapper**: Full coverage of PowerWorld's API through modular mixins
- **Native Pandas Integration**: All data operations return DataFrames for immediate analysis
- **Transient Stability Support**: Fluent API for dynamic simulation with ``TS`` field intellisense
- **Advanced Analysis Apps**: Built-in modules for GIC, network topology, and forced oscillation detection

Quick Start
-----------

.. code-block:: bash

    pip install esapp

.. code-block:: python

    from esapp import GridWorkBench
    from esapp.components import Bus, Gen

    wb = GridWorkBench("path/to/case.pwb")

    # Read data
    voltages = wb[Bus, "BusPUVolt"]

    # Solve and analyze
    V = wb.pflow()
    violations = wb.violations(v_min=0.95)

    # Modify and save
    wb[Gen, "GenMW"] = 100.0
    wb.save()

Citation
--------

If you use this toolkit in your research, please cite:

.. code-block:: bibtex

    @article{esa2020,
      title={Easy SimAuto (ESA): A Python Package for PowerWorld Simulator Automation},
      author={Mao, Zeyu and Thayer, Brandon and Liu, Yijing and Birchfield, Adam},
      year={2020}
    }

Authors
-------

ESA++ is developed and maintained by **Luke Lowery** and **Adam Birchfield** at Texas A&M University.

- `Birchfield Research Group <https://birchfield.engr.tamu.edu/>`_
- `Luke Lowery's Research <https://lukelowry.github.io/>`_

License
-------

Distributed under the `Apache License 2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_.
