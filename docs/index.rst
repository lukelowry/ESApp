ESA++ Documentation
===================

.. This pulls the main description from your project's README file.
.. include:: ../README.rst
   :start-after: ====================================
   :end-before: Documentation

What is ESA++?
==============

**ESA++** (Easy SimAuto Plus Plus) is a comprehensive Python toolkit for power systems analysis built on top of PowerWorld Simulator. It provides an intuitive, Pythonic interface to PowerWorld's automation capabilities, enabling researchers, engineers, and analysts to conduct power flow analysis, contingency studies, optimization, and other grid operations programmatically.

Key Features
~~~~~~~~~~~~

PowerWorld v24
    Supports the latest PowerWorld Simulator version 24 with full SimAuto compatibility
Simple Data Access
    Use NumPy-style indexing (e.g., ``wb[Bus, "BusPUVolt"]``) to read and write power system data
High Coverage
    Full access to PowerWorld's SimAuto API via the SAW class with high reliability.
Pandas Integration
    All data retrievals return Pandas DataFrames or Series for easy analysis and manipulation

.. important::
   ESA++ requires a licensed installation of PowerWorld Simulator with SimAuto (COM interface) enabled.

.. toctree::
    :maxdepth: 2
    :caption: Get Started

    guide/install
    guide/examples
    guide/usage


.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/api

.. toctree::
   :maxdepth: 1
   :caption: Development & Testing

   dev/components 
   dev/tests