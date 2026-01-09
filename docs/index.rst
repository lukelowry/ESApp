ESA++ Documentation
===================

.. This pulls the main description from your project's README file.
.. include:: ../README.rst
   :end-before: Documentation

What is ESA++?
==============

**ESA++** (Electric Systems Analysis Plus Plus) is a comprehensive Python toolkit for power systems analysis built on top of PowerWorld Simulator. It provides an intuitive, Pythonic interface to PowerWorld's automation capabilities, enabling researchers, engineers, and analysts to conduct power flow analysis, contingency studies, optimization, and other grid operations programmatically.

Key Features
~~~~~~~~~~~~

Simple Data Access
    Use NumPy-style indexing (e.g., ``wb[Bus, "BusPUVolt"]``) to read and write power system data
Comprehensive Analysis
    Power flow, contingency analysis, optimal power flow, transient stability, GIC analysis, and more
Network Analysis
    Extract and analyze system matrices (Y-Bus, incidence matrices) and topology
Automatic Code Generation
    Auto-generated component classes ensure compatibility with all PowerWorld field definitions
Pandas Integration
    All data retrieval returns standard Pandas DataFrames for seamless integration with Python data science tools
Rich Testing Suite
    Comprehensive unit and integration tests with real PowerWorld cases

.. important::
   ESA++ requires a licensed installation of PowerWorld Simulator with SimAuto (COM interface) enabled.

.. toctree::
    :maxdepth: 2
    :caption: User Guide

    guide/install
    examples
    guide/usage


.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api

.. toctree::
   :maxdepth: 1
   :caption: Development & Testing

   dev/components 
   dev/tests

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`