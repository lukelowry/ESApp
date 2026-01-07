ESA++ Documentation
===================

ESA++ is a high-performance, open-source power-grid toolkit designed to streamline the automation of PowerWorld Simulator. It serves as a "syntax-sugar" fork of the original Easy SimAuto (ESA) library, focusing on making data access and system manipulation as intuitive and Pythonic as possible.

Why ESA++?
----------

*   **Pythonic Syntax**: Use indexing and slicing to interact with grid components (e.g., ``wb[Bus, :]``).
*   **Pandas Native**: All data retrieval operations return Pandas DataFrames, making analysis and visualization seamless.
*   **Performance**: Optimized COM communication patterns to reduce overhead when dealing with large-scale models.
*   **Extensible**: A modular architecture that allows for easy integration of custom analysis scripts and "Apps".

Core Components
---------------

*   **GridWorkBench**: The primary entry point for loading cases and executing high-level commands.
*   **IndexTool**: The engine behind the intuitive data access syntax.
*   **Adapter**: A collection of helper functions for common tasks like power flow, GIC, and contingency analysis.
*   **SAW**: The underlying SimAuto Wrapper providing full access to the PowerWorld API.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   tutorial
   usage
   auto_examples/index

   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`